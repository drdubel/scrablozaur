from __future__ import annotations

import os
import random
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from scrablozaur import Board, Dawg, set_num_threads
from web.difficulty import DEFAULT_LEVEL, EngineMode, clamp_level, engine_mode, max_level_for
from web.engine import DEFAULT_LANGUAGE, get_pack

# smart_player is a standalone script-style package (like board_reader, see
# web/scan.py), not importable as a normal module -- add its dir to sys.path
# so its sibling-style internal imports (`from model import ...`) resolve.
SMART_PLAYER_SRC = Path(__file__).resolve().parent.parent / "smart_player"
if str(SMART_PLAYER_SRC) not in sys.path:
    sys.path.insert(0, str(SMART_PLAYER_SRC))

from board_features import encode_board  # noqa: E402
from player import DEFAULT_LEAVE_WEIGHT, choose_move, leave_values, remove_used  # noqa: E402
from sim_player import choose_move_sim, get_net  # noqa: E402
from rules import (  # noqa: E402
    Streaks,
    TurnResult,
    apply_end_of_game_scoring,
    went_out,
)
from strategy import pick_by_rank, rank_window  # noqa: E402

# Rollouts per candidate when a human asks for `sim`-sorted suggestions. Lower
# than the bot's own 200 because someone is waiting on the response: at 100 the
# ordering is stable and the call lands near half a second.
SUGGEST_SIM_ITERATIONS = 100

# The endgame search is bounded by a node budget. The CLI's 300k default has a
# measured 9.4s worst case -- fine for a benchmark, not for someone waiting on a
# web request. The median endgame needs ~16k nodes, so this keeps essentially
# all of the strength while bounding the tail.
WEB_ENDGAME_NODES = 50_000


class GameMode(str, Enum):
    SANDBOX = "sandbox"
    SANDBOX_AUTO = "sandbox_auto"
    COMPETITIVE = "competitive"



@dataclass
class TileBag:
    tiles: list[str]

    @classmethod
    def full(cls, spec) -> TileBag:
        """A shuffled bag for `spec`'s distribution. The counts come from
        `languages/<code>.json`, the same file the engine's own bag is built
        from -- this used to be a second hand-maintained copy."""
        bag = list(spec.bag)
        random.shuffle(bag)
        return cls(tiles=bag)

    def draw(self, n: int) -> list[str]:
        n = min(n, len(self.tiles))
        drawn, self.tiles = self.tiles[:n], self.tiles[n:]
        return drawn

    def exchange(self, return_tiles: list[str]) -> list[str]:
        """Return `return_tiles` to the bag, shuffle, then draw the same
        count of new tiles. Caller must check Board.can_exchange first."""
        self.tiles.extend(return_tiles)
        random.shuffle(self.tiles)
        return self.draw(len(return_tiles))

    def remaining(self) -> int:
        return len(self.tiles)


@dataclass
class ComputerMoveInfo:
    word: str
    score: int
    row: int
    col: int
    horizontal: bool
    passed: bool = False


@dataclass
class Player:
    name: str
    is_computer: bool
    score: int = 0
    letters: str = ""
    # Custom difficulty level (see web/difficulty.py): 1 = weakest, 10 = the
    # Monte-Carlo player, and every step in between actually exists -- this
    # used to be a four-value enum with unadjustable gaps.
    difficulty: int = DEFAULT_LEVEL


@dataclass
class UndoEntry:
    # A real deep copy of the position, not a grid to rebuild from. The
    # `str()` -> `from_grid()` round trip this used to do loses the engine's
    # own bag, has to be told about blanks separately, and rejects positions
    # outright when a played blank makes a letter appear more often than the
    # distribution allows.
    board: Board
    player_scores: list[int]
    player_letters: list[str]
    current_player_idx: int
    is_first_move: bool
    move_number: int
    tile_bag_tiles: list[str] | None
    last_computer_move: ComputerMoveInfo | None
    tile_owners: list[list[int | None]]
    streaks: tuple[int, int]


@dataclass
class GameSession:
    session_id: str
    board: Board
    players: list[Player]
    #: Which language this game is played in. Fixed at creation -- the board,
    #: the bag and the dictionary all have to agree, so it cannot change later.
    language: str = DEFAULT_LANGUAGE
    current_player_idx: int = 0
    is_first_move: bool = True
    move_number: int = 0
    move_history: list[UndoEntry] = field(default_factory=list)
    game_mode: GameMode = GameMode.SANDBOX
    tile_bag: TileBag | None = None
    last_computer_move: ComputerMoveInfo | None = None
    tile_owners: list[list[int | None]] = field(default_factory=lambda: [[None] * 15 for _ in range(15)])
    game_over: bool = False
    streaks: Streaks = field(default_factory=Streaks)
    passed_players: set[int] = field(default_factory=set)
    last_move_rating: int | None = None

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def advance_turn(self) -> None:
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        self.move_number += 1

    def board_grid(self) -> list[list[str]]:
        return [row.split(" ") for row in str(self.board).strip().split("\n")]

    def push_undo(self) -> None:
        self.move_history.append(
            UndoEntry(
                board=self.board.copy(),
                player_scores=[p.score for p in self.players],
                player_letters=[p.letters for p in self.players],
                current_player_idx=self.current_player_idx,
                is_first_move=self.is_first_move,
                move_number=self.move_number,
                tile_bag_tiles=list(self.tile_bag.tiles) if self.tile_bag else None,
                last_computer_move=self.last_computer_move,
                tile_owners=[row[:] for row in self.tile_owners],
                streaks=(self.streaks.no_play, self.streaks.no_score),
            )
        )

    def pop_undo(self) -> bool:
        if not self.move_history:
            return False
        entry = self.move_history.pop()
        self.board = entry.board
        for i, p in enumerate(self.players):
            p.score = entry.player_scores[i]
            p.letters = entry.player_letters[i]
        self.current_player_idx = entry.current_player_idx
        self.is_first_move = entry.is_first_move
        self.move_number = entry.move_number
        if entry.tile_bag_tiles is not None and self.tile_bag:
            self.tile_bag.tiles = entry.tile_bag_tiles
        self.last_computer_move = entry.last_computer_move
        self.tile_owners = [row[:] for row in entry.tile_owners]
        self.streaks = Streaks(*entry.streaks)
        self.game_over = False
        return True

    def record_placement(self, word: str, row: int, col: int, horizontal: bool, player_idx: int) -> None:
        """Mark newly placed cells as owned by player_idx."""
        pre_grid = self.board_grid()
        for i, _ in enumerate(word):
            r = row if horizontal else row + i
            c = col + i if horizontal else col
            if pre_grid[r][c] == "-":
                self.tile_owners[r][c] = player_idx


_DEFAULT_PLAYERS = [
    Player(name="Gracz", is_computer=False),
    Player(name="Komputer", is_computer=True),
]


def _deal_new_game(players: list[Player], game_mode: GameMode, pack) -> GameSession:
    """Build a fresh GameSession for *players*, dealing a real bag + random
    racks for the modes that use one. Shared by SessionStore.create (a real,
    registered game) and run_benchmark (ephemeral simulated games that never
    touch the session store)."""
    tile_bag: TileBag | None = None
    first_player_idx = 0

    # COMPETITIVE (1 human + 1 computer) and SANDBOX_AUTO (2-4 computers)
    # both play with a real bag and random racks -- only the referee-style
    # plain SANDBOX mode has no bag at all.
    board = Board(pack.lang)
    if game_mode in (GameMode.COMPETITIVE, GameMode.SANDBOX_AUTO):
        tile_bag = TileBag.full(pack.spec)
        # Standard rule: each player draws one tile, closest to 'A'
        # (blank beats everything) goes first; drawn tiles go back to
        # the bag and get reshuffled in before dealing real racks.
        draws = tile_bag.draw(len(players))
        first_player_idx = board.first_draw_winner(draws)
        tile_bag.tiles.extend(draws)
        random.shuffle(tile_bag.tiles)
        for p in players:
            p.letters = "".join(tile_bag.draw(7))

    return GameSession(
        session_id=str(uuid.uuid4()),
        board=board,
        players=players,
        language=pack.code,
        current_player_idx=first_player_idx,
        game_mode=game_mode,
        tile_bag=tile_bag,
    )


class SessionStore:
    _sessions: dict[str, GameSession] = {}

    @classmethod
    def create(
        cls,
        players: list[Player] | None = None,
        game_mode: GameMode = GameMode.SANDBOX,
        pack=None,
    ) -> GameSession:
        pack = pack or get_pack()
        # Clamp to what this language can actually field: without a trained
        # leave net there is no level 9 or 10 to give anyone.
        ceiling = max_level_for(pack.spec)
        player_list = [
            Player(p.name, p.is_computer, difficulty=clamp_level(p.difficulty, ceiling))
            for p in (players or _DEFAULT_PLAYERS)
        ]
        session = _deal_new_game(player_list, game_mode, pack)
        cls._sessions[session.session_id] = session
        return session

    @classmethod
    def get(cls, sid: str) -> GameSession | None:
        return cls._sessions.get(sid)

    @classmethod
    def delete(cls, sid: str) -> None:
        cls._sessions.pop(sid, None)


# ── Tile management ───────────────────────────────────────────────────────────


def validate_rack_for_word(
    rack: str,
    word: str,
    board_grid: list[list[str]],
    row: int,
    col: int,
    horizontal: bool,
) -> bool:
    """Return True if rack contains tiles needed to place new letters in word."""
    rack_chars = list(rack)
    for i, ch in enumerate(word):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if board_grid[r][c] != "-":
            continue
        if ch in rack_chars:
            rack_chars.remove(ch)
        elif "?" in rack_chars:
            rack_chars.remove("?")
        else:
            return False
    return True


def rack_contains(rack: str, letters: str) -> bool:
    """Return True if rack contains at least the exact tiles in *letters*
    (no blank substitution — exchanging a blank means giving up that exact
    tile, not standing in for something else)."""
    rack_chars = list(rack)
    for ch in letters:
        if ch not in rack_chars:
            return False
        rack_chars.remove(ch)
    return True


def _tiles_used_for_word(
    rack: str,
    word: str,
    board_grid: list[list[str]],
    row: int,
    col: int,
    horizontal: bool,
) -> list[str]:
    """Which tiles the play takes off the rack: one entry per newly covered
    square, in placement order, `'?'` where a blank has to stand in for a
    letter the rack doesn't hold.

    Same shape the engine's own `Board.get_best_words` returns, and what
    `Board.place_word` needs in order to remember which squares hold blanks.
    """
    remaining = list(rack)
    used: list[str] = []
    for i, ch in enumerate(word):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if board_grid[r][c] != "-":
            continue
        if ch in remaining:
            remaining.remove(ch)
            used.append(ch)
        else:
            remaining.remove("?")
            used.append("?")
    return used


def _leave_after_word(
    rack: str,
    word: str,
    board_grid: list[list[str]],
    row: int,
    col: int,
    horizontal: bool,
) -> str:
    """Return the rack remaining after playing word at (row, col), without
    mutating rack. Cells already filled on the board don't consume a rack
    tile; a letter not held literally is assumed to come from a blank."""
    remaining = list(rack)
    for tile in _tiles_used_for_word(rack, word, board_grid, row, col, horizontal):
        remaining.remove(tile)
    return "".join(remaining)


def _deduct_tiles(
    player: Player,
    word: str,
    board_grid: list[list[str]],
    row: int,
    col: int,
    horizontal: bool,
) -> None:
    player.letters = _leave_after_word(player.letters, word, board_grid, row, col, horizontal)


def _refill_rack(session: GameSession, player: Player) -> None:
    if not session.tile_bag:
        return
    needed = 7 - len(player.letters)
    if needed > 0:
        player.letters += "".join(session.tile_bag.draw(needed))


# ── End of game (COMPETITIVE only — SANDBOX has no real bag/opponent) ────────

# Standard rule: the game ends once nobody has played a word for this many
# consecutive turns in a row. Only a genuine no-action turn (skip, or a
def _check_game_over(session: GameSession, just_played_idx: int) -> None:
    """Call after every real turn (move/pass/exchange) in COMPETITIVE mode.

    The rules themselves -- when a streak ends a game, and how the final rack
    adjustment works -- come from `src/rules.py`, so an interactive game ends on
    exactly the same conditions as one played through `arena.py`. This file used
    to carry its own copy with no scoreless-turn cap at all, which meant two
    players exchanging at each other could keep a game alive indefinitely.
    """
    if session.game_over or session.tile_bag is None:
        return
    player = session.players[just_played_idx]
    if went_out(player, session.tile_bag.remaining()):
        apply_end_of_game_scoring(session.board, session.players, went_out_idx=just_played_idx)
        session.game_over = True
    elif session.streaks.exhausted:
        apply_end_of_game_scoring(session.board, session.players, went_out_idx=None)
        session.game_over = True


def _pick_engine_move(session: GameSession, dawg: Dawg, mode: EngineMode) -> dict | None:
    """The top levels' move choice, delegated to `smart_player`.

    SMART and SIM do not pick from the suggestion list at all -- they run the
    same decision the CLI players run, so there is exactly one implementation
    of it. This replaced a hand-rolled copy here that had drifted: it ignored
    `leave_weight`, scored candidates one at a time instead of batching, and
    never searched the endgame.

    `bag_remaining` comes from `session.tile_bag`, not from the engine Board --
    the web app keeps its own bag and never draws from the Board's, so the
    Board's count is meaningless here and would leave the endgame search
    permanently switched off (or wrongly switched on).
    """
    rack = session.current_player.letters
    bag_remaining = session.tile_bag.remaining() if session.tile_bag else 0
    # Explicitly this language's net. `smart_player`'s own default comes from a
    # process-global language, which is meaningless here: one server process
    # serves games in every installed language.
    model_path = _leave_net_path(session.language)

    if mode is EngineMode.SIM:
        choice = choose_move_sim(
            session.board,
            dawg,
            rack,
            bag_remaining,
            model_path=model_path,
            endgame_max_nodes=WEB_ENDGAME_NODES,
        )
    else:
        choice = choose_move(
            session.board,
            dawg,
            rack,
            bag_remaining,
            model_path=model_path,
            endgame_max_nodes=WEB_ENDGAME_NODES,
        )
    if not choice.word:
        return None
    row, col, horizontal = choice.position
    return {
        "word": choice.word,
        "score": choice.score,
        "row": row,
        "col": col,
        "horizontal": horizontal,
        "cells": [
            (row, col + i) if horizontal else (row + i, col) for i in range(len(choice.word))
        ],
    }


def computer_auto_play(session: GameSession, dawg: Dawg) -> ComputerMoveInfo:
    """Play a move for the computer at its own difficulty level, then advance the turn."""
    player_idx = session.current_player_idx
    level = session.current_player.difficulty
    mode = engine_mode(level)

    if mode is not EngineMode.RANKED:
        # The top levels make the full decision themselves rather than sampling
        # a shortlist -- they weigh the leave, and search the endgame outright.
        sug = _pick_engine_move(session, dawg, mode)
    else:
        # Every other level ranks candidates exactly as StrategicPlayer does
        # and then reaches further down the list, so a weaker bot plays a *worse
        # move* rather than a differently-chosen one. `pick_by_rank` is shared
        # with `RankedPlayer`, so a level can actually be benchmarked.
        _, worst_rank = rank_window(level)
        suggestions = get_suggestions(session, dawg, n=max(worst_rank, 1))
        sug = pick_by_rank(suggestions, level) if suggestions else None

    if sug is None:
        session.streaks.record(TurnResult.NO_ACTION)
        _check_game_over(session, player_idx)
        if not session.game_over:
            session.advance_turn()
        return ComputerMoveInfo(word="", score=0, row=0, col=0, horizontal=True, passed=True)

    word, row, col, horizontal = sug["word"], sug["row"], sug["col"], sug["horizontal"]

    grid = session.board_grid()
    used = _tiles_used_for_word(session.current_player.letters, word, grid, row, col, horizontal)
    session.record_placement(word, row, col, horizontal, player_idx)
    session.board.place_word(word, row, col, horizontal, used)
    session.current_player.score += sug["score"]
    session.is_first_move = False

    _deduct_tiles(session.current_player, word, grid, row, col, horizontal)
    _refill_rack(session, session.current_player)
    session.streaks.record(TurnResult.PLAYED)
    _check_game_over(session, player_idx)
    if not session.game_over:
        session.advance_turn()

    return ComputerMoveInfo(word=word, score=sug["score"], row=row, col=col, horizontal=horizontal)


# ── Suggestion generation ─────────────────────────────────────────────────────


def _engine_suggestions(board: Board, dawg: Dawg, letters: str, n: int) -> list[dict]:
    """Top `n` plays, straight from the engine's GADDAG generator.

    This used to be two hand-rolled searches: a centre-covering scan for the
    opening, and a per-board-span scan for everything after. The span scan kept
    only the *best word per span* before ranking, so it systematically missed
    plays -- measured against the engine over 1167 real positions, it returned a
    lower top score in 5.1% of them, losing 35 points on average when it did, or
    about 1.8 points per move overall. That handicapped every difficulty level
    and both human-facing hint endpoints.

    `Board.get_best_words` enumerates every legal play and returns the true top
    `n`, and picks the opening path itself when the board is empty, so one call
    replaces both.
    """
    return [
        {
            "word": word,
            "score": score,
            "row": row,
            "col": col,
            "horizontal": horizontal,
            "cells": [
                (row, col + i) if horizontal else (row + i, col) for i in range(len(word))
            ],
        }
        for word, score, (row, col, horizontal), _used in board.get_best_words(dawg, letters, n)
    ]


SORT_MODES = ("score", "smart", "sim")
#: Orderings that consult the learned rack-leave evaluator, and so are only
#: available in a language that has one trained.
LEAVE_NET_SORTS = ("smart", "sim")


def _leave_net_path(language: str) -> str:
    """This language's trained checkpoint. Callers must have checked
    `has_leave_net` first -- there is deliberately no fallback to another
    language's net, which would read the wrong features."""
    spec = get_pack(language).spec
    if spec.leave_net is None:
        raise ValueError(f"language '{language}' has no trained leave net")
    return str(spec.leave_net.checkpoint)


def has_leave_net(language: str) -> bool:
    """Whether `language` has a trained rack-leave evaluator.

    The `smart` and `sim` orderings both consult one, and a net is only ever
    valid for the tile alphabet it was trained on -- feeding an English rack to
    a Polish net does not fail, it silently drops the letters Polish lacks and
    scores the rest against the wrong distribution.
    """
    return get_pack(language).spec.leave_net is not None


def rank_suggestions(
    board: Board, dawg: Dawg, letters: str, n: int, sort: str = "score"
) -> list[dict]:
    """Top `n` plays, ordered by `sort`.

    The three modes answer genuinely different questions, which is why offering
    them is worth the cost:

    - `score`   what scores most this turn.
    - `smart`   what is worth most counting the rack it leaves behind, which is
                what separates two similar-scoring plays.
    - `sim`     what survives the opponent's reply -- the only one that can see
                that a play scoring four more points opens the triple-word lane.

    Each result carries `value`, the number it was ranked by, so the UI can show
    what it is being told rather than an unexplained reordering.
    """
    if sort not in SORT_MODES:
        raise ValueError(f"unknown sort {sort!r} (expected one of {SORT_MODES})")
    if sort in LEAVE_NET_SORTS and not has_leave_net(board.lang.code):
        raise ValueError(
            f"'{sort}' ordering needs a trained leave net, and '{board.lang.code}' has none"
        )
    if not letters:
        return []

    if sort == "sim":
        # One simulation ranks the whole set; `batch == iterations` keeps every
        # candidate rather than only the survivors of pruning.
        results = board.simulate(
            dawg,
            get_net(_leave_net_path(board.lang.code), lang=board.lang),
            letters,
            candidates=max(n, 1),
            iterations=SUGGEST_SIM_ITERATIONS,
            plies=1,
            batch=SUGGEST_SIM_ITERATIONS,
        )
        return [
            {
                "word": word,
                "score": score,
                "row": row,
                "col": col,
                "horizontal": horizontal,
                "cells": [(row, col + i) if horizontal else (row + i, col) for i in range(len(word))],
                "value": round(equity, 1),
            }
            for word, score, (row, col, horizontal), _used, equity, _se, _n in results
        ][:n]

    suggestions = _engine_suggestions(board, dawg, letters, n)
    if sort == "score" or not suggestions:
        for s in suggestions:
            s["value"] = float(s["score"])
        return suggestions

    # `smart`: score plus what the play leaves behind, the same quantity
    # `choose_move` ranks by, so the list agrees with what the bot would do.
    grid = [row.split(" ") for row in str(board).strip().split("\n")]
    board_features = encode_board(board)
    unseen = sum(board.unseen_tile_counts(letters))
    leaves = [
        remove_used(
            letters,
            _tiles_used_for_word(letters, s["word"], grid, s["row"], s["col"], s["horizontal"]),
        )
        for s in suggestions
    ]
    values = leave_values(leaves, unseen, board_features, _leave_net_path(board.lang.code))
    for s, leave_value in zip(suggestions, values):
        s["value"] = round(s["score"] + DEFAULT_LEAVE_WEIGHT * leave_value, 1)
    suggestions.sort(key=lambda s: -s["value"])
    return suggestions


def get_suggestions(session: GameSession, dawg: Dawg, n: int = 10, sort: str = "score") -> list[dict]:
    return rank_suggestions(session.board, dawg, session.current_player.letters, n, sort)


def get_suggestions_for_letters(
    session: GameSession, dawg: Dawg, letters: str, n: int = 10, sort: str = "score"
) -> list[dict]:
    """Like get_suggestions but uses the supplied letters instead of current player's rack."""
    return rank_suggestions(session.board, dawg, letters, n, sort)


def compute_move_rating(session: GameSession, dawg: Dawg, letters: str, actual_score: int) -> int:
    """Rate actual_score 0–100 relative to best and worst possible scores with *letters*."""
    all_moves = get_suggestions_for_letters(session, dawg, letters, n=999)
    if not all_moves:
        return 100
    scores = [m["score"] for m in all_moves]
    best = max(scores)
    worst = min(scores)
    if best == worst:
        return 100
    rating = (actual_score - worst) / (best - worst) * 100
    return max(0, min(100, round(rating)))


# ── Benchmark simulation (SANDBOX_AUTO, no human ever involved) ──────────────

# Defensive cap on moves per simulated game -- the no-play/scoreless limits and
# going-out-with-an-empty-bag always end a real game long before this, it
# only guards against an unforeseen non-terminating edge case.
MAX_BENCHMARK_GAME_MOVES = 200


@dataclass
class BenchmarkMoveRecord:
    player_idx: int
    word: str
    score: int
    row: int
    col: int
    horizontal: bool
    passed: bool
    board: list[list[str]]
    scores_after: list[int]
    letters_after: list[str]
    tile_owners: list[list[int | None]]


@dataclass
class BenchmarkPlayerStats:
    name: str
    difficulty: int
    games_played: int = 0
    wins: int = 0
    ties: int = 0
    total_score: int = 0
    high_score: int = 0
    low_score: int = 0
    words_played: int = 0
    total_word_score: int = 0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.games_played if self.games_played else 0.0

    @property
    def avg_word_score(self) -> float:
        return self.total_word_score / self.words_played if self.words_played else 0.0


@dataclass
class BenchmarkBestGame:
    winner_name: str
    winner_score: int
    final_players: list[Player]
    moves: list[BenchmarkMoveRecord]


@dataclass
class BenchmarkResult:
    games_played: int
    duration_ms: int
    player_stats: list[BenchmarkPlayerStats]
    best_game: BenchmarkBestGame | None
    avg_game_length: float
    longest_word: str | None
    longest_word_score: int | None
    highest_single_move_score: int | None


@dataclass
class _SimulatedGame:
    players: list[Player]
    moves: list[BenchmarkMoveRecord]
    move_count: int


# Each simulated game runs in its own process (games are independent, so this
# is embarrassingly parallel). `Dawg`/`Board` are Rust (pyo3) objects that
# can't be pickled across the process boundary, so instead of passing one in,
# each worker resolves the language itself and `web.engine.get_pack` caches the
# result process-locally -- mirrors how src/main.py's `benchmark` gets a fresh
# `d = Dawg(...)` per worker for free via module re-import under spawn.
def _init_worker(engine_threads: int) -> None:
    # Split the cores between workers instead of letting every worker build the
    # engine's default min(8, cores) rayon pool -- with one pool per worker that
    # oversubscribes the box several times over (and every thread costs stack +
    # arena memory). Same call as smart_player/arena.py.
    try:
        set_num_threads(engine_threads)
    except RuntimeError:
        pass  # pool already built -- nothing to do in a fresh worker anyway


def _simulate_game(player_specs: list[tuple[str, int]], language: str) -> _SimulatedGame:
    # `get_pack` caches per process, so the first game in a worker pays the
    # dictionary load and the rest are free -- and a worker that has served two
    # languages simply holds both, which is cheaper than a pool per language.
    pack = get_pack(language)
    players = [Player(name=name, is_computer=True, difficulty=level) for name, level in player_specs]
    session = _deal_new_game(players, GameMode.SANDBOX_AUTO, pack)

    moves: list[BenchmarkMoveRecord] = []
    move_count = 0
    while not session.game_over and move_count < MAX_BENCHMARK_GAME_MOVES:
        player_idx = session.current_player_idx
        move = computer_auto_play(session, pack.dawg)
        move_count += 1
        moves.append(
            BenchmarkMoveRecord(
                player_idx=player_idx,
                word=move.word,
                score=move.score,
                row=move.row,
                col=move.col,
                horizontal=move.horizontal,
                passed=move.passed,
                board=session.board_grid(),
                scores_after=[p.score for p in players],
                letters_after=[p.letters for p in players],
                tile_owners=[row[:] for row in session.tile_owners],
            )
        )

    return _SimulatedGame(players=players, moves=moves, move_count=move_count)


# A benchmark worker is *expensive*: re-importing this module pulls in torch and
# the leave model, and every worker loads its own copy of the DAWG + GADDAG
# (~300 MB RSS per process, measured). Defaulting to os.cpu_count() workers and
# keeping them alive for the lifetime of the server therefore pinned several GB
# after the first benchmark and never gave it back. So: cap the pool well below
# the core count and tear it down once it has been idle for a while.
BENCHMARK_WORKERS = max(1, int(os.environ.get("SCRABLOZAUR_BENCH_WORKERS") or 0)
                        or min(4, os.cpu_count() or 1))
# Rayon threads *inside* each worker (see _init_worker): share the cores out
# rather than giving every worker a full-width pool.
BENCHMARK_ENGINE_THREADS = max(1, int(os.environ.get("SCRABLOZAUR_BENCH_THREADS") or 0)
                               or min(4, (os.cpu_count() or 1) // BENCHMARK_WORKERS))
# Seconds of idleness after which the pool is shut down and its memory returned
# to the OS. 0 (or less) means "shut down as soon as a run finishes".
BENCHMARK_POOL_IDLE_TIMEOUT = float(os.environ.get("SCRABLOZAUR_BENCH_POOL_IDLE") or 120.0)

_executor: ProcessPoolExecutor | None = None
_executor_lock = threading.Lock()
_executor_users = 0
_executor_reaper: threading.Timer | None = None


def _shutdown_pool_locked() -> ProcessPoolExecutor | None:
    """Detach the current pool (caller must hold _executor_lock, and must do
    the actual shutdown outside it)."""
    global _executor, _executor_reaper
    if _executor_reaper is not None:
        _executor_reaper.cancel()
        _executor_reaper = None
    executor, _executor = _executor, None
    return executor


def _reap_idle_pool() -> None:
    with _executor_lock:
        if _executor_users:  # a run started while the timer was firing
            return
        executor = _shutdown_pool_locked()
    if executor is not None:
        executor.shutdown(wait=False)


def shutdown_benchmark_pool() -> None:
    """Drop the worker pool now (server shutdown / tests)."""
    with _executor_lock:
        executor = _shutdown_pool_locked()
    if executor is not None:
        executor.shutdown(wait=False)


@contextmanager
def _benchmark_pool() -> Iterator[ProcessPoolExecutor]:
    """Lazily-created pool shared by concurrent benchmark runs so back-to-back
    benchmarks (e.g. clicking "Run again") don't pay process-spawn +
    Dawg-load cost every time -- but which is reaped once nobody is using it,
    so idle workers don't sit on hundreds of MB each."""
    global _executor, _executor_users, _executor_reaper
    with _executor_lock:
        if _executor_reaper is not None:
            _executor_reaper.cancel()
            _executor_reaper = None
        if _executor is None:
            _executor = ProcessPoolExecutor(
                max_workers=BENCHMARK_WORKERS,
                initializer=_init_worker,
                initargs=(BENCHMARK_ENGINE_THREADS,),
            )
        _executor_users += 1
        executor = _executor
    try:
        yield executor
    finally:
        idle_executor = None
        with _executor_lock:
            _executor_users -= 1
            if _executor_users == 0:
                if BENCHMARK_POOL_IDLE_TIMEOUT > 0:
                    _executor_reaper = threading.Timer(
                        BENCHMARK_POOL_IDLE_TIMEOUT, _reap_idle_pool
                    )
                    _executor_reaper.daemon = True
                    _executor_reaper.start()
                else:
                    idle_executor = _shutdown_pool_locked()
        if idle_executor is not None:
            idle_executor.shutdown(wait=False)


def run_benchmark(
    player_specs: list[tuple[str, int]],
    games: int,
    on_game_done: Callable[[int], None] | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> BenchmarkResult:
    """Simulate *games* full SANDBOX_AUTO games with the given (name,
    level) players end-to-end using the same engine primitives as a
    live game (_deal_new_game + computer_auto_play), never touching
    SessionStore. Games are independent, so they're farmed out across a
    process pool (see _benchmark_pool) instead of run one at a time. Returns
    aggregate per-player stats plus the full move-by-move detail of whichever
    single game had the highest final score for any one player."""
    start = time.perf_counter()
    stats = [BenchmarkPlayerStats(name=name, difficulty=level) for name, level in player_specs]
    best_game: BenchmarkBestGame | None = None
    best_score = -1
    total_moves = 0
    longest_word: str | None = None
    longest_word_score = 0
    highest_single_move_score = 0

    with _benchmark_pool() as executor:
        pending: set[Future[_SimulatedGame]] = {
            executor.submit(_simulate_game, player_specs, language) for _ in range(games)
        }
        try:
            # `as_completed` gets its own copy of the futures and releases each
            # one as it yields it; `pending` is drained in step so that a
            # finished game's move-by-move detail (a full board + tile-owner
            # grid *per move*, i.e. megabytes per game) is freed as soon as it
            # has been folded into the stats, instead of every game's detail
            # piling up until the whole run ends.
            for done, future in enumerate(as_completed(set(pending)), start=1):
                pending.discard(future)
                sim = future.result()
                players, moves = sim.players, sim.moves
                total_moves += sim.move_count

                for move in moves:
                    if not move.passed:
                        s = stats[move.player_idx]
                        s.words_played += 1
                        s.total_word_score += move.score
                        highest_single_move_score = max(highest_single_move_score, move.score)
                        if longest_word is None or len(move.word) > len(longest_word):
                            longest_word, longest_word_score = move.word, move.score

                top_score = max(p.score for p in players)
                winners = [p for p in players if p.score == top_score]
                for i, p in enumerate(players):
                    s = stats[i]
                    s.games_played += 1
                    s.total_score += p.score
                    if s.games_played == 1:
                        s.high_score, s.low_score = p.score, p.score
                    else:
                        s.high_score = max(s.high_score, p.score)
                        s.low_score = min(s.low_score, p.score)
                    if p.score == top_score:
                        if len(winners) > 1:
                            s.ties += 1
                        else:
                            s.wins += 1

                if top_score > best_score:
                    best_score = top_score
                    best_game = BenchmarkBestGame(
                        winner_name=winners[0].name if len(winners) == 1 else "Remis",
                        winner_score=top_score,
                        final_players=list(players),
                        moves=moves,
                    )

                if on_game_done:
                    on_game_done(done)

                # Only the best game's detail is kept; drop this one's now
                # rather than waiting for the next loop iteration to rebind.
                del sim, future, players, moves
        finally:
            for f in pending:
                f.cancel()
            pending.clear()

    duration_ms = int((time.perf_counter() - start) * 1000)
    return BenchmarkResult(
        games_played=games,
        duration_ms=duration_ms,
        player_stats=stats,
        best_game=best_game,
        avg_game_length=total_moves / games if games else 0.0,
        longest_word=longest_word,
        longest_word_score=longest_word_score if longest_word else None,
        highest_single_move_score=highest_single_move_score or None,
    )
