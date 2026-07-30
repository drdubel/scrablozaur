from __future__ import annotations

import random
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from scrablozaur import Board, Dawg
from web.difficulty import DEFAULT_LEVEL, EngineMode, clamp_level, engine_mode
from web.engine import DAWG_PATH, GADDAG_PATH

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



# Polish Scrabble tile distribution: letter → count (100 tiles total)
TILE_COUNTS: dict[str, int] = {
    "a": 9,
    "ą": 1,
    "b": 2,
    "c": 3,
    "ć": 1,
    "d": 3,
    "e": 7,
    "ę": 1,
    "f": 1,
    "g": 2,
    "h": 2,
    "i": 8,
    "j": 2,
    "k": 3,
    "l": 3,
    "ł": 2,
    "m": 3,
    "n": 5,
    "ń": 1,
    "o": 6,
    "ó": 1,
    "p": 3,
    "r": 4,
    "s": 4,
    "ś": 1,
    "t": 3,
    "u": 2,
    "w": 4,
    "y": 4,
    "z": 5,
    "ź": 1,
    "ż": 1,
    "?": 2,
}


@dataclass
class TileBag:
    tiles: list[str]

    @classmethod
    def full(cls) -> TileBag:
        bag = [letter for letter, count in TILE_COUNTS.items() for _ in range(count)]
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


def _deal_new_game(players: list[Player], game_mode: GameMode) -> GameSession:
    """Build a fresh GameSession for *players*, dealing a real bag + random
    racks for the modes that use one. Shared by SessionStore.create (a real,
    registered game) and run_benchmark (ephemeral simulated games that never
    touch the session store)."""
    tile_bag: TileBag | None = None
    first_player_idx = 0

    # COMPETITIVE (1 human + 1 computer) and SANDBOX_AUTO (2-4 computers)
    # both play with a real bag and random racks -- only the referee-style
    # plain SANDBOX mode has no bag at all.
    if game_mode in (GameMode.COMPETITIVE, GameMode.SANDBOX_AUTO):
        tile_bag = TileBag.full()
        # Standard rule: each player draws one tile, closest to 'A'
        # (blank beats everything) goes first; drawn tiles go back to
        # the bag and get reshuffled in before dealing real racks.
        draws = tile_bag.draw(len(players))
        first_player_idx = Board.first_draw_winner(draws)
        tile_bag.tiles.extend(draws)
        random.shuffle(tile_bag.tiles)
        for p in players:
            p.letters = "".join(tile_bag.draw(7))

    return GameSession(
        session_id=str(uuid.uuid4()),
        board=Board(),
        players=players,
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
    ) -> GameSession:
        player_list = [
            Player(p.name, p.is_computer, difficulty=clamp_level(p.difficulty))
            for p in (players or _DEFAULT_PLAYERS)
        ]
        session = _deal_new_game(player_list, game_mode)
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
        apply_end_of_game_scoring(session.players, went_out_idx=just_played_idx)
        session.game_over = True
    elif session.streaks.exhausted:
        apply_end_of_game_scoring(session.players, went_out_idx=None)
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

    if mode is EngineMode.SIM:
        choice = choose_move_sim(
            session.board, dawg, rack, bag_remaining, endgame_max_nodes=WEB_ENDGAME_NODES
        )
    else:
        choice = choose_move(
            session.board, dawg, rack, bag_remaining, endgame_max_nodes=WEB_ENDGAME_NODES
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
    if not letters:
        return []

    if sort == "sim":
        # One simulation ranks the whole set; `batch == iterations` keeps every
        # candidate rather than only the survivors of pruning.
        results = board.simulate(
            dawg,
            get_net(),
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
    values = leave_values(leaves, unseen, board_features)
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
# every worker loads its own copy once at startup and keeps it in a
# process-local global -- mirrors how src/main.py's `benchmark` gets a fresh
# `d = Dawg(...)` per worker for free via module re-import under spawn.
_worker_dawg: Dawg | None = None


def _init_worker(dawg_path: str, gaddag_path: str) -> None:
    global _worker_dawg
    _worker_dawg = Dawg(dawg_path, gaddag_path)


def _simulate_game(player_specs: list[tuple[str, int]]) -> _SimulatedGame:
    assert _worker_dawg is not None, "worker executor initializer did not run"
    players = [Player(name=name, is_computer=True, difficulty=level) for name, level in player_specs]
    session = _deal_new_game(players, GameMode.SANDBOX_AUTO)

    moves: list[BenchmarkMoveRecord] = []
    move_count = 0
    while not session.game_over and move_count < MAX_BENCHMARK_GAME_MOVES:
        player_idx = session.current_player_idx
        move = computer_auto_play(session, _worker_dawg)
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


# Lazily-created, process-lifetime pool shared by every benchmark run so
# repeated benchmarks (e.g. clicking "Run again" on the website) don't pay
# process-spawn + Dawg-load cost every time.
_executor: ProcessPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ProcessPoolExecutor(
                initializer=_init_worker, initargs=(str(DAWG_PATH), str(GADDAG_PATH))
            )
        return _executor


def run_benchmark(
    player_specs: list[tuple[str, int]],
    games: int,
    on_game_done: Callable[[int], None] | None = None,
) -> BenchmarkResult:
    """Simulate *games* full SANDBOX_AUTO games with the given (name,
    level) players end-to-end using the same engine primitives as a
    live game (_deal_new_game + computer_auto_play), never touching
    SessionStore. Games are independent, so they're farmed out across a
    process pool (see _get_executor) instead of run one at a time. Returns
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

    executor = _get_executor()
    futures = [executor.submit(_simulate_game, player_specs) for _ in range(games)]
    for done, future in enumerate(as_completed(futures), start=1):
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
