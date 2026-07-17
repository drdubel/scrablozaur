from __future__ import annotations

import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from scrablozaur import Board, Dawg


class GameMode(str, Enum):
    SANDBOX = "sandbox"
    SANDBOX_AUTO = "sandbox_auto"
    COMPETITIVE = "competitive"


class Difficulty(str, Enum):
    EASY       = "easy"
    MEDIUM     = "medium"
    HARD       = "hard"
    IMPOSSIBLE = "impossible"


# Polish Scrabble tile distribution: letter → count (100 tiles total)
TILE_COUNTS: dict[str, int] = {
    "a": 9, "ą": 1, "b": 2, "c": 3, "ć": 1, "d": 3, "e": 7, "ę": 1,
    "f": 1, "g": 2, "h": 2, "i": 8, "j": 2, "k": 3, "l": 3, "ł": 2,
    "m": 3, "n": 5, "ń": 1, "o": 6, "ó": 1, "p": 3, "r": 4, "s": 4,
    "ś": 1, "t": 3, "u": 2, "w": 4, "y": 4, "z": 5, "ź": 1, "ż": 1,
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
    difficulty: Difficulty = Difficulty.HARD


@dataclass
class UndoEntry:
    board_grid: list[list[str]]
    player_scores: list[int]
    player_letters: list[str]
    current_player_idx: int
    is_first_move: bool
    move_number: int
    tile_bag_tiles: list[str] | None
    last_computer_move: ComputerMoveInfo | None
    tile_owners: list[list[int | None]]
    consecutive_no_play: int


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
    tile_owners: list[list[int | None]] = field(
        default_factory=lambda: [[None] * 15 for _ in range(15)]
    )
    game_over: bool = False
    consecutive_no_play: int = 0
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
        self.move_history.append(UndoEntry(
            board_grid=self.board_grid(),
            player_scores=[p.score for p in self.players],
            player_letters=[p.letters for p in self.players],
            current_player_idx=self.current_player_idx,
            is_first_move=self.is_first_move,
            move_number=self.move_number,
            tile_bag_tiles=list(self.tile_bag.tiles) if self.tile_bag else None,
            last_computer_move=self.last_computer_move,
            tile_owners=[row[:] for row in self.tile_owners],
            consecutive_no_play=self.consecutive_no_play,
        ))

    def pop_undo(self) -> bool:
        if not self.move_history:
            return False
        entry = self.move_history.pop()
        self.board = Board(entry.board_grid)
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
        self.consecutive_no_play = entry.consecutive_no_play
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
        board=Board([["-"] * 15 for _ in range(15)]),
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
            Player(p.name, p.is_computer, difficulty=p.difficulty)
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


def _deduct_tiles(
    player: Player,
    word: str,
    board_grid: list[list[str]],
    row: int,
    col: int,
    horizontal: bool,
) -> None:
    rack = list(player.letters)
    for i, ch in enumerate(word):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if board_grid[r][c] != "-":
            continue
        if ch in rack:
            rack.remove(ch)
        else:
            rack.remove("?")
    player.letters = "".join(rack)


def _refill_rack(session: GameSession, player: Player) -> None:
    if not session.tile_bag:
        return
    needed = 7 - len(player.letters)
    if needed > 0:
        player.letters += "".join(session.tile_bag.draw(needed))


# ── End of game (COMPETITIVE only — SANDBOX has no real bag/opponent) ────────

# Standard rule: the game ends once nobody has played a word for this many
# consecutive turns in a row (pass/exchange both count as "no play").
CONSECUTIVE_NO_PLAY_LIMIT = 2


def _apply_end_of_game_scoring(session: GameSession, went_out_idx: int | None) -> None:
    """Standard end-of-game score adjustment. If `went_out_idx` is given,
    that player gains the summed rack value of every other player and is
    the only one who doesn't lose their own (they have none left); with no
    `went_out_idx` (game ended by mutual no-play), every player loses their
    own remaining rack value instead."""
    if went_out_idx is not None:
        others_value = sum(
            Board.rack_value(p.letters) for i, p in enumerate(session.players) if i != went_out_idx
        )
        session.players[went_out_idx].score += others_value
        for i, p in enumerate(session.players):
            if i != went_out_idx:
                p.score -= Board.rack_value(p.letters)
    else:
        for p in session.players:
            p.score -= Board.rack_value(p.letters)


def _check_game_over(session: GameSession, just_played_idx: int) -> None:
    """Call after every real turn (move/pass/exchange) in COMPETITIVE mode.
    Ends the game immediately if the player who just moved emptied their
    rack with an empty bag (going out), or if enough consecutive turns have
    passed with no one playing a word."""
    if session.game_over or session.tile_bag is None:
        return
    player = session.players[just_played_idx]
    if not player.letters and session.tile_bag.remaining() == 0:
        _apply_end_of_game_scoring(session, went_out_idx=just_played_idx)
        session.game_over = True
    elif session.consecutive_no_play >= CONSECUTIVE_NO_PLAY_LIMIT:
        _apply_end_of_game_scoring(session, went_out_idx=None)
        session.game_over = True


def _pick_by_difficulty(suggestions: list[dict], difficulty: Difficulty) -> dict:
    """Weighted-random choice: lower difficulty → higher weight for worse moves."""
    if difficulty == Difficulty.IMPOSSIBLE or len(suggestions) == 1:
        return suggestions[0]

    n = len(suggestions)
    scores = [s["score"] for s in suggestions]
    best, worst = max(scores), min(scores)

    if best == worst:
        return random.choice(suggestions)

    # Normalise each score to [0, 1] relative to range (0 = worst, 1 = best)
    norm = [(s - worst) / (best - worst) for s in scores]

    # Weight = normalised_score raised to an exponent:
    #   easy   → exponent = -2  (inverts so bad moves get high weight)
    #   medium → exponent =  0  (uniform — flat distribution)
    #   hard   → exponent =  3  (good moves get much higher weight)
    if difficulty == Difficulty.EASY:
        # Invert: weight proportional to (1 - norm)^2 so worst = highest weight
        weights = [(1.0 - v) ** 2 + 0.05 for v in norm]
    elif difficulty == Difficulty.MEDIUM:
        # Triangular peak around the middle of the range
        weights = [1.0 - abs(v - 0.5) * 1.5 + 0.1 for v in norm]
    else:  # HARD
        weights = [v ** 3 + 0.02 for v in norm]

    return random.choices(suggestions, weights=weights, k=1)[0]


def computer_auto_play(session: GameSession, dawg: Dawg) -> ComputerMoveInfo:
    """Play a move for the computer weighted by its own difficulty, then advance the turn."""
    player_idx = session.current_player_idx
    difficulty = session.current_player.difficulty
    # Fetch enough candidates so difficulty has real choices; cap at 30 for speed
    n_candidates = 1 if difficulty == Difficulty.IMPOSSIBLE else 30
    suggestions = get_suggestions(session, dawg, n=n_candidates)

    if not suggestions:
        session.consecutive_no_play += 1
        _check_game_over(session, player_idx)
        if not session.game_over:
            session.advance_turn()
        return ComputerMoveInfo(word="", score=0, row=0, col=0, horizontal=True, passed=True)

    sug = _pick_by_difficulty(suggestions, difficulty)
    word, row, col, horizontal = sug["word"], sug["row"], sug["col"], sug["horizontal"]

    grid = session.board_grid()
    session.record_placement(word, row, col, horizontal, player_idx)
    session.board.place_word(word, row, col, horizontal)
    session.current_player.score += sug["score"]
    session.is_first_move = False

    _deduct_tiles(session.current_player, word, grid, row, col, horizontal)
    _refill_rack(session, session.current_player)
    session.consecutive_no_play = 0
    _check_game_over(session, player_idx)
    if not session.game_over:
        session.advance_turn()

    return ComputerMoveInfo(
        word=word, score=sug["score"], row=row, col=col, horizontal=horizontal
    )


# ── Suggestion generation ─────────────────────────────────────────────────────

def get_suggestions(session: GameSession, dawg: Dawg, n: int = 10) -> list[dict]:
    letters = session.current_player.letters
    if not letters:
        return []
    if session.is_first_move:
        return _first_move_suggestions(session.board, dawg, letters, n)
    return _subsequent_suggestions(session.board, dawg, letters, n)


def _first_move_suggestions(board: Board, dawg: Dawg, letters: str, n: int) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, int, int, bool]] = set()

    for word in dawg.search("*", letters):
        word_len = len(word)
        for offset in range(min(7, word_len)):
            col = 7 - offset
            if col + word_len - 1 > 14:
                continue
            key = (word, 7, col, True)
            if key in seen:
                continue
            seen.add(key)
            try:
                score = board.calculate_word_points(word, 7, col, True, letters)
                if score > 0:
                    candidates.append({
                        "word": word,
                        "score": score,
                        "row": 7,
                        "col": col,
                        "horizontal": True,
                        "cells": [(7, col + i) for i in range(word_len)],
                    })
            except Exception:
                pass

    candidates.sort(key=lambda x: -x["score"])
    return candidates[:n]


def _subsequent_suggestions(board: Board, dawg: Dawg, letters: str, n: int) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, int, int, bool]] = set()

    for index, start, end, horizontal in board.get_all_patterns():
        row, col = (index, start) if horizontal else (start, index)
        word = board.best_word_from_pattern(dawg, row, col, end, horizontal, letters)
        if not word:
            continue
        key = (word, row, col, horizontal)
        if key in seen:
            continue
        seen.add(key)
        try:
            score = board.calculate_word_points(word, row, col, horizontal, letters)
            if score <= 0:
                continue
            word_len = len(word)
            cells = (
                [(row, col + i) for i in range(word_len)]
                if horizontal
                else [(row + i, col) for i in range(word_len)]
            )
            candidates.append({
                "word": word, "score": score, "row": row, "col": col,
                "horizontal": horizontal, "cells": cells,
            })
        except Exception:
            pass

    candidates.sort(key=lambda x: -x["score"])
    return candidates[:n]


def get_suggestions_for_letters(session: GameSession, dawg: Dawg, letters: str, n: int = 10) -> list[dict]:
    """Like get_suggestions but uses the supplied letters instead of current player's rack."""
    if not letters:
        return []
    if session.is_first_move:
        return _first_move_suggestions(session.board, dawg, letters, n)
    return _subsequent_suggestions(session.board, dawg, letters, n)


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

# Defensive cap on moves per simulated game -- CONSECUTIVE_NO_PLAY_LIMIT and
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
    difficulty: str
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


def run_benchmark(
    player_specs: list[tuple[str, Difficulty]],
    games: int,
    dawg: Dawg,
    on_game_done: Callable[[int], None] | None = None,
) -> BenchmarkResult:
    """Simulate *games* full SANDBOX_AUTO games with the given (name,
    difficulty) players end-to-end using the same engine primitives as a
    live game (_deal_new_game + computer_auto_play), never touching
    SessionStore. Returns aggregate per-player stats plus the full
    move-by-move detail of whichever single game had the highest final
    score for any one player."""
    start = time.perf_counter()
    stats = [
        BenchmarkPlayerStats(name=name, difficulty=difficulty.value)
        for name, difficulty in player_specs
    ]
    best_game: BenchmarkBestGame | None = None
    best_score = -1
    total_moves = 0
    longest_word: str | None = None
    longest_word_score = 0
    highest_single_move_score = 0

    for game_idx in range(games):
        players = [
            Player(name=name, is_computer=True, difficulty=difficulty)
            for name, difficulty in player_specs
        ]
        session = _deal_new_game(players, GameMode.SANDBOX_AUTO)

        moves: list[BenchmarkMoveRecord] = []
        move_count = 0
        while not session.game_over and move_count < MAX_BENCHMARK_GAME_MOVES:
            player_idx = session.current_player_idx
            move = computer_auto_play(session, dawg)
            move_count += 1
            total_moves += 1
            if not move.passed:
                s = stats[player_idx]
                s.words_played += 1
                s.total_word_score += move.score
                highest_single_move_score = max(highest_single_move_score, move.score)
                if longest_word is None or len(move.word) > len(longest_word):
                    longest_word, longest_word_score = move.word, move.score
            moves.append(BenchmarkMoveRecord(
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
            ))

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
            on_game_done(game_idx + 1)

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
