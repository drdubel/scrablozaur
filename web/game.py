from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum

from scrablozaur import Board, Dawg


class GameMode(str, Enum):
    SANDBOX = "sandbox"
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
    difficulty: Difficulty = Difficulty.HARD
    tile_bag: TileBag | None = None
    last_computer_move: ComputerMoveInfo | None = None
    tile_owners: list[list[int | None]] = field(
        default_factory=lambda: [[None] * 15 for _ in range(15)]
    )
    game_over: bool = False
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


class SessionStore:
    _sessions: dict[str, GameSession] = {}

    @classmethod
    def create(
        cls,
        players: list[Player] | None = None,
        game_mode: GameMode = GameMode.SANDBOX,
        difficulty: Difficulty = Difficulty.HARD,
    ) -> GameSession:
        player_list = [Player(p.name, p.is_computer) for p in (players or _DEFAULT_PLAYERS)]
        tile_bag: TileBag | None = None

        if game_mode == GameMode.COMPETITIVE:
            tile_bag = TileBag.full()
            for p in player_list:
                p.letters = "".join(tile_bag.draw(7))

        session = GameSession(
            session_id=str(uuid.uuid4()),
            board=Board([["-"] * 15 for _ in range(15)]),
            players=player_list,
            game_mode=game_mode,
            difficulty=difficulty,
            tile_bag=tile_bag,
        )
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
    """Play a move for the computer weighted by difficulty, then advance the turn."""
    # Fetch enough candidates so difficulty has real choices; cap at 30 for speed
    n_candidates = 1 if session.difficulty == Difficulty.IMPOSSIBLE else 30
    suggestions = get_suggestions(session, dawg, n=n_candidates)

    if not suggestions:
        session.advance_turn()
        return ComputerMoveInfo(word="", score=0, row=0, col=0, horizontal=True, passed=True)

    sug = _pick_by_difficulty(suggestions, session.difficulty)
    word, row, col, horizontal = sug["word"], sug["row"], sug["col"], sug["horizontal"]

    grid = session.board_grid()
    session.record_placement(word, row, col, horizontal, session.current_player_idx)
    session.board.place_word(word, row, col, horizontal)
    session.current_player.score += sug["score"]
    session.is_first_move = False

    _deduct_tiles(session.current_player, word, grid, row, col, horizontal)
    _refill_rack(session, session.current_player)
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
