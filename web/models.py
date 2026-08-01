from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from web.difficulty import DEFAULT_LEVEL, MAX_LEVEL, MIN_LEVEL

# Custom difficulty is a single integer dial rather than a set of named tiers
# (see web/difficulty.py). Declared once so every request/response that carries
# a level validates it the same way.
DifficultyLevelField = Field(DEFAULT_LEVEL, ge=MIN_LEVEL, le=MAX_LEVEL)

# ── Requests ─────────────────────────────────────────────────────────────────


class NewPlayerConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=20)
    is_computer: bool = False
    difficulty: int = DifficultyLevelField


class NewGameRequest(BaseModel):
    players: list[NewPlayerConfig] = Field(..., min_length=1, max_length=4)
    game_mode: Literal["sandbox", "sandbox_auto", "competitive"] = "sandbox"
    difficulty: int = DifficultyLevelField
    # A plain str, not a Literal, so adding a language stays a matter of
    # dropping a file in `languages/`. Validated against the registry in the
    # handler, which knows what is actually installed.
    language: str | None = None


class PlaceHumanWordRequest(BaseModel):
    word: str = Field(..., min_length=2, max_length=15)
    row: int = Field(..., ge=0, le=14)
    col: int = Field(..., ge=0, le=14)
    horizontal: bool


class SetComputerLettersRequest(BaseModel):
    letters: str = Field(..., max_length=7)


class ExchangeTilesRequest(BaseModel):
    letters: str = Field(..., min_length=1, max_length=7)


class ScanConfirmRequest(BaseModel):
    board: list[list[str]] = Field(..., min_length=15, max_length=15)


class ScanSuggestRequest(BaseModel):
    letters: str = Field(..., min_length=1, max_length=7)
    sort: Literal["score", "smart", "sim"] = "score"


class ScanRecheckRequest(BaseModel):
    board: list[list[str]] = Field(..., min_length=15, max_length=15)
    locked: list[list[bool]] = Field(..., min_length=15, max_length=15)


class PlaceComputerWordRequest(BaseModel):
    word: str
    row: int = Field(..., ge=0, le=14)
    col: int = Field(..., ge=0, le=14)
    horizontal: bool
    score: int = Field(..., ge=0)


class BenchmarkPlayerConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=20)
    difficulty: int = DifficultyLevelField


class BenchmarkRequest(BaseModel):
    players: list[BenchmarkPlayerConfig] = Field(..., min_length=2, max_length=4)
    games: int = Field(20, ge=1)
    language: str | None = None


# ── Responses ─────────────────────────────────────────────────────────────────


class PlayerState(BaseModel):
    name: str
    is_computer: bool
    score: int
    letters: str
    difficulty: int = DEFAULT_LEVEL


class DifficultyLevelInfo(BaseModel):
    """One notch of the difficulty slider, described well enough that a player
    can tell what they are choosing before the game starts. Served by
    `GET /api/game/difficulty-levels` so the UI text is derived from the real
    rank windows instead of a hand-maintained copy in JS."""

    level: int
    name: str
    emoji: str
    summary: str
    expect: str
    engine: Literal["ranked", "smart", "sim"]
    rank_best: int | None = None
    rank_worst: int | None = None
    slow: bool = False


class DifficultyLevelsResponse(BaseModel):
    min_level: int
    max_level: int
    default_level: int
    levels: list[DifficultyLevelInfo]


class LanguageInfo(BaseModel):
    """One entry of the language picker, plus the tables the client needs to
    render a board in that language. `letter_values` replaces what used to be a
    hand-kept copy of the point table in `web/static/js/board.js`."""

    code: str
    name: str
    flag: str
    alphabet: str
    blank: str
    letter_values: dict[str, int]
    tile_counts: dict[str, int]
    total_tiles: int
    #: Strongest difficulty this language can field -- capped where no leave
    #: net has been trained yet (see web.difficulty.max_level_for).
    max_level: int
    #: Whether the board-photo scanner has models for this language.
    has_ocr: bool
    #: True where those models were trained on rendered fonts alone, never
    #: checked against photographs of real tiles.
    ocr_experimental: bool
    #: Whether a trained rack-leave evaluator exists. Without one there are no
    #: levels 9-10, and the `smart`/`sim` suggestion orderings are unavailable.
    has_leave_net: bool


class LanguagesResponse(BaseModel):
    default: str
    languages: list[LanguageInfo]


class LastComputerMove(BaseModel):
    word: str
    score: int
    row: int
    col: int
    horizontal: bool
    passed: bool


class Suggestion(BaseModel):
    word: str
    score: int
    row: int
    col: int
    horizontal: bool
    cells: list[tuple[int, int]]
    # What the list was ordered by: the raw score, the score plus the leave the
    # play would keep, or the simulated equity. Shown so a reordering is
    # explained rather than mysterious.
    value: float | None = None


class BoardStateResponse(BaseModel):
    language: str = "pl"
    board: list[list[str]]
    # Which occupied squares hold a blank. The grid renders a blank as the
    # letter it stands in for, so without this the UI cannot tell a blank from
    # a real tile -- and they score very differently for anything played
    # through them later.
    board_blanks: list[list[bool]] = Field(default_factory=list)
    players: list[PlayerState]
    current_player_idx: int
    is_first_move: bool
    move_number: int
    session_id: str
    can_undo: bool
    game_mode: str
    tiles_remaining: int
    last_computer_move: LastComputerMove | None = None
    tile_owners: list[list[int | None]] = Field(default_factory=list)
    game_over: bool = False
    winner_name: str | None = None
    last_move_rating: int | None = None


class PreviewScoreResponse(BaseModel):
    score: int | None = None
    error: str | None = None


class DefinitionResponse(BaseModel):
    word: str
    definitions: list[str]
    found: bool


class SuggestionsResponse(BaseModel):
    suggestions: list[Suggestion]
    letters: str


class ScanCell(BaseModel):
    letter: str
    confidence: float = 0.0
    alternatives: list[str] = Field(default_factory=list)
    flagged: bool = False
    carried_over: bool = False


class ScanBoardResponse(BaseModel):
    cells: list[list[ScanCell]] = Field(default_factory=list)
    flagged_count: int = 0
    error: str | None = None


class ScanStateResponse(BaseModel):
    board: list[list[str]]
    has_session: bool


class ScanRecheckResponse(BaseModel):
    flagged: list[list[bool]]


class SaveTrainingResponse(BaseModel):
    id: int
    difficulty: Literal["e", "m", "h"]
    matched: int
    total: int
    match_ratio: float


# ── Benchmark ─────────────────────────────────────────────────────────────────


class BenchmarkMoveRecord(BaseModel):
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


class BenchmarkPlayerStats(BaseModel):
    name: str
    difficulty: int
    games_played: int
    wins: int
    ties: int
    avg_score: float
    high_score: int
    low_score: int
    words_played: int
    avg_word_score: float


class BenchmarkBestGame(BaseModel):
    winner_name: str
    winner_score: int
    final_scores: list[PlayerState]
    moves: list[BenchmarkMoveRecord]


class BenchmarkResultResponse(BaseModel):
    games_played: int
    duration_ms: int
    player_stats: list[BenchmarkPlayerStats]
    best_game: BenchmarkBestGame | None = None
    avg_game_length: float
    longest_word: str | None = None
    longest_word_score: int | None = None
    highest_single_move_score: int | None = None


class BenchmarkJobStartResponse(BaseModel):
    job_id: str


class BenchmarkJobStatusResponse(BaseModel):
    status: Literal["running", "done", "error"]
    games_done: int
    games_total: int
    result: BenchmarkResultResponse | None = None
    error: str | None = None
