from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Requests ─────────────────────────────────────────────────────────────────


class NewPlayerConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=20)
    is_computer: bool = False
    difficulty: Literal["easy", "medium", "hard", "impossible"] = "hard"


class NewGameRequest(BaseModel):
    players: list[NewPlayerConfig] = Field(..., min_length=1, max_length=4)
    game_mode: Literal["sandbox", "sandbox_auto", "competitive"] = "sandbox"
    difficulty: Literal["easy", "medium", "hard", "impossible"] = "hard"


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
    difficulty: Literal["easy", "medium", "hard", "impossible"] = "hard"


class BenchmarkRequest(BaseModel):
    players: list[BenchmarkPlayerConfig] = Field(..., min_length=2, max_length=4)
    games: int = Field(20, ge=1)


# ── Responses ─────────────────────────────────────────────────────────────────


class PlayerState(BaseModel):
    name: str
    is_computer: bool
    score: int
    letters: str
    difficulty: str = "hard"


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


class BoardStateResponse(BaseModel):
    board: list[list[str]]
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
    difficulty: str
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
