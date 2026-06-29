from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Requests ─────────────────────────────────────────────────────────────────

class NewPlayerConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=20)
    is_computer: bool = False


class NewGameRequest(BaseModel):
    players: list[NewPlayerConfig] = Field(..., min_length=1, max_length=4)
    game_mode: Literal["sandbox", "competitive"] = "sandbox"
    difficulty: Literal["easy", "medium", "hard", "impossible"] = "hard"


class PlaceHumanWordRequest(BaseModel):
    word: str = Field(..., min_length=2, max_length=15)
    row: int = Field(..., ge=0, le=14)
    col: int = Field(..., ge=0, le=14)
    horizontal: bool


class SetComputerLettersRequest(BaseModel):
    letters: str = Field(..., max_length=7)


class PlaceComputerWordRequest(BaseModel):
    word: str
    row: int = Field(..., ge=0, le=14)
    col: int = Field(..., ge=0, le=14)
    horizontal: bool
    score: int = Field(..., ge=0)


# ── Responses ─────────────────────────────────────────────────────────────────

class PlayerState(BaseModel):
    name: str
    is_computer: bool
    score: int
    letters: str


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
