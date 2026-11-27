"""The shared leaderboard (`web/leaderboard.py`): the Android app uploads its
finished games here, on the player's request, and anyone reads the standings."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from web import leaderboard

router = APIRouter(prefix="/leaderboard")


class UploadRequest(BaseModel):
    #: The uploading device's own id: its games are kept apart from another's.
    source: str = Field(min_length=1, max_length=64)
    #: Records in the app's history format.
    games: list[dict] = Field(max_length=leaderboard.MAX_UPLOAD_GAMES)


def _require_enabled() -> None:
    if not leaderboard.enabled():
        raise HTTPException(status_code=404, detail="Ranking jest na tym serwerze wyłączony.")


@router.get("")
async def standings(language: str | None = None) -> dict:
    """Every player's results, the best scores, and which languages were played."""
    if not leaderboard.enabled():
        return {"enabled": False, "games": 0, "languages": [], "players": [], "top_scores": []}
    return {"enabled": True, **await run_in_threadpool(leaderboard.standings, language)}


@router.post("/games")
async def upload(body: UploadRequest, x_leaderboard_token: str | None = Header(default=None)) -> dict:
    """Keep an app's finished games, replacing any it sent before."""
    _require_enabled()
    expected = leaderboard.token_required()
    if expected and x_leaderboard_token != expected:
        raise HTTPException(status_code=403, detail="Nieprawidłowy klucz rankingu.")
    kept = await run_in_threadpool(leaderboard.add, f"app:{body.source}", body.games)
    return {"received": len(body.games), "kept": kept}
