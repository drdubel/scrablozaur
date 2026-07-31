"""Shared FastAPI dependencies.

`_require_session` used to live in `web/routers/game.py` and be imported by the
other routers, which made a sibling router a de-facto utility module. It lives
here instead, alongside the language resolution every route now needs.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from web.engine import Dawg, LanguagePack, available, get_pack, is_available
from web.game import GameSession, SessionStore

SESSION_COOKIE = "scrablozaur_session"


def optional_session(request: Request) -> GameSession | None:
    """The game session if there is one, without demanding it. The scan flow
    works standalone, but borrows the game's language when a game is running."""
    sid = request.cookies.get(SESSION_COOKIE)
    return SessionStore.get(sid) if sid else None


def require_session(request: Request) -> GameSession:
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        raise HTTPException(status_code=401, detail="Brak sesji. Rozpocznij nową grę.")
    session = SessionStore.get(sid)
    if not session:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji. Rozpocznij nową grę.")
    return session


def session_pack(request: Request) -> LanguagePack:
    """The language pack for the current session.

    Every route that consults the dictionary goes through here, so a game
    started in one language can never be served by another's lexicon.
    """
    return get_pack(require_session(request).language)


def session_dawg(request: Request) -> Dawg:
    """Just the dictionary — what most routes actually want. Routes that also
    need the point table or the definition sources take `session_pack`."""
    return session_pack(request).dawg


def resolve_language(code: str | None) -> LanguagePack:
    """Validate a client-supplied language code and load its pack.

    Checked here rather than with a `Literal` on the request model so adding a
    language stays a matter of dropping a JSON file in `languages/`.
    """
    if code is not None and not is_available(code):
        raise HTTPException(
            status_code=400,
            detail=f"Nieznany język: {code}. Dostępne: {', '.join(available())}.",
        )
    return get_pack(code)
