from __future__ import annotations

from starlette.concurrency import run_in_threadpool

from fastapi import APIRouter, HTTPException, Request, Response

from web.deps import require_session, resolve_language
from web.difficulty import DEFAULT_LEVEL, MIN_LEVEL, all_levels, max_level_for
from web.engine import DEFAULT_LANGUAGE, Dawg, available
from languages import load as load_spec
from web.game import GameMode, GameSession, Player, SessionStore, computer_auto_play
from web.models import (BoardStateResponse, DifficultyLevelInfo, DifficultyLevelsResponse,
                        LanguageInfo, LanguagesResponse, LastComputerMove, NewGameRequest,
                        PlayerState)

router = APIRouter(prefix="/game")


def _state_response(session: GameSession) -> BoardStateResponse:
    return BoardStateResponse(
        language=session.language,
        board=session.board_grid(),
        board_blanks=session.board.blank_mask(),
        players=[
            PlayerState(
                name=p.name,
                is_computer=p.is_computer,
                score=p.score,
                # hide computer's rack from client in competitive mode
                letters=(
                    ""
                    if session.game_mode == GameMode.COMPETITIVE and p.is_computer
                    else p.letters
                ),
                difficulty=p.difficulty,
            )
            for p in session.players
        ],
        current_player_idx=session.current_player_idx,
        is_first_move=session.is_first_move,
        move_number=session.move_number,
        session_id=session.session_id,
        can_undo=bool(session.move_history),
        game_mode=session.game_mode.value,
        tiles_remaining=session.tile_bag.remaining() if session.tile_bag else 0,
        last_computer_move=(
            LastComputerMove(
                word=session.last_computer_move.word,
                score=session.last_computer_move.score,
                row=session.last_computer_move.row,
                col=session.last_computer_move.col,
                horizontal=session.last_computer_move.horizontal,
                passed=session.last_computer_move.passed,
            )
            if session.last_computer_move
            else None
        ),
        tile_owners=session.tile_owners,
        game_over=session.game_over,
        winner_name=_winner_name(session) if session.game_over else None,
        last_move_rating=session.last_move_rating,
    )


def _winner_name(session: GameSession) -> str | None:
    if not session.players:
        return None
    best = max(session.players, key=lambda p: p.score)
    tied = [p for p in session.players if p.score == best.score]
    if len(tied) > 1:
        return "Remis"
    return best.name


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie("scrablozaur_session", session_id, httponly=True, samesite="lax")


def _players_from_request(body: NewGameRequest) -> list[Player]:
    if body.game_mode == "competitive":
        non_computer = [p for p in body.players if not p.is_computer]
        if len(non_computer) != 1:
            raise HTTPException(
                status_code=400,
                detail="Tryb rywalizacji wymaga dokładnie jednego gracza-człowieka.",
            )
        return [
            Player(name=non_computer[0].name, is_computer=False),
            Player(name="Komputer", is_computer=True, difficulty=body.difficulty),
        ]
    if body.game_mode == "sandbox_auto":
        if len(body.players) < 2:
            raise HTTPException(
                status_code=400,
                detail="Tryb automatyczny wymaga co najmniej dwóch graczy-komputerów.",
            )
        return [
            Player(name=p.name, is_computer=True, difficulty=p.difficulty)
            for p in body.players
        ]
    computer_count = sum(1 for p in body.players if p.is_computer)
    if computer_count != 1:
        raise HTTPException(status_code=400, detail="Exactly one player must be the computer.")
    return [
        Player(name=p.name, is_computer=p.is_computer, difficulty=p.difficulty)
        for p in body.players
    ]


async def _play_opening_computer_move(session: GameSession, dawg: Dawg) -> None:
    """First-player draw (SessionStore.create) can land on the computer --
    every other auto-play trigger is nested inside a human-initiated
    endpoint, so without this the game would just sit stuck waiting for a
    human turn that isn't next."""
    if session.game_mode == GameMode.COMPETITIVE and session.current_player.is_computer:
        session.last_computer_move = await run_in_threadpool(computer_auto_play, session, dawg)


@router.post("/new", response_model=BoardStateResponse)
async def new_game(body: NewGameRequest, response: Response) -> BoardStateResponse:
    pack = resolve_language(body.language)
    players = _players_from_request(body)
    session = SessionStore.create(players, game_mode=GameMode(body.game_mode), pack=pack)
    await _play_opening_computer_move(session, pack.dawg)
    _set_session_cookie(response, session.session_id)
    return _state_response(session)


@router.get("/difficulty-levels", response_model=DifficultyLevelsResponse)
async def difficulty_levels(language: str | None = None) -> DifficultyLevelsResponse:
    """Every notch of the custom-difficulty slider, with the feedback text the
    setup dialog shows. Server-side so the descriptions stay tied to the rank
    windows the bot actually plays by -- and so a language without a trained
    leave net simply serves a shorter list, which the slider honours without
    needing to know why."""
    spec = resolve_language(language).spec
    levels = all_levels(spec)
    return DifficultyLevelsResponse(
        min_level=MIN_LEVEL,
        max_level=max_level_for(spec),
        default_level=DEFAULT_LEVEL,
        levels=[
            DifficultyLevelInfo(
                level=info.level,
                name=info.name,
                emoji=info.emoji,
                summary=info.summary,
                expect=info.expect,
                engine=info.engine.value,
                rank_best=info.rank_best,
                rank_worst=info.rank_worst,
                slow=info.slow,
            )
            for info in levels
        ],
    )


@router.get("/state", response_model=BoardStateResponse)
async def get_state(request: Request) -> BoardStateResponse:
    return _state_response(_require_session(request))


@router.post("/reset", response_model=BoardStateResponse)
async def reset_game(
    body: NewGameRequest, request: Request, response: Response
) -> BoardStateResponse:
    pack = resolve_language(body.language)
    sid = request.cookies.get("scrablozaur_session")
    if sid:
        SessionStore.delete(sid)
    players = _players_from_request(body)
    session = SessionStore.create(players, game_mode=GameMode(body.game_mode), pack=pack)
    await _play_opening_computer_move(session, pack.dawg)
    _set_session_cookie(response, session.session_id)
    return _state_response(session)


@router.get("/languages", response_model=LanguagesResponse)
async def list_languages() -> LanguagesResponse:
    """The language picker's options, plus each language's point and count
    tables. Serving the tables means the client never carries its own copy --
    `board.js` used to hold a hand-maintained duplicate of the point values."""
    infos = []
    for code in available():
        # Read the definition file, not `get_pack` -- the picker only needs
        # metadata, and loading every language's dictionary to build a dropdown
        # would cost ~60-80 MB apiece for nothing.
        spec = load_spec(code)
        infos.append(
            LanguageInfo(
                code=spec.code,
                name=spec.name,
                flag=spec.flag,
                alphabet=spec.alphabet,
                blank=spec.blank,
                letter_values=spec.points,
                tile_counts=spec.counts,
                total_tiles=spec.total_tiles,
                max_level=max_level_for(spec),
                has_ocr=spec.has_ocr,
                ocr_experimental=spec.ocr_is_experimental,
                has_leave_net=spec.leave_net is not None,
            )
        )
    return LanguagesResponse(default=DEFAULT_LANGUAGE, languages=infos)


# Re-exported: the other routers imported this from here before it moved to
# `web/deps.py`, and this keeps that import working.
_require_session = require_session
