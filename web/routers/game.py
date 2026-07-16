from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from web.engine import Dawg, get_dawg
from web.game import Difficulty, GameMode, GameSession, Player, SessionStore, computer_auto_play
from web.models import (BoardStateResponse, LastComputerMove, NewGameRequest,
                        PlayerState)

router = APIRouter(prefix="/game")


def _state_response(session: GameSession) -> BoardStateResponse:
    return BoardStateResponse(
        board=session.board_grid(),
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
            Player(name="Komputer", is_computer=True),
        ]
    computer_count = sum(1 for p in body.players if p.is_computer)
    if computer_count != 1:
        raise HTTPException(status_code=400, detail="Exactly one player must be the computer.")
    return [Player(name=p.name, is_computer=p.is_computer) for p in body.players]


def _play_opening_computer_move(session: GameSession, dawg: Dawg) -> None:
    """First-player draw (SessionStore.create) can land on the computer --
    every other auto-play trigger is nested inside a human-initiated
    endpoint, so without this the game would just sit stuck waiting for a
    human turn that isn't next."""
    if session.game_mode == GameMode.COMPETITIVE and session.current_player.is_computer:
        session.last_computer_move = computer_auto_play(session, dawg)


@router.post("/new", response_model=BoardStateResponse)
async def new_game(body: NewGameRequest, response: Response, dawg: Dawg = Depends(get_dawg)) -> BoardStateResponse:
    players = _players_from_request(body)
    session = SessionStore.create(
        players, game_mode=GameMode(body.game_mode), difficulty=Difficulty(body.difficulty)
    )
    _play_opening_computer_move(session, dawg)
    _set_session_cookie(response, session.session_id)
    return _state_response(session)


@router.get("/state", response_model=BoardStateResponse)
async def get_state(request: Request) -> BoardStateResponse:
    return _state_response(_require_session(request))


@router.post("/reset", response_model=BoardStateResponse)
async def reset_game(
    body: NewGameRequest, request: Request, response: Response, dawg: Dawg = Depends(get_dawg)
) -> BoardStateResponse:
    sid = request.cookies.get("scrablozaur_session")
    if sid:
        SessionStore.delete(sid)
    players = _players_from_request(body)
    session = SessionStore.create(
        players, game_mode=GameMode(body.game_mode), difficulty=Difficulty(body.difficulty)
    )
    _play_opening_computer_move(session, dawg)
    _set_session_cookie(response, session.session_id)
    return _state_response(session)


def _require_session(request: Request) -> GameSession:
    sid = request.cookies.get("scrablozaur_session")
    if not sid:
        raise HTTPException(status_code=401, detail="No session. Start a new game.")
    session = SessionStore.get(sid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Start a new game.")
    return session
