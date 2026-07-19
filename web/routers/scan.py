from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile

from web.engine import Board, Dawg, get_dawg
from web.game import _first_move_suggestions, _subsequent_suggestions
from web.models import (SaveTrainingResponse, ScanBoardResponse, ScanCell, ScanConfirmRequest,
                        ScanStateResponse, ScanSuggestRequest, Suggestion, SuggestionsResponse)
from web.scan import (GRID, POLISH_LOWER, ScanSessionStore, board_is_empty, empty_board,
                      evaluate_raw_recognition, save_training_example, scan_board_image)

router = APIRouter(prefix="/scan")

_MAX_UPLOAD_BYTES = 15 * 1024 * 1024
_COOKIE_NAME = "scrablozaur_scan_session"


def _get_session(request: Request):
    return ScanSessionStore.get(request.cookies.get(_COOKIE_NAME))


def _set_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(_COOKIE_NAME, session_id, httponly=True, samesite="lax")


async def _read_image_upload(file: UploadFile) -> bytes:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Prześlij plik graficzny (zdjęcie planszy).")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Pusty plik.")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Zdjęcie jest zbyt duże (limit 15 MB).")
    return data


def _validate_grid(raw_grid: object) -> list[list[str]]:
    if not isinstance(raw_grid, list) or len(raw_grid) != GRID or any(
        not isinstance(row, list) or len(row) != GRID for row in raw_grid
    ):
        raise HTTPException(status_code=400, detail="Nieprawidłowy rozmiar planszy.")
    grid = [[(ch or "-").lower() for ch in row] for row in raw_grid]
    for row in grid:
        for ch in row:
            if ch != "-" and ch not in POLISH_LOWER:
                raise HTTPException(status_code=400, detail=f"Nieprawidłowy znak na planszy: '{ch}'.")
    return grid


@router.post("/board", response_model=ScanBoardResponse)
async def scan_board(request: Request, file: UploadFile = File(...)) -> ScanBoardResponse:
    """Read a photo of the board. If a ScanSession already exists (i.e. this
    isn't the first photo), its last confirmed board is passed in as a prior
    to help recognise tiles this photo alone reads poorly -- see web/scan.py.
    Doesn't touch the session yet; the result still has to be reviewed and
    POSTed to /scan/confirm."""
    data = await _read_image_upload(file)

    session = _get_session(request)
    prior_board = session.board if session and not board_is_empty(session.board) else None

    suffix = Path(file.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        result = scan_board_image(tmp.name, prior_board=prior_board)

    if "error" in result:
        return ScanBoardResponse(error=result["error"])

    cells = [[ScanCell(**cell) for cell in row] for row in result["cells"]]
    flagged_count = sum(1 for row in cells for cell in row if cell.flagged)
    return ScanBoardResponse(cells=cells, flagged_count=flagged_count)


@router.post("/confirm", response_model=ScanStateResponse)
async def confirm_scan(body: ScanConfirmRequest, request: Request, response: Response) -> ScanStateResponse:
    """Commit a (possibly hand-edited) board as this ScanSession's new
    current state, creating the session on the first-ever confirm."""
    grid = _validate_grid(body.board)

    session = _get_session(request)
    if session is None:
        session = ScanSessionStore.create(board=grid)
        _set_cookie(response, session.session_id)
    else:
        session.board = grid

    return ScanStateResponse(board=session.board, has_session=True)


@router.get("/state", response_model=ScanStateResponse)
async def scan_state(request: Request) -> ScanStateResponse:
    session = _get_session(request)
    if session is None:
        return ScanStateResponse(board=empty_board(), has_session=False)
    return ScanStateResponse(board=session.board, has_session=True)


@router.post("/reset")
async def reset_scan_session(request: Request, response: Response) -> ScanStateResponse:
    """Discard the current ScanSession -- the next photo starts fresh with
    no prior state (e.g. the user is starting to track a new physical game)."""
    ScanSessionStore.delete(request.cookies.get(_COOKIE_NAME))
    response.delete_cookie(_COOKIE_NAME)
    return ScanStateResponse(board=empty_board(), has_session=False)


@router.post("/suggest", response_model=SuggestionsResponse)
async def suggest_for_scan(
    body: ScanSuggestRequest,
    request: Request,
    dawg: Dawg = Depends(get_dawg),
) -> SuggestionsResponse:
    session = _get_session(request)
    if session is None:
        raise HTTPException(status_code=400, detail="Najpierw zeskanuj i zatwierdź planszę.")

    letters = body.letters.lower()
    board = Board(session.board)
    fn = _first_move_suggestions if board_is_empty(session.board) else _subsequent_suggestions
    raw = fn(board, dawg, letters, 10)
    suggestions = [Suggestion(**s) for s in raw]
    return SuggestionsResponse(suggestions=suggestions, letters=letters)


@router.post("/save-training", response_model=SaveTrainingResponse)
async def save_training(
    file: UploadFile = File(...),
    board: str = Form(...),
) -> SaveTrainingResponse:
    """Opt-in: append this photo and the board the user just confirmed for
    it to board_reader/'s own eval/retraining set (see
    [[project-ocr-pipeline]]), difficulty-tagged by how well the *raw*
    classifier (no dictionary correction, no prior-state help) did against
    that confirmed board on its own."""
    data = await _read_image_upload(file)

    try:
        raw_grid = json.loads(board)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Nieprawidłowe dane planszy.") from exc
    grid = _validate_grid(raw_grid)

    suffix = Path(file.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            difficulty, stats = evaluate_raw_recognition(tmp.name, grid)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        image_id = save_training_example(data, grid, difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SaveTrainingResponse(id=image_id, difficulty=difficulty, **stats)
