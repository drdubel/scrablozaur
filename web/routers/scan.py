from __future__ import annotations

import json
import tempfile
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile

from web.deps import optional_session
from web.engine import DEFAULT_LANGUAGE, Board, Dawg, LanguagePack, get_pack
from web.game import LEAVE_NET_SORTS, SORT_MODES, has_leave_net, rank_suggestions
from web.models import (
    SaveTrainingResponse,
    ScanBoardResponse,
    ScanCell,
    ScanConfirmRequest,
    ScanRecheckRequest,
    ScanRecheckResponse,
    ScanStateResponse,
    ScanSuggestRequest,
    Suggestion,
    SuggestionsResponse,
)
from web.scan import (
    GRID,
    ScanSessionStore,
    board_is_empty,
    empty_board,
    evaluate_raw_recognition,
    flag_invalid,
    save_training_example,
    scan_board_image,
)

router = APIRouter(prefix="/scan")

_MAX_UPLOAD_BYTES = 15 * 1024 * 1024
_COOKIE_NAME = "scrablozaur_scan_session"


def _get_session(request: Request):
    return ScanSessionStore.get(request.cookies.get(_COOKIE_NAME))


def _scan_language(request: Request) -> str:
    """The language this scan is in.

    A scan session remembers its own; a fresh one borrows the running game's,
    so scanning a board mid-game reads it in the language being played. The
    scan flow works with no game at all, so this must never demand one.
    """
    session = _get_session(request)
    if session is not None:
        return session.language
    game = optional_session(request)
    return game.language if game is not None else DEFAULT_LANGUAGE


def _scan_pack(request: Request) -> LanguagePack:
    return get_pack(_scan_language(request))


def _scan_dawg(request: Request) -> Dawg:
    return _scan_pack(request).dawg


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


def _validate_grid(
    raw_grid: object, alphabet: str, *, allow_unknown: bool = False
) -> list[list[str]]:
    """*allow_unknown* additionally accepts '?' -- scan_board_image()'s marker
    for a tile the OCR couldn't read at all. /confirm and /save-training
    reject it (a persisted or exported board should never contain an
    unresolved tile), but /recheck runs against a board the user is still
    mid-edit on, where some other cell can easily still be an unresolved
    '?' the user simply hasn't gotten to yet -- that shouldn't 400 out the
    recheck of the cell they just fixed."""
    if (
        not isinstance(raw_grid, list)
        or len(raw_grid) != GRID
        or any(not isinstance(row, list) or len(row) != GRID for row in raw_grid)
    ):
        raise HTTPException(status_code=400, detail="Nieprawidłowy rozmiar planszy.")
    letters = set(alphabet)
    grid = [[(ch or "-").lower() for ch in row] for row in raw_grid]
    for row in grid:
        for ch in row:
            if ch != "-" and ch not in letters and not (allow_unknown and ch == "?"):
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

    language = _scan_language(request)
    session = _get_session(request)
    prior_board = session.board if session and not board_is_empty(session.board) else None

    suffix = Path(file.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        result = scan_board_image(tmp.name, prior_board=prior_board, language=language)

    if "error" in result:
        return ScanBoardResponse(error=result["error"])

    cells = [[ScanCell(**cell) for cell in row] for row in result["cells"]]
    flagged_count = sum(1 for row in cells for cell in row if cell.flagged)
    return ScanBoardResponse(cells=cells, flagged_count=flagged_count)


@router.post("/confirm", response_model=ScanStateResponse)
async def confirm_scan(body: ScanConfirmRequest, request: Request, response: Response) -> ScanStateResponse:
    """Commit a (possibly hand-edited) board as this ScanSession's new
    current state, creating the session on the first-ever confirm."""
    language = _scan_language(request)
    grid = _validate_grid(body.board, get_pack(language).spec.alphabet)

    session = _get_session(request)
    if session is None:
        session = ScanSessionStore.create(board=grid, language=language)
        _set_cookie(response, session.session_id)
    else:
        session.board = grid

    return ScanStateResponse(board=session.board, has_session=True)


@router.post("/recheck", response_model=ScanRecheckResponse)
async def recheck_scan_board(
    body: ScanRecheckRequest,
    dawg: Dawg = Depends(_scan_dawg),
) -> ScanRecheckResponse:
    """Re-run the dictionary flagging check (no auto-correction -- see
    web/scan.py's flag_invalid()) over a board the user is still editing in
    the review step, so a hand-typed letter's effect on its own and any
    crossing words is reflected immediately rather than only on the cell
    that was actually touched. Stateless: doesn't touch the ScanSession."""
    grid = _validate_grid(body.board, dawg.lang.alphabet, allow_unknown=True)
    if len(body.locked) != GRID or any(len(row) != GRID for row in body.locked):
        raise HTTPException(status_code=400, detail="Nieprawidłowy rozmiar maski zablokowanych pól.")
    locked = {(r, c) for r in range(GRID) for c in range(GRID) if body.locked[r][c]}

    flagged_positions = flag_invalid(dawg, grid, locked=locked)
    flagged = [[(r, c) in flagged_positions for c in range(GRID)] for r in range(GRID)]
    return ScanRecheckResponse(flagged=flagged)


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
    dawg: Dawg = Depends(_scan_dawg),
) -> SuggestionsResponse:
    session = _get_session(request)
    if session is None:
        raise HTTPException(status_code=400, detail="Najpierw zeskanuj i zatwierdź planszę.")

    letters = body.letters.lower()
    board = Board.from_grid(get_pack(session.language).lang, session.board)
    # `get_best_words` picks the opening path itself from the board's own
    # first-move flag, which `from_grid` derives from the grid.
    if body.sort not in SORT_MODES:
        raise HTTPException(status_code=400, detail=f"Nieznany sposób sortowania: {body.sort}")
    if body.sort in LEAVE_NET_SORTS and not has_leave_net(session.language):
        raise HTTPException(
            status_code=400,
            detail=f"Sortowanie „{body.sort}” nie jest dostępne dla tego języka (brak wytrenowanego modelu).",
        )
    raw = await run_in_threadpool(rank_suggestions, board, dawg, letters, 10, body.sort)
    suggestions = [Suggestion(**s) for s in raw]
    return SuggestionsResponse(suggestions=suggestions, letters=letters)


@router.post("/save-training", response_model=SaveTrainingResponse)
async def save_training(
    request: Request,
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
    grid = _validate_grid(raw_grid, get_pack(_scan_language(request)).spec.alphabet)

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
