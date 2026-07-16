from __future__ import annotations

import re
import urllib.parse
import urllib.request

from fastapi import APIRouter, Depends, HTTPException, Request

from web.engine import Board, Dawg, get_dawg
from web.game import (GameMode, _check_game_over, _deduct_tiles, _refill_rack,
                      computer_auto_play, compute_move_rating, get_suggestions,
                      get_suggestions_for_letters, rack_contains, validate_rack_for_word)
from web.models import (BoardStateResponse, DefinitionResponse, ExchangeTilesRequest,
                        PlaceComputerWordRequest, PlaceHumanWordRequest, PreviewScoreResponse,
                        SetComputerLettersRequest, Suggestion, SuggestionsResponse)
from web.routers.game import _require_session, _state_response

router = APIRouter(prefix="/board")

_SANDBOX_ONLY = "This endpoint is only available in sandbox mode."


def _check_connectivity(session, word: str, row: int, col: int, horizontal: bool) -> None:
    """Raise ValueError if the word is not connected to existing tiles (or first-move rules)."""
    grid = session.board_grid()
    word_len = len(word)

    if session.is_first_move:
        # Must pass through center (7, 7)
        touches_center = any(
            (row == 7 and col + i == 7) if horizontal else (row + i == 7 and col == 7)
            for i in range(word_len)
        )
        if not touches_center:
            raise ValueError("Pierwsze słowo musi przechodzić przez środek planszy (pole oznaczone ★).")
        return

    # Subsequent moves: must share or touch at least one existing tile
    for i in range(word_len):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if grid[r][c] != "-":
            return  # overlaps existing tile — connected
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 15 and 0 <= nc < 15 and grid[nr][nc] != "-":
                return  # adjacent to existing tile — connected
    raise ValueError("Słowo musi być połączone z już istniejącym słowem na planszy.")


@router.post("/human-move", response_model=BoardStateResponse)
async def place_human_word(
    body: PlaceHumanWordRequest,
    request: Request,
    dawg: Dawg = Depends(get_dawg),
) -> BoardStateResponse:
    session = _require_session(request)
    word = body.word.lower()

    if not dawg.contains(word):
        raise HTTPException(status_code=400, detail=f"'{body.word}' nie ma w słowniku.")

    try:
        _check_connectivity(session, word, body.row, body.col, body.horizontal)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        session.board.check_word_placement(dawg, word, body.row, body.col, body.horizontal)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Capture grid and rack before any mutation
    pre_grid = session.board_grid()
    pre_letters = session.current_player.letters

    if session.game_mode == GameMode.COMPETITIVE:
        if not validate_rack_for_word(
            pre_letters, word, pre_grid, body.row, body.col, body.horizontal
        ):
            raise HTTPException(status_code=400, detail="Nie masz wymaganych liter na stojaku.")

    # Pass real rack in competitive (engine uses it to score blanks at 0),
    # pass the word itself in sandbox (all tiles score at face value).
    letters_for_scoring = (
        pre_letters if session.game_mode == GameMode.COMPETITIVE else word
    )
    try:
        score = session.board.calculate_word_points(
            word, body.row, body.col, body.horizontal, letters_for_scoring
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Rate the move before mutating the board (rating needs pre-move board state)
    if session.game_mode == GameMode.COMPETITIVE and pre_letters:
        session.last_move_rating = compute_move_rating(session, dawg, pre_letters, score)
    else:
        session.last_move_rating = None

    session.push_undo()
    session.record_placement(word, body.row, body.col, body.horizontal, session.current_player_idx)
    session.board.place_word(word, body.row, body.col, body.horizontal)
    session.current_player.score += score
    session.is_first_move = False

    if session.game_mode == GameMode.COMPETITIVE:
        _deduct_tiles(session.current_player, word, pre_grid, body.row, body.col, body.horizontal)
        _refill_rack(session, session.current_player)
        session.consecutive_no_play = 0
        _check_game_over(session, session.current_player_idx)

    if not session.game_over:
        session.advance_turn()

    if session.game_mode == GameMode.COMPETITIVE and not session.game_over:
        session.last_computer_move = computer_auto_play(session, dawg)

    return _state_response(session)


@router.post("/skip", response_model=BoardStateResponse)
async def skip_turn(
    request: Request,
    dawg: Dawg = Depends(get_dawg),
) -> BoardStateResponse:
    session = _require_session(request)
    if session.game_over:
        raise HTTPException(status_code=400, detail="Gra już się zakończyła.")
    session.push_undo()
    session.advance_turn()

    if session.game_mode == GameMode.COMPETITIVE:
        session.last_computer_move = computer_auto_play(session, dawg)

    return _state_response(session)


@router.post("/exchange", response_model=BoardStateResponse)
async def exchange_tiles(
    body: ExchangeTilesRequest,
    request: Request,
    dawg: Dawg = Depends(get_dawg),
) -> BoardStateResponse:
    """Return the given tiles to the bag and draw the same number of new
    ones instead of playing a word — only legal in COMPETITIVE mode while
    at least 7 tiles remain in the bag (Board.can_exchange)."""
    session = _require_session(request)
    if session.game_over:
        raise HTTPException(status_code=400, detail="Gra już się zakończyła.")
    if session.game_mode != GameMode.COMPETITIVE or not session.tile_bag:
        raise HTTPException(status_code=400, detail="Wymiana liter jest dostępna tylko w trybie rywalizacji.")

    letters = body.letters.lower()
    if not Board.can_exchange(session.tile_bag.remaining()):
        raise HTTPException(status_code=400, detail="Za mało liter w worku, żeby wymienić (potrzeba co najmniej 7).")

    player = session.current_player
    if not rack_contains(player.letters, letters):
        raise HTTPException(status_code=400, detail="Nie masz tych liter na stojaku.")

    session.push_undo()
    rack_chars = list(player.letters)
    for ch in letters:
        rack_chars.remove(ch)
    player.letters = "".join(rack_chars) + "".join(session.tile_bag.exchange(list(letters)))
    session.consecutive_no_play += 1
    _check_game_over(session, session.current_player_idx)

    if not session.game_over:
        session.advance_turn()
        session.last_computer_move = computer_auto_play(session, dawg)

    return _state_response(session)


@router.post("/pass", response_model=BoardStateResponse)
async def pass_turn(request: Request, dawg: Dawg = Depends(get_dawg)) -> BoardStateResponse:
    """Current player passes their turn (plays no word). Standard rule: the
    game ends once nobody has played a word for CONSECUTIVE_NO_PLAY_LIMIT
    turns in a row, not after a single pass."""
    session = _require_session(request)
    if session.game_over:
        raise HTTPException(status_code=400, detail="Gra już się zakończyła.")

    session.push_undo()

    if session.game_mode == GameMode.COMPETITIVE:
        session.consecutive_no_play += 1
        _check_game_over(session, session.current_player_idx)

    if not session.game_over:
        session.advance_turn()
        if session.game_mode == GameMode.COMPETITIVE:
            session.last_computer_move = computer_auto_play(session, dawg)

    return _state_response(session)


@router.post("/undo", response_model=BoardStateResponse)
async def undo_move(request: Request) -> BoardStateResponse:
    session = _require_session(request)
    if not session.pop_undo():
        raise HTTPException(status_code=400, detail="Brak ruchów do cofnięcia.")
    return _state_response(session)


# ── Sandbox-only endpoints ────────────────────────────────────────────────────

@router.post("/set-letters", response_model=BoardStateResponse)
async def set_computer_letters(
    body: SetComputerLettersRequest,
    request: Request,
) -> BoardStateResponse:
    session = _require_session(request)
    if session.game_mode != GameMode.SANDBOX:
        raise HTTPException(status_code=400, detail=_SANDBOX_ONLY)
    session.current_player.letters = body.letters.lower()
    return _state_response(session)


@router.post("/suggest", response_model=SuggestionsResponse)
async def suggest_moves(
    request: Request,
    dawg: Dawg = Depends(get_dawg),
) -> SuggestionsResponse:
    session = _require_session(request)
    if session.game_mode != GameMode.SANDBOX:
        raise HTTPException(status_code=400, detail=_SANDBOX_ONLY)
    raw = get_suggestions(session, dawg, n=10)
    suggestions = [Suggestion(**s) for s in raw]
    return SuggestionsResponse(suggestions=suggestions, letters=session.current_player.letters)


@router.get("/hints", response_model=SuggestionsResponse)
async def get_hints(
    request: Request,
    dawg: Dawg = Depends(get_dawg),
) -> SuggestionsResponse:
    session = _require_session(request)
    if session.game_mode != GameMode.COMPETITIVE:
        raise HTTPException(status_code=400, detail="Podpowiedzi dostępne tylko w trybie rywalizacji.")
    letters = session.current_player.letters
    raw = get_suggestions_for_letters(session, dawg, letters, n=20)
    suggestions = [Suggestion(**s) for s in raw]
    return SuggestionsResponse(suggestions=suggestions, letters=letters)


@router.post("/computer-move", response_model=BoardStateResponse)
async def place_computer_word(
    body: PlaceComputerWordRequest,
    request: Request,
    dawg: Dawg = Depends(get_dawg),
) -> BoardStateResponse:
    session = _require_session(request)
    if session.game_mode != GameMode.SANDBOX:
        raise HTTPException(status_code=400, detail=_SANDBOX_ONLY)

    word = body.word.lower()

    try:
        session.board.check_word_placement(dawg, word, body.row, body.col, body.horizontal)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        score = session.board.calculate_word_points(
            word, body.row, body.col, body.horizontal, session.current_player.letters
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    grid = session.board_grid()
    session.push_undo()
    session.record_placement(word, body.row, body.col, body.horizontal, session.current_player_idx)
    session.board.place_word(word, body.row, body.col, body.horizontal)
    session.current_player.score += score
    session.is_first_move = False

    rack = list(session.current_player.letters)
    for i, ch in enumerate(word):
        r = body.row if body.horizontal else body.row + i
        c = body.col + i if body.horizontal else body.col
        if grid[r][c] == "-":
            if ch in rack:
                rack.remove(ch)
            elif "?" in rack:
                rack.remove("?")
    session.current_player.letters = "".join(rack)

    session.advance_turn()
    return _state_response(session)


@router.post("/preview-score", response_model=PreviewScoreResponse)
async def preview_score(
    body: PlaceHumanWordRequest,
    request: Request,
    dawg: Dawg = Depends(get_dawg),
) -> PreviewScoreResponse:
    session = _require_session(request)
    word = body.word.lower()

    if not dawg.contains(word):
        return PreviewScoreResponse(error="not_in_dict")

    try:
        _check_connectivity(session, word, body.row, body.col, body.horizontal)
        session.board.check_word_placement(dawg, word, body.row, body.col, body.horizontal)
    except Exception:
        return PreviewScoreResponse(error="invalid_placement")

    letters_for_scoring = (
        session.current_player.letters
        if session.game_mode == GameMode.COMPETITIVE
        else word
    )
    try:
        score = session.board.calculate_word_points(
            word, body.row, body.col, body.horizontal, letters_for_scoring
        )
        return PreviewScoreResponse(score=score)
    except Exception:
        return PreviewScoreResponse(error="score_error")


_SKIP_P = re.compile(
    r"^(SŁOWNIK SJP|KOMENTARZE|PROSIMY|POWIĄZANE|dopuszczal|niedopuszczal|function |-$|\(brak\)|dodaj$|OK$|nazwisko$|imię$)",
    re.IGNORECASE,
)
_HTML_TAG = re.compile(r"<[^>]+>")
_ENTITIES = {"&quot;": '"', "&amp;": "&", "&nbsp;": " ", "&lt;": "<", "&gt;": ">"}


def _clean_html(s: str) -> str:
    s = _HTML_TAG.sub("", s)
    for ent, ch in _ENTITIES.items():
        s = s.replace(ent, ch)
    return " ".join(s.split())


def _fetch_sjp(word: str) -> list[tuple[str, str]]:
    """Fetch entries from sjp.pl for *word*.

    Returns list of (lemma, definition) tuples — sjp.pl transparently maps
    inflected forms to their base form, so "zwie" yields ("zwać", "...") etc.
    """
    url = f"https://sjp.pl/{urllib.parse.quote(word)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scrablozaur/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8")
    except Exception:
        return []

    raw = [
        _clean_html(p)
        for p in re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
    ]
    # Stop at comments section — everything after it is user noise
    paragraphs = []
    for p in raw:
        if re.match(r"^KOMENTARZE", p, re.IGNORECASE) or re.match(r"^POWIĄZANE", p, re.IGNORECASE):
            break
        if p and not _SKIP_P.search(p):
            paragraphs.append(p)

    entries: list[tuple[str, str]] = []
    i = 0
    while i < len(paragraphs):
        t = paragraphs[i]
        is_lemma = (
            not t.startswith("znaczenie")
            and len(t) < 60
            and not re.search(r"\d\.", t)
        )
        if is_lemma:
            # Skip proper nouns (capitalized lemmas not matching the searched word)
            if t[0].isupper() and t.lower() != word:
                i += 1
                continue
            # Skip the "znaczenie: info (N)" line if present
            j = i + 1
            if j < len(paragraphs) and paragraphs[j].startswith("znaczenie"):
                j += 1
            if j < len(paragraphs):
                defn = paragraphs[j]
                if not _SKIP_P.search(defn):
                    entries.append((t, defn))
                    i = j + 1
                    continue
        i += 1
    return entries[:3]


@router.get("/definition/{word}", response_model=DefinitionResponse)
async def get_definition(word: str) -> DefinitionResponse:
    word = word.lower()
    entries = _fetch_sjp(word)
    if not entries:
        return DefinitionResponse(word=word, definitions=[], found=False)
    definitions = [
        f"{lemma} — {defn}" if lemma.lower() != word else defn
        for lemma, defn in entries
    ]
    return DefinitionResponse(word=word, definitions=definitions, found=True)
