from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from web.engine import Dawg, LanguagePack
from rules import TurnResult
from web.game import (LEAVE_NET_SORTS, SORT_MODES, GameMode, has_leave_net, _check_game_over, _deduct_tiles, _refill_rack,
                      _tiles_used_for_word, computer_auto_play, compute_move_rating,
                      get_suggestions, get_suggestions_for_letters, rack_contains,
                      validate_rack_for_word)
from web.models import (BoardStateResponse, DefinitionResponse, ExchangeTilesRequest,
                        PlaceComputerWordRequest, PlaceHumanWordRequest, PreviewScoreResponse,
                        SetComputerLettersRequest, Suggestion, SuggestionsResponse)
from web.definitions import lookup
from web.deps import require_session as _require_session, session_dawg, session_pack
from web.routers.game import _state_response

router = APIRouter(prefix="/board")

_SANDBOX_ONLY = "This endpoint is only available in sandbox mode."


def _check_sort(sort: str, language: str) -> str:
    """Validate the suggestion sort mode, rejecting rather than silently
    falling back -- a typo'd mode should not quietly return score order, and
    nor should an ordering this language cannot actually compute."""
    if sort not in SORT_MODES:
        raise HTTPException(status_code=400, detail=f"Nieznany sposób sortowania: {sort}")
    if sort in LEAVE_NET_SORTS and not has_leave_net(language):
        raise HTTPException(
            status_code=400,
            detail=f"Sortowanie „{sort}” nie jest dostępne dla tego języka (brak wytrenowanego modelu).",
        )
    return sort


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
    dawg: Dawg = Depends(session_dawg),
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

    # Only competitive mode has a real rack, so only there can a play involve a
    # blank. Sandbox scores every tile at face value (see letters_for_scoring
    # above), so it places none.
    used = (
        _tiles_used_for_word(pre_letters, word, pre_grid, body.row, body.col, body.horizontal)
        if session.game_mode == GameMode.COMPETITIVE
        else None
    )

    session.push_undo()
    session.record_placement(word, body.row, body.col, body.horizontal, session.current_player_idx)
    session.board.place_word(word, body.row, body.col, body.horizontal, used)
    session.current_player.score += score
    session.is_first_move = False

    if session.game_mode == GameMode.COMPETITIVE:
        _deduct_tiles(session.current_player, word, pre_grid, body.row, body.col, body.horizontal)
        _refill_rack(session, session.current_player)
        session.streaks.record(TurnResult.PLAYED)
        _check_game_over(session, session.current_player_idx)

    if not session.game_over:
        session.advance_turn()

    if session.game_mode == GameMode.COMPETITIVE and not session.game_over:
        session.last_computer_move = await run_in_threadpool(computer_auto_play, session, dawg)

    return _state_response(session)


@router.post("/skip", response_model=BoardStateResponse)
async def skip_turn(
    request: Request,
    dawg: Dawg = Depends(session_dawg),
) -> BoardStateResponse:
    """Current player skips their turn (plays no word) -- the standard
    Scrabble "pass". The game only ends once nobody has played a word for
    the no-play limit in src/rules.py, not after a single skip."""
    session = _require_session(request)
    if session.game_over:
        raise HTTPException(status_code=400, detail="Gra już się zakończyła.")
    session.push_undo()

    if session.game_mode == GameMode.COMPETITIVE:
        session.streaks.record(TurnResult.NO_ACTION)
        _check_game_over(session, session.current_player_idx)

    if not session.game_over:
        session.advance_turn()
        if session.game_mode == GameMode.COMPETITIVE:
            session.last_computer_move = await run_in_threadpool(computer_auto_play, session, dawg)

    return _state_response(session)


@router.post("/next-move", response_model=BoardStateResponse)
async def next_auto_move(
    request: Request,
    dawg: Dawg = Depends(session_dawg),
) -> BoardStateResponse:
    """Advance a SANDBOX_AUTO game by one turn -- every player in this mode
    is a computer, so this is the only kind of turn it has. The live
    auto-play UI calls this once per step, or repeatedly for autoplay."""
    session = _require_session(request)
    if session.game_mode != GameMode.SANDBOX_AUTO:
        raise HTTPException(status_code=400, detail="Dostępne tylko w trybie automatycznym.")
    if session.game_over:
        raise HTTPException(status_code=400, detail="Gra już się zakończyła.")

    session.push_undo()
    session.last_computer_move = await run_in_threadpool(computer_auto_play, session, dawg)
    return _state_response(session)


@router.post("/exchange", response_model=BoardStateResponse)
async def exchange_tiles(
    body: ExchangeTilesRequest,
    request: Request,
    dawg: Dawg = Depends(session_dawg),
) -> BoardStateResponse:
    """Return the given tiles to the bag and draw the same number of new
    ones instead of playing a word — only legal in COMPETITIVE mode while
    at least 7 tiles remain in the bag (the standard exchange rule). A real,
    repeatable action: unlike /board/skip it never counts toward the no-play
    streak, so a player can exchange as many turns in a row as they want
    without that alone ending the game. It is still scoreless, so the separate
    scoreless cap in src/rules.py does eventually bound it."""
    session = _require_session(request)
    if session.game_over:
        raise HTTPException(status_code=400, detail="Gra już się zakończyła.")
    if session.game_mode != GameMode.COMPETITIVE or not session.tile_bag:
        raise HTTPException(status_code=400, detail="Wymiana liter jest dostępna tylko w trybie rywalizacji.")

    letters = body.letters.lower()
    # Board.can_exchange() is an instance method reading the *engine's own*
    # internal bag, which the web app doesn't use for bag tracking (it has
    # its own TileBag, session.tile_bag) -- inline the same standard-rule
    # threshold (>= 7 tiles) that method documents instead of misapplying it.
    if session.tile_bag.remaining() < 7:
        raise HTTPException(status_code=400, detail="Za mało liter w worku, żeby wymienić (potrzeba co najmniej 7).")

    player = session.current_player
    if not rack_contains(player.letters, letters):
        raise HTTPException(status_code=400, detail="Nie masz tych liter na stojaku.")

    session.push_undo()
    rack_chars = list(player.letters)
    for ch in letters:
        rack_chars.remove(ch)
    player.letters = "".join(rack_chars) + "".join(session.tile_bag.exchange(list(letters)))
    # A real, repeatable action, so it never counts toward the no-play streak --
    # but it scores nothing, and the scoreless cap is what stops two players
    # exchanging at each other forever.
    session.streaks.record(TurnResult.EXCHANGED)
    _check_game_over(session, session.current_player_idx)

    if not session.game_over:
        session.advance_turn()
        session.last_computer_move = await run_in_threadpool(computer_auto_play, session, dawg)

    return _state_response(session)


@router.post("/pass", response_model=BoardStateResponse)
async def pass_turn(request: Request) -> BoardStateResponse:
    """Current human player resigns ("Poddaj się") -- a deliberate,
    immediate stop, distinct from /board/skip's ordinary Scrabble pass. If
    all humans resign, the game ends. Not to be confused with a standard
    pass, which never ends the game after just one turn."""
    session = _require_session(request)
    if session.game_over:
        raise HTTPException(status_code=400, detail="Gra już się zakończyła.")

    session.passed_players.add(session.current_player_idx)
    human_idxs = {i for i, p in enumerate(session.players) if not p.is_computer}

    if human_idxs <= session.passed_players:
        session.game_over = True
    else:
        session.advance_turn()

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
    sort: str = "score",
    dawg: Dawg = Depends(session_dawg),
) -> SuggestionsResponse:
    session = _require_session(request)
    if session.game_mode != GameMode.SANDBOX:
        raise HTTPException(status_code=400, detail=_SANDBOX_ONLY)
    # `sim` spends real time thinking, so it goes to a worker thread like the
    # bot does rather than stalling the event loop.
    raw = await run_in_threadpool(get_suggestions, session, dawg, 10, _check_sort(sort, session.language))
    suggestions = [Suggestion(**s) for s in raw]
    return SuggestionsResponse(suggestions=suggestions, letters=session.current_player.letters)


@router.get("/hints", response_model=SuggestionsResponse)
async def get_hints(
    request: Request,
    sort: str = "score",
    dawg: Dawg = Depends(session_dawg),
) -> SuggestionsResponse:
    session = _require_session(request)
    if session.game_mode != GameMode.COMPETITIVE:
        raise HTTPException(status_code=400, detail="Podpowiedzi dostępne tylko w trybie rywalizacji.")
    letters = session.current_player.letters
    raw = await run_in_threadpool(
        get_suggestions_for_letters, session, dawg, letters, 20, _check_sort(sort, session.language)
    )
    suggestions = [Suggestion(**s) for s in raw]
    return SuggestionsResponse(suggestions=suggestions, letters=letters)


@router.post("/computer-move", response_model=BoardStateResponse)
async def place_computer_word(
    body: PlaceComputerWordRequest,
    request: Request,
    dawg: Dawg = Depends(session_dawg),
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
    # Sandbox lets the rack be set arbitrarily and tolerates a play the rack
    # can't actually cover, so only claim to know which tiles were blanks when
    # the rack really does cover it.
    used = (
        _tiles_used_for_word(
            session.current_player.letters, word, grid, body.row, body.col, body.horizontal
        )
        if validate_rack_for_word(
            session.current_player.letters, word, grid, body.row, body.col, body.horizontal
        )
        else None
    )

    session.push_undo()
    session.record_placement(word, body.row, body.col, body.horizontal, session.current_player_idx)
    session.board.place_word(word, body.row, body.col, body.horizontal, used)
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
    dawg: Dawg = Depends(session_dawg),
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


@router.get("/definition/{word}", response_model=DefinitionResponse)
async def get_definition(
    word: str, pack: LanguagePack = Depends(session_pack)
) -> DefinitionResponse:
    word = word.lower()
    definitions = await run_in_threadpool(lookup, word, pack.spec.definitions)
    return DefinitionResponse(
        word=word, definitions=definitions, found=bool(definitions)
    )
