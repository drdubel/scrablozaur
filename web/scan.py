"""Photo-to-board-state pipeline for the "scan board" web feature.

This is a *scanning assistant*, not a game: the user is playing a real
physical game (with real opponents), and the workflow is photo -> best word
suggestion -> place it on the real board -> photo again. A ScanSession just
tracks the last confirmed board state across that loop so each new photo can
lean on it: since tiles are only ever added in Scrabble, never removed or
changed, any cell the previous confirmed state had a tile in should still
have that same tile now. That's a much stronger signal than the dictionary
alone, so it takes priority -- see scan_board_image()'s `locked` set.

A language with the live scanner's models (`"scanner"` in its definition)
is read the way the app reads camera frames: the engine's grid finder, cell
reader and legal-board decoder (`scrablozaur.scan_frame`), which on the test
photos gets far fewer squares wrong than the classical pipeline. Any other
language goes through board_reader's OCR pipeline (a standalone script-style
package, not importable as a normal module) and a dictionary-driven
correction pass for genuinely *new* tiles: every horizontal/vertical run of
tiles is checked against the Dawg, and words that don't match get a
single-letter substitution attempted from the OCR's own ranked alternatives
before giving up and flagging the cells for the user to fix by hand.
"""

from __future__ import annotations

import re
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import scrablozaur
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOARD_READER_SRC = PROJECT_ROOT / "board_reader" / "src"
if str(BOARD_READER_SRC) not in sys.path:
    sys.path.insert(0, str(BOARD_READER_SRC))

from letter_classifier import alphabet as ocr_alphabet, set_language as set_ocr_language  # noqa: E402
from premium_layout import GRID  # noqa: E402
from read_board import read_board  # noqa: E402
from read_letters import classify_tiles  # noqa: E402

from web.engine import DEFAULT_LANGUAGE, Dawg, get_pack  # noqa: E402


def _use_language(language: str):
    """Point the classifier at `language` and return its spec.

    The classifier keeps its models and templates in per-language caches, so
    this is cheap after the first scan in each language.
    """
    spec = get_pack(language).spec
    set_ocr_language(spec)
    return spec


# How many rounds of single-letter correction to run: fixing one word can
# change a crossing word's letter too, so a couple of extra rounds lets
# those knock-on fixes settle instead of only fixing whichever word happens
# to be processed first.
_CORRECTION_ROUNDS = 3
_MAX_ALTERNATIVES_SHOWN = 4
# fuse_predictions() always returns 5 ranked alternatives padded out with
# whatever's left of the probability mass, even when the model had no real
# opinion beyond its top pick -- on a confident read the bottom few are
# essentially uniform noise (~1/32 each), not genuine second guesses. Without
# this floor, _verify_and_correct() will happily "fix" an already-correct
# letter into a noise-level alternative whenever that alternative happens to
# complete some other short dictionary word (observed on real output: a
# correct vertical "AĄ" -- not a word, but neither is it OCR error -- got
# corrected to "SĄ" purely because 'S' was sitting at 5.5% probability).
_MIN_ALTERNATIVE_PROB = 0.15


def empty_board() -> list[list[str]]:
    return [["-"] * GRID for _ in range(GRID)]


def board_is_empty(board: list[list[str]]) -> bool:
    return not any(cell != "-" for row in board for cell in row)


@dataclass
class ScanSession:
    session_id: str
    board: list[list[str]] = field(default_factory=empty_board)
    #: Which language's dictionary corrects the OCR output and validates the
    #: confirmed grid. Taken from the game session when there is one.
    language: str = DEFAULT_LANGUAGE


class ScanSessionStore:
    _sessions: dict[str, ScanSession] = {}

    @classmethod
    def create(
        cls, board: list[list[str]] | None = None, language: str = DEFAULT_LANGUAGE
    ) -> ScanSession:
        session = ScanSession(
            session_id=str(uuid.uuid4()),
            board=board or empty_board(),
            language=language,
        )
        cls._sessions[session.session_id] = session
        return session

    @classmethod
    def get(cls, sid: str | None) -> ScanSession | None:
        if not sid:
            return None
        return cls._sessions.get(sid)

    @classmethod
    def delete(cls, sid: str | None) -> None:
        if sid:
            cls._sessions.pop(sid, None)


_NO_BOARD = (
    "Nie udało się znaleźć planszy na zdjęciu. Spróbuj sfotografować "
    "całą planszę z góry, przy dobrym oświetleniu."
)

# The engine's scanner nets per language, loaded on first use.
_scanners: dict[str, tuple[scrablozaur.GridFinder, scrablozaur.CellReader]] = {}
_scanners_lock = threading.Lock()
# Photos are read at most this large: the scanner was checked on photos this
# size and on 1080p frames, and a 50-megapixel upload only costs memory.
_MAX_PHOTO_SIDE = 4096


def _scanner(
    language: str,
) -> tuple[scrablozaur.GridFinder, scrablozaur.CellReader] | None:
    """The live scanner's grid finder and cell reader for `language`, if it has them."""
    pack = get_pack(language)
    if not pack.spec.has_scanner:
        return None
    with _scanners_lock:
        if language not in _scanners:
            paths = pack.spec.scanner
            _scanners[language] = (
                scrablozaur.GridFinder(str(paths.grid_finder)),
                scrablozaur.CellReader(pack.lang, str(paths.cell_reader)),
            )
        return _scanners[language]


def _focal_px(path: str, size: tuple[int, int]) -> float:
    """The photo's focal length in pixels at `size`, from its EXIF 35 mm
    equivalent (a 36 mm wide frame); 26 mm, a typical phone's, without one.
    As `board_reader/scripts/export_scan_dataset.py` computes it for training."""
    try:
        exif = Image.open(path).getexif().get_ifd(0x8769)
        f35 = float(exif.get(0xA405) or 26)  # FocalLengthIn35mmFilm
    except (OSError, ValueError):
        f35 = 26.0
    return f35 / 36 * max(size)


def _scan_with_engine(
    path: str,
    prior_board: list[list[str]] | None,
    language: str,
    nets: tuple[scrablozaur.GridFinder, scrablozaur.CellReader],
) -> dict:
    """scan_board_image() through the engine's scanner, the app's: the most
    probable legal board, keeping every tile of *prior_board*."""
    bgr = cv2.imread(path)
    if bgr is None:
        return {"error": "Nie udało się odczytać zdjęcia."}
    scale = _MAX_PHOTO_SIDE / max(bgr.shape[:2])
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    prior = (
        prior_board
        if prior_board is not None and not board_is_empty(prior_board)
        else None
    )
    finder, reader = nets
    out = scrablozaur.scan_frame(
        finder,
        reader,
        get_pack(language).dawg,
        rgb.tobytes(),
        w,
        h,
        _focal_px(path, (w, h)),
        prior,
    )
    if out is None:
        return {"error": _NO_BOARD}
    decoded = out.decoded
    grid = [list(row) for row in decoded.letters]
    cells = []
    for r in range(GRID):
        row = []
        for c in range(GRID):
            p_tile, letters, p_blank = out.readings[r][c]
            letter = grid[r][c]
            ranked = sorted(letters, key=lambda lp: -lp[1])
            # What this photo alone makes of the square: empty, a blank ('?'), or a letter.
            seen = (
                "-"
                if p_tile < 0.5
                else "?"
                if p_blank > 0.5 or not ranked
                else ranked[0][0]
            )
            row.append(
                {
                    "letter": letter,
                    "confidence": round(
                        1 - p_tile
                        if letter == "-"
                        else p_tile * dict(ranked).get(letter, 0.0),
                        3,
                    ),
                    "alternatives": [
                        ch
                        for ch, p in ranked
                        if p >= _MIN_ALTERNATIVE_PROB and ch != letter
                    ][:_MAX_ALTERNATIVES_SHOWN],
                    # In a non-word -- or, on a board made legal only by
                    # overriding letters the reader was sure of, overridden.
                    "flagged": decoded.to_check[r][c],
                    # A tile of the last confirmed board this photo read otherwise.
                    "carried_over": prior is not None
                    and prior[r][c] != "-"
                    and seen != prior[r][c],
                }
            )
        cells.append(row)
    return {"cells": cells, "board": grid}


def scan_board_image(
    path: str,
    prior_board: list[list[str]] | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> dict:
    """Read the photo at *path* and return either {"error": str} or
    {"cells": [[...]], "board": [[str]]}.

    *prior_board* is the previous confirmed state from the same ScanSession,
    if any -- every cell it has a tile in overrides this photo's own
    reading (see module docstring), and is excluded from dictionary
    correction/flagging since it was already validated when first confirmed.
    """
    nets = _scanner(language)
    if nets is not None:
        return _scan_with_engine(path, prior_board, language, nets)

    rotated, mesh, _cells, verdicts, shift = read_board(path, show=False)
    if verdicts is None:
        return {"error": _NO_BOARD}

    readings = classify_tiles(rotated, mesh, verdicts, global_shift=shift)
    if not readings:
        return {"error": "Nie wykryto żadnych kafelków na planszy."}

    spec = _use_language(language)
    letters = set(spec.alphabet)
    grid = empty_board()
    confidence = [[0.0] * GRID for _ in range(GRID)]
    alternatives: list[list[list[str]]] = [
        [[] for _ in range(GRID)] for _ in range(GRID)
    ]
    for (r, c), (letter, conf, alts) in readings.items():
        lowered = letter.lower() if letter else ""
        grid[r][c] = lowered if lowered in letters else "?"
        confidence[r][c] = conf
        alternatives[r][c] = [
            a.lower()
            for a, p in alts
            if p >= _MIN_ALTERNATIVE_PROB and a.lower() in letters
        ]

    locked: set[tuple[int, int]] = set()
    carried_over: set[tuple[int, int]] = set()
    if prior_board is not None:
        for r in range(GRID):
            for c in range(GRID):
                prior_letter = prior_board[r][c]
                if prior_letter == "-":
                    continue
                locked.add((r, c))
                if grid[r][c] != prior_letter:
                    # Disagreement with history: trust the prior confirmed
                    # state (tiles don't change once played), but keep this
                    # photo's own fresh reading as the first-offered
                    # alternative in case the *prior* was actually the
                    # mistake (e.g. a misreview last time).
                    fresh = grid[r][c]
                    if fresh not in ("-", "?") and fresh not in alternatives[r][c]:
                        alternatives[r][c] = [fresh, *alternatives[r][c]]
                    grid[r][c] = prior_letter
                    carried_over.add((r, c))

    pack = get_pack(language)
    flagged = _verify_and_correct(pack.dawg, grid, alternatives, locked=locked)

    cells = [
        [
            {
                "letter": grid[r][c],
                "confidence": round(confidence[r][c], 3),
                "alternatives": [a for a in alternatives[r][c] if a != grid[r][c]][
                    :_MAX_ALTERNATIVES_SHOWN
                ],
                "flagged": (r, c) in flagged,
                "carried_over": (r, c) in carried_over,
            }
            for c in range(GRID)
        ]
        for r in range(GRID)
    ]
    return {"cells": cells, "board": grid}


# ── Training-set export ───────────────────────────────────────────────────────
# Lets a user opt in, per photo, to append it to board_reader/'s own eval/
# retraining set (see [[project-ocr-pipeline]]) -- same imgN_<difficulty>.jpg
# + boardN.txt layout board_reader/tests/ground_truth.py already expects, so
# no changes are needed there to pick these up.

BOARD_READER_TEST_IN = PROJECT_ROOT / "board_reader" / "test" / "in"
BOARD_READER_TEST_OUT = PROJECT_ROOT / "board_reader" / "test" / "out"

_IMG_ID_RE = re.compile(r"img(\d+)_[emh]\.jpg$")
_BOARD_ID_RE = re.compile(r"board(\d+)\.txt$")


def evaluate_raw_recognition(
    path: str, confirmed_board: list[list[str]], language: str = DEFAULT_LANGUAGE
) -> tuple[str, dict]:
    """Compare this photo's *unassisted* OCR reading (no dictionary
    correction, no prior-state help -- just what the classifier alone says)
    against the user's final confirmed board, and bucket the result into a
    difficulty label for the training set: 'e' if the raw reader got
    (essentially) every tile right on its own, 'm' if a minority were
    wrong, 'h' if a majority were wrong. Only cells the confirmed board
    actually has a tile in count -- there's nothing to grade where both
    agree on empty."""
    _use_language(language)
    rotated, mesh, _cells, verdicts, shift = read_board(path, show=False)
    if verdicts is None:
        raise ValueError("Nie udało się znaleźć planszy na zdjęciu.")

    readings = classify_tiles(rotated, mesh, verdicts, global_shift=shift)
    letters = ocr_alphabet()
    raw = empty_board()
    for (r, c), (letter, _conf, _alts) in readings.items():
        raw[r][c] = letter.lower() if letter and letter in letters else "?"

    total = matched = 0
    for r in range(GRID):
        for c in range(GRID):
            if confirmed_board[r][c] == "-":
                continue
            total += 1
            if raw[r][c] == confirmed_board[r][c]:
                matched += 1

    if total == 0:
        return "e", {"matched": 0, "total": 0, "match_ratio": 1.0}

    ratio = matched / total
    difficulty = "e" if ratio == 1.0 else ("m" if ratio >= 0.5 else "h")
    return difficulty, {
        "matched": matched,
        "total": total,
        "match_ratio": round(ratio, 3),
    }


def _next_training_id() -> int:
    """One counter shared across test/in and test/out so the two never
    drift out of sync even if a previous save only half-completed."""
    ids = [
        int(m.group(1))
        for p in BOARD_READER_TEST_IN.glob("img*_*.jpg")
        if (m := _IMG_ID_RE.match(p.name))
    ]
    ids += [
        int(m.group(1))
        for p in BOARD_READER_TEST_OUT.glob("board*.txt")
        if (m := _BOARD_ID_RE.match(p.name))
    ]
    return max(ids, default=-1) + 1


def _format_ground_truth(board: list[list[str]]) -> str:
    """board_reader/tests/ground_truth.py's expected format: one row per
    line, space-separated single-character tokens, uppercase letters
    (matching classify_board()'s own output alphabet), '-' for empty."""
    lines = (
        " ".join(cell.upper() if cell != "-" else "-" for cell in row) for row in board
    )
    return "\n".join(lines) + "\n"


def save_training_example(
    image_bytes: bytes, board: list[list[str]], difficulty: str
) -> int:
    """Append this photo + its user-confirmed board to board_reader/'s test
    set. Re-encodes through cv2 rather than writing *image_bytes* verbatim
    so the file on disk is genuinely a .jpg regardless of what the browser
    actually sent (phones/pickers don't always send real JPEGs)."""
    BOARD_READER_TEST_IN.mkdir(parents=True, exist_ok=True)
    BOARD_READER_TEST_OUT.mkdir(parents=True, exist_ok=True)

    image_id = _next_training_id()

    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Nie udało się odczytać zdjęcia.")
    cv2.imwrite(str(BOARD_READER_TEST_IN / f"img{image_id}_{difficulty}.jpg"), img)
    (BOARD_READER_TEST_OUT / f"board{image_id}.txt").write_text(
        _format_ground_truth(board), encoding="utf-8"
    )

    return image_id


def _runs(grid: list[list[str]]) -> list[list[tuple[int, int]]]:
    """Every contiguous horizontal/vertical run of tiles with length >= 2."""
    runs = []
    for r in range(GRID):
        c = 0
        while c < GRID:
            if grid[r][c] == "-":
                c += 1
                continue
            start = c
            while c < GRID and grid[r][c] != "-":
                c += 1
            if c - start >= 2:
                runs.append([(r, cc) for cc in range(start, c)])
    for c in range(GRID):
        r = 0
        while r < GRID:
            if grid[r][c] == "-":
                r += 1
                continue
            start = r
            while r < GRID and grid[r][c] != "-":
                r += 1
            if r - start >= 2:
                runs.append([(rr, c) for rr in range(start, r)])
    return runs


def _isolated_tiles(grid: list[list[str]]) -> list[tuple[int, int]]:
    """Tiles with no neighbour in either direction -- never valid on a real
    board (the minimum word length is 2), so these are almost always a
    false-positive tile detection rather than a misread letter."""
    in_run = {pos for run in _runs(grid) for pos in run}
    return [
        (r, c)
        for r in range(GRID)
        for c in range(GRID)
        if grid[r][c] != "-" and (r, c) not in in_run
    ]


def _verify_and_correct(
    dawg: Dawg,
    grid: list[list[str]],
    alternatives: list[list[list[str]]],
    locked: set[tuple[int, int]] = frozenset(),
) -> set[tuple[int, int]]:
    """Mutates *grid* in place, substituting a cell's letter with one of its
    OCR-ranked alternatives wherever that turns an invalid word into a valid
    one. Only tries one substitution per word per round -- multi-letter
    misreads within the same word are rare enough at the OCR's ~97% letter
    accuracy that a full combinatorial search isn't worth it; those cases
    are left for the user to fix in the review step instead. Returns the
    set of cells still part of an invalid word (or fully isolated) after
    correction, for the frontend to flag.

    *locked* cells (carried over from a previous confirmed ScanSession
    state) are never substituted and never flagged -- they were already
    validated when first confirmed, so re-litigating them on every
    subsequent photo would just be noise. A run that mixes locked and new
    cells is still checked (a misread new tile can combine with old ones
    into an invalid word) and still flagged, but only its new cells are
    reported -- the locked ones stay untouched either way.
    """
    for _ in range(_CORRECTION_ROUNDS):
        changed = False
        for run in _runs(grid):
            if all(pos in locked for pos in run):
                continue
            word = "".join(grid[r][c] for r, c in run)
            if dawg.contains(word):
                continue
            for r, c in run:
                if (r, c) in locked:
                    continue
                original = grid[r][c]
                fixed_here = False
                for cand in alternatives[r][c]:
                    if cand == original:
                        continue
                    grid[r][c] = cand
                    if dawg.contains("".join(grid[rr][cc] for rr, cc in run)):
                        fixed_here = True
                        break
                    grid[r][c] = original
                if fixed_here:
                    changed = True
                    break
        if not changed:
            break

    return flag_invalid(dawg, grid, locked=locked)


def flag_invalid(
    dawg: Dawg,
    grid: list[list[str]],
    locked: set[tuple[int, int]] = frozenset(),
) -> set[tuple[int, int]]:
    """Just the flagging half of _verify_and_correct(), with no substitution
    attempted -- used to re-check the board after the user hand-edits a
    cell in the review step, where silently substituting some *other*
    cell's letter out from under them would fight their own correction
    rather than help it."""
    flagged: set[tuple[int, int]] = set()
    for run in _runs(grid):
        if all(pos in locked for pos in run):
            continue
        word = "".join(grid[r][c] for r, c in run)
        if not dawg.contains(word):
            flagged.update(pos for pos in run if pos not in locked)
    flagged.update(pos for pos in _isolated_tiles(grid) if pos not in locked)
    return flagged
