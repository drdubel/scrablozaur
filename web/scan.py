"""Photo-to-board-state pipeline for the "scan board" web feature.

This is a *scanning assistant*, not a game: the user is playing a real
physical game (with real opponents), and the workflow is photo -> best word
suggestion -> place it on the real board -> photo again. A ScanSession just
tracks the last confirmed board state across that loop so each new photo can
lean on it: since tiles are only ever added in Scrabble, never removed or
changed, any cell the previous confirmed state had a tile in should still
have that same tile now. That's a much stronger signal than the dictionary
alone, so it takes priority -- see scan_board_image()'s `locked` set.

Also wraps board_reader's OCR pipeline (a standalone script-style package,
not importable as a normal module) and adds a dictionary-driven correction
pass for genuinely *new* tiles: every horizontal/vertical run of tiles is
checked against the Dawg, and words that don't match get a single-letter
substitution attempted from the OCR's own ranked alternatives before giving
up and flagging the cells for the user to fix by hand.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOARD_READER_SRC = PROJECT_ROOT / "board_reader" / "src"
if str(BOARD_READER_SRC) not in sys.path:
    sys.path.insert(0, str(BOARD_READER_SRC))

from letter_classifier import POLISH_ALPHABET  # noqa: E402
from premium_layout import GRID  # noqa: E402
from read_board import read_board  # noqa: E402
from read_letters import classify_tiles  # noqa: E402

from web.engine import Dawg, get_dawg  # noqa: E402

POLISH_LOWER = set(POLISH_ALPHABET.lower())

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


class ScanSessionStore:
    _sessions: dict[str, ScanSession] = {}

    @classmethod
    def create(cls, board: list[list[str]] | None = None) -> ScanSession:
        session = ScanSession(session_id=str(uuid.uuid4()), board=board or empty_board())
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


def scan_board_image(path: str, prior_board: list[list[str]] | None = None) -> dict:
    """Run the OCR pipeline on the photo at *path* and return either
    {"error": str} or {"cells": [[...]], "board": [[str]]}.

    *prior_board* is the previous confirmed state from the same ScanSession,
    if any -- every cell it has a tile in overrides this photo's own
    reading (see module docstring), and is excluded from dictionary
    correction/flagging since it was already validated when first confirmed.
    """
    rotated, mesh, _cells, verdicts, shift = read_board(path, show=False)
    if verdicts is None:
        return {
            "error": (
                "Nie udało się znaleźć planszy na zdjęciu. Spróbuj sfotografować "
                "całą planszę z góry, przy dobrym oświetleniu."
            )
        }

    readings = classify_tiles(rotated, mesh, verdicts, global_shift=shift)
    if not readings:
        return {"error": "Nie wykryto żadnych kafelków na planszy."}

    grid = empty_board()
    confidence = [[0.0] * GRID for _ in range(GRID)]
    alternatives: list[list[list[str]]] = [[[] for _ in range(GRID)] for _ in range(GRID)]
    for (r, c), (letter, conf, alts) in readings.items():
        grid[r][c] = letter.lower() if letter and letter in POLISH_ALPHABET else "?"
        confidence[r][c] = conf
        alternatives[r][c] = [
            a.lower() for a, p in alts if p >= _MIN_ALTERNATIVE_PROB and a.lower() in POLISH_LOWER
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

    dawg = get_dawg()
    flagged = _verify_and_correct(dawg, grid, alternatives, locked=locked)

    cells = [
        [
            {
                "letter": grid[r][c],
                "confidence": round(confidence[r][c], 3),
                "alternatives": [a for a in alternatives[r][c] if a != grid[r][c]][:_MAX_ALTERNATIVES_SHOWN],
                "flagged": (r, c) in flagged,
                "carried_over": (r, c) in carried_over,
            }
            for c in range(GRID)
        ]
        for r in range(GRID)
    ]
    return {"cells": cells, "board": grid}


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

    flagged: set[tuple[int, int]] = set()
    for run in _runs(grid):
        if all(pos in locked for pos in run):
            continue
        word = "".join(grid[r][c] for r, c in run)
        if not dawg.contains(word):
            flagged.update(pos for pos in run if pos not in locked)
    flagged.update(pos for pos in _isolated_tiles(grid) if pos not in locked)
    return flagged
