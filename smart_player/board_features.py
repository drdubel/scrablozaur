"""Board-state summary features for the leave-value model (see model.py).

The engine's bonus-square layout (`BONUS_TABLE` in src/lib.rs) isn't
exposed to Python via pyo3. Rather than re-transcribe it from the Rust
source, this ports the already-validated JS copy at
web/static/js/board.js:4-32 (itself derived from and comment-annotated
against lib.rs's BONUS_TABLE, and running in production in the web app) --
safer than a fresh transcription.
"""

from scrablozaur import Board

# (letter_multiplier, word_multiplier) for one quadrant; the full 15x15
# board folds onto this 8x8 table via r2=min(r,14-r), c2=min(c,14-c).
_QUADRANT = [
    [(1, 3), (1, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 3)],
    [(1, 1), (1, 2), (1, 1), (1, 1), (1, 1), (3, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (1, 2), (1, 1), (1, 1), (1, 1), (2, 1), (1, 1)],
    [(2, 1), (1, 1), (1, 1), (1, 2), (1, 1), (1, 1), (1, 1), (2, 1)],
    [(1, 1), (1, 1), (1, 1), (1, 1), (1, 2), (1, 1), (1, 1), (1, 1)],
    [(1, 1), (3, 1), (1, 1), (1, 1), (1, 1), (3, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1), (2, 1), (1, 1)],
    [(1, 3), (1, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 2)],
]


def _classify(row: int, col: int) -> str:
    r2, c2 = min(row, 14 - row), min(col, 14 - col)
    lm, wm = _QUADRANT[r2][c2]
    if wm == 3:
        return "tw"
    if wm == 2:
        return "dw"
    if lm == 3:
        return "tl"
    if lm == 2:
        return "dl"
    return "plain"


# Precomputed once at import time: classification grid and per-type totals
# (8 TW / 17 DW / 12 TL / 24 DL / 164 plain squares -- the standard board).
_GRID = [[_classify(r, c) for c in range(15)] for r in range(15)]
_TOTALS = {"tw": 8, "dw": 17, "tl": 12, "dl": 24}


def encode_board(board: Board) -> tuple[float, float, float, float, float]:
    """(tw_open, dw_open, tl_open, dl_open, board_fill) for the board's
    current state: fraction of each premium-square type still unclaimed,
    plus the fraction of all 225 cells occupied. A leave's value depends on
    board context (e.g. a leftover S is worth more when bingo-enabling
    premium squares are still reachable) that the leave alone can't
    express."""
    rows = [row.split(" ") for row in str(board).strip().split("\n")]
    open_counts = {"tw": 0, "dw": 0, "tl": 0, "dl": 0}
    filled = 0
    for r in range(15):
        for c in range(15):
            empty = rows[r][c] == "-"
            if not empty:
                filled += 1
            kind = _GRID[r][c]
            if kind != "plain" and empty:
                open_counts[kind] += 1

    return (
        open_counts["tw"] / _TOTALS["tw"],
        open_counts["dw"] / _TOTALS["dw"],
        open_counts["tl"] / _TOTALS["tl"],
        open_counts["dl"] / _TOTALS["dl"],
        filled / 225,
    )
