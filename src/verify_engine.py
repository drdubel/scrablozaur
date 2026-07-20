"""Regression checks for the core Rust engine (src/lib.rs) — scoring and
pattern-generation correctness, independent of any move-selection strategy.
Covers two real bugs found and fixed during a rules-compliance audit:
cross-word scoring incorrectly inheriting the main word's word-multiplier,
and pattern generation silently gluing a new word onto an adjacent existing
tile (a candidate ending right before an existing letter formed an invalid
combined word once placed on the board).

Run: uv run python src/verify_engine.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from scrablozaur import Board, Dawg  # noqa: E402

DAWG_PATH = _ROOT / "words" / "dawg.bin"


def board_grid(board: Board) -> list[list[str]]:
    return [row.split(" ") for row in str(board).strip().split("\n")]


def check_cross_word_scoring() -> list[str]:
    """Each word (main + every cross-word it forms) must be scored with its
    own word multiplier, not the main word's accumulated one. 'ba' placed
    vertically at (0,0), then 'cdf' placed vertically at column 1 crossing
    a DW at (1,1): main word (c+d+f)*2=18, cross-word at row0 (b+c)=5,
    cross-word at row1 (a+d)*2=6 (also on the DW) -- total 29, not the
    buggy 26 you'd get by multiplying everything by the main word's mult."""
    board = Board()
    board.place_word("ba", 0, 0, False)
    score = board.calculate_word_points("cdf", 0, 1, False, "cdf")
    if score != 29:
        return [f"expected cross-word-aware score 29, got {score}"]
    return []


def check_first_move_covers_every_offset() -> list[str]:
    """get_best_word(first=True) must consider every offset through the
    centre square, including the word's first letter landing exactly on
    it (offset 0) -- previously the search started at offset 1 and missed
    this case entirely."""
    dawg = Dawg(str(DAWG_PATH))
    board = Board()
    letters = "yclgaup"
    _, best_score, _, _ = board.get_best_word(dawg, letters, first=True, parallel=False)

    better_at_offset_zero = [
        (w, board.calculate_word_points(w, 7, 7, True, letters)) for w in dawg.search("*", letters) if len(w) <= 8
    ]
    max_at_offset_zero = max((s for _, s in better_at_offset_zero), default=0)
    if max_at_offset_zero > best_score:
        return [f"get_best_word found {best_score}, but offset-0 alone reaches {max_at_offset_zero}"]
    return []


def check_pattern_boundaries() -> list[str]:
    """Every pattern from get_all_patterns() must be bounded by an empty
    cell (or the board edge) on both sides -- otherwise a word placed at
    that span silently glues onto an adjacent existing tile on the board
    (e.g. "nitowa" placed right before an existing 'c' becomes "nitowac"
    when the row is read as a whole), producing an invalid word that was
    never validated. Includes the exact real-game repro: a same-row
    boundary tile plus a perpendicular tile to trigger a parallel pattern
    in the first place."""
    grid = [["-"] * 15 for _ in range(15)]
    grid[5][10] = "c"
    grid[4][3] = "x"
    board = Board.from_grid(grid)

    errors = []
    bgrid = board_grid(board)
    for index, start, end, horizontal in board.get_all_patterns():
        if horizontal:
            before = bgrid[index][start - 1] if start > 0 else "-"
            after = bgrid[index][end + 1] if end < 14 else "-"
        else:
            before = bgrid[start - 1][index] if start > 0 else "-"
            after = bgrid[end + 1][index] if end < 14 else "-"
        if before != "-":
            errors.append(
                f"pattern (idx={index}, {start}-{end}, horiz={horizontal}): starts right after a tile ({before!r})"
            )
        if after != "-":
            errors.append(
                f"pattern (idx={index}, {start}-{end}, horiz={horizontal}): ends right before a tile ({after!r})"
            )
    return errors


def check_min_word_length() -> list[str]:
    """A play must be at least 2 letters -- rejected explicitly, not just
    because the dictionary happens to have no 1-letter entries."""
    dawg = Dawg(str(DAWG_PATH))
    board = Board()
    errors = []
    try:
        board.check_word_placement(dawg, "a", 7, 7, True)
        errors.append("expected a 1-letter word to be rejected")
    except Exception:
        pass
    try:
        board.check_word_placement(dawg, "ok", 7, 7, True)
    except Exception as e:
        errors.append(f"expected a 2-letter word to be accepted, got: {e}")
    return errors


def main() -> None:
    checks = [
        ("cross-word scoring uses its own word multiplier", check_cross_word_scoring),
        ("first-move search covers offset 0 (word starting exactly on centre)", check_first_move_covers_every_offset),
        ("every pattern is bounded by an empty cell or the board edge", check_pattern_boundaries),
        ("a play must be at least 2 letters", check_min_word_length),
    ]

    failed = False
    for label, check in checks:
        errors = check()
        if errors:
            failed = True
            print(f"FAILED: {label}")
            for e in errors[:20]:
                print(" -", e)
        else:
            print(f"OK: {label}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
