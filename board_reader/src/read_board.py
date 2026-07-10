"""Top-level orchestration: photo -> 15x15 board of letters.

Chains detect_board.py (stage 1: find + warp the board), grid_reader.py
(stage 2: orient, find + warp to the precise grid), and
letter_classifier.py (stage 3: per-cell local binarization + occupancy +
letter classification) into read_board(), the "run everything" entry
point at this module's level -- mirrors detect_board.py's own role for
stage 1 alone.

Run directly to evaluate against the ground truth in test/out/*.txt
(board<N>.txt matches test/in/img<N>_*.jpg):
    python src/read_board.py                  # difficulty "em" (easy+medium)
    python src/read_board.py -d e              # easy only
    python src/read_board.py -d emh            # easy + medium + hard

Hard ("h") photos are excluded by default -- some are close to unreadable
even by eye, so they aren't the current focus; pass -d emh to include them.
"""

import argparse
import glob
import os
import re
import sys

import cv2

from detect_board import find_board_quad, warp_board
from grid_reader import extract_cells, find_grid_quad, orient_to_bottom, warp_to_grid
from hsv_config import load_params
from letter_classifier import PARAM_DEFAULTS, classify_cell, render_digit_glyphs, render_reference_glyphs

GRID = 15


def read_board(image_path, refs=None, digit_refs=None):
    """Full pipeline: photo -> (15x15 letters, 15x15 confidence scores).
    Letters use '-' for empty, '?' for an occupied cell with no plausible
    glyph. Returns (None, None) if the board or grid couldn't be found."""
    if refs is None:
        refs = render_reference_glyphs()
    if digit_refs is None:
        digit_refs = render_digit_glyphs()
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    corners = find_board_quad(image)
    if corners is None:
        return None, None
    warped = warp_board(image, corners)
    oriented, _ = orient_to_bottom(warped)
    grid_corners, _ = find_grid_quad(oriented)
    if grid_corners is None:
        return None, None
    grid_warp = warp_to_grid(oriented, grid_corners)
    # Color cells, not grid_reader.binarize_tiles()'s global-threshold
    # output -- classify_cell() binarizes each one locally now; see
    # letter_classifier.py's module docstring for why.
    expand_frac = load_params("letter_params", PARAM_DEFAULTS)["expand_frac"]
    cells = extract_cells(grid_warp, GRID, expand_frac=expand_frac)

    board, scores = [], []
    for row in cells:
        letters, row_scores = [], []
        for cell in row:
            letter, score = classify_cell(cell, refs, digit_refs)
            letters.append(letter)
            row_scores.append(score)
        board.append(letters)
        scores.append(row_scores)
    return board, scores


def load_ground_truth(path):
    """Parse a test/out/board<N>.txt file into a 15x15 list of single-
    character strings, matching read_board()'s '-' for empty convention."""
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]


def board_number(image_path):
    m = re.search(r"img(\d+)_", os.path.basename(image_path))
    return m.group(1) if m else None


def evaluate(paths, test_out_dir="test/out"):
    """Run read_board() over `paths` and score against test/out/board<N>.txt
    ground truth, printing a per-image line plus overall/occupied-cell
    accuracy totals."""
    refs = render_reference_glyphs()
    digit_refs = render_digit_glyphs()

    total_cells = total_correct = 0
    total_occupied = occupied_correct = 0
    boards_found = 0
    for path in paths:
        num = board_number(path)
        gt_path = os.path.join(test_out_dir, f"board{num}.txt") if num else None
        if not gt_path or not os.path.exists(gt_path):
            print(f"{os.path.basename(path)}: SKIP (no ground truth)")
            continue
        gt = load_ground_truth(gt_path)

        board, _ = read_board(path, refs, digit_refs)
        if board is None:
            print(f"{os.path.basename(path)}: FAILED (no board/grid found)")
            continue
        boards_found += 1

        correct = occ_correct = occ_total = 0
        for r in range(GRID):
            for c in range(GRID):
                total_cells += 1
                match = board[r][c] == gt[r][c]
                correct += match
                if gt[r][c] != "-":
                    total_occupied += 1
                    occ_total += 1
                    occ_correct += match
        total_correct += correct
        occupied_correct += occ_correct
        acc = correct / (GRID * GRID)
        occ_acc = occ_correct / occ_total if occ_total else 1.0
        print(f"{os.path.basename(path)}: {acc:.1%} overall, {occ_correct}/{occ_total} ({occ_acc:.1%}) occupied cells correct")

    print(f"\n{boards_found}/{len(paths)} images had a board+grid found.")
    if total_cells:
        print(f"Overall cell accuracy: {total_correct}/{total_cells} ({total_correct / total_cells:.1%})")
    if total_occupied:
        print(f"Occupied-cell letter accuracy: {occupied_correct}/{total_occupied} ({occupied_correct / total_occupied:.1%})")


def _parse_args():
    p = argparse.ArgumentParser(description="Read a Scrabble board photo into a 15x15 grid and evaluate against test/out/")
    p.add_argument(
        "-d", "--difficulty", default="em",
        help="difficulty suffixes to include: any of 'e' (easy), 'm' (medium), 'h' (hard), e.g. -d emh "
             "(default: em -- hard photos can be near-impossible to read and aren't the current focus)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    seen = set()
    paths = []
    for c in args.difficulty:
        for path in sorted(glob.glob(f"test/in/img*_{c}.jpg")):
            if path not in seen:
                seen.add(path)
                paths.append(path)
    if not paths:
        print(f"No images matched difficulty {args.difficulty!r}")
        sys.exit(1)
    evaluate(paths)


if __name__ == "__main__":
    main()
