"""Interactive tuner for STAGE 3: per-cell occupancy + letter classification
-- letter_classifier.py's thresholds, tuned on top of grid_tuner.py's stage-2
output. Runs stage 1 + stage 2 itself first (using whatever hsv_tuner.py/
grid_tuner.py already saved to hsv_config.json), so it always starts from a
real grid-aligned, binarized board.

Everything lives in one window. Sliders stack at the top (see PARAM_SPECS
below for what each one does -- select one with '[' / ']' to see its
description on screen). Three labelled previews sit below the sliders,
left to right:
    ORIGINAL       the raw source photo, exactly as loaded from disk
    BINARIZED       stage 2's output: tile = white (including a black
                    letter-ink hole), everything else (grid lines, empty
                    cells) = black
    PREDICTIONS     the grid-aligned colour warp with this stage's guess
                    for each cell drawn on it -- this *is* "the board
                    state that was read". Green text = matches
                    test/out/board<N>.txt ground truth, red = mismatch,
                    yellow = no ground truth exists for this image. A
                    red dot marks a cell ground truth says is occupied
                    but that came back empty here.
The recognized board is also printed as a 15x15 text grid in the terminal
every time you switch images, and per-image + running accuracy are shown
in the window and printed to the terminal.

Usage (run from board_reader/, same convention as hsv_tuner.py):
    python src/letter_tuner.py                    # difficulty "em" (easy+medium)
    python src/letter_tuner.py -d emh              # easy + medium + hard
    python src/letter_tuner.py "some/glob/*.jpg"   # explicit pattern, overrides -d

    Hard ("h") photos are excluded by default -- some are close to
    unreadable even by eye, so they aren't the current focus.

Keys:
    n / p    next / previous image
    w        save current sliders to hsv_config.json right now
    [ / ]    select the previous / next individual slider
    0        reset only the selected slider back to its seed
    r        reset every slider back to its seed (saved config or defaults)
    q / Esc  quit
"""

import argparse
import glob
import sys
from collections import namedtuple

import cv2
import numpy as np

from detect_board import (
    find_board_quad,
    signal_handler,  # noqa: F401  (registers SIGINT handler on import)
    warp_board,
)
from grid_reader import (
    binarize_tiles,
    extract_cells,
    find_grid_quad,
    orient_to_bottom,
    warp_to_grid,
)
from hsv_config import load_params, save_params
from letter_classifier import (
    PARAM_DEFAULTS,
    classify_cell,
    render_digit_glyphs,
    render_reference_glyphs,
)
from read_board import board_number, load_ground_truth

WINDOW = "Letter Tuner (original | binarized | predictions)"
GRID = 15
PANEL_HEIGHT = 700
LABEL_BAR = 34  # px reserved above each panel for its name

# (json key, trackbar label, trackbar max position, scale, what it does).
# Trackbars are integer-only, so fractional values are stored scaled up;
# `scale` divides the raw trackbar position back down to the real value.
ParamSpec = namedtuple("ParamSpec", "key label max_pos scale desc")
SliderRef = namedtuple("SliderRef", "label seed scale max_pos desc")
PARAM_SPECS = [
    ParamSpec("min_white_frac", "min white x1000", 500, 1000,
              "Cheap pre-filter: a cell needs at least this fraction of white pixels to even be considered occupied."),
    ParamSpec("min_hole_area_frac", "min hole x1000", 300, 1000,
              "The real occupied/empty decision: a cell needs an ink hole at least this big (as a fraction of "
              "cell area) to count as a letter -- rejects premium-square label text, which fragments into "
              "several smaller holes instead of one dominant one."),
    ParamSpec("max_hole_area_frac", "max hole x1000", 1000, 1000,
              "An ink hole bigger than this fraction of the cell isn't a plausible single letter."),
    ParamSpec("digit_corner_frac", "digit corner x1000", 500, 1000,
              "Size of the bottom-right corner region reserved for the tile's printed point-value digit -- "
              "excluded when looking for the letter, required when looking for the digit."),
    ParamSpec("min_digit_area_frac", "min digit x1000", 100, 1000,
              "Smallest ink-hole size (fraction of cell area) that counts as a plausible score digit."),
    ParamSpec("max_digit_area_frac", "max digit x1000", 200, 1000,
              "Largest ink-hole size (fraction of cell area) that counts as a plausible score digit."),
    ParamSpec("ambiguity_margin", "ambig margin x1000", 500, 1000,
              "How close two letter candidates' match scores need to be to count as 'genuinely ambiguous' -- "
              "only then is the tile's point-value digit used to break the tie (e.g. L=2pts vs LŁ=3pts)."),
    ParamSpec("expand_frac", "cell expand x1000", 400, 1000,
              "Grows each cell's crop by this fraction of a cell's size per side before reading it, so a tile "
              "that's shifted or perspective-stretched off its ideal cell isn't clipped. Larger values shrink a "
              "real letter's hole-to-cell-area ratio too -- may need 'min hole' lowered to compensate."),
]


def _nothing(_):
    pass


def _create_param_trackbars(seed_params):
    for spec in PARAM_SPECS:
        pos = int(round(seed_params.get(spec.key, PARAM_DEFAULTS[spec.key]) * spec.scale))
        pos = max(0, min(spec.max_pos, pos))
        cv2.createTrackbar(spec.label, WINDOW, pos, spec.max_pos, _nothing)


def _set_param_trackbars(params):
    for spec in PARAM_SPECS:
        pos = int(round(params.get(spec.key, PARAM_DEFAULTS[spec.key]) * spec.scale))
        cv2.setTrackbarPos(spec.label, WINDOW, max(0, min(spec.max_pos, pos)))


def _read_params():
    values = {}
    for spec in PARAM_SPECS:
        pos = cv2.getTrackbarPos(spec.label, WINDOW)
        values[spec.key] = pos / spec.scale
    return values


def _build_slider_refs(seed_params):
    return [SliderRef(spec.label, seed_params.get(spec.key, PARAM_DEFAULTS[spec.key]), spec.scale, spec.max_pos, spec.desc)
            for spec in PARAM_SPECS]


def _reset_single(ref):
    pos = int(round(ref.seed * ref.scale))
    cv2.setTrackbarPos(ref.label, WINDOW, max(0, min(ref.max_pos, pos)))


def _selection_status(ref):
    pos = cv2.getTrackbarPos(ref.label, WINDOW)
    current = pos / ref.scale
    changed = abs(current - ref.seed) > 1e-9
    return f"selected: {ref.label}  current={current:g}  seed={ref.seed:g}{'  (changed)' if changed else ''}  -- {ref.desc}"


def _stage12(image):
    """Run stage 1 + stage 2 (using whatever's already saved to
    hsv_config.json) to get to a binarized, grid-aligned board. Returns
    (grid_warp_bgr, binarized) or (None, None)."""
    corners = find_board_quad(image)
    if corners is None:
        return None, None
    warped = warp_board(image, corners)
    oriented, _ = orient_to_bottom(warped)
    grid_corners, _ = find_grid_quad(oriented)
    if grid_corners is None:
        return None, None
    grid_warp = warp_to_grid(oriented, grid_corners)
    return grid_warp, binarize_tiles(grid_warp)


def _panel(img, label, height=PANEL_HEIGHT):
    """Resize to a common height and stamp a name banner above it, so
    every preview is self-explanatory without reading the module docstring."""
    h, w = img.shape[:2]
    scale = height / h
    resized = cv2.resize(img, (max(1, int(w * scale)), height))
    bar = np.zeros((LABEL_BAR, resized.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (8, LABEL_BAR - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return np.vstack([bar, resized])


def _predictions_overlay(grid_warp, binarized, refs, digit_refs, gt, params):
    """grid_warp with each cell's predicted letter drawn on it, colour-
    coded green (matches ground truth) / red (mismatch) / yellow (no
    ground truth for this image). Returns (overlay, board, correct, total)."""
    h, w = binarized.shape[:2]
    cells = extract_cells(binarized, GRID, expand_frac=params.get("expand_frac", 0.0))
    overlay = grid_warp.copy()
    board = []
    correct = total = 0
    for r in range(GRID):
        row = []
        for c in range(GRID):
            letter, _ = classify_cell(cells[r][c], refs, digit_refs, **params)
            row.append(letter)
            cy0, cy1 = r * h // GRID, (r + 1) * h // GRID
            cx0, cx1 = c * w // GRID, (c + 1) * w // GRID
            cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), (60, 60, 60), 1)
            if gt is not None:
                total += 1
                match = letter == gt[r][c]
                correct += match
                color = (0, 200, 0) if match else (0, 0, 220)
            else:
                color = (0, 220, 220)
            if letter != "-":
                cv2.putText(overlay, letter, (cx0 + 5, cy1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                           0.9, color, 2)
            elif gt is not None and gt[r][c] != "-":
                cv2.circle(overlay, ((cx0 + cx1) // 2, (cy0 + cy1) // 2), 6, (0, 0, 220), -1)
        board.append(row)
    return overlay, board, correct, total


def _parse_args():
    p = argparse.ArgumentParser(description="Interactive tuner for letter classification")
    p.add_argument("pattern", nargs="?", default=None, help="glob pattern for images (overrides --difficulty)")
    p.add_argument(
        "-d", "--difficulty", default="em",
        help="difficulty suffixes to include: any of 'e' (easy), 'm' (medium), 'h' (hard), e.g. -d emh "
             "(default: em -- hard photos can be near-impossible to read and aren't the current focus)",
    )
    return p.parse_args()


def _load(path):
    """Returns (original_bgr, grid_warp_bgr, binarized) or (None, None, None)."""
    image = cv2.imread(path)
    if image is None:
        return None, None, None
    grid_warp, binarized = _stage12(image)
    if grid_warp is None:
        return image, None, None
    return image, grid_warp, binarized


def main():
    args = _parse_args()
    if args.pattern:
        paths = sorted(glob.glob(args.pattern))
    else:
        seen = set()
        paths = []
        for c in args.difficulty:
            for path in sorted(glob.glob(f"test/in/img*_{c}.jpg")):
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
    if not paths:
        print(f"No images matched (pattern={args.pattern!r}, difficulty={args.difficulty!r})")
        sys.exit(1)

    seed_params = load_params("letter_params", PARAM_DEFAULTS)
    print(f"seed params: {seed_params} ({'from hsv_config.json' if seed_params != PARAM_DEFAULTS else 'hardcoded default'})")
    for spec in PARAM_SPECS:
        print(f"  {spec.label}: {spec.desc}")
    refs = render_reference_glyphs()
    digit_refs = render_digit_glyphs()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    _create_param_trackbars(seed_params)
    slider_refs = _build_slider_refs(seed_params)
    selected_idx = 0

    print(f"\n{len(paths)} image(s). n/p: switch image, w: save now, [/]: select slider, "
          f"0: reset selected, r: reset all, q/Esc: quit.")

    def _gt_for(path):
        num = board_number(path)
        if num is None:
            return None
        try:
            return load_ground_truth(f"test/out/board{num}.txt")
        except OSError:
            return None

    idx = 0
    original, grid_warp, binarized = _load(paths[idx])
    while grid_warp is None and idx < len(paths) - 1:
        idx += 1
        original, grid_warp, binarized = _load(paths[idx])
    if grid_warp is None:
        print("Could not find a board+grid (via stage 1+2) on any of the matched images.")
        sys.exit(1)

    gt = _gt_for(paths[idx])
    cv2.setWindowTitle(WINDOW, f"{WINDOW} - {paths[idx]}")
    print(f"\n[{idx + 1}/{len(paths)}] {paths[idx]}")

    running_correct = running_total = 0
    while True:
        params = _read_params()
        overlay, board, correct, total = _predictions_overlay(grid_warp, binarized, refs, digit_refs, gt, params)
        composite = np.hstack([
            _panel(original, "ORIGINAL PHOTO"),
            _panel(cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR), "BINARIZED"),
            _panel(overlay, "PREDICTIONS (green=correct red=wrong yellow=no ground truth)"),
        ])

        acc_text = f"this image: {correct}/{total} ({correct / total:.1%})" if total else "this image: no ground truth"
        cv2.putText(composite, acc_text, (10, composite.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(composite, _selection_status(slider_refs[selected_idx]), (10, composite.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.imshow(WINDOW, composite)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("w"):
            save_params("letter_params", params)
            print(f"  wrote hsv_config.json letter_params: {params}")
        elif key == ord("r"):
            _set_param_trackbars(seed_params)
            print("  reset all sliders to seed values")
        elif key == ord("["):
            selected_idx = (selected_idx - 1) % len(slider_refs)
        elif key == ord("]"):
            selected_idx = (selected_idx + 1) % len(slider_refs)
        elif key == ord("0"):
            ref = slider_refs[selected_idx]
            _reset_single(ref)
            print(f"  reset {ref.label} to seed {ref.seed:g}")
        elif key in (ord("n"), ord("p")) and len(paths) > 1:
            if total:
                running_correct += correct
                running_total += total
            step = 1 if key == ord("n") else -1
            new_original = new_warp = new_bin = None
            for _ in range(len(paths)):
                idx = (idx + step) % len(paths)
                new_original, new_warp, new_bin = _load(paths[idx])
                if new_warp is not None:
                    break
            if new_warp is not None:
                original, grid_warp, binarized = new_original, new_warp, new_bin
                gt = _gt_for(paths[idx])
                cv2.setWindowTitle(WINDOW, f"{WINDOW} - {paths[idx]}")
                running_note = f" | running: {running_correct}/{running_total} ({running_correct / running_total:.1%})" if running_total else ""
                print(f"\n[{idx + 1}/{len(paths)}] {paths[idx]}{running_note}")
            else:
                print("  no other image has a stage-1+2 board to show -- staying put")
                continue

        # Board state as text, printed once per image switch (not every
        # frame -- 30fps of 15-line prints would flood the terminal).
        if key in (ord("n"), ord("p")):
            print(format_board(board))


def format_board(board):
    return "\n".join(" ".join(row) for row in board)


if __name__ == "__main__":
    main()
