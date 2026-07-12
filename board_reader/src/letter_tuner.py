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
    BINARIZED       what the classifier actually sees: each cell's own
                    per-cell Otsu threshold (letter_classifier.
                    local_binarize()), stitched back together -- tile =
                    white (including a black letter-ink hole), everything
                    else (grid lines, empty cells) = black. NOT
                    grid_reader.binarize_tiles()'s single whole-board
                    threshold; see letter_classifier.py's module docstring
                    for why that changed.
    PREDICTIONS     the grid-aligned colour warp with this stage's guess
                    for each cell drawn on it -- this *is* "the board
                    state that was read". Green text = matches
                    test/out/board<N>.txt ground truth, red = mismatch,
                    yellow = no ground truth exists for this image. A
                    red dot marks a cell ground truth says is occupied
                    but that came back empty here.
A fourth ERROR WALKTHROUGH panel (toggle with 'e', step through with 'j'/
'k') zooms into one wrongly-classified cell at a time -- the small 15x15
grid makes it hard to see *why* a specific cell is wrong, so this shows
that cell's color crop, its own local_binarize() mask, and its isolated
ink-hole glyph (exactly what ocr_classify_glyph()/classify_glyph() are
given), plus whether is_tile_present() even considered it occupied and
what each recognizer guessed -- so you can tell whether a miss is an
occupancy problem, a glyph-isolation problem, or a recognition problem
before reaching for a slider. It stays on the same cell as you tune
(doesn't jump away once the cell turns correct), so you can see sliders
fix a specific miss in real time; 'j'/'k' move to the next/previous wrong
cell within the current image, wrapping around.

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
    j / k    next / previous wrongly-classified cell in this image (error
             walkthrough panel), wrapping around; also shows the panel
    e        toggle the error walkthrough panel on/off without moving
    w        save current sliders to hsv_config.json right now
    [ / ]    select the previous / next individual slider
    0        reset only the selected slider back to its seed
    r        reset every slider back to its seed (saved config or defaults)
    q / Esc  quit

Recognition now runs through Tesseract OCR by default (letter_classifier.
ocr_classify_glyph(), template matching only as its fallback), which is
far more accurate but calls out to a subprocess per occupied cell -- the
full-board recompute below is therefore cached and only re-run when the
sliders or the loaded image actually change, not on every redraw tick.

Two more sliders ("shift x x1000" / "shift y x1000") control grid_reader.
extract_cells()'s tile-height parallax correction -- physical tiles sit
above the flat board plane, so an angled photo leaves them shifted off
their true cell (see grid_reader.estimate_tile_shift()'s docstring).
Unlike every other slider, these are reseeded per image (to a fresh
estimate_tile_shift() guess for that specific photo) rather than from one
session-global saved value -- the status line at the bottom of the window
shows both the current slider values and what was auto-estimated, and 'r'
resets them back to that estimate (not to 0).
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
    estimate_tile_shift,
    extract_cells,
    find_grid_quad,
    orient_to_bottom,
    rotate_shift,
    warp_to_grid,
)
from hsv_config import load_params, save_params
from letter_classifier import (
    PARAM_DEFAULTS,
    board_saturation_reference,
    classify_board,
    classify_glyph,
    extract_glyph,
    is_tile_present,
    local_binarize,
    ocr_classify_glyph,
    premium_class,
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
    ParamSpec("tile_sat_ratio", "tile sat ratio x1000", 1000, 1000,
              "The real occupied/empty decision, checked on the raw color cell: a physical tile is lower-"
              "saturation than the board (normal or premium, any printed label). Compared against this "
              "photo's own board_saturation_reference() rather than a fixed value, since absolute saturation "
              "shifts a lot with lighting -- a cell counts as tile-present if its saturation is below this "
              "fraction of that reference."),
    ParamSpec("min_hole_area_frac", "min hole x1000", 300, 1000,
              "The letter-vs-empty decision for a cell already confirmed tile-present: needs an ink hole at "
              "least this big (as a fraction of cell area) to count as a letter."),
    ParamSpec("max_hole_area_frac", "max hole x1000", 1000, 1000,
              "An ink hole bigger than this fraction of the cell isn't a plausible single letter."),
    ParamSpec("min_dominance_ratio", "dominance x10", 200, 10,
              "The best ink hole must be at least this many times bigger than the runner-up -- real letters "
              "are strongly dominant (~5-14x), decorative premium-square content isn't (~1-3.4x)."),
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
    ParamSpec("premium_min_speck_frac", "premium speck x1000", 50, 1000,
              "Lower area floor (fraction of cell area) for the row-clustering premium-label check below -- "
              "needs to see individual small text-fragment holes, not just letter-sized ones."),
    ParamSpec("premium_row_gap_frac", "premium row gap x1000", 500, 1000,
              "Vertical gap (fraction of tile height) that starts a new row when clustering hole y-centers -- "
              "separates one text line's holes from the next."),
    ParamSpec("premium_min_rows", "premium min rows", 6, 1,
              "Holes clustering into at least this many rows means multi-line label text, not a single letter -- "
              "rejected regardless of the dominance-ratio check. Every premium label was observed as 3 lines."),
]


# Tile-shift sliders (estimate_tile_shift()'s auto-estimate, overridable)
# live outside PARAM_SPECS: unlike every other slider, their seed is
# per-image (a fresh estimate_tile_shift() call per photo), not one
# session-global value from hsv_config.json, so they need their own
# create/set/read helpers instead of the generic PARAM_SPECS loop below.
SHIFT_LABELS = ("shift x x1000", "shift y x1000")
SHIFT_RANGE = 0.3  # sliders represent [-SHIFT_RANGE, +SHIFT_RANGE]
SHIFT_SCALE = 1000


def _shift_to_pos(frac):
    return max(0, min(int(2 * SHIFT_RANGE * SHIFT_SCALE), int(round((frac + SHIFT_RANGE) * SHIFT_SCALE))))


def _pos_to_shift(pos):
    return pos / SHIFT_SCALE - SHIFT_RANGE


def _create_shift_trackbars(seed_shift):
    max_pos = int(2 * SHIFT_RANGE * SHIFT_SCALE)
    for label, seed in zip(SHIFT_LABELS, seed_shift):
        cv2.createTrackbar(label, WINDOW, _shift_to_pos(seed), max_pos, _nothing)


def _set_shift_trackbars(shift):
    for label, frac in zip(SHIFT_LABELS, shift):
        cv2.setTrackbarPos(label, WINDOW, _shift_to_pos(frac))


def _read_shift():
    return tuple(_pos_to_shift(cv2.getTrackbarPos(label, WINDOW)) for label in SHIFT_LABELS)


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
    hsv_config.json) to get to a grid-aligned, color board. Returns
    (grid_warp_bgr, (shift_x_frac, shift_y_frac)), or (None, None) if the
    board or grid couldn't be found. The shift is estimate_tile_shift()'s
    per-photo parallax guess (see grid_reader.py's docstring for why it's
    measured on stage 1's raw, unrotated board quad rather than stage 2's
    grid quad), rotated into grid_warp's actual coordinate frame via
    rotate_shift() -- orient_to_bottom() may have rotated the image
    90/180/270 relative to the quad the estimate was measured on."""
    corners = find_board_quad(image)
    if corners is None:
        return None, None
    warped = warp_board(image, corners)
    oriented, panel_edge = orient_to_bottom(warped)
    grid_corners, _ = find_grid_quad(oriented)
    if grid_corners is None:
        return None, None
    grid_warp = warp_to_grid(oriented, grid_corners)
    shift = rotate_shift(*estimate_tile_shift(corners), panel_edge)
    return grid_warp, shift


def _panel(img, label, height=PANEL_HEIGHT):
    """Resize to a common height and stamp a name banner above it, so
    every preview is self-explanatory without reading the module docstring."""
    h, w = img.shape[:2]
    scale = height / h
    resized = cv2.resize(img, (max(1, int(w * scale)), height))
    bar = np.zeros((LABEL_BAR, resized.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (8, LABEL_BAR - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return np.vstack([bar, resized])


def _predictions_overlay(grid_warp, refs, digit_refs, gt, params):
    """grid_warp with each cell's predicted letter drawn on it, colour-
    coded green (matches ground truth) / red (mismatch) / yellow (no
    ground truth for this image). Also builds the BINARIZED panel (each
    cell's own local_binarize() output, stitched back together) in the
    same pass, since both need the same per-cell extraction.
    Returns (overlay, binarized_composite, board, correct, total)."""
    h, w = grid_warp.shape[:2]
    cells = extract_cells(
        grid_warp, GRID, expand_frac=params.get("expand_frac", 0.0),
        shift_x_frac=params.get("shift_x_frac", 0.0), shift_y_frac=params.get("shift_y_frac", 0.0),
    )
    sat_ref_by_class = board_saturation_reference(cells)
    # classify_cell() mostly waits on a Tesseract subprocess per occupied
    # cell -- calling it 225x sequentially here is what made the tuner
    # crawl after every slider change (see classify_board()'s docstring).
    board, _ = classify_board(cells, refs, digit_refs, sat_ref_by_class, **params)
    overlay = grid_warp.copy()
    binarized_composite = np.zeros((h, w), np.uint8)
    correct = total = 0
    for r in range(GRID):
        for c in range(GRID):
            letter = board[r][c]
            cy0, cy1 = r * h // GRID, (r + 1) * h // GRID
            cx0, cx1 = c * w // GRID, (c + 1) * w // GRID
            cell_bin = local_binarize(cells[r][c], **params)
            ch, cw = cell_bin.shape[:2]
            th, tw = cy1 - cy0, cx1 - cx0
            # Cells may be larger than the plain grid slice (expand_frac's
            # margin) or *smaller* (a shift pushed the read window past
            # the image border, so extract_cells() clipped it) -- crop or
            # pad around the center either way so the composite stays a
            # clean, non-overlapping 15x15 mosaic regardless.
            sh, sw = min(ch, th), min(cw, tw)
            oy, ox = max(0, (ch - th) // 2), max(0, (cw - tw) // 2)
            dy, dx = max(0, (th - ch) // 2), max(0, (tw - cw) // 2)
            binarized_composite[cy0 + dy:cy0 + dy + sh, cx0 + dx:cx0 + dx + sw] = cell_bin[oy:oy + sh, ox:ox + sw]
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
    return overlay, binarized_composite, board, correct, total


def _mismatches(board, gt):
    """(row, col) of every cell where the current prediction disagrees
    with ground truth, in reading order. Empty if there's no ground truth
    for this image."""
    if gt is None:
        return []
    return [(r, c) for r in range(GRID) for c in range(GRID) if board[r][c] != gt[r][c]]


def _zoom(img, size):
    """Nearest-neighbor upscale to a size x size square (crisp pixel
    edges -- these crops are tiny, and INTER_NEAREST shows the actual
    pixels instead of blurring them into mush). Converts grayscale to BGR
    so it stacks with color panels."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _focus_cell_data(grid_warp, r, c, refs, params):
    """Every intermediate classify_cell() produces for one cell, computed
    directly rather than via classify_cell() itself so the walkthrough can
    show *why* -- was the cell even considered tile-present, was a glyph
    isolated, what did each recognizer guess -- not just the final
    letter. Returns (cell_bgr, cell_bin, glyph, present, ocr_guess,
    tmpl_guess, tmpl_score)."""
    cells = extract_cells(
        grid_warp, GRID, expand_frac=params.get("expand_frac", 0.0),
        shift_x_frac=params.get("shift_x_frac", 0.0), shift_y_frac=params.get("shift_y_frac", 0.0),
    )
    cell_bgr = cells[r][c]
    board_sat_ref = board_saturation_reference(cells)[premium_class(r, c, GRID)]
    present = is_tile_present(cell_bgr, board_sat_ref, **params)
    cell_bin = local_binarize(cell_bgr, **params)
    glyph = extract_glyph(cell_bin, **params) if present else None
    ocr_guess = ocr_classify_glyph(glyph) if glyph is not None else None
    tmpl_guess, tmpl_score = classify_glyph(glyph, refs) if glyph is not None else (None, None)
    return cell_bgr, cell_bin, glyph, present, ocr_guess, tmpl_guess, tmpl_score


def _error_panel(focus, data, board, gt, image_name, mismatches, height=PANEL_HEIGHT):
    """Zoomed color/binarized/glyph crops for the cell at `focus`, plus a
    text summary of every stage's verdict -- see the module docstring for
    why. `data` is _focus_cell_data()'s output, or None if there's no
    focused cell yet (nothing to show but a placeholder).

    Total height (including the label bar) must come out to exactly
    LABEL_BAR + PANEL_HEIGHT, matching every other panel's _panel()
    output, or np.hstack() in main() raises -- so the three image squares
    and the text block are sized to add up exactly (`sq * 3 + text_h ==
    height`), not just approximately via separate rounded constants.
    """
    n_lines = 5
    text_h = 22 * n_lines + 12
    sq = (height - text_h) // 3
    text_h = height - sq * 3  # absorb the floor-division remainder here

    bar = np.zeros((LABEL_BAR, sq, 3), np.uint8)
    cv2.putText(bar, "ERROR WALKTHROUGH", (8, LABEL_BAR - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if focus is None or data is None:
        body = np.zeros((height, sq, 3), np.uint8)
        cv2.putText(body, "press j/k to jump", (10, sq), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(body, "to a wrong cell", (10, sq + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return np.vstack([bar, body])

    r, c = focus
    cell_bgr, cell_bin, glyph, present, ocr_guess, tmpl_guess, tmpl_score = data
    color_sq, bin_sq = _zoom(cell_bgr, sq), _zoom(cell_bin, sq)
    glyph_sq = _zoom(glyph, sq) if glyph is not None else np.zeros((sq, sq, 3), np.uint8)
    for sq_img, label in ((color_sq, "color"), (bin_sq, "binarized"),
                          (glyph_sq, "isolated glyph" if glyph is not None else "no glyph isolated")):
        cv2.putText(sq_img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    pred = board[r][c]
    truth = gt[r][c] if gt is not None else "?"
    fixed = pred == truth
    lines = [
        (f"{image_name}  ({r},{c})  error {mismatches.index(focus) + 1}/{len(mismatches)}" if focus in mismatches
         else f"{image_name}  ({r},{c})  FIXED", (0, 200, 0) if fixed else (255, 255, 0)),
        (f"tile_present: {'yes' if present else 'no'}   glyph found: {'yes' if glyph is not None else 'no'}", (200, 200, 200)),
        (f"OCR guess: {ocr_guess!r}" if ocr_guess is not None else "OCR guess: (no confident answer)", (200, 200, 200)),
        (f"template guess: {tmpl_guess!r} (score={tmpl_score:.2f})" if tmpl_guess is not None else "template guess: n/a", (200, 200, 200)),
        (f"prediction: {pred!r}   ground truth: {truth!r}", (0, 200, 0) if fixed else (0, 0, 220)),
    ]
    text_block = np.zeros((text_h, sq, 3), np.uint8)
    for i, (line, color) in enumerate(lines):
        cv2.putText(text_block, line, (6, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    return np.vstack([bar, color_sq, bin_sq, glyph_sq, text_block])


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
    """Returns (original_bgr, grid_warp_bgr, auto_shift) or (None, None,
    None); grid_warp_bgr/auto_shift are None if a board+grid couldn't be
    found even when the photo loaded. auto_shift is _stage12()'s
    (shift_x_frac, shift_y_frac) estimate for this specific photo."""
    image = cv2.imread(path)
    if image is None:
        return None, None, None
    grid_warp, auto_shift = _stage12(image)
    return image, grid_warp, auto_shift


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

    print(f"\n{len(paths)} image(s). n/p: switch image, j/k: next/prev wrong cell, e: toggle error "
          f"panel, w: save now, [/]: select slider, 0: reset selected, r: reset all, q/Esc: quit.")

    def _gt_for(path):
        num = board_number(path)
        if num is None:
            return None
        try:
            return load_ground_truth(f"test/out/board{num}.txt")
        except OSError:
            return None

    idx = 0
    original, grid_warp, auto_shift = _load(paths[idx])
    while grid_warp is None and idx < len(paths) - 1:
        idx += 1
        original, grid_warp, auto_shift = _load(paths[idx])
    if grid_warp is None:
        print("Could not find a board+grid (via stage 1+2) on any of the matched images.")
        sys.exit(1)

    gt = _gt_for(paths[idx])
    cv2.setWindowTitle(WINDOW, f"{WINDOW} - {paths[idx]}")
    print(f"\n[{idx + 1}/{len(paths)}] {paths[idx]}")
    _create_shift_trackbars(auto_shift)

    running_correct = running_total = 0
    focus = None            # (row, col) currently zoomed in on, or None
    show_walkthrough = False
    board_cache = None      # (params_key, overlay, binarized, board, correct, total)
    focus_cache = None      # ((focus, params_key), data)

    while True:
        params = _read_params()
        shift_x_frac, shift_y_frac = _read_shift()
        params["shift_x_frac"], params["shift_y_frac"] = shift_x_frac, shift_y_frac
        params_key = tuple(sorted(params.items()))

        # classify_cell() now runs Tesseract OCR per occupied cell (a
        # subprocess spawn each), so recomputing all 225 cells on every
        # ~30ms redraw tick -- as the pre-OCR version did unconditionally
        # -- makes the whole window crawl. Only recompute when the
        # sliders (or the loaded image, which forces board_cache=None
        # below) actually changed.
        if board_cache is None or board_cache[0] != params_key:
            overlay, binarized, board, correct, total = _predictions_overlay(grid_warp, refs, digit_refs, gt, params)
            board_cache = (params_key, overlay, binarized, board, correct, total)
        else:
            _, overlay, binarized, board, correct, total = board_cache
        mismatches = _mismatches(board, gt)

        panels = [
            _panel(original, "ORIGINAL PHOTO"),
            _panel(cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR), "BINARIZED"),
            _panel(overlay, "PREDICTIONS (green=correct red=wrong yellow=no ground truth)"),
        ]
        if show_walkthrough:
            data = None
            if focus is not None:
                focus_key = (focus, params_key)
                if focus_cache is None or focus_cache[0] != focus_key:
                    focus_cache = (focus_key, _focus_cell_data(grid_warp, focus[0], focus[1], refs, params))
                data = focus_cache[1]
            panels.append(_error_panel(focus, data, board, gt, paths[idx], mismatches))
        composite = np.hstack(panels)

        acc_text = f"this image: {correct}/{total} ({correct / total:.1%})" if total else "this image: no ground truth"
        if mismatches:
            acc_text += f"  |  {len(mismatches)} wrong cell(s), j/k to step through"
        acc_text += f"  |  tile shift: x={shift_x_frac:+.3f} y={shift_y_frac:+.3f} (auto={auto_shift[0]:+.3f},{auto_shift[1]:+.3f})"
        cv2.putText(composite, acc_text, (10, composite.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(composite, _selection_status(slider_refs[selected_idx]), (10, composite.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.imshow(WINDOW, composite)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("w"):
            # shift_x_frac/shift_y_frac are photo-specific (see the
            # tile-shift docstring block above) -- estimate_tile_shift()
            # recomputes them fresh every time, so saving whatever this
            # image's sliders happened to read would just leave stale,
            # misleading values sitting in the shared preset.
            to_save = {k: v for k, v in params.items() if k not in ("shift_x_frac", "shift_y_frac")}
            save_params("letter_params", to_save)
            print(f"  wrote hsv_config.json letter_params: {to_save}")
        elif key == ord("r"):
            _set_param_trackbars(seed_params)
            _set_shift_trackbars(auto_shift)
            print("  reset all sliders to seed values (shift -> this image's auto-estimate)")
        elif key == ord("["):
            selected_idx = (selected_idx - 1) % len(slider_refs)
        elif key == ord("]"):
            selected_idx = (selected_idx + 1) % len(slider_refs)
        elif key == ord("0"):
            ref = slider_refs[selected_idx]
            _reset_single(ref)
            print(f"  reset {ref.label} to seed {ref.seed:g}")
        elif key == ord("e"):
            show_walkthrough = not show_walkthrough
        elif key in (ord("j"), ord("k")):
            show_walkthrough = True
            if not mismatches:
                print("  no wrongly-classified cells in this image" if gt is not None
                      else "  no ground truth for this image -- nothing to walk through")
            else:
                step = 1 if key == ord("j") else -1
                if focus in mismatches:
                    focus = mismatches[(mismatches.index(focus) + step) % len(mismatches)]
                else:
                    focus = mismatches[0] if key == ord("j") else mismatches[-1]
        elif key in (ord("n"), ord("p")) and len(paths) > 1:
            if total:
                running_correct += correct
                running_total += total
            step = 1 if key == ord("n") else -1
            new_original = new_warp = new_shift = None
            for _ in range(len(paths)):
                idx = (idx + step) % len(paths)
                new_original, new_warp, new_shift = _load(paths[idx])
                if new_warp is not None:
                    break
            if new_warp is not None:
                original, grid_warp, auto_shift = new_original, new_warp, new_shift
                gt = _gt_for(paths[idx])
                board_cache = None
                focus_cache = None
                focus = None
                _set_shift_trackbars(auto_shift)
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
