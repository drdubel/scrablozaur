"""Stage 2: orientation, the precise 15x15 grid, and tile/ink binarization.

Everything here runs on the *already warped* board that detect_board.py's
find_board_quad() + a perspective warp produces (stage 1) -- never on the
raw photo. In particular find_panel()/find_panel_edge()/orient_to_bottom()
must be called on that warped image: the red "SCRABBLE" panel search
assumes a roughly board-filling frame, which only holds once stage 1 has
already cropped/warped the photo down to the board.

Pipeline (see read_grid() for the reference order):
    warped board (stage 1)
        -> orient_to_bottom(): rotate so the SCRABBLE panel is at the bottom
        -> find_grid_quad(): the *white grid lines* between cells give a
           much tighter outline than the outer bezel+teal quad stage 1
           finds (which includes the bezel/panel margin) -- warp again onto
           just that quad
        -> binarize_tiles(): tiles = white (letters on them = black, being
           holes in the tile blob), everything else (grid lines, empty
           cells) = black

Tunable via grid_tuner.py; parameters are saved to hsv_config.json under
"grid_params" (kept separate from detect_board.py's "board_params" preset).
"""

import cv2
import numpy as np

from detect_board import find_quad_candidates, order_corners
from hsv_config import load_params

# All of this stage's tunable knobs, overridden by a preset saved to
# hsv_config.json (name "grid_params") if one exists.
PARAM_DEFAULTS = {
    # panel / orientation (red ink) -- same rationale as detect_board.py's
    # old red_* params (see find_panel()'s docstring for the hue-wrap note)
    "red_hue_min": 172,
    "red_hue_max": 179,
    "red_sat_min": 120,
    "red_val_min": 60,
    "red_min_area_frac": 0.00005,
    "red_aspect_threshold": 1.6,

    # white grid lines: hue-agnostic (white has no meaningful hue), just a
    # saturation ceiling and value floor
    "white_sat_max": 60,
    "white_val_min": 150,
    "grid_dilate_kernel": 3,
    "grid_close_kernel": 9,
    "grid_close_iterations": 2,
    "grid_open_kernel": 3,

    # quad-validity for the grid quad -- reuses detect_board.is_valid_quad,
    # but the grid is expected to fill most of the (already warped) board,
    # so its own, much larger minimum area applies than stage 1's board quad
    "quad_side_ratio_max": 1.3,
    "quad_angle_tolerance": 20,
    "quad_min_area_frac": 0.5,

    # tile/ink binarization
    "tile_open_kernel": 7,
}


def _params(overrides=None):
    """Merge hsv_config.json's saved "grid_params" preset with any explicit
    overrides -- used by grid_tuner.py to preview live, not-yet-saved
    slider values without every function needing its own long parameter
    list."""
    merged = load_params("grid_params", PARAM_DEFAULTS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def find_panel(bgr, **param_overrides):
    """Find the board's red "SCRABBLE" panel, ignoring the smaller,
    roughly-square premium-square icons the board also prints in red --
    only the elongated panel is useful (for orientation), so the
    premium-square icons are filtered out rather than classified.

    Must run on an already-warped board (see module docstring), not the
    raw photo -- the aspect/area thresholds assume the panel occupies a
    board-relative fraction of the frame, not a photo-relative one.

    Board red only ever falls on the *high* side of OpenCV's hue wrap
    (~172-179), never the low side (~0-6) -- even though both look "red" to
    the eye. Reddish-brown wood grain sits on the low side and is easily
    mistaken for board red there (see detect_board.py's git history for the
    original 1055-vs-32-candidate-blobs comparison this was tuned against).

    Returns (centroid, bbox) for the largest sufficiently elongated
    (aspect >= red_aspect_threshold) red blob, or None if nothing matched.
    """
    p = _params(param_overrides)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, (int(p["red_hue_min"]), int(p["red_sat_min"]), int(p["red_val_min"])), (int(p["red_hue_max"]), 255, 255)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = p["red_min_area_frac"] * bgr.shape[0] * bgr.shape[1]
    best, best_area = None, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area <= best_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        bbox = cv2.boundingRect(contour)
        _, _, w, h = bbox
        aspect = max(w, h) / max(1, min(w, h))
        if aspect < p["red_aspect_threshold"]:
            continue  # too square to be the panel -- a premium-square icon
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        best, best_area = (centroid, bbox), area
    return best


# Rotation that brings the edge nearest the panel to the bottom of the
# image (panel canonically sits along the board's bottom edge).
_ROTATION_TO_BOTTOM = {
    "left": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "top": cv2.ROTATE_180,
    "right": cv2.ROTATE_90_CLOCKWISE,
    "bottom": None,
}


def find_panel_edge(bgr, **param_overrides):
    """Which edge of `bgr` the "SCRABBLE" panel is nearest to, or None if
    no panel was found. `bgr` must already be a warped board (see module
    docstring) -- the panel's position pins down the board's rotation."""
    panel = find_panel(bgr, **param_overrides)
    if panel is None:
        return None
    (cx, cy), _ = panel
    img_h, img_w = bgr.shape[:2]
    distances = {"left": cx, "right": img_w - cx, "top": cy, "bottom": img_h - cy}
    return min(distances, key=distances.get)


def orient_to_bottom(bgr, **param_overrides):
    """Rotate `bgr` (a warped board) so the SCRABBLE panel ends up at the
    bottom edge. Returns (rotated_image, edge_found); edge_found is None
    (image unchanged) when no panel could be located."""
    edge = find_panel_edge(bgr, **param_overrides)
    if edge is None:
        return bgr, None
    rotation = _ROTATION_TO_BOTTOM[edge]
    return (bgr if rotation is None else cv2.rotate(bgr, rotation)), edge


def white_grid_mask(bgr, **param_overrides):
    """Isolate the thin white grid lines between cells.

    Hue-agnostic (white has no meaningful hue): just a saturation ceiling
    and value floor. The raw mask is dilated first since thin lines often
    break into disconnected segments at typical source resolution, then
    closed to bridge the remaining gaps (especially at cell-corner
    intersections) into one connected lattice, then opened to drop small
    noise flecks that pass the floor/ceiling but aren't part of the grid.
    """
    p = _params(param_overrides)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, int(p["white_val_min"])), (179, int(p["white_sat_max"]), 255))

    dilate_size = max(1, int(p["grid_dilate_kernel"]))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)))

    close_size = max(1, int(p["grid_close_kernel"]))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
        iterations=max(1, int(p["grid_close_iterations"])),
    )

    open_size = max(1, int(p["grid_open_kernel"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)))
    return mask


def find_grid_quad(bgr, **param_overrides):
    """Find the precise 15x15 grid's outer quad within an already
    warped+oriented board, using the white grid-line lattice instead of
    the coarser outer bezel+teal quad detect_board.py finds -- that outer
    quad includes the bezel/panel margin, which is what made drawing a
    uniform grid directly on stage 1's warp land on the frame instead of
    the play area. Reuses detect_board.find_quad_candidates/is_valid_quad
    (same quad-plausibility logic as stage 1) with this stage's own,
    much larger minimum-area threshold, since the grid should fill most
    of an already-cropped board.

    Returns (corners_or_None, mask_shown).
    """
    p = _params(param_overrides)
    mask = white_grid_mask(bgr, **p)
    area = bgr.shape[0] * bgr.shape[1]
    candidates = find_quad_candidates(mask, area, **p)
    if not candidates:
        return None, mask
    best = max(candidates, key=cv2.contourArea)
    return order_corners(best), mask


def warp_to_grid(bgr, grid_corners):
    """Perspective-warp `bgr` (an already warped+oriented board) onto
    `grid_corners` (from find_grid_quad), producing a crop aligned to the
    actual 15x15 play area instead of the coarser outer quad."""
    w, h = bgr.shape[1], bgr.shape[0]
    new_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(grid_corners, new_corners)
    return cv2.warpPerspective(bgr, matrix, (w, h), flags=cv2.INTER_LINEAR)


def binarize_tiles(bgr, **param_overrides):
    """Render a grid-aligned board as tiles=white (including black-ink
    letter holes) and everything else (grid lines, empty cells, any bezel
    remnant) = black.

    Tiles and the white grid lines share the same (low-saturation,
    high-value) colour range, so they can't be told apart by colour --
    only by size: a morphological open with a kernel between the grid-line
    width and a tile's size erodes away the thin line network while
    leaving each tile's solid blob -- and its interior letter-ink holes,
    too small relative to the kernel to be affected -- intact.
    """
    p = _params(param_overrides)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, (0, 0, int(p["white_val_min"])), (179, int(p["white_sat_max"]), 255))
    open_size = max(1, int(p["tile_open_kernel"]))
    return cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)))


def extract_cells(binarized, n=15, expand_frac=0.0):
    """Uniform n x n slice of binarize_tiles()'s output into per-cell crops.

    The cell *boundaries* are exact by construction, not an approximation:
    warp_to_grid() already mapped find_grid_quad()'s detected corners onto
    this image's own four corners (that's what a perspective warp does),
    so as long as that quad was correct, plain w/n, h/n arithmetic lands
    exactly on the true cell boundaries -- no further grid-line detection
    is needed at this stage. But a single quad only corrects the *outer*
    corners; residual bowing/keystone between them, or a tile that's
    physically sitting a little off-centre in its square, means the
    physical tile isn't always perfectly flush with its ideal cell -- an
    exact crop can clip part of it. `expand_frac` grows every cell's crop
    by that fraction of a cell's size on each side (crops of neighbouring
    cells then overlap a little), so a slightly shifted or perspective-
    stretched tile is still captured whole; letter_classifier.py's contour
    logic still finds the *centre* tile correctly since it's much larger
    than any neighbouring tile's sliver at the crop's edge.

    Returns a list of n rows, each a list of n cell crops (row-major, i.e.
    cells[row][col]).
    """
    h, w = binarized.shape[:2]
    cell_h, cell_w = h / n, w / n
    pad_h, pad_w = int(cell_h * expand_frac), int(cell_w * expand_frac)
    return [
        [
            binarized[
                max(0, int(r * cell_h) - pad_h):min(h, int((r + 1) * cell_h) + pad_h),
                max(0, int(c * cell_w) - pad_w):min(w, int((c + 1) * cell_w) + pad_w),
            ]
            for c in range(n)
        ]
        for r in range(n)
    ]


def read_grid(warped_board_bgr):
    """Reference pipeline: orient -> find+warp to the precise grid ->
    binarize. `warped_board_bgr` must already be stage 1's output (see
    module docstring). Returns a dict of every intermediate result, for
    debugging/display; `binarized` is None if no grid quad was found."""
    oriented, panel_edge = orient_to_bottom(warped_board_bgr)
    grid_corners, grid_mask = find_grid_quad(oriented)
    if grid_corners is None:
        return {
            "oriented": oriented, "panel_edge": panel_edge,
            "grid_mask": grid_mask, "grid_corners": None,
            "grid_warp": None, "binarized": None,
        }
    grid_warp = warp_to_grid(oriented, grid_corners)
    binarized = binarize_tiles(grid_warp)
    return {
        "oriented": oriented, "panel_edge": panel_edge,
        "grid_mask": grid_mask, "grid_corners": grid_corners,
        "grid_warp": grid_warp, "binarized": binarized,
    }
