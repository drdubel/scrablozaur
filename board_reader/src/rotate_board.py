import glob

import cv2
import numpy as np
from cv_utils import ParamSpec
from hsv_config import load_params

SPECS = [
    ParamSpec(
        "percentile", "percentile", 100, 1, "keep the reddest N% of pixels (relative to this image, not a fixed cutoff)"
    ),
    ParamSpec(
        "redness_min",
        "min redness",
        255,
        1,
        "absolute floor so a board with no red rectangle doesn't false-positive on noise",
    ),
    ParamSpec(
        "close_size",
        "close size",
        41,
        1,
        "morphological closing kernel size -- bridges glare holes/notches in the mask",
    ),
]

RED_RECT_DEFAULTS = {"percentile": 95, "redness_min": 5, "close_size": 30}

MIN_ASPECT_RATIO = 4  # the physical marker is a long strip, never square -- rejects blobby false positives
MIN_AREA = 10000
MIN_FILL_RATIO = 0.7  # contour area vs. its minAreaRect box -- rejects non-rectangular blobs


def _params(overrides=None):
    """Merge hsv_config.json's saved "red_rectangle_params" preset with any
    explicit overrides -- so red_rectangle_mask() picks up whatever was
    last tuned by default, instead of the hardcoded fallback values."""
    merged = load_params("red_rectangle_params", RED_RECT_DEFAULTS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def red_rectangle_mask(board, **param_overrides):
    """Return a mask of the red rectangle in the board, if any.

    Ranks pixels by how much red dominates over green/blue and keeps the
    reddest N% of *this* image, rather than testing each pixel against a
    fixed HSV cutoff. A glare-blown patch on the rectangle rarely regains
    an absolute "red" saturation/value, but it's still redder than
    anything on the green background, so the percentile threshold still
    catches it. Morphological closing then bridges whatever hole/notch
    the glare leaves in the mask before contour extraction.
    """
    p = _params(param_overrides)
    b, g, r = cv2.split(board.astype(np.int16))
    redness = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)

    threshold = max(p["redness_min"], np.percentile(redness, p["percentile"]))
    mask = np.where(redness >= threshold, 255, 0).astype(np.uint8)

    close_size = int(p["close_size"]) | 1  # kernel size must be odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def find_red_rectangle(board, **param_overrides):
    """Find the largest red rectangle in the board and return its corners.

    The marker's corners are visibly rounded, not sharp 90-degree corners,
    so approxPolyDP rarely collapses its contour to exactly 4 points (a
    rounded corner gets approximated as several short segments instead of
    one vertex). minAreaRect doesn't care about corner sharpness -- it
    fits the smallest enclosing rotated rectangle around the contour
    either way -- so that's used for the corners directly, and shape is
    validated via aspect ratio plus how much of that box the contour
    actually fills (rejects e.g. an L-shaped or diamond blob that happens
    to share the right bounding-box aspect ratio).
    """
    mask = red_rectangle_mask(board, **param_overrides)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if min(w, h) == 0:
            continue

        if max(w, h) / min(w, h) < MIN_ASPECT_RATIO:
            continue

        if area / (w * h) < MIN_FILL_RATIO:
            continue

        return cv2.boxPoints(rect).astype(np.int32)

    return None


def get_board_orientation(board):
    """Determine the orientation of the board (0, 90, 180, or 270 degrees) based on the red rectangle's position. Returns None if no rectangle is found."""
    corners = find_red_rectangle(board)
    if corners is None:
        return None

    # Calculate the center of the red rectangle
    center = np.mean(corners, axis=0)

    # Determine the orientation based on the position of the center
    height, width, _ = board.shape
    if center[0] < width / 4:
        return 90  # Left side
    elif center[0] > width * 3 / 4:
        return 270  # Right side
    elif center[1] < height / 4:
        return 180  # Top side
    else:
        return 0  # Bottom side


def rotate_board(board):
    """Rotate the board to ensure the red rectangle is in the top-left corner."""
    orientation = get_board_orientation(board)
    if orientation is None:
        print("No red rectangle found; cannot determine orientation.")
        return board  # Return the original board if no rectangle is found

    # Rotate the board based on the determined orientation
    if orientation == 0:
        return board  # No rotation needed
    elif orientation == 90:
        return cv2.rotate(board, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 180:
        return cv2.rotate(board, cv2.ROTATE_180)
    elif orientation == 270:
        return cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE)
