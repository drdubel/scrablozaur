import glob
import signal
import sys

import cv2
import numpy as np
from hsv_config import load_params, load_range

# Defaults match hsv_tuner.py's seed values; overridden by a tuned preset
# saved to hsv_config.json (name "board_teal"), if one exists.
TEAL_LOWER_DEFAULT = (70, 35, 85)
TEAL_UPPER_DEFAULT = (125, 255, 255)

# Every other board-detection knob (dilation/close/open kernels, Canny blur
# and thresholds, quad-validity thresholds), overridden by a tuned preset
# saved to hsv_config.json (name "board_params"), if one exists.
# hsv_tuner.py exposes all of these as trackbars. Orientation (red panel)
# and grid-level parameters live in grid_reader.py / grid_tuner.py instead
# -- this module only finds the board's *outer* quad, nothing past that.
PARAM_DEFAULTS = {
    "dark_s_max": 110,
    "dark_v_max": 90,
    "near_teal_kernel": 25,
    "close_kernel": 15,
    "close_iterations": 2,
    "open_kernel": 5,
    "canny_blur_kernel": 9,
    "canny_blur_sigma": 5,
    "canny_low": 10,
    "canny_high": 100,
    "canny_dilate_kernel": 7,
    "quad_side_ratio_max": 1.6,
    "quad_angle_tolerance": 45,
    "quad_min_area_frac": 0.08,
}


def _params(overrides=None):
    """Merge hsv_config.json's saved "board_params" preset with any
    explicit overrides -- used by hsv_tuner.py to preview live, not-yet-
    saved slider values without every function needing its own long
    parameter list."""
    merged = load_params("board_params", PARAM_DEFAULTS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def make_opening(image):
    kernel = np.ones((7, 7), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def make_canny(image):
    return cv2.Canny(image, 100, 200)


def contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def signal_handler(sig, frame):
    """Handle Ctrl+C interruption gracefully"""
    print("\nClosing...")
    cv2.destroyAllWindows()
    sys.exit(0)


def show_image(title, image):
    """Display an image in a resizable window."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    # Use shorter wait time to be responsive to Ctrl+C
    while True:
        key = cv2.waitKey(100)  # Wait 100ms
        if key != -1:  # If a key was pressed
            break
    cv2.destroyAllWindows()


# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


# Detection runs on a copy downscaled so its longer side is this many pixels;
# only the search stage is downscaled, the final warp uses the full-res image.
DETECT_MAX_SIDE = 1400


def order_corners(approx):
    """Sort 4 corner points into TL, TR, BR, BL order.

    approxPolyDP returns corners in whatever order it traced the contour,
    which is not necessarily TL-first, so mapping them onto an axis-aligned
    rectangle without sorting can flip or skew a rotated board's warp.
    """
    corners = approx.reshape(-1, 2).astype(np.float32)
    total = corners.sum(axis=1)
    diff = corners[:, 0] - corners[:, 1]
    top_left = corners[np.argmin(total)]
    bottom_right = corners[np.argmax(total)]
    top_right = corners[np.argmax(diff)]
    bottom_left = corners[np.argmin(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def approx_quad(contour):
    """Simplify a contour to a convex quadrilateral, or None if it isn't one.

    A single fixed epsilon either fails to collapse noisy/rotated outlines to
    exactly 4 points or over-simplifies and loses a corner, so a small sweep
    is tried and the first value that yields a clean 4-point hull is kept.
    """
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon_fraction in (0.02, 0.05, 0.08, 0.10):
        candidate = cv2.approxPolyDP(hull, epsilon_fraction * perimeter, True)
        if len(candidate) == 4 and cv2.isContourConvex(candidate):
            return candidate
    return None


def _corner_angle(a, b, c):
    """Interior angle at vertex b, in degrees."""
    v1, v2 = a - b, c - b
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def is_valid_quad(approx, img_area, **param_overrides):
    """Reject candidates that aren't plausibly a (perspective-skewed) square.

    A Scrabble board is square, so its true outline -- however skewed by
    perspective -- keeps a bounded side-length ratio and near-90 degree
    corners. This is what actually separates the real board from same-sized
    background blobs (e.g. couch fabric matching the colour mask): area
    alone doesn't, since a background region can easily be the larger one.
    """
    p = _params(param_overrides)
    corners = approx.reshape(-1, 2).astype(np.float64)
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    if max(sides) / min(sides) > p["quad_side_ratio_max"]:
        return False

    angles = [_corner_angle(corners[(i - 1) % 4], corners[i], corners[(i + 1) % 4]) for i in range(4)]
    if max(abs(angle - 90) for angle in angles) > p["quad_angle_tolerance"]:
        return False

    return cv2.contourArea(approx) >= img_area * p["quad_min_area_frac"]


def board_color_mask(bgr, teal_lower=None, teal_upper=None, **param_overrides):
    """Mask the board's own colours: teal interior + black bezel.

    Their union is one connected blob whose *outer* contour is the true
    board edge (bezel and red panel included) -- there is no separate inner
    quad to prefer by mistake, unlike a Canny edge map of the grid lines.
    Value/saturation bounds (not hue alone) keep dim, desaturated furniture
    (e.g. a dark blue couch, which can share the board's hue) out of the mask.

    "Dark" alone (low value, low saturation) matches far more than the thin
    bezel -- shadows, a dark couch, an unlit wall -- so the dark mask is
    restricted to pixels near the teal region before being unioned in,
    instead of being taken globally across the whole photo.

    `teal_lower`/`teal_upper` and any of PARAM_DEFAULTS' keys override the
    saved/default values when given -- used by hsv_tuner.py to preview this
    exact function against live, not-yet-saved slider values instead of
    duplicating its logic.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if teal_lower is None or teal_upper is None:
        teal_lower, teal_upper = load_range("board_teal", TEAL_LOWER_DEFAULT, TEAL_UPPER_DEFAULT)
    p = _params(param_overrides)
    teal = cv2.inRange(hsv, teal_lower, teal_upper)
    dark = cv2.inRange(hsv, (0, 0, 0), (179, int(p["dark_s_max"]), int(p["dark_v_max"])))

    near_teal_size = max(1, int(p["near_teal_kernel"]))
    near_teal_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_teal_size, near_teal_size))
    near_teal = cv2.dilate(teal, near_teal_kernel)
    bezel = cv2.bitwise_and(dark, near_teal)

    mask = cv2.bitwise_or(teal, bezel)

    close_size = max(1, int(p["close_kernel"]))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=max(1, int(p["close_iterations"])))
    open_size = max(1, int(p["open_kernel"]))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return mask


def canny_edge_mask(bgr, **param_overrides):
    p = _params(param_overrides)
    gray = get_grayscale(bgr)
    blur_size = max(1, int(p["canny_blur_kernel"])) | 1  # GaussianBlur kernel must be odd
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), float(p["canny_blur_sigma"]))
    edged = cv2.Canny(blurred, float(p["canny_low"]), float(p["canny_high"]))
    dilate_size = max(1, int(p["canny_dilate_kernel"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
    return cv2.dilate(edged, kernel, iterations=1)


def find_quad_candidates(mask, img_area, **param_overrides):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        approx = approx_quad(contour)
        if approx is None or not is_valid_quad(approx, img_area, **param_overrides):
            continue
        candidates.append(approx)
    return candidates


def find_board_quad(image):
    """Locate the board's outer quadrilateral in full-resolution image coordinates.

    Search runs on a downscaled copy for speed (detection only needs
    approximate corner locations, so full resolution buys nothing there);
    the winning corners are scaled back up for the caller's full-res warp.
    Colour segmentation is tried first since it targets the true outer edge
    directly; the gradient-based approach is a fallback for unusual lighting
    where colour segmentation fails to produce any candidate.
    """
    h, w = image.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else image
    small_area = small.shape[0] * small.shape[1]

    candidates = find_quad_candidates(board_color_mask(small), small_area)
    if not candidates:
        candidates = find_quad_candidates(canny_edge_mask(small), small_area)
    if not candidates:
        return None

    best = max(candidates, key=cv2.contourArea)
    return order_corners(best) / scale


def warp_board(image, corners):
    """Perspective-warp `image` onto `corners` (from find_board_quad),
    producing the "Warped Board": the warp itself targets a canvas shaped
    like the source image (the board's true corners don't generally span a
    square region of the photo), so a following resize is what actually
    squares it off. Shared by detect_board(), hsv_tuner.py, and
    grid_tuner.py so this exact stage-1 geometry only lives in one place.
    """
    w, h = image.shape[1], image.shape[0]
    new_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, new_corners)
    warped = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
    return cv2.resize(warped, (warped.shape[1], warped.shape[1]))


def detect_board(image_path):
    """Stage 1 only: find + warp the board's outer quad. Orientation (red
    panel), the white-grid refinement, and tile/ink binarization are
    grid_reader.py's job -- see grid_tuner.py to tune and preview those on
    top of this function's output."""
    image = cv2.imread(image_path)
    corners = find_board_quad(image)

    show_image("Original Image", image)

    contours_img = image.copy()

    if corners is not None:
        cv2.polylines(contours_img, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        resized_image = warp_board(image, corners)

        show_image("Detected Board", contours_img)
        show_image("Warped Board", resized_image)
    else:
        print(f"No board found in {image_path}")


def main():
    path = "test/in/*_e.jpg"
    paths = glob.glob(path)
    print(paths)

    for path in paths:
        detect_board(path)


if __name__ == "__main__":
    main()
