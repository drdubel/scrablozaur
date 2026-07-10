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

# Every other detection knob (dilation/close/open kernels, Canny blur and
# thresholds, red-ink range, quad-validity thresholds), overridden by a
# tuned preset saved to hsv_config.json (name "board_params"), if one
# exists. hsv_tuner.py exposes all of these as trackbars.
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
    "red_hue_min": 172,
    "red_hue_max": 179,
    "red_sat_min": 120,
    "red_val_min": 60,
    "red_min_area_frac": 0.00005,
    "red_aspect_threshold": 1.6,
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


def find_panel(bgr, **param_overrides):
    """Find the board's red "SCRABBLE" panel, ignoring the smaller,
    roughly-square premium-square icons the board also prints in red --
    only the elongated panel is useful (for orientation), so the
    premium-square icons are filtered out rather than classified.

    Board red only ever falls on the *high* side of OpenCV's hue wrap
    (~172-179), never the low side (~0-6) -- even though both look "red" to
    the eye. Reddish-brown wood grain sits on the low side and is easily
    mistaken for board red there: on one test photo, a hue range covering
    both sides of the wrap produced 1055 candidate blobs, almost all of them
    wood grain on the table below the board (confirmed by drawing their
    bounding boxes -- the two largest landed squarely on the table, not the
    board). Restricting to the high side alone dropped that to 32 blobs,
    with the board's own 6-8 premium squares and the panel landing cleanly
    among the largest of them (aspect ~1.0-1.1 for the squares, ~4 for the
    panel) and no wood-grain contamination.

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
    no panel was found. Meant to run on an already-warped board, where the
    panel's position pins down the board's rotation."""
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


def detect_board(image_path, show_panel=False):
    image = cv2.imread(image_path)
    corners = find_board_quad(image)

    show_image("Original Image", image)

    contours_img = image.copy()

    if show_panel:
        panel = find_panel(image)
        if panel is not None:
            _, (x, y, w, h) = panel
            cv2.rectangle(contours_img, (x, y), (x + w, y + h), (0, 255, 255), 4)

    if corners is not None:
        cv2.polylines(contours_img, [corners.astype(np.int32)], True, (0, 255, 0), 3)

        new_corners = np.array(
            [
                [0, 0],
                [image.shape[1], 0],
                [image.shape[1], image.shape[0]],
                [0, image.shape[0]],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(corners, new_corners)
        warped_image = cv2.warpPerspective(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        warped_image, panel_edge = orient_to_bottom(warped_image)
        print(
            f"SCRABBLE panel: "
            f"{'not found, orientation unchanged' if panel_edge is None else f'found near {panel_edge}, rotated to bottom'}"
        )

        resized_image = cv2.resize(warped_image, (warped_image.shape[1], warped_image.shape[1]))

        show_image("Detected Board with Panel", contours_img)
        show_image("Warped Board", resized_image)
    else:
        print(f"No board found in {image_path}")


def main():
    path = "test/in/*_m.jpg"
    paths = glob.glob(path)
    print(paths)

    for path in paths:
        detect_board(path, show_panel=True)


if __name__ == "__main__":
    main()
