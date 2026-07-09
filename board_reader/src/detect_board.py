import glob
import signal
import sys

import cv2
import numpy as np


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


def is_valid_quad(approx, img_area):
    """Reject candidates that aren't plausibly a (perspective-skewed) square.

    A Scrabble board is square, so its true outline -- however skewed by
    perspective -- keeps a bounded side-length ratio and near-90 degree
    corners. This is what actually separates the real board from same-sized
    background blobs (e.g. couch fabric matching the colour mask): area
    alone doesn't, since a background region can easily be the larger one.
    """
    corners = approx.reshape(-1, 2).astype(np.float64)
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    if max(sides) / min(sides) > 1.6:
        return False

    angles = [_corner_angle(corners[(i - 1) % 4], corners[i], corners[(i + 1) % 4]) for i in range(4)]
    if max(abs(angle - 90) for angle in angles) > 45:
        return False

    return cv2.contourArea(approx) >= img_area * 0.08


def board_color_mask(bgr):
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
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    teal = cv2.inRange(hsv, (70, 35, 85), (125, 255, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 110, 90))

    near_teal_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    near_teal = cv2.dilate(teal, near_teal_kernel)
    bezel = cv2.bitwise_and(dark, near_teal)

    mask = cv2.bitwise_or(teal, bezel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return mask


def canny_edge_mask(bgr):
    gray = get_grayscale(bgr)
    blurred = cv2.GaussianBlur(gray, (9, 9), 5)
    edged = cv2.Canny(blurred, 10, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.dilate(edged, kernel, iterations=1)


def find_quad_candidates(mask, img_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        approx = approx_quad(contour)
        if approx is not None and is_valid_quad(approx, img_area):
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


def detect_board(image_path):
    image = cv2.imread(image_path)
    corners = find_board_quad(image)

    contours_img = image.copy()

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

        resized_image = cv2.resize(warped_image, (warped_image.shape[1], warped_image.shape[1]))

        # cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
        # cv2.imshow("Board", resized_image)

        # Use shorter wait time to be responsive to Ctrl+C
        # while True:
        #    key = cv2.waitKey(100)  # Wait 100ms
        #    if key != -1:  # If a key was pressed
        #        break
        # cv2.destroyAllWindows()
    else:
        print(f"No board found in {image_path}")

    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
    cv2.imshow("Contours", contours_img)

    # Use shorter wait time to be responsive to Ctrl+C
    while True:
        key = cv2.waitKey(100)  # Wait 100ms
        if key != -1:  # If a key was pressed
            break
    cv2.destroyAllWindows()


def main():
    path = "test/in/*_e.jpg"
    paths = glob.glob(path)
    print(paths)

    for path in paths:
        detect_board(path)


if __name__ == "__main__":
    main()
