import glob

import cv2
from cv_utils import ParamSpec, run_tuner
from detect_board import find_board_quad, warp_board
from hsv_config import load_params

SPECS = [
    ParamSpec("hue_low", "hue low", 179, 1, "lower red hue band (wraps at 180)"),
    ParamSpec("sat_min", "sat min", 255, 1, "minimum saturation to count as red"),
    ParamSpec("val_min", "val min", 255, 1, "minimum value/brightness to count as red"),
]

RED_RECT_DEFAULTS = {"hue_low": 10, "sat_min": 100, "val_min": 100}


def _params(overrides=None):
    """Merge hsv_config.json's saved "red_rectangle_params" preset with any
    explicit overrides -- so red_rectangle_mask() picks up whatever was
    last tuned by default, instead of the hardcoded fallback values."""
    merged = load_params("red_rectangle_params", RED_RECT_DEFAULTS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def red_rectangle_mask(board, **param_overrides):
    """Return a mask of the red rectangle in the board, if any."""
    p = _params(param_overrides)
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    lower_red1 = (0, p["sat_min"], p["val_min"])
    upper_red1 = (p["hue_low"], 255, 255)
    lower_red2 = (160, p["sat_min"], p["val_min"])
    upper_red2 = (180, 255, 255)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)


def find_red_rectangle(board, **param_overrides):
    """Find the largest red rectangle in the board and return its corners."""
    mask = red_rectangle_mask(board, **param_overrides)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        return approx.reshape(4, 2)

    return None


def rotate_board(board):
    return board  # TODO: implement rotation detection


def detect_grid(path):
    """Detect the board in the given image and return a warped, square version
    of it. Returns None if no board was found."""
    image = cv2.imread(path)
    corners = find_board_quad(image)
    if corners is None:
        print(f"No board found in {path}")
        return None

    board = warp_board(image, corners)

    def render(params):
        corners = find_red_rectangle(board, **params)
        if corners is not None:
            cv2.polylines(board, [corners], True, (255, 255, 0), 5)

        return [board, red_rectangle_mask(board, **params)]

    run_tuner(SPECS, render, RED_RECT_DEFAULTS, window="Red Rectangle Tuner", config_name="red_rectangle_params")
    return board


def main():
    path = "test/in/*_e.jpg"
    paths = glob.glob(path)
    print(paths)

    for path in paths:
        detect_grid(path)


if __name__ == "__main__":
    main()
