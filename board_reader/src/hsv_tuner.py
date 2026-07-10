"""Interactive HSV color-range tuner.

Drag the trackbars until the "Board Detection Result" window outlines the
board correctly on the current image, then press 's' to record that
image's working range. Repeat across images with 'n'/'p'; on quit, if any
ranges were recorded, the tuple that covers all of them (component-wise
min of the lower bounds, max of the upper bounds) is computed and
re-validated against every recorded image, so you get a single range
plus a report of whether it actually still finds the board everywhere.

Usage (run from board_reader/, same convention as detect_board.py):
    python src/hsv_tuner.py               # test/in/*_e.jpg
    python src/hsv_tuner.py "some/glob/*.jpg"

Keys:
    n / p    next / previous image (keeps current slider positions)
    s        record current slider values as this image's working range
    q / Esc  quit; compute + validate the combined range if any were recorded
"""

import glob
import sys

import cv2
import numpy as np

from detect_board import find_quad_candidates, order_corners
from detect_board import signal_handler  # noqa: F401  (registers SIGINT handler on import)

WINDOW = "HSV Tuner (mask | original)"
DETECTION_WINDOW = "Board Detection Result"
DISPLAY_MAX_SIDE = 900

# Trackbar name -> (max value, initial value). Initial values are seeded
# from board_color_mask's current teal bounds so tuning starts from a
# known-working range instead of from scratch.
TRACKBARS = [
    ("H min", 179, 70), ("H max", 179, 125),
    ("S min", 255, 35), ("S max", 255, 255),
    ("V min", 255, 85), ("V max", 255, 255),
]


def _nothing(_):
    pass


def _load(path):
    image = cv2.imread(path)
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = min(1.0, DISPLAY_MAX_SIDE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "test/in/*_e.jpg"
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No images matched {pattern!r}")
        sys.exit(1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DETECTION_WINDOW, cv2.WINDOW_NORMAL)
    for name, maxval, default in TRACKBARS:
        cv2.createTrackbar(name, WINDOW, default, maxval, _nothing)

    print(f"{len(paths)} image(s). n/p: switch image, s: record working range, "
          f"q/Esc: quit and combine.")

    recordings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    idx = 0
    image = _load(paths[idx])
    while image is None and idx < len(paths) - 1:
        idx += 1
        image = _load(paths[idx])
    if image is None:
        print("Could not read any of the matched images.")
        sys.exit(1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print(f"[{idx + 1}/{len(paths)}] {paths[idx]}")

    lower = upper = None
    while True:
        lower = np.array([cv2.getTrackbarPos(n, WINDOW) for n in ("H min", "S min", "V min")])
        upper = np.array([cv2.getTrackbarPos(n, WINDOW) for n in ("H max", "S max", "V max")])
        mask = cv2.inRange(hsv, lower, upper)
        preview = np.hstack([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), image])
        status = f"recorded {len(recordings)}/{len(paths)} | this image: " \
                 f"{'SAVED' if paths[idx] in recordings else 'not saved'}"
        cv2.putText(preview, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.imshow(WINDOW, preview)

        detected = image.copy()
        candidates = find_quad_candidates(mask, mask.shape[0] * mask.shape[1])
        if candidates:
            best = max(candidates, key=cv2.contourArea)
            corners = order_corners(best)
            cv2.polylines(detected, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        else:
            cv2.putText(detected, "No board found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(DETECTION_WINDOW, detected)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            recordings[paths[idx]] = (lower.copy(), upper.copy())
            note = "" if candidates else "  (warning: no board found at these values)"
            print(f"  saved {paths[idx]}: lower={tuple(int(v) for v in lower)} "
                  f"upper={tuple(int(v) for v in upper)}{note}")
        elif key in (ord("n"), ord("p")) and len(paths) > 1:
            step = 1 if key == ord("n") else -1
            new_image = None
            for _ in range(len(paths)):
                idx = (idx + step) % len(paths)
                new_image = _load(paths[idx])
                if new_image is not None:
                    break
            if new_image is not None:
                image = new_image
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saved_note = " (already saved)" if paths[idx] in recordings else ""
                print(f"[{idx + 1}/{len(paths)}] {paths[idx]}{saved_note}")

    cv2.destroyAllWindows()

    if not recordings:
        print(f"lower = {tuple(int(v) for v in lower)}")
        print(f"upper = {tuple(int(v) for v in upper)}")
        return

    lowers = np.stack([lo for lo, _ in recordings.values()])
    uppers = np.stack([up for _, up in recordings.values()])
    combined_lower = lowers.min(axis=0)
    combined_upper = uppers.max(axis=0)
    print(f"\n{len(recordings)} image(s) recorded.")
    print(f"combined lower = {tuple(int(v) for v in combined_lower)}")
    print(f"combined upper = {tuple(int(v) for v in combined_upper)}")

    print("\nValidating combined range against every recorded image:")
    ok = 0
    for path in recordings:
        img = _load(path)
        if img is None:
            print(f"  SKIP could not reload {path}")
            continue
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_mask = cv2.inRange(img_hsv, combined_lower, combined_upper)
        found = bool(find_quad_candidates(img_mask, img_mask.shape[0] * img_mask.shape[1]))
        ok += found
        print(f"  {'OK  ' if found else 'FAIL'} {path}")
    print(f"\n{ok}/{len(recordings)} recorded images still detect the board "
          f"with the combined range.")


if __name__ == "__main__":
    main()
