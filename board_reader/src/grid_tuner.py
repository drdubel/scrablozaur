"""Interactive tuner for STAGE 2: orientation (red SCRABBLE panel), the
white grid, and tile/ink binarization -- everything grid_reader.py does on
top of hsv_tuner.py's stage-1 output. This tool runs stage 1 itself first
(find_board_quad + warp_board, using whatever hsv_tuner.py already saved to
hsv_config.json) so it always starts from a real warped board, exactly like
grid_reader.find_panel()/orient_to_bottom() require -- the panel and grid
are searched for on that transformed image, never the raw photo.

Everything lives in one window: all sliders (white value/saturation range,
then red-panel hue/sat/val/area/aspect, grid-line dilate/close/open
kernels, grid quad-validity thresholds, tile-open kernel) stack at the top,
and three live previews sit side by side below them -- the white-grid mask,
the oriented board with the found grid quad outlined (green) or a "no grid
found" placeholder, and binarize_tiles()'s output (tiles white with letters
on them as black holes, everything else black).

Sliders are seeded from hsv_config.json's saved "grid_params" preset if one
exists, otherwise from grid_reader.py's hardcoded defaults.

Press 's' to record the current white range as this image's working value,
repeat across images with 'n'/'p', then 'q'/Esc to combine everything
recorded (min of the value floors, max of the saturation ceilings) and
re-validate against every recorded image. The combined white range plus
whatever the other sliders currently show -- or, if nothing was recorded,
just the current white sliders -- gets written to hsv_config.json ('w'
saves that at any time), which grid_reader.py's functions read
automatically. 'r' resets every slider back to its seed.

Usage (run from board_reader/, same convention as hsv_tuner.py):
    python src/grid_tuner.py                    # difficulty "e" (easy) only
    python src/grid_tuner.py -d emh              # easy + medium + hard
    python src/grid_tuner.py "some/glob/*.jpg"   # explicit pattern, overrides -d

    (run with plain `python`, not `ipython` -- see hsv_tuner.py's note)

Keys:
    n / p    next / previous image (keeps current slider positions)
    s        record current white range as this image's working value
    w        save current sliders (white range + parameters) right now
    [ / ]    select the previous / next individual slider
    0        reset only the selected slider back to its seed
    r        reset every slider back to its seed (saved config or defaults)
    q / Esc  quit; compute + validate + save the combined range if any were recorded
"""

import argparse
import glob
import sys
from collections import namedtuple

import cv2
import numpy as np
from detect_board import signal_handler  # noqa: F401  (registers SIGINT handler on import)
from detect_board import find_board_quad, warp_board
from grid_reader import PARAM_DEFAULTS, binarize_tiles, find_grid_quad, orient_to_bottom, warp_to_grid
from hsv_config import load_params, save_params

WINDOW = "Grid Tuner (white mask | oriented+grid | binarized)"

WHITE_TRACKBARS = ("val min", "sat max")

# Each parameter's UI: (json key, trackbar label, trackbar max position,
# scale). Trackbars are integer-only, so non-integer/small-fraction values
# are stored scaled up; `scale` divides the raw trackbar position back down
# to the real value, or is 1 for plain integer parameters. Labels are kept
# short -- cv2 clips trackbar labels to the window width.
ParamSpec = namedtuple("ParamSpec", "key label max_pos scale")
SliderRef = namedtuple("SliderRef", "label seed scale max_pos")
PARAM_SPECS = [
    ParamSpec("red_hue_min", "red hue min", 179, 1),
    ParamSpec("red_hue_max", "red hue max", 179, 1),
    ParamSpec("red_sat_min", "red sat min", 255, 1),
    ParamSpec("red_val_min", "red val min", 255, 1),
    ParamSpec("red_min_area_frac", "red area x1e-6", 2000, 1_000_000),
    ParamSpec("red_aspect_threshold", "red aspect x0.1", 100, 10),
    ParamSpec("grid_dilate_kernel", "grid dilate k", 31, 1),
    ParamSpec("grid_close_kernel", "grid close k", 61, 1),
    ParamSpec("grid_close_iterations", "grid close iters", 5, 1),
    ParamSpec("grid_open_kernel", "grid open k", 31, 1),
    ParamSpec("quad_side_ratio_max", "quad side x0.1", 100, 10),
    ParamSpec("quad_angle_tolerance", "quad angle deg", 90, 1),
    ParamSpec("quad_min_area_frac", "quad area x1e-3", 1000, 1000),
    ParamSpec("tile_open_kernel", "tile open k", 31, 1),
]

# Each preview panel is resized to this height before being hstacked into
# one composite image -- keeps the single window's width driven by the
# (wide) image content, which incidentally keeps every trackbar label
# above it fully visible without needing a manual minimum window size.
PANEL_HEIGHT = 520


def _nothing(_):
    pass


def _set_white_trackbars(val_min, sat_max):
    cv2.setTrackbarPos("val min", WINDOW, int(val_min))
    cv2.setTrackbarPos("sat max", WINDOW, int(sat_max))


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
        values[spec.key] = pos / spec.scale if spec.scale != 1 else int(pos)
    return values


def _build_slider_refs(seed_val_min, seed_sat_max, seed_params):
    """One SliderRef per individual trackbar (white range + every
    parameter), so any single one can be selected and reset on its own
    instead of only all-at-once via 'r'."""
    refs = [
        SliderRef("val min", int(seed_val_min), 1, 255),
        SliderRef("sat max", int(seed_sat_max), 1, 255),
    ]
    for spec in PARAM_SPECS:
        seed_val = seed_params.get(spec.key, PARAM_DEFAULTS[spec.key])
        refs.append(SliderRef(spec.label, seed_val, spec.scale, spec.max_pos))
    return refs


def _reset_single(ref):
    pos = int(round(ref.seed * ref.scale))
    cv2.setTrackbarPos(ref.label, WINDOW, max(0, min(ref.max_pos, pos)))


def _selection_status(ref):
    """Status line for the currently selected slider ('[' / ']' / '0')."""
    pos = cv2.getTrackbarPos(ref.label, WINDOW)
    current = pos / ref.scale if ref.scale != 1 else pos
    changed = abs(current - ref.seed) > 1e-9
    return f"selected: {ref.label}  current={current:g}  seed={ref.seed:g}{'  (changed)' if changed else ''}"


def _stage1_warp(image):
    """Run stage 1 (find_board_quad + warp_board) using whatever
    hsv_tuner.py already saved to hsv_config.json -- this tool tunes stage
    2 only, so stage 1's own sliders aren't re-exposed here."""
    corners = find_board_quad(image)
    if corners is None:
        return None
    return warp_board(image, corners)


def _panel(img, height=PANEL_HEIGHT):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (max(1, int(w * scale)), height))


def _compose(grid_mask, display, binarized):
    """Resize every preview to a common height and hstack them into one
    composite image, so all three previews and every trackbar live in a
    single window instead of several separate cv2 windows."""
    return np.hstack([_panel(cv2.cvtColor(grid_mask, cv2.COLOR_GRAY2BGR)), _panel(display), _panel(binarized)])


def _parse_args():
    p = argparse.ArgumentParser(description="Interactive tuner for orientation + white grid + binarization")
    p.add_argument("pattern", nargs="?", default=None, help="glob pattern for images (overrides --difficulty)")
    p.add_argument(
        "-d",
        "--difficulty",
        default="e",
        help="difficulty suffixes to include: any of 'e' (easy), 'm' (medium), 'h' (hard), e.g. -d emh (default: e)",
    )
    return p.parse_args()


def _load(path):
    image = cv2.imread(path)
    if image is None:
        return None
    return _stage1_warp(image)


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

    # Single "grid_params" preset covers everything grid_reader.py's own
    # _params() merges from, white range included -- keeping it split
    # across two preset names here would mean this tool's white-range
    # tuning silently never reaches grid_reader.py's defaults.
    seed_params = load_params("grid_params", PARAM_DEFAULTS)
    seed_val_min, seed_sat_max = seed_params["white_val_min"], seed_params["white_sat_max"]
    print(
        f"seed white range: val_min={seed_val_min} sat_max={seed_sat_max} "
        f"({'from hsv_config.json' if seed_params != PARAM_DEFAULTS else 'hardcoded default'})"
    )

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("val min", WINDOW, int(seed_val_min), 255, _nothing)
    cv2.createTrackbar("sat max", WINDOW, int(seed_sat_max), 255, _nothing)
    _create_param_trackbars(seed_params)

    slider_refs = _build_slider_refs(seed_val_min, seed_sat_max, seed_params)
    selected_idx = 0

    print(
        f"{len(paths)} image(s). n/p: switch image, s: record working range, w: save now, "
        f"[/]: select slider, 0: reset selected, r: reset all, q/Esc: quit and combine."
    )

    recordings: dict[str, tuple[int, int]] = {}
    idx = 0
    warped = _load(paths[idx])
    while warped is None and idx < len(paths) - 1:
        idx += 1
        warped = _load(paths[idx])
    if warped is None:
        print("Could not find a board (via stage 1) on any of the matched images.")
        sys.exit(1)
    cv2.setWindowTitle(WINDOW, f"{WINDOW} - {paths[idx]}")
    print(f"[{idx + 1}/{len(paths)}] {paths[idx]}")

    val_min = sat_max = None
    while True:
        val_min = cv2.getTrackbarPos("val min", WINDOW)
        sat_max = cv2.getTrackbarPos("sat max", WINDOW)
        params = _read_params()
        white = {"white_val_min": val_min, "white_sat_max": sat_max}

        oriented, panel_edge = orient_to_bottom(warped, **params)
        grid_corners, grid_mask = find_grid_quad(oriented, **white, **params)

        display = oriented.copy()
        panel_label = "panel: not found" if panel_edge is None else f"panel was near {panel_edge} -> rotated"
        cv2.putText(display, panel_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if grid_corners is not None:
            cv2.polylines(display, [grid_corners.astype(np.int32)], True, (0, 255, 0), 3)
            grid_warp = warp_to_grid(oriented, grid_corners)
            binarized = cv2.cvtColor(binarize_tiles(grid_warp, **white, **params), cv2.COLOR_GRAY2BGR)
        else:
            cv2.putText(display, "No grid found", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            binarized = cv2.cvtColor(np.zeros(oriented.shape[:2], np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.putText(binarized, "No grid found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        composite = _compose(grid_mask, display, binarized)
        status = (
            f"recorded {len(recordings)}/{len(paths)} | this image: "
            f"{'SAVED' if paths[idx] in recordings else 'not saved'}"
        )
        cv2.putText(composite, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(
            composite,
            _selection_status(slider_refs[selected_idx]),
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
        )
        cv2.imshow(WINDOW, composite)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            recordings[paths[idx]] = (val_min, sat_max)
            note = "" if grid_corners is not None else "  (warning: no grid found at these values)"
            print(f"  saved {paths[idx]}: val_min={val_min} sat_max={sat_max}{note}")
        elif key == ord("w"):
            save_params("grid_params", {**params, **white})
            print(f"  wrote hsv_config.json: white val_min={val_min} sat_max={sat_max}, {len(params)} other params")
        elif key == ord("r"):
            _set_white_trackbars(seed_val_min, seed_sat_max)
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
            step = 1 if key == ord("n") else -1
            new_warped = None
            for _ in range(len(paths)):
                idx = (idx + step) % len(paths)
                new_warped = _load(paths[idx])
                if new_warped is not None:
                    break
            if new_warped is not None:
                warped = new_warped
                cv2.setWindowTitle(WINDOW, f"{WINDOW} - {paths[idx]}")
                saved_note = " (already saved)" if paths[idx] in recordings else ""
                print(f"[{idx + 1}/{len(paths)}] {paths[idx]}{saved_note}")
            else:
                print("  no other image has a stage-1 board to warp to -- staying put")

    cv2.destroyAllWindows()

    if not recordings:
        print(f"val_min = {val_min}")
        print(f"sat_max = {sat_max}")
        return

    val_mins = [v for v, _ in recordings.values()]
    sat_maxes = [s for _, s in recordings.values()]
    combined_val_min = min(val_mins)
    combined_sat_max = max(sat_maxes)
    print(f"\n{len(recordings)} image(s) recorded.")
    print(f"combined val_min = {combined_val_min}")
    print(f"combined sat_max = {combined_sat_max}")

    # Reuse `params` as last read inside the loop (before the windows were
    # destroyed) -- calling _read_params() again here would call
    # cv2.getTrackbarPos() on already-destroyed windows and silently read
    # back -1 for every parameter (see hsv_tuner.py's git history).
    combined_white = {"white_val_min": combined_val_min, "white_sat_max": combined_sat_max}
    print("\nValidating combined range against every recorded image:")
    ok = 0
    for path in recordings:
        img = cv2.imread(path)
        w1 = _stage1_warp(img) if img is not None else None
        if w1 is None:
            print(f"  SKIP could not re-warp {path}")
            continue
        oriented, _ = orient_to_bottom(w1, **params)
        grid_corners, _ = find_grid_quad(oriented, **combined_white, **params)
        found = grid_corners is not None
        ok += found
        print(f"  {'OK  ' if found else 'FAIL'} {path}")
    print(f"\n{ok}/{len(recordings)} recorded images still find the grid with the combined range.")

    save_params("grid_params", {**params, **combined_white})
    print("\nSaved to hsv_config.json -- grid_reader.py's functions will use it automatically.")


if __name__ == "__main__":
    main()
