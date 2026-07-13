"""Interactive tuners for board_reader's detection pipeline -- one entry
point, pick which stage to tune with a subcommand.

    python src/tuner.py board [-d em] [pattern]
        Tune find_board_quad()'s teal HSV range and every other
        board-detection knob (dark-bezel thresholds, dilation/close/open
        kernel sizes, Canny blur/thresholds, quad-validity thresholds).
        All sliders stack at the top of one window; three live previews
        (colour/Canny mask, detected outline, warped board) sit below.
        Sliders seed from hsv_config.json's saved presets if any exist,
        otherwise from detect_board.py's hardcoded defaults.

        Press 's' to record the current teal range as this image's
        working value, repeat across images with 'n'/'p', then 'q'/Esc
        to combine everything recorded (component-wise min of the lower
        bounds, max of the upper bounds) and re-validate the result
        against every recorded image. The combined teal range plus
        whatever the other sliders currently show -- or, if nothing was
        recorded, just the current teal sliders -- gets written to
        hsv_config.json ('w' saves that at any time).

        Every mask/detection step calls detect_board.py's actual
        functions (board_color_mask, canny_edge_mask,
        find_quad_candidates) at the same DETECT_MAX_SIDE scale
        detect_board.py itself searches at, instead of a simplified
        re-implementation -- otherwise a range that looks perfect here
        can still perform worse once it's actually used by
        detect_board(). The colour-mask stage is tried first and, if it
        finds nothing, the Canny fallback is tried too, exactly like
        find_board_quad().

        Keys: n/p next/previous image, s record working range, w save
        now, [/] select slider, 0 reset selected, r reset all, q/Esc
        quit (combine + validate + save).

    python src/tuner.py red_rectangle [-d em] [pattern]
        Tune red_rectangle_mask()/find_red_rectangle()'s params (the
        orientation marker on the board) via cv_utils.run_tuner()'s
        generic live-preview harness -- opens one tuner window per
        matched image, in sequence. Sliders seed from hsv_config.json's
        saved "red_rectangle_params" preset if any exists, otherwise
        rotate_board.py's RED_RECT_DEFAULTS.

        Keys: w save now, [/] select slider, 0 reset selected, r reset
        all, q/Esc quit (moves to the next image, if any).

    python src/tuner.py tile_detector [-d em] [pattern]
        Tune tile_detector.detect_tiles()'s params (per-class colour
        z-score, tile-colour-model seeding, accept/reject distance & z
        bounds, glyph floors) via cv_utils.run_tuner()'s generic
        live-preview harness -- opens one tuner window per matched image,
        in sequence. Each preview draws read_board.py's tile/empty overlay
        (green = tile, red = empty) so a slider change's effect on the
        whole board is visible at once, not just one cell. Sliders seed
        from hsv_config.json's saved "tile_detector_params" preset if any
        exists, otherwise tile_detector.py's PARAM_DEFAULTS.

        Keys: w save now, [/] select slider, 0 reset selected, r reset
        all, q/Esc quit (moves to the next image, if any).

    All three subcommands accept -d/--difficulty (any combination of 'e'
    easy, 'm' medium, 'h' hard, default 'e') to match against test/in's
    imgN_<difficulty>.jpg suffix, or an explicit glob `pattern` positional
    that overrides it.

Run with plain `python`, not `ipython` -- ipython swallows leading dashes
as its own flags; use `ipython src/tuner.py -- board -d h` if you want
ipython specifically.
"""

import argparse
import glob
import sys
from collections import namedtuple

import cv2
import numpy as np
from cv_utils import (  # noqa: F401  (signal_handler registers SIGINT handler on import)
    ParamSpec,
    compose_panels,
    run_tuner,
    signal_handler,
)
from detect_board import (
    DETECT_MAX_SIDE,
    PARAM_DEFAULTS,
    TEAL_LOWER_DEFAULT,
    TEAL_UPPER_DEFAULT,
    board_color_mask,
    canny_edge_mask,
    find_board_quad,
    find_quad_candidates,
    order_corners,
    warp_board,
)
from grid_detector import detect_grid
from hsv_config import load_params, load_range, save_params, save_range
from read_board import draw_tile_overlay, extract_cells, find_parallax_shift
from rotate_board import RED_RECT_DEFAULTS, SPECS, find_red_rectangle, red_rectangle_mask, rotate_board
from tile_detector import PARAM_DEFAULTS as TILE_PARAM_DEFAULTS
from tile_detector import detect_tiles

# ---------------------------------------------------------------------------
# "board" subcommand -- find_board_quad()'s teal range + Canny/quad params.
# Uses its own bespoke multi-image record/combine loop rather than
# cv_utils.run_tuner(), since finding one HSV range that's robust across
# many photos needs recording several images' working values and combining
# them, not just live-previewing a single image.

WINDOW = "HSV Tuner (mask | detection | warp)"

TEAL_TRACKBARS = ("H min", "H max", "S min", "S max", "V min", "V max")

# Each parameter's UI: (json key, trackbar label, trackbar max position,
# scale). Trackbars are integer-only, so non-integer/small-fraction values
# are stored scaled up (e.g. a fraction of 0.00005 as a "x1e-6" slider at
# position 50); `scale` divides the raw trackbar position back down to the
# real value, or is 1 for plain integer parameters. Labels are kept short --
# cv2 clips trackbar labels to the window width.
BoardParamSpec = namedtuple("BoardParamSpec", "key label max_pos scale")
# One entry per individually selectable/resettable slider, built once in
# tune_board() from that run's seed values.
BoardSliderRef = namedtuple("BoardSliderRef", "label seed scale max_pos")
PARAM_SPECS = [
    BoardParamSpec("dark_s_max", "dark S max", 255, 1),
    BoardParamSpec("dark_v_max", "dark V max", 255, 1),
    BoardParamSpec("near_teal_kernel", "bezel dilate k", 61, 1),
    BoardParamSpec("close_kernel", "close kernel", 61, 1),
    BoardParamSpec("close_iterations", "close iters", 5, 1),
    BoardParamSpec("open_kernel", "open kernel", 31, 1),
    BoardParamSpec("canny_blur_kernel", "canny blur k", 31, 1),
    BoardParamSpec("canny_blur_sigma", "canny blur sig", 20, 1),
    BoardParamSpec("canny_low", "canny low", 300, 1),
    BoardParamSpec("canny_high", "canny high", 300, 1),
    BoardParamSpec("canny_dilate_kernel", "canny dilate k", 31, 1),
    BoardParamSpec("quad_side_ratio_max", "quad side x0.1", 100, 10),
    BoardParamSpec("quad_angle_tolerance", "quad angle deg", 90, 1),
    BoardParamSpec("quad_min_area_frac", "quad area x1e-3", 500, 1000),
]

# Each preview panel is resized to this height before being hstacked into
# one composite image -- keeps the single window's width driven by the
# (wide) image content, which incidentally keeps every trackbar label
# above it fully visible without needing a manual minimum window size.
PANEL_HEIGHT = 520


def _resolve_paths(args):
    """`args.pattern`, if given, overrides `args.difficulty` outright;
    otherwise every imgN_<c>.jpg for each difficulty letter in
    args.difficulty, de-duplicated and sorted -- shared by all three
    subcommands so -d/--difficulty behaves identically everywhere."""
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
    return paths


def _nothing(_):
    pass


def _load(path):
    """Return (full, search, scale). `search` is downscaled to
    DETECT_MAX_SIDE -- the exact scale find_board_quad() searches at -- so
    masks/candidates computed on it match production. `full` is the
    original resolution, used for the actual warp once a quad is found:
    detect_board()'s own pipeline searches on a downscaled copy but always
    warps the original image, so this tool's "Warped Board" preview does
    the same instead of quietly capping every downstream step (and,
    eventually, letter recognition) at search resolution."""
    full = cv2.imread(path)
    if full is None:
        return None, None, 1.0
    h, w = full.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    search = cv2.resize(full, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else full
    return full, search, scale


def _set_teal_trackbars(lower, upper):
    for name, val in zip(TEAL_TRACKBARS[0::2], lower):
        cv2.setTrackbarPos(name, WINDOW, int(val))
    for name, val in zip(TEAL_TRACKBARS[1::2], upper):
        cv2.setTrackbarPos(name, WINDOW, int(val))


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


def _build_slider_refs(seed_lower, seed_upper, seed_params):
    """One BoardSliderRef per individual trackbar (teal channels + every
    parameter), so any single one can be selected and reset on its own
    instead of only all-at-once via 'r'."""
    refs = [
        BoardSliderRef("H min", int(seed_lower[0]), 1, 179),
        BoardSliderRef("H max", int(seed_upper[0]), 1, 179),
        BoardSliderRef("S min", int(seed_lower[1]), 1, 255),
        BoardSliderRef("S max", int(seed_upper[1]), 1, 255),
        BoardSliderRef("V min", int(seed_lower[2]), 1, 255),
        BoardSliderRef("V max", int(seed_upper[2]), 1, 255),
    ]
    for spec in PARAM_SPECS:
        seed_val = seed_params.get(spec.key, PARAM_DEFAULTS[spec.key])
        refs.append(BoardSliderRef(spec.label, seed_val, spec.scale, spec.max_pos))
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


def _find_candidates(image, lower, upper, params):
    """Mirror find_board_quad()'s two-stage search (colour mask, then Canny
    fallback if that finds nothing) so both stages -- and their sliders --
    are actually exercised by the live preview. Returns
    (candidates, mask_shown, stage_label)."""
    area = image.shape[0] * image.shape[1]
    mask = board_color_mask(image, lower, upper, **params)
    candidates = find_quad_candidates(mask, area, **params)
    if candidates:
        return candidates, mask, "colour"
    canny_mask = canny_edge_mask(image, **params)
    candidates = find_quad_candidates(canny_mask, area, **params)
    return candidates, canny_mask, "canny (colour found nothing)"


def _warp_to_board(image, corners):
    """detect_board.warp_board(), with a "no board found" placeholder when
    there are no corners to warp onto. Not oriented yet -- rotate_board.py's
    job."""
    if corners is None:
        blank = np.zeros_like(image)
        cv2.putText(blank, "No board found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return blank
    return warp_board(image, corners)


def _compose(mask, detected, warped):
    """Resize every preview to a common height and stack them into one
    composite image (cv_utils.compose_panels()), so all three previews and
    every trackbar live in a single window instead of several separate
    cv2 windows."""
    return compose_panels([mask, detected, warped], height=PANEL_HEIGHT)


def tune_board(args):
    paths = _resolve_paths(args)

    seed_lower, seed_upper = load_range("board_teal", TEAL_LOWER_DEFAULT, TEAL_UPPER_DEFAULT)
    seed_params = load_params("board_params", PARAM_DEFAULTS)
    print(
        f"seed teal range: lower={seed_lower} upper={seed_upper} "
        f"({'from hsv_config.json' if (seed_lower, seed_upper) != (TEAL_LOWER_DEFAULT, TEAL_UPPER_DEFAULT) else 'hardcoded default'})"
    )

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    maxvals = (179, 179, 255, 255, 255, 255)
    defaults = (seed_lower[0], seed_upper[0], seed_lower[1], seed_upper[1], seed_lower[2], seed_upper[2])
    for name, maxval, default in zip(TEAL_TRACKBARS, maxvals, defaults):
        cv2.createTrackbar(name, WINDOW, default, maxval, _nothing)
    _create_param_trackbars(seed_params)

    slider_refs = _build_slider_refs(seed_lower, seed_upper, seed_params)
    selected_idx = 0

    print(
        f"{len(paths)} image(s). n/p: switch image, s: record working range, w: save now, "
        f"[/]: select slider, 0: reset selected, r: reset all, q/Esc: quit and combine."
    )

    recordings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    idx = 0
    full_image, image, search_scale = _load(paths[idx])
    while image is None and idx < len(paths) - 1:
        idx += 1
        full_image, image, search_scale = _load(paths[idx])
    if image is None:
        print("Could not read any of the matched images.")
        sys.exit(1)
    cv2.setWindowTitle(WINDOW, f"{WINDOW} - {paths[idx]}")
    print(f"[{idx + 1}/{len(paths)}] {paths[idx]}")

    lower = upper = None
    while True:
        lower = np.array([cv2.getTrackbarPos(n, WINDOW) for n in ("H min", "S min", "V min")])
        upper = np.array([cv2.getTrackbarPos(n, WINDOW) for n in ("H max", "S max", "V max")])
        params = _read_params()
        lower_t, upper_t = tuple(int(v) for v in lower), tuple(int(v) for v in upper)

        candidates, mask, stage = _find_candidates(image, lower_t, upper_t, params)

        detected = image.copy()
        corners = None
        if candidates:
            best = max(candidates, key=cv2.contourArea)
            corners = order_corners(best)
            cv2.polylines(detected, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            cv2.putText(detected, f"found via {stage}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(detected, "No board found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        full_corners = corners / search_scale if corners is not None else None
        composite = _compose(mask, detected, _warp_to_board(full_image, full_corners))
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
            recordings[paths[idx]] = (lower.copy(), upper.copy())
            note = "" if candidates else "  (warning: no board found at these values)"
            print(f"  saved {paths[idx]}: lower={lower_t} upper={upper_t}{note}")
        elif key == ord("w"):
            save_range("board_teal", lower, upper)
            save_params("board_params", params)
            print(f"  wrote hsv_config.json: teal lower={lower_t} upper={upper_t}, {len(params)} params")
        elif key == ord("r"):
            _set_teal_trackbars(seed_lower, seed_upper)
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
            new_image = None
            for _ in range(len(paths)):
                idx = (idx + step) % len(paths)
                new_full, new_image, new_scale = _load(paths[idx])
                if new_image is not None:
                    break
            if new_image is not None:
                full_image, image, search_scale = new_full, new_image, new_scale
                cv2.setWindowTitle(WINDOW, f"{WINDOW} - {paths[idx]}")
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

    # Reuse `params` as last read inside the loop (before the windows were
    # destroyed) -- calling _read_params() again here would call
    # cv2.getTrackbarPos() on already-destroyed windows and silently read
    # back -1 for every parameter.
    print("\nValidating combined range against every recorded image:")
    ok = 0
    for path in recordings:
        _, img, _ = _load(path)
        if img is None:
            print(f"  SKIP could not reload {path}")
            continue
        img_mask = board_color_mask(
            img, tuple(int(v) for v in combined_lower), tuple(int(v) for v in combined_upper), **params
        )
        found = bool(find_quad_candidates(img_mask, img_mask.shape[0] * img_mask.shape[1], **params))
        ok += found
        print(f"  {'OK  ' if found else 'FAIL'} {path}")
    print(f"\n{ok}/{len(recordings)} recorded images still detect the board with the combined range.")

    save_range("board_teal", combined_lower, combined_upper)
    save_params("board_params", params)
    print("\nSaved to hsv_config.json -- detect_board.py's functions will use it automatically.")


# ---------------------------------------------------------------------------
# "red_rectangle" subcommand -- red_rectangle_mask()/find_red_rectangle()'s
# params (the orientation marker on the board). Small enough to just lean
# on cv_utils.run_tuner()'s generic live-preview harness instead of a
# bespoke loop like tune_board() needs.


def tune_red_rectangle(args):
    paths = _resolve_paths(args)
    print(paths)

    for path in paths:
        image = cv2.imread(path)
        corners = find_board_quad(image)
        if corners is None:
            print(f"No board found in {path}")
            continue

        board = warp_board(image, corners)

        def render(params):
            preview = board.copy()
            rect_corners = find_red_rectangle(board, **params)
            if rect_corners is not None:
                cv2.polylines(preview, [rect_corners], True, (0, 255, 0), 5)
            return [preview, red_rectangle_mask(board, **params)]

        run_tuner(SPECS, render, RED_RECT_DEFAULTS, window="Red Rectangle Tuner", config_name="red_rectangle_params")


# ---------------------------------------------------------------------------
# "tile_detector" subcommand -- detect_tiles()'s params (per-class colour
# z-score, tile-colour-model seeding, accept/reject bounds, glyph floors).
# Same shape as tune_red_rectangle(): small enough to lean on
# cv_utils.run_tuner()'s generic live-preview harness.

TILE_SPECS = [
    ParamSpec(
        "color_z_threshold", "color z x10", 100, 10, "per-class robust z-score above which a cell is a colour outlier"
    ),
    ParamSpec("seed_glyph_high", "seed glyph hi x100", 100, 100, "glyph bar for a confident pass-2 tile-colour seed"),
    ParamSpec("seed_glyph_low", "seed glyph lo x100", 100, 100, "relaxed glyph bar when too few confident seeds exist"),
    ParamSpec("seed_glyph_fallback", "seed glyph fb x100", 100, 100, "glyph-only bar when colour can't seed the model at all"),
    ParamSpec("seed_max", "seed max", 20, 1, "max candidates used to build the photo's tile-colour model"),
    ParamSpec("seed_min", "seed min", 10, 1, "below this many seeds, fall back to the ivory prior"),
    ParamSpec("same_tile_dist", "same-tile d x10", 50, 10, "distance-to-tile-colour accepted outright"),
    ParamSpec("d_accept", "d accept x10", 80, 10, "permissive distance-accept bound"),
    ParamSpec("z_accept", "z accept x10", 80, 10, "permissive class-z accept bound"),
    ParamSpec("d_accept_strict", "d accept strict x10", 80, 10, "strict distance-accept bound"),
    ParamSpec("z_accept_strict", "z accept strict x10", 80, 10, "strict class-z accept bound"),
    ParamSpec("definitely_empty_d", "empty d x10", 100, 10, "distance above which a cell is confidently empty"),
    ParamSpec("definitely_empty_z", "empty z x10", 50, 10, "z below which a cell is confidently empty"),
    ParamSpec("ambiguous_glyph_min", "ambig glyph x100", 100, 100, "glyph bar arbitrating the ambiguous band"),
    ParamSpec("strict_glyph_min", "strict glyph x100", 100, 100, "strict-mode glyph floor"),
    ParamSpec("permissive_glyph_min", "permissive glyph x100", 100, 100, "glyph floor for the d/z-accept branch (glare guard)"),
]


def tune_tile_detector(args):
    paths = _resolve_paths(args)
    print(paths)

    for path in paths:
        image = cv2.imread(path)
        corners = find_board_quad(image)
        if corners is None:
            print(f"No board found in {path}")
            continue

        rotated = rotate_board(warp_board(image, corners))
        grid = detect_grid(rotated)
        if grid is None:
            print(f"No grid found in {path}")
            continue
        shift = find_parallax_shift(rotated, grid.mesh)
        cells = extract_cells(rotated, grid.mesh, global_shift=shift)

        def render(params):
            verdicts = detect_tiles(cells, **params)
            overlay = draw_tile_overlay(rotated, grid.mesh, verdicts)
            cv2.putText(
                overlay,
                f"{sum(v.is_tile for v in verdicts)} tiles",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
            )
            return overlay

        run_tuner(
            TILE_SPECS, render, TILE_PARAM_DEFAULTS, window="Tile Detector Tuner", config_name="tile_detector_params"
        )


def main():
    parser = argparse.ArgumentParser(description="Interactive tuners for board_reader's detection pipeline")
    sub = parser.add_subparsers(dest="target", required=True)

    difficulty_help = (
        "difficulty suffixes to include: any of 'e' (easy), 'm' (medium), 'h' (hard), e.g. -d emh (default: e)"
    )

    board_p = sub.add_parser("board", help="tune find_board_quad()'s teal range + Canny/quad params")
    board_p.add_argument("pattern", nargs="?", default=None, help="glob pattern for images (overrides --difficulty)")
    board_p.add_argument("-d", "--difficulty", default="e", help=difficulty_help)

    rect_p = sub.add_parser("red_rectangle", help="tune red_rectangle_mask()/find_red_rectangle()'s params")
    rect_p.add_argument("pattern", nargs="?", default=None, help="glob pattern for images (overrides --difficulty)")
    rect_p.add_argument("-d", "--difficulty", default="e", help=difficulty_help)

    tile_p = sub.add_parser("tile_detector", help="tune detect_tiles()'s params")
    tile_p.add_argument("pattern", nargs="?", default=None, help="glob pattern for images (overrides --difficulty)")
    tile_p.add_argument("-d", "--difficulty", default="e", help=difficulty_help)

    args = parser.parse_args()
    if args.target == "board":
        tune_board(args)
    elif args.target == "red_rectangle":
        tune_red_rectangle(args)
    else:
        tune_tile_detector(args)


if __name__ == "__main__":
    main()
