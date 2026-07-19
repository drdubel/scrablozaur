"""Small, generic OpenCV/UI helpers shared across board_reader's scripts --
nothing board-specific lives here, just the display/plumbing bits every
detection script tends to need.

Importing this module registers a Ctrl+C handler (see signal_handler())
so an interrupted script closes its cv2 windows instead of leaving them
stuck open.
"""

import signal
import sys
from collections import namedtuple

import cv2
import numpy as np
from hsv_config import load_params, save_params

LABEL_BAR = 34  # px reserved above a panel for its name, in show_images()


def signal_handler(sig, frame):
    """Close every open cv2 window and exit -- registered below so Ctrl+C
    during a blocking show_image()/waitKey() loop doesn't leave a stuck
    window behind."""
    print("\nClosing...")
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_image(title, image):
    """Display a single image in a resizable window until any key is
    pressed."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    while True:
        key = cv2.waitKey(100)  # short wait so Ctrl+C stays responsive
        if key != -1:
            break
    cv2.destroyAllWindows()


def resize_to_height(image, height):
    """Resize `image` to `height` pixels tall, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = height / h
    return cv2.resize(image, (max(1, int(w * scale)), height))


def compose_panels(images, labels=None, height=520):
    """Resize each of `images` to a common height and stack them side by
    side into one image -- the building block behind show_images(). Same
    idea as tuner.py's own panel-composing (which predates this
    shared version): several previews sharing one window/one set of
    trackbars beats juggling several separate cv2 windows.

    Grayscale (2D) images are converted to BGR first so they stack
    cleanly with color ones. If `labels` is given (one string per image,
    or None for no label on that one), a name banner is stamped above
    each labeled panel.
    """
    if labels is None:
        labels = [None] * len(images)
    panels = []
    for image, label in zip(images, labels):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        panel = resize_to_height(image, height)
        if label is not None:
            bar = np.zeros((LABEL_BAR, panel.shape[1], 3), np.uint8)
            cv2.putText(bar, label, (8, LABEL_BAR - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            panel = np.vstack([bar, panel])
        panels.append(panel)
    return np.hstack(panels)


def show_images(images, labels=None, title="Preview", height=520):
    """show_image(), but for several images side by side in one window --
    see compose_panels() for the `labels` convention. Handy for a quick
    "here's the mask, here's the detection, here's the result" look
    without opening a separate window per image."""
    show_image(title, compose_panels(images, labels, height))


# ---------------------------------------------------------------------------
# Generic parameter-tuner window -- the trackbar/slider machinery tuner.py
# and letter_tuner.py each built for themselves, factored out so a new tuner
# is "describe the sliders + write a render function", not several hundred
# lines of trackbar bookkeeping.

# One entry per slider: `key` is what shows up in the params dict passed to
# your render function (and the name saved to hsv_config.json); `label` is
# the on-screen trackbar name (cv2 clips long labels, keep them short);
# `max_pos` is the trackbar's integer range (always starts at 0); trackbars
# are integer-only, so a fractional value is stored scaled up (e.g. a real
# range of 0-0.5 as a "x1000" slider with max_pos=500, scale=1000) --
# `scale` divides the raw trackbar position back down to the real value,
# or use 1 for a plain integer parameter. `desc` is shown when the slider
# is selected (see '[' / ']' below); pass "" if you don't want that.
ParamSpec = namedtuple("ParamSpec", "key label max_pos scale desc")
SliderRef = namedtuple("SliderRef", "label seed scale max_pos desc")


def create_param_trackbars(window, specs, seed_params, defaults):
    """Create one trackbar per spec in `window` (already created via
    cv2.namedWindow), seeded from `seed_params` (falling back to
    `defaults` for any missing key)."""
    for spec in specs:
        pos = int(round(seed_params.get(spec.key, defaults[spec.key]) * spec.scale))
        pos = max(0, min(spec.max_pos, pos))
        cv2.createTrackbar(spec.label, window, pos, spec.max_pos, lambda _: None)


def read_params(window, specs):
    """Current value of every slider in `window`, as {key: value}."""
    return {spec.key: cv2.getTrackbarPos(spec.label, window) / spec.scale for spec in specs}


def set_param_trackbars(window, specs, params, defaults):
    """Move every trackbar in `window` to match `params` (falling back to
    `defaults` for any missing key) -- e.g. for a 'reset all' key."""
    for spec in specs:
        pos = int(round(params.get(spec.key, defaults[spec.key]) * spec.scale))
        cv2.setTrackbarPos(spec.label, window, max(0, min(spec.max_pos, pos)))


def build_slider_refs(specs, seed_params, defaults):
    """One SliderRef per spec, remembering its seed value -- for selecting
    ('[' / ']') and resetting a single slider ('0') independently of the
    others."""
    return [
        SliderRef(spec.label, seed_params.get(spec.key, defaults[spec.key]), spec.scale, spec.max_pos, spec.desc)
        for spec in specs
    ]


def reset_single_slider(window, ref):
    """Move just this one trackbar back to its seed value."""
    pos = int(round(ref.seed * ref.scale))
    cv2.setTrackbarPos(ref.label, window, max(0, min(ref.max_pos, pos)))


def selection_status(window, ref):
    """One-line status string for the currently selected slider -- current
    value, seed value, and its description."""
    pos = cv2.getTrackbarPos(ref.label, window)
    current = pos / ref.scale
    changed = abs(current - ref.seed) > 1e-9
    suffix = f" -- {ref.desc}" if ref.desc else ""
    return f"selected: {ref.label}  current={current:g}  seed={ref.seed:g}{'  (changed)' if changed else ''}{suffix}"


def run_tuner(specs, render, defaults, *, window="Tuner", config_name=None, on_key=None, help_text=""):
    """Run a generic interactive trackbar-tuner loop, modeled on tuner.py
    /letter_tuner.py's window (both now thin wrappers around this).

    specs: list of ParamSpec -- the sliders to show.
    render(params) -> image, or list of images: called every frame with the
        sliders' current values (as {key: value}); return what to display,
        via show_image()/compose_panels() (a single image, or several to
        show side by side in the same window -- e.g. a mask next to the
        detection result).
    defaults: {key: value} fallback for any slider missing from the saved
        preset (or from `config_name=None`, meaning there's no preset at
        all -- sliders always start from `defaults` in that case).
    config_name: hsv_config.json section to load the seed from and save to
        ('w'). None means no persistence -- sliders always seed from
        `defaults`, and 'w' just prints the current values instead of
        writing them anywhere, for a tuner you don't want to (or can't
        yet) wire up to a real preset.
    on_key: optional {ord('x'): callback} for extra keys beyond the
        standard ones below. `callback(params)` runs when that key is
        pressed; anything it needs to remember between frames (e.g. which
        image is currently loaded) should live in a variable it closes
        over, not in a return value -- run_tuner() doesn't inspect it.
    help_text: extra line appended to the standard keys list, printed once
        at startup (e.g. to document your own on_key bindings).

    Standard keys:
        w        save current sliders to hsv_config.json (if config_name given)
        [ / ]    select the previous / next individual slider
        0        reset only the selected slider back to its seed
        r        reset every slider back to its seed
        q / Esc  quit
    """
    seed_params = load_params(config_name, defaults) if config_name else dict(defaults)
    print(
        f"seed params: {seed_params} ({'from hsv_config.json' if config_name else 'no config_name -- using defaults'})"
    )
    for spec in specs:
        if spec.desc:
            print(f"  {spec.label}: {spec.desc}")

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    create_param_trackbars(window, specs, seed_params, defaults)
    slider_refs = build_slider_refs(specs, seed_params, defaults)
    selected_idx = 0

    print(
        "w: save, [/]: select slider, 0: reset selected, r: reset all, q/Esc: quit."
        + (f" {help_text}" if help_text else "")
    )

    params = seed_params
    last_params = None
    base_composite = None
    while True:
        params = read_params(window, specs)
        if base_composite is None or params != last_params:
            # Only re-run render() -- the actual image processing -- when a
            # slider moved (or on the first frame). Without this, the loop
            # below redoes the full detection pipeline on every waitKey()
            # tick (~33x/second) even while the tuner just sits idle.
            result = render(params)
            images = result if isinstance(result, list) else [result]
            base_composite = compose_panels(images) if len(images) > 1 else images[0]
            last_params = dict(params)

        composite = base_composite.copy()
        cv2.putText(
            composite,
            selection_status(window, slider_refs[selected_idx]),
            (10, composite.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1,
        )
        cv2.imshow(window, composite)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("w"):
            if config_name:
                save_params(config_name, params)
                print(f"  wrote hsv_config.json {config_name}: {params}")
            else:
                print(f"  no config_name given, not saved: {params}")
        elif key == ord("r"):
            set_param_trackbars(window, specs, seed_params, defaults)
            print("  reset all sliders to seed values")
        elif key == ord("["):
            selected_idx = (selected_idx - 1) % len(slider_refs)
        elif key == ord("]"):
            selected_idx = (selected_idx + 1) % len(slider_refs)
        elif key == ord("0"):
            ref = slider_refs[selected_idx]
            reset_single_slider(window, ref)
            print(f"  reset {ref.label} to seed {ref.seed:g}")
        elif on_key and key in on_key:
            on_key[key](params)
            base_composite = None  # callback may have changed state outside params -- force a redraw

    cv2.destroyAllWindows()
    return params
