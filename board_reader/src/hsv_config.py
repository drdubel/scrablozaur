"""Shared load/save for detection parameters tuned interactively with
hsv_tuner.py.

Values are stored by name in a small JSON file next to this module, so
detect_board.py can pick up a tuned preset automatically instead of
someone hand-copying printed numbers into its source.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "hsv_config.json")


def load_range(name, default_lower, default_upper):
    """Return (lower, upper) tuples for `name`, falling back to the given
    defaults if the config file or entry doesn't exist yet."""
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        entry = data[name]
        return tuple(entry["lower"]), tuple(entry["upper"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return default_lower, default_upper


def save_range(name, lower, upper):
    data = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data[name] = {
        "lower": [int(v) for v in lower],
        "upper": [int(v) for v in upper],
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_params(name, defaults):
    """Return a dict for `name`, merging any saved values over `defaults`.

    Unlike load_range, a missing or extra key doesn't invalidate the whole
    entry -- so adding a new tunable parameter later doesn't require every
    existing hsv_config.json to be regenerated.
    """
    merged = dict(defaults)
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        merged.update(data[name])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        pass
    return merged


def save_params(name, values):
    data = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data[name] = dict(values)
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
