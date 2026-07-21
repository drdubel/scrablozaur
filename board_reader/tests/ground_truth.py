"""Ground truth for the photos in test/in/, loaded from test/out/board<i>.txt.

The .txt files are authoritative: one row per line, space-separated tokens,
'-' for an empty square.
"""

import glob
import os
import re

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test")
IN_DIR = os.path.join(TEST_DIR, "in")
OUT_DIR = os.path.join(TEST_DIR, "out")


def _load(idx: int) -> dict[tuple[int, int], str]:
    path = os.path.join(OUT_DIR, f"board{idx}.txt")
    tiles: dict[tuple[int, int], str] = {}
    with open(path, encoding="utf-8") as f:
        for r, line in enumerate(f):
            for c, tok in enumerate(line.split()):
                if tok != "-":
                    tiles[(r, c)] = tok
    return tiles


def _image_path(idx: int) -> str | None:
    matches = glob.glob(os.path.join(IN_DIR, f"img{idx}_*.jpg"))
    return matches[0] if matches else None


def _difficulty(path: str) -> str:
    """'e'/'m'/'h' suffix from an imgN_<difficulty>.jpg filename."""
    match = re.search(r"_([emh])\.jpg$", path)
    return match.group(1) if match else "e"


def _available() -> list[int]:
    ids = []
    for f in os.listdir(OUT_DIR):
        m = re.fullmatch(r"board(\d+)\.txt", f)
        if m and _image_path(int(m.group(1))) is not None:
            ids.append(int(m.group(1)))
    return sorted(ids)


GROUND_TRUTH = {i: _load(i) for i in _available()}
IMAGE_PATHS = {i: _image_path(i) for i in GROUND_TRUTH}
DIFFICULTY = {i: _difficulty(path) for i, path in IMAGE_PATHS.items() if path is not None}
