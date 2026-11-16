"""Export what the classical pipeline sees in every test photo -- the teacher
for the learned scanner.

For each photo in test/in/ that has ground truth, runs the board_reader stages
and writes to --out (default test/derived/scan_dataset/):

    board<i>.jpg   the playing grid straightened by ONE homography from its four
                   corners -- the same rectification the app performs on a camera
                   frame -- in true board orientation: SIZE x SIZE pixels, cells
                   CELL px square, a MARGIN of context around the grid
    board<i>.json  {
                     "photo": "imgN_e.jpg", "difficulty": "e", "layout": n,
                     "photo_size": [w, h],
                     "grid_corners": [[x, y], ...],  # TL, TR, BR, BL of the 15x15 grid, photo px
                     "mesh": 16 x 16 x [x, y],       # every grid-line intersection, photo px
                     "rotation_k": k,                # letter-based correction read_letters applied
                     "registration": 0.97,           # see registration(); boards under 0.7 are not exported
                     "truth": [15 strings],          # '-' empty, upper-case letters
                     "tiles": {"r,c": [is_tile, confidence]},             # detector, every cell
                     "readings": {"r,c": [letter, confidence, [[alt, p]]]}  # classifier, tiles only
                   }
    index.json     every exported board with its difficulty and layout

`layout` numbers boards that show the same tile arrangement: the test photos
are a few layouts shot many times, so a fair held-out split is by layout.

Run from board_reader/:  python scripts/export_scan_dataset.py
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

import read_letters  # noqa: E402
from detect_board import find_board_quad, warp_board  # noqa: E402
from grid_detector import detect_grid  # noqa: E402
from ground_truth import DIFFICULTY, GROUND_TRUTH, IMAGE_PATHS  # noqa: E402
from read_board import extract_cells, find_parallax_shift  # noqa: E402
from rotate_board import get_board_orientation, rotate_board  # noqa: E402
from tile_detector import detect_tiles  # noqa: E402

GRID = 15
CELL = 64
MARGIN = 32
SIZE = GRID * CELL + 2 * MARGIN

# classify_tiles() settles the board's orientation internally and does not
# return it; record what _resolve_rotation() decided instead of re-deriving it.
_resolve_rotation = read_letters._resolve_rotation
_last_rotation = {"k": 0}


def _recording_resolve_rotation(*args, **kwargs):
    k = _resolve_rotation(*args, **kwargs)
    _last_rotation["k"] = k
    return k


read_letters._resolve_rotation = _recording_resolve_rotation


def _unrotate(points, angle, n):
    """Points in rotate_board()'s output back into the unrotated warp. `angle`
    is get_board_orientation()'s verdict; see rotate_board() for the mapping."""
    x, y = points[..., 0], points[..., 1]
    if angle == 90:  # rotated counter-clockwise
        return np.stack([n - 1 - y, x], axis=-1)
    if angle == 180:
        return np.stack([n - 1 - x, n - 1 - y], axis=-1)
    if angle == 270:  # rotated clockwise
        return np.stack([y, n - 1 - x], axis=-1)
    return points


def _to_photo(points, corners, photo_w, photo_h, angle):
    """Points in the rotated, squared warp back to photo pixels: undo the
    rotation, the squaring resize (height W -> H), then the perspective."""
    pts = _unrotate(points.astype(np.float64), angle, photo_w)
    pts = pts * np.array([1.0, photo_h / photo_w])
    canvas = np.array([[0, 0], [photo_w, 0], [photo_w, photo_h], [0, photo_h]], dtype=np.float32)
    inverse = cv2.getPerspectiveTransform(canvas, corners.astype(np.float32))
    flat = pts.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(flat, inverse).reshape(pts.shape)


def _rotate_index(r, c, k):
    for _ in range(k % 4):
        r, c = c, GRID - 1 - r
    return r, c


def quad_area(corners) -> float:
    x, y = np.asarray(corners, dtype=np.float64).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def registration(straight, truth) -> float:
    """How well the straightened grid matches the labels, from colour alone:
    the chance that a labelled tile's square is less saturated than a labelled
    empty one (tiles are cream, the board teal and red/blue). About 0.9-1.0
    when they line up, whatever the lighting; near 0.5 when the grid is
    misplaced or upside down."""
    sat = cv2.cvtColor(straight, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    inner = CELL // 4
    med = np.array(
        [
            [
                np.median(
                    sat[
                        MARGIN + r * CELL + inner : MARGIN + (r + 1) * CELL - inner,
                        MARGIN + c * CELL + inner : MARGIN + (c + 1) * CELL - inner,
                    ]
                )
                for c in range(GRID)
            ]
            for r in range(GRID)
        ]
    )
    is_tile = np.array([[ch != "-" for ch in row] for row in truth])
    tiles, empty = med[is_tile], med[~is_tile]
    if not len(tiles) or not len(empty):
        return 1.0
    return float((tiles[:, None] < empty[None, :]).mean() + 0.5 * (tiles[:, None] == empty[None, :]).mean())


# Below this, a board's labels do not belong to the squares they name. Every
# exported photo that fell below it was visibly wrong: corners collapsed to a
# point, or the grid shifted or upside down; the lowest correct one scored 0.80.
MIN_REGISTRATION = 0.7


def straighten(photo, grid_corners):
    dst = np.array(
        [[MARGIN, MARGIN], [SIZE - MARGIN, MARGIN], [SIZE - MARGIN, SIZE - MARGIN], [MARGIN, SIZE - MARGIN]],
        dtype=np.float32,
    )
    h = cv2.getPerspectiveTransform(np.asarray(grid_corners, dtype=np.float32), dst)
    return cv2.warpPerspective(photo, h, (SIZE, SIZE), flags=cv2.INTER_AREA)


def export_one(idx, layout, out_dir):
    path = IMAGE_PATHS[idx]
    photo = cv2.imread(path)
    photo_h, photo_w = photo.shape[:2]
    corners = find_board_quad(photo)
    if corners is None:
        return f"board{idx}: no board found"
    board = warp_board(photo, corners)
    angle = get_board_orientation(board)
    rotated = rotate_board(board)
    grid = detect_grid(rotated)
    if grid is None:
        return f"board{idx}: no grid found"
    shift = find_parallax_shift(rotated, grid.mesh)
    verdicts = detect_tiles(extract_cells(rotated, grid.mesh, global_shift=shift))
    _last_rotation["k"] = 0
    readings = read_letters.classify_tiles(rotated, grid.mesh, verdicts, global_shift=shift)
    k = _last_rotation["k"]

    # Into photo pixels, then into the board's true orientation: rotating the
    # board k quarter-turns clockwise moves intersection (i, j) to (j, 15 - i).
    mesh = _to_photo(np.asarray(grid.mesh, dtype=np.float64), corners, photo_w, photo_h, angle)
    mesh = np.rot90(mesh, -k, axes=(0, 1))
    grid_corners = [mesh[0, 0], mesh[0, GRID], mesh[GRID, GRID], mesh[GRID, 0]]
    if quad_area(grid_corners) < 0.05 * photo_w * photo_h:
        return f"board{idx}: the grid collapsed ({quad_area(grid_corners):.0f} px²)"

    truth = ["".join(GROUND_TRUTH[idx].get((r, c), "-") for c in range(GRID)) for r in range(GRID)]
    straight = straighten(photo, grid_corners)
    match = registration(straight, truth)
    if match < MIN_REGISTRATION:
        return f"board{idx}: the grid does not match the labels (registration {match:.2f})"
    tiles = {}
    for v in verdicts:
        r, c = _rotate_index(v.row, v.col, k)
        tiles[f"{r},{c}"] = [bool(v.is_tile), round(float(v.confidence), 4)]
    out_readings = {
        f"{r},{c}": [letter, round(float(conf), 4), [[a, round(float(p), 4)] for a, p in alts]]
        for (r, c), (letter, conf, alts) in readings.items()
    }
    record = {
        "photo": os.path.basename(path),
        "difficulty": DIFFICULTY[idx],
        "layout": layout,
        "photo_size": [photo_w, photo_h],
        "grid_corners": [[round(float(x), 2), round(float(y), 2)] for x, y in grid_corners],
        "mesh": [[[round(float(x), 2), round(float(y), 2)] for x, y in row] for row in mesh],
        "rotation_k": int(k),
        "registration": round(match, 3),
        "truth": truth,
        "tiles": tiles,
        "readings": out_readings,
    }
    with open(os.path.join(out_dir, f"board{idx}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    cv2.imwrite(os.path.join(out_dir, f"board{idx}.jpg"), straight, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = os.path.join(HERE, "..", "test", "derived", "scan_dataset")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--only", type=int, nargs="*", help="export just these board numbers")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Same tile arrangement -> same layout number, in board order.
    layouts, index = {}, []
    for idx in sorted(GROUND_TRUTH):
        key = tuple(sorted(GROUND_TRUTH[idx].items()))
        layout = layouts.setdefault(key, len(layouts))
        if args.only and idx not in args.only:
            continue
        start = time.time()
        problem = export_one(idx, layout, args.out)
        print(f"board{idx}: {'FAILED ' + problem if problem else 'ok'} ({time.time() - start:.1f} s)", flush=True)
        if problem is not None:
            # Do not leave an earlier run's export of this board behind.
            for ext in ("json", "jpg"):
                stale = os.path.join(args.out, f"board{idx}.{ext}")
                if os.path.exists(stale):
                    os.remove(stale)
        else:
            index.append({"board": idx, "difficulty": DIFFICULTY[idx], "layout": layout})
    if not args.only:
        with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=1)
    print(f"{len(index)} boards exported to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
