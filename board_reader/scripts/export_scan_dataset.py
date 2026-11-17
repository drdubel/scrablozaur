"""Export what the classical pipeline sees in every test photo -- the teacher
for the learned scanner.

For each photo in test/in/ that has ground truth, runs the board_reader stages
and writes to --out (default test/derived/scan_dataset/):

    board<i>.jpg   the tiles' top faces straightened, in true board
                   orientation: SIZE x SIZE pixels, cells CELL px square, a
                   MARGIN of context around the grid -- so that every tile's
                   letter sits in the square its label names. Tiles stand
                   TILE_HEIGHT above the board, so in a photo taken at an
                   angle their tops are displaced from the printed grid:
                   by the camera's pose (from the grid's corners and the
                   photo's focal length) and, where the pipeline's mesh is
                   itself off, by a residual measured against the labels
    board<i>.json  {
                     "photo": "imgN_e.jpg", "difficulty": "e", "layout": n,
                     "photo_size": [w, h],
                     "grid_corners": [[x, y], ...],  # TL, TR, BR, BL of the 15x15 grid, photo px
                     "mesh": 16 x 16 x [x, y],       # every grid-line intersection, photo px
                     "rotation_k": k,                # letter-based correction read_letters applied
                     "registration": 0.97,           # see registration(); boards under 0.7 are not exported
                     "bend": 0.12,                   # see bend(): squares off a four-corner homography
                     "focal_px": 2607,               # from EXIF, the 35 mm equivalent
                     "parallax": [dx, dy],           # tile tops' displacement at the centre, squares
                     "residual": [dx, dy],           # the further offset the labels needed, squares
                     "truth": [15 strings],          # '-' empty, upper-case letters
                     "tiles": {"r,c": [is_tile, confidence]},             # detector, every cell
                     "readings": {"r,c": [letter, confidence, [[alt, p]]]}  # classifier, tiles only
                   }
    index.json     every exported board with its difficulty and layout

`layout` numbers boards that show the same tile arrangement: the test photos
are a few layouts shot many times, so a fair held-out split is by layout.

Scans the app saved for training ("Zapisz do treningu OCR", brought in by
import_app_scans.py into test/app/) are exported too, numbered from
APP_BASE, difficulty "c" (camera), "source": "app". Their grid comes from the
corners the app's grid finder found, not from the classical pipeline -- they
are camera frames, a quarter of a photo's resolution, at any rotation -- and
their truth is the board the player confirmed. They carry no "tiles" or
"readings": the classical pipeline never saw them. The scans of one game,
each holding the last one's tiles, share a layout.

Run from board_reader/:  python scripts/export_scan_dataset.py
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

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
APP_DIR = os.path.join(HERE, "..", "test", "app")
# App scans are numbered from here, clear of the photos' board numbers.
APP_BASE = 10000
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


# A tile's top face above the board, in squares: fitted to the test photos
# (0.253; a tile is about 4.9 mm thick on a 19.5 mm square).
TILE_HEIGHT = 0.25


def focal_px(path, photo_size) -> float:
    """The photo's focal length in pixels, from its EXIF 35 mm equivalent
    (a 36 mm wide frame); 26 mm, a typical phone's, if it has none."""
    try:
        exif = Image.open(path).getexif().get_ifd(0x8769)
        f35 = float(exif.get(0xA405) or 26)  # FocalLengthIn35mmFilm
    except (OSError, ValueError):
        f35 = 26.0
    return f35 / 36 * max(photo_size)


def tile_tops(mesh, photo_size, f_px):
    """Per mesh point, where the top of a tile on that point appears,
    in grid squares relative to the point itself: the camera's pose comes
    from the homography of the grid's corners and the focal length, and a
    plane TILE_HEIGHT above the board is projected with it. (16, 16, 2)."""
    m = np.asarray(mesh, np.float64)
    w, h = photo_size
    K = np.array([[f_px, 0, w / 2], [0, f_px, h / 2], [0, 0, 1]])
    grid = np.array([[0, 0], [GRID, 0], [GRID, GRID], [0, GRID]], np.float64)
    H, _ = cv2.findHomography(grid, np.array([m[0, 0], m[0, GRID], m[GRID, GRID], m[GRID, 0]]))
    B = np.linalg.inv(K) @ H
    scale = 2 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    r1, r2, t = scale * B[:, 0], scale * B[:, 1], scale * B[:, 2]
    if t[2] < 0:  # the board must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    if (np.stack([r1, r2, r3]) @ -t)[2] < 0:  # and tiles rise toward the camera
        r3 = -r3
    tops = K @ np.stack([r1, r2, t + TILE_HEIGHT * r3], 1)
    xy = np.stack(np.meshgrid(np.arange(GRID + 1), np.arange(GRID + 1)), -1).astype(np.float64)
    seen = np.einsum("ij,...j->...i", np.linalg.inv(H) @ tops, np.concatenate([xy, np.ones_like(xy[..., :1])], -1))
    return seen[..., :2] / seen[..., 2:] - xy


def extended(mesh):
    """The mesh with one more ring of points, extrapolated: (18, 18, 2),
    point (i, j) at grid coordinates (i - 1, j - 1)."""
    m = np.asarray(mesh, dtype=np.float64)
    m = np.concatenate([2 * m[:1] - m[1:2], m, 2 * m[-1:] - m[-2:-1]], 0)
    return np.concatenate([2 * m[:, :1] - m[:, 1:2], m, 2 * m[:, -1:] - m[:, -2:-1]], 1)


def at(ext, gx, gy):
    """Photo points at grid coordinates (gx, gy), interpolating the extended mesh."""
    gx = np.clip(np.asarray(gx, np.float64) + 1, 0, GRID + 2 - 1e-6)
    gy = np.clip(np.asarray(gy, np.float64) + 1, 0, GRID + 2 - 1e-6)
    i, j = np.minimum(gy.astype(int), GRID + 1), np.minimum(gx.astype(int), GRID + 1)
    v, u = (gy - i)[..., None], (gx - j)[..., None]
    return (
        (1 - u) * (1 - v) * ext[i, j]
        + u * (1 - v) * ext[i, j + 1]
        + (1 - u) * v * ext[i + 1, j]
        + u * v * ext[i + 1, j + 1]
    )


def offset(img, truth, reach=0.9):
    """The offset, in squares and at most `reach` each way, at which square
    windows best separate the labelled tiles from the labelled empty squares
    by their share of tile-face pixels (bright, unsaturated): where the
    labels' tiles are."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    face = ((sat < np.percentile(sat, 30) + 25) & (val > np.percentile(val, 50))).astype(np.float32)
    integral = cv2.integral(face)
    is_tile = np.array([[ch != "-" for ch in row] for row in truth])
    if is_tile.all() or not is_tile.any():
        return 0.0, 0.0
    best = (-np.inf, 0.0, 0.0)
    steps = np.arange(-reach, reach + 0.01, 0.05)
    for sy in steps:
        for sx in steps:
            y0 = np.clip(np.round(MARGIN + (np.arange(GRID) + sy) * CELL).astype(int), 0, SIZE)
            x0 = np.clip(np.round(MARGIN + (np.arange(GRID) + sx) * CELL).astype(int), 0, SIZE)
            y1, x1 = np.clip(y0 + CELL, 0, SIZE), np.clip(x0 + CELL, 0, SIZE)
            Y0, X0, Y1, X1 = y0[:, None], x0[None, :], y1[:, None], x1[None, :]
            share = (integral[Y1, X1] - integral[Y0, X1] - integral[Y1, X0] + integral[Y0, X0]) / CELL**2
            score = share[is_tile].mean() - share[~is_tile].mean()
            if score > best[0]:
                best = (score, float(sx), float(sy))
    return round(best[1], 2), round(best[2], 2)


def straighten(photo, mesh, shift=None, smooth=0.8):
    """The grid straightened through its mesh, each output pixel found in
    the photo by interpolating the mesh at its grid coordinates -- moved by
    `shift`, per mesh point in squares (16, 16, 2), if given. `smooth` is the
    blur, in photo pixels, taken off first."""
    ext = extended(mesh)
    g = (np.arange(SIZE) + 0.5 - MARGIN) / CELL
    gx, gy = np.meshgrid(g, g)
    if shift is not None:
        s = extended(shift)  # extrapolated the same way, as a smooth field
        moved = at(s, gx, gy)
        gx, gy = gx + moved[..., 0], gy + moved[..., 1]
    pts = at(ext, gx, gy).astype(np.float32)
    # A photo's squares are about twice the export's: smooth first, or
    # sampling them would alias. A camera frame's are about its size.
    src = cv2.GaussianBlur(photo, (0, 0), smooth) if smooth else photo
    return cv2.remap(src, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR)


def bend(mesh) -> float:
    """The largest distance, in squares, between a mesh point and where the
    homography from the grid's four corners puts it: about 0 for a flat
    board, near 1 for one bowed along its fold."""
    m = np.asarray(mesh, dtype=np.float32)
    corners = np.array([m[0, 0], m[0, GRID], m[GRID, GRID], m[GRID, 0]], dtype=np.float32)
    unit = np.array([[0, 0], [GRID, 0], [GRID, GRID], [0, GRID]], dtype=np.float32)
    to_grid = cv2.getPerspectiveTransform(corners, unit)
    ideal = np.stack(np.meshgrid(np.arange(GRID + 1), np.arange(GRID + 1)), -1).astype(np.float32)  # (x, y)
    seen = cv2.perspectiveTransform(m.reshape(-1, 1, 2), to_grid).reshape(GRID + 1, GRID + 1, 2)
    return float(np.linalg.norm(seen - ideal, axis=-1).max())


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
    f_px = focal_px(path, (photo_w, photo_h))
    tops = tile_tops(mesh, (photo_w, photo_h), f_px)
    first = straighten(photo, mesh, tops)
    residual = offset(first, truth)
    straight = straighten(photo, mesh, tops + np.array(residual)) if any(residual) else first
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
        "bend": round(bend(mesh), 3),
        "focal_px": round(f_px, 1),
        "parallax": [round(float(v), 3) for v in tops[GRID // 2, GRID // 2]],
        "residual": list(residual),
        "truth": truth,
        "tiles": tiles,
        "readings": out_readings,
    }
    with open(os.path.join(out_dir, f"board{idx}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    cv2.imwrite(os.path.join(out_dir, f"board{idx}.jpg"), straight, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return None


def app_scans():
    """The app's scans in test/app/ (scan_<ms>.json and .jpg), oldest first."""
    if not os.path.isdir(APP_DIR):
        return []
    return sorted(f[: -len(".json")] for f in os.listdir(APP_DIR) if f.startswith("scan_") and f.endswith(".json"))


def app_truth(scan):
    """The confirmed board as truth rows: upper-case letters, '?' a blank, '-' empty."""
    letters, blanks = scan["letters"], scan["blanks"]

    def square(i):
        return "-" if not letters[i] else "?" if blanks[i] else letters[i].upper()

    return ["".join(square(r * GRID + c) for c in range(GRID)) for r in range(GRID)]


def same_game(earlier, later) -> bool:
    """Whether `later` is `earlier` with tiles added: nearly all of its
    tiles still there, the same letters. A confirmed board can differ from
    the last by a correction, so not all."""
    tiles = [(r, c) for r in range(GRID) for c in range(GRID) if earlier[r][c] != "-"]
    kept = sum(later[r][c] == earlier[r][c] for r, c in tiles)
    return bool(tiles) and kept >= 0.9 * len(tiles)


def export_app(idx, name, layout, out_dir):
    with open(os.path.join(APP_DIR, name + ".json"), encoding="utf-8") as f:
        scan = json.load(f)
    photo = cv2.imread(os.path.join(APP_DIR, name + ".jpg"))
    if photo is None:
        return f"{name}: no frame"
    photo_h, photo_w = photo.shape[:2]
    # The app's corners start from the board's own top left: no turning.
    corners = np.asarray(scan["corners"], np.float32).reshape(4, 2)
    if quad_area(corners) < 0.05 * photo_w * photo_h:
        return f"{name}: the grid is too small in the frame ({quad_area(corners):.0f} px²)"
    unit = np.array([[0, 0], [GRID, 0], [GRID, GRID], [0, GRID]], np.float32)
    xy = np.stack(np.meshgrid(np.arange(GRID + 1), np.arange(GRID + 1)), -1).astype(np.float32)
    to_frame = cv2.getPerspectiveTransform(unit, corners)
    mesh = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), to_frame).reshape(GRID + 1, GRID + 1, 2).astype(np.float64)

    truth = app_truth(scan)
    f_px = float(scan["focal_px"])
    tops = tile_tops(mesh, (photo_w, photo_h), f_px)
    straight = straighten(photo, mesh, tops, smooth=0)
    match = registration(straight, truth)
    # The grid finder's corners are within a fifth of a square nine times in
    # ten, so the residual only ever trims. Searched further, a sparse board
    # with glare finds spurious shifts near a whole square; so it reaches
    # less, and is kept only where the labels then match better.
    residual = offset(straight, truth, reach=0.4)
    if any(residual):
        shifted = straighten(photo, mesh, tops + np.array(residual), smooth=0)
        better = registration(shifted, truth)
        if better > match:
            straight, match = shifted, better
        else:
            residual = (0.0, 0.0)
    if match < MIN_REGISTRATION:
        return f"{name}: the grid does not match the labels (registration {match:.2f})"
    record = {
        "source": "app",
        "photo": "../app/" + name + ".jpg",
        "difficulty": "c",
        "layout": layout,
        "photo_size": [photo_w, photo_h],
        "grid_corners": [[round(float(x), 2), round(float(y), 2)] for x, y in corners],
        "mesh": [[[round(float(x), 2), round(float(y), 2)] for x, y in row] for row in mesh],
        "rotation_k": 0,
        "registration": round(match, 3),
        "bend": 0.0,
        "focal_px": round(f_px, 1),
        "parallax": [round(float(v), 3) for v in tops[GRID // 2, GRID // 2]],
        "residual": list(residual),
        "truth": truth,
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
    jobs = []
    for idx in sorted(GROUND_TRUTH):
        key = tuple(sorted(GROUND_TRUTH[idx].items()))
        jobs.append((idx, layouts.setdefault(key, len(layouts)), DIFFICULTY[idx], None))
    # Then the app's scans, one layout per game.
    game, previous = len(layouts) - 1, None
    for i, name in enumerate(app_scans()):
        with open(os.path.join(APP_DIR, name + ".json"), encoding="utf-8") as f:
            truth = app_truth(json.load(f))
        if previous is None or not same_game(previous, truth):
            game += 1
        previous = truth
        jobs.append((APP_BASE + i, game, "c", name))
    for idx, layout, difficulty, app_name in jobs:
        if args.only and idx not in args.only:
            continue
        start = time.time()
        if app_name is None:
            problem = export_one(idx, layout, args.out)
        else:
            problem = export_app(idx, app_name, layout, args.out)
        print(f"board{idx}: {'FAILED ' + problem if problem else 'ok'} ({time.time() - start:.1f} s)", flush=True)
        if problem is not None:
            # Do not leave an earlier run's export of this board behind.
            for ext in ("json", "jpg"):
                stale = os.path.join(args.out, f"board{idx}.{ext}")
                if os.path.exists(stale):
                    os.remove(stale)
        else:
            index.append({"board": idx, "difficulty": difficulty, "layout": layout})
    if not args.only:
        with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=1)
    print(f"{len(index)} boards exported to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
