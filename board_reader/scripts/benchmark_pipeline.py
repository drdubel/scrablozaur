"""Time board_reader's photo-to-board-state pipeline, stage by stage.

Two instrumentation layers, both call-site only -- nothing under src/ is
modified:

  Layer 1: a short re-call of read_board()'s own body (image load, board
  detection/warp, rotation, grid detection, parallax-shift search, cell
  extraction, tile detection), each step timed individually, followed by
  one black-box-timed call to classify_tiles() -- mirroring exactly what
  web/scan.py's scan_board_image() does, so the reported total is a real
  photo's cost, not an inflated one from calling internals twice.

  Layer 2: monkeypatched timing wrappers around the specific bound names
  used *inside* detect_tiles()/classify_tiles() (glyph_score, gn.normalize,
  classify_cnn_batch, etc.), to attribute time within those two black
  boxes. These buckets nest inside the Layer 1 stages that call them
  (e.g. classify_tiles_total contains gn.normalize + classify_cnn_batch
  time) -- they're a further breakdown, not additional wall time, so don't
  add Layer 2 sums on top of Layer 1 sums when eyeballing a grand total.

Usage:
    python scripts/benchmark_pipeline.py                 # all 89 photos (easy+medium+hard)
    python scripts/benchmark_pipeline.py -d em             # easy+medium only
    python scripts/benchmark_pipeline.py 0 7 10             # specific ids
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from ground_truth import DIFFICULTY, GROUND_TRUTH, IMAGE_PATHS  # noqa: E402

import glyph_normalizer  # noqa: E402
import letter_classifier  # noqa: E402
import read_board  # noqa: E402
import read_letters  # noqa: E402
import tile_detector  # noqa: E402
from detect_board import find_board_quad, warp_board  # noqa: E402
from grid_detector import detect_grid  # noqa: E402
from rotate_board import rotate_board  # noqa: E402

STAGE_ORDER = [
    "image_load",
    "find_board_quad",
    "warp_board",
    "rotate_board",
    "detect_grid",
    "find_parallax_shift",
    "extract_cells",
    "detect_tiles",
    "classify_tiles_total",
]


class Instrumentor:
    """Wraps named functions on given modules with a timing+call-count
    probe, resettable per photo. See module docstring for why patching
    happens at the *call site's* module namespace, not the defining one."""

    def __init__(self):
        self._patches = []  # (module, attr_name, orig)
        self.stats = defaultdict(lambda: [0, 0.0])  # bucket_name -> [count, total_seconds]

    def patch(self, module, attr_name, bucket_name=None):
        bucket_name = bucket_name or attr_name
        orig = getattr(module, attr_name)

        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            r = orig(*a, **kw)
            dt = time.perf_counter() - t0
            entry = self.stats[bucket_name]
            entry[0] += 1
            entry[1] += dt
            return r

        setattr(module, attr_name, wrapped)
        self._patches.append((module, attr_name, orig))

    def restore(self):
        for module, attr_name, orig in self._patches:
            setattr(module, attr_name, orig)
        self._patches.clear()

    def reset_photo(self):
        self.stats = defaultdict(lambda: [0, 0.0])


def _fmt(name, arr, width=34):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return f"{name:<{width}s}  (no data)"
    return (
        f"{name:<{width}s} mean={arr.mean() * 1000:8.1f}ms  median={np.median(arr) * 1000:8.1f}ms  "
        f"p95={np.percentile(arr, 95) * 1000:8.1f}ms  sum={arr.sum():7.2f}s  n={len(arr)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", type=int, help="specific image indices to benchmark (overrides --difficulty)")
    parser.add_argument(
        "-d",
        "--difficulty",
        default="emh",
        help="difficulty suffixes to include: any of 'e' (easy), 'm' (medium), 'h' (hard) (default: emh, all photos)",
    )
    args = parser.parse_args()
    ids = args.ids or [i for i in sorted(GROUND_TRUTH) if DIFFICULTY[i] in args.difficulty]
    print(f"Benchmarking {len(ids)} photos: {ids}\n")

    # Deterministic warmup: force all four lazy singletons to load now, so
    # the first photo's numbers aren't contaminated by one-time model/
    # template load cost. Don't rely on photo 0 incidentally triggering all
    # four -- template matching is confidence-gated and may not fire on an
    # easy first photo.
    letter_classifier._get_cnn()
    letter_classifier._get_digit_cnn()
    letter_classifier._get_templates()
    letter_classifier._get_digit_templates()
    print(f"CNN device: {letter_classifier._cnn_device}  digit CNN device: {letter_classifier._digit_cnn_device}\n")

    instr = Instrumentor()
    instr.patch(tile_detector, "glyph_score", "glyph_score[detect_tiles]")
    instr.patch(read_board, "glyph_score", "glyph_score[parallax _shift_score]")
    instr.patch(read_board, "features_batch", "features_batch[parallax _shift_score]")
    instr.patch(read_board, "_shift_score", "_shift_score")
    instr.patch(read_letters, "extract_tile_patches", "extract_tile_patches")
    instr.patch(read_letters, "_resolve_rotation", "_resolve_rotation")
    instr.patch(glyph_normalizer, "normalize", "gn.normalize (main+reprocess+resolve_rotation combined)")
    instr.patch(read_letters, "classify_cnn_batch", "classify_cnn_batch")
    instr.patch(read_letters, "classify_digit_cnn_batch", "classify_digit_cnn_batch")
    instr.patch(read_letters, "classify_templates", "classify_templates")
    instr.patch(read_letters, "classify_digit_templates", "classify_digit_templates")

    stage_times = defaultdict(list)
    bucket_times = defaultdict(list)
    bucket_calls = defaultdict(list)
    total_times = []
    tile_counts = []
    parallax_triggered = []
    failures = []

    try:
        for idx in ids:
            path = IMAGE_PATHS[idx]
            instr.reset_photo()
            t_start = time.perf_counter()

            t0 = time.perf_counter()
            image = cv2.imread(path)
            stage_times["image_load"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            corners = find_board_quad(image)
            stage_times["find_board_quad"].append(time.perf_counter() - t0)
            if corners is None:
                print(f"img{idx:<3}: FAILED (no board found)")
                failures.append(idx)
                continue

            t0 = time.perf_counter()
            board = warp_board(image, corners)
            stage_times["warp_board"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            rotated = rotate_board(board)
            stage_times["rotate_board"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            grid = detect_grid(rotated)
            stage_times["detect_grid"].append(time.perf_counter() - t0)
            if grid is None:
                print(f"img{idx:<3}: FAILED (no grid found)")
                failures.append(idx)
                continue

            t0 = time.perf_counter()
            shift = read_board.find_parallax_shift(rotated, grid.mesh)
            stage_times["find_parallax_shift"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            cells = read_board.extract_cells(rotated, grid.mesh, global_shift=shift)
            stage_times["extract_cells"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            verdicts = read_board.detect_tiles(cells)
            stage_times["detect_tiles"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            readings = read_letters.classify_tiles(rotated, grid.mesh, verdicts, global_shift=shift)
            stage_times["classify_tiles_total"].append(time.perf_counter() - t0)

            total_times.append(time.perf_counter() - t_start)
            tile_counts.append(sum(v.is_tile for v in verdicts))
            shift_calls = instr.stats["_shift_score"][0]
            parallax_triggered.append(shift_calls > 1)

            for name, (count, secs) in instr.stats.items():
                bucket_times[name].append(secs)
                bucket_calls[name].append(count)

            print(f"img{idx:<3}: {tile_counts[-1]:3d} tiles, total {total_times[-1] * 1000:7.1f}ms" f" ({len(readings)} read)")
    finally:
        instr.restore()

    n_ok = len(total_times)
    print(f"\n{'=' * 100}")
    print(f"Layer 1: per-photo stage timings ({n_ok} succeeded / {len(ids)} attempted)")
    print("=" * 100)
    for name in STAGE_ORDER:
        if name in stage_times:
            print(_fmt(name, stage_times[name]))
    if total_times:
        print(_fmt("TOTAL (image_load..classify_tiles_total)", total_times))

    print(f"\n{'=' * 100}")
    print("Layer 2: sub-stage buckets inside detect_tiles()/classify_tiles() (nested -- see module docstring)")
    print("=" * 100)
    for name, arr in sorted(bucket_times.items(), key=lambda kv: -sum(kv[1])):
        calls = bucket_calls[name]
        print(_fmt(name, arr) + f"  calls/photo mean={np.mean(calls):5.1f}")

    print(f"\n{'=' * 100}")
    print("Supporting stats")
    print("=" * 100)
    if tile_counts:
        tc = np.array(tile_counts)
        print(f"tiles/photo: mean={tc.mean():.1f}  median={np.median(tc):.0f}  min={tc.min()}  max={tc.max()}")
    if parallax_triggered:
        n_trig = sum(parallax_triggered)
        print(
            f"parallax shift search triggered (baseline score < {read_board.SHIFT_SEARCH_TRIGGER}): "
            f"{n_trig}/{len(parallax_triggered)} photos ({100 * n_trig / len(parallax_triggered):.1f}%)"
        )
    print(f"board/grid detection failures: {len(failures)} {failures}")


if __name__ == "__main__":
    main()
