"""Evaluate tile-presence detection against the hand-labelled ground truth
in test/out/board<i>.txt (no letter OCR exists in board_reader yet, so this
only scores tile presence/absence, not letter identity).

Usage:
    python tests/eval_tile_detection.py                 # all easy+medium images
    python tests/eval_tile_detection.py 0 7 10           # selected images (any difficulty)
    python tests/eval_tile_detection.py -d e              # easy only
    python tests/eval_tile_detection.py -d emh             # include hard too

Hard ('h') images are excluded by default -- some are close to impossible to
read (extreme angle/lighting) and would just add noise to the aggregate
score; pass -d emh (or list their ids explicitly) to include them anyway.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ground_truth import DIFFICULTY, GROUND_TRUTH, IMAGE_PATHS  # noqa: E402
from read_board import read_board  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", type=int, help="specific image indices to evaluate (overrides --difficulty)")
    parser.add_argument(
        "-d",
        "--difficulty",
        default="em",
        help="difficulty suffixes to include: any of 'e' (easy), 'm' (medium), 'h' (hard), e.g. -d emh (default: em)",
    )
    args = parser.parse_args()

    ids = args.ids or [i for i in sorted(GROUND_TRUTH) if DIFFICULTY[i] in args.difficulty]

    tot_tp = tot_fp = tot_fn = 0
    tot_cells = tot_cells_ok = 0
    failures = []

    for idx in ids:
        gt = set(GROUND_TRUTH[idx])
        _rotated, _mesh, _cells, verdicts, _shift = read_board(IMAGE_PATHS[idx], show=False)
        if verdicts is None:
            print(f"img{idx:<3}: FAILED (no board found)")
            failures.append(idx)
            continue

        pred = {(v.row, v.col) for v in verdicts if v.is_tile}
        tp, fp, fn = len(pred & gt), len(pred - gt), len(gt - pred)
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_cells += 225
        tot_cells_ok += 225 - fp - fn
        print(f"img{idx:<3}: {tp}/{len(gt)} tiles found, {fp} extra | cell_acc {(225 - fp - fn) / 225:.1%}")

    precision = tot_tp / max(1, tot_tp + tot_fp)
    recall = tot_tp / max(1, tot_tp + tot_fn)
    cell_acc = tot_cells_ok / max(1, tot_cells)
    print(
        f"\nprecision {precision:.1%}  recall {recall:.1%}  cell accuracy {cell_acc:.1%}"
        f"  ({len(failures)} board-detection failures: {failures})"
    )


if __name__ == "__main__":
    main()
