"""Evaluate letter classification against the hand-labelled ground truth in
test/out/board<i>.txt.

Usage:
    python tests/eval_letters.py                 # all easy+medium images
    python tests/eval_letters.py 0 7 10           # selected images (any difficulty)
    python tests/eval_letters.py -d e              # easy only
    python tests/eval_letters.py -d emh             # include hard too

Hard ('h') images are excluded by default, matching eval_tile_detection.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ground_truth import DIFFICULTY, GROUND_TRUTH, IMAGE_PATHS  # noqa: E402
from read_board import read_board  # noqa: E402
from read_letters import classify_board  # noqa: E402


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

    tot_tp = tot_fp = tot_fn = tot_correct = 0
    tot_cells = tot_cells_ok = 0
    failures = []

    for idx in ids:
        gt = GROUND_TRUTH[idx]
        gt_tiles = set(gt)
        rotated, mesh, _cells, verdicts, shift = read_board(IMAGE_PATHS[idx], show=False)
        if verdicts is None:
            print(f"img{idx:<3}: FAILED (no board found)")
            failures.append(idx)
            continue

        board = classify_board(rotated, mesh, verdicts, global_shift=shift)

        pred_tiles = {(v.row, v.col) for v in verdicts if v.is_tile}
        tp, fp, fn = pred_tiles & gt_tiles, pred_tiles - gt_tiles, gt_tiles - pred_tiles
        correct = sum(1 for r, c in tp if board[r][c] == gt[(r, c)])
        wrong = {(r, c): (board[r][c], gt[(r, c)]) for r, c in tp if board[r][c] != gt[(r, c)]}

        tot_tp += len(tp)
        tot_fp += len(fp)
        tot_fn += len(fn)
        tot_correct += correct
        tot_cells += 225
        tot_cells_ok += 225 - len(fp) - len(fn) - len(wrong)

        letter_acc = correct / len(tp) if tp else 0.0
        msg = f"img{idx:<3}: tiles {len(tp)}/{len(gt_tiles)} found, {len(fp)} extra | letters {correct}/{len(tp)} ({letter_acc:.1%})"
        if wrong:
            msg += f" | wrong: {wrong}"
        print(msg)

    precision = tot_tp / max(1, tot_tp + tot_fp)
    recall = tot_tp / max(1, tot_tp + tot_fn)
    letter_acc = tot_correct / max(1, tot_tp)
    cell_acc = tot_cells_ok / max(1, tot_cells)
    print(
        f"\nprecision {precision:.1%}  recall {recall:.1%}  letter accuracy {letter_acc:.1%}"
        f"  cell accuracy {cell_acc:.1%}  ({len(failures)} board-detection failures: {failures})"
    )


if __name__ == "__main__":
    main()
