"""Harvest candidate real digit-crop templates (the tile's printed point
value, bottom-right corner) from board_reader's own test photos, into a
staging directory for manual review (review_templates.py --digits) before
they feed scripts/train_digit_classifier.py.

The digit's label needs no working classifier and no ambiguity check:
once a tile is at its ground-truth position, its printed point value is
already pinned down for free by LETTER_POINTS (a fixed, correct table --
ground_truth.py gives the letter, the table gives the digit). Review here
is purely about *crop quality*, not label correctness -- the same
"chopped in half" risk harvest_templates.py guards against for letters
applies to the digit region glyph_normalizer.py isolates too.

Usage:
    python scripts/harvest_digit_templates.py                # all em test photos
    python scripts/harvest_digit_templates.py -d emh          # include hard photos
    python scripts/harvest_digit_templates.py 0 5 6 7          # specific ids
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from ground_truth import DIFFICULTY, GROUND_TRUTH, IMAGE_PATHS  # noqa: E402
from letter_classifier import LETTER_POINTS  # noqa: E402
from read_board import extract_tile_patches, read_board  # noqa: E402

import glyph_normalizer as gn  # noqa: E402

STAGING_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "data", "real_digit_templates_staging")


def harvest_one(idx, path, gt):
    rotated, mesh, _cells, verdicts, shift = read_board(path, show=False)
    if verdicts is None:
        print(f"img{idx}: no board found, skipped")
        return 0
    patches = extract_tile_patches(rotated, mesh, verdicts, global_shift=shift)
    n = 0
    for v in verdicts:
        if not v.is_tile or (v.row, v.col) not in gt:
            continue
        pts = LETTER_POINTS.get(gt[(v.row, v.col)])
        if pts is None:
            continue
        glyph = gn.normalize(patches[(v.row, v.col)])
        if glyph.digit_gray is None:
            continue
        out_dir = os.path.join(STAGING_DIR, str(pts))
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"img{idx}_{v.row:02d}_{v.col:02d}.png"), glyph.digit_gray)
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", type=int, help="specific image indices to harvest (overrides --difficulty)")
    parser.add_argument(
        "-d",
        "--difficulty",
        default="em",
        help="difficulty suffixes to include: any of 'e' (easy), 'm' (medium), 'h' (hard), e.g. -d emh (default: em)",
    )
    args = parser.parse_args()

    ids = args.ids or [i for i in sorted(GROUND_TRUTH) if DIFFICULTY[i] in args.difficulty]

    total = 0
    for idx in ids:
        n = harvest_one(idx, IMAGE_PATHS[idx], GROUND_TRUTH[idx])
        print(f"img{idx:<3}: {n} digit crops harvested")
        total += n

    print(f"\n{total} digit crops harvested to {STAGING_DIR}")
    print("Run scripts/review_templates.py --digits next to inspect them before training.")


if __name__ == "__main__":
    main()
