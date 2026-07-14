"""Harvest candidate real-tile glyph templates from board_reader's own test
photos, into a staging directory for manual review (review_templates.py)
before they're promoted into src/data/real_templates/ -- the directory the
template matcher reads live, and generate_synthetic_dataset.py mixes into
CNN training data.

Ground truth gives the label for free, so this needs no working
classifier: a tile only needs to be correctly DETECTED (tile_detector.py's
is_tile, at its ground-truth position) to be harvested, independent of
whatever letter_classifier.py currently predicts for it. This is also why
review matters -- nothing here has checked that the *glyph extraction*
itself came out clean (glyph_normalizer.py can still chop a letter in half
or lose a diacritic on a bad crop), only that the tile's presence and
position are right.

Usage:
    python scripts/harvest_templates.py                # all em test photos
    python scripts/harvest_templates.py -d emh          # include hard photos
    python scripts/harvest_templates.py 0 5 6 7          # specific ids
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from ground_truth import DIFFICULTY, GROUND_TRUTH, IMAGE_PATHS  # noqa: E402
from read_board import extract_tile_patches, read_board  # noqa: E402

import glyph_normalizer as gn  # noqa: E402

STAGING_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "data", "real_templates_staging")
# A tile with almost no ink is probably a failed extraction (blank crop);
# one with very heavy ink is probably a mis-crop that grabbed a neighbour
# or a tile edge, not a clean single glyph -- same bounds ocr/'s own
# harvest_templates.py uses.
MIN_INK_FRACTION = 0.04
MAX_INK_FRACTION = 0.45


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
        glyph = gn.normalize(patches[(v.row, v.col)])
        if not glyph.has_glyph or not (MIN_INK_FRACTION < glyph.ink_fraction < MAX_INK_FRACTION):
            continue
        letter = gt[(v.row, v.col)]
        out_dir = os.path.join(STAGING_DIR, letter)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"img{idx}_{v.row:02d}_{v.col:02d}.png"), glyph.mask)
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
        print(f"img{idx:<3}: {n} glyphs harvested")
        total += n

    print(f"\n{total} glyphs harvested to {STAGING_DIR}")
    print("Run scripts/review_templates.py next to inspect them before they're used for anything.")


if __name__ == "__main__":
    main()
