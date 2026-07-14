"""Automatically strip stray dot/line artifacts from harvested real
templates in src/data/real_templates/.

glyph_normalizer.py's component selection doesn't know which letter it's
looking at, so a stray fragment (a bit of grid line, a bevel shadow corner)
positioned near the top of the glyph can get kept as if it were a
legitimate diacritic. But which letters can *legitimately* have a second
ink component is exactly known and fixed: only Polish's 9 diacritic
letters (a dot/acute above, or an ogonek below) ever have one. For the
other 23 letters, any component beyond the main glyph body is definitely
noise, not a diacritic -- there's nothing to guess.

So this keeps, per template: the single largest ink component for a
plain letter, or the largest plus one more (the next-largest, standing in
for its accent) for a diacritic letter. Anything past that is dropped.

This is a blunt instrument, not a substitute for review_templates.py: it
can't tell a real diacritic from a same-sized noise blob on a diacritic
letter (keeps both, since it only counts, doesn't judge shape/position),
and if the *main* glyph itself is corrupted this won't fix that. Originals
are always preserved in src/data/real_templates_before_cleanup/ (nothing
is edited without a backup), and any template whose cleaned result still
looks suspicious (tiny or line-thin main component) is moved back into the
staging directory for review_templates.py to re-check by hand.

Usage:
    python scripts/clean_templates.py
"""

import os
import shutil
import sys
import unicodedata

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from letter_classifier import POLISH_ALPHABET  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_DIR = os.path.join(SCRIPT_DIR, "..", "src", "data", "real_templates")
BACKUP_DIR = os.path.join(SCRIPT_DIR, "..", "src", "data", "real_templates_before_cleanup")
STAGING_DIR = os.path.join(SCRIPT_DIR, "..", "src", "data", "real_templates_staging")

DIACRITIC_LETTERS = set("ĄĆĘŁŃÓŚŹŻ")

# After cleaning, a main component below this fraction of the canvas area,
# or with either side under this fraction of the canvas width/height, is
# probably not a real letter at all (a line fragment or speck that
# happened to be the largest thing in a near-empty template) -- route it
# back to staging instead of trusting it.
MIN_AREA_FRAC = 0.03
MIN_SIDE_FRAC = 0.12


def _clean_mask(mask, max_components):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask, False
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    keep_ids = set(order[:max_components])
    if len(keep_ids) == n - 1:
        return mask, False  # nothing dropped
    cleaned = np.zeros_like(mask)
    for i in keep_ids:
        cleaned[labels == i] = 255
    return cleaned, True


def _is_suspicious(mask):
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return True  # nothing left at all
    h, w = mask.shape
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = int(np.argmax(areas)) + 1
    x, y, bw, bh, a = stats[main]
    return a < MIN_AREA_FRAC * h * w or bw < MIN_SIDE_FRAC * w or bh < MIN_SIDE_FRAC * h


def clean_letter(letter):
    letter_dir = os.path.join(LIVE_DIR, letter)
    if not os.path.isdir(letter_dir):
        return 0, 0, 0
    max_components = 2 if letter in DIACRITIC_LETTERS else 1

    n_changed = n_flagged = n_total = 0
    for fname in sorted(os.listdir(letter_dir)):
        if not fname.endswith(".png"):
            continue
        n_total += 1
        path = os.path.join(letter_dir, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cleaned, changed = _clean_mask(mask, max_components)
        if not changed:
            continue
        n_changed += 1

        backup_dir = os.path.join(BACKUP_DIR, letter)
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(backup_dir, fname))

        if _is_suspicious(cleaned):
            stage_dir = os.path.join(STAGING_DIR, letter)
            os.makedirs(stage_dir, exist_ok=True)
            shutil.move(path, os.path.join(stage_dir, fname))
            n_flagged += 1
        else:
            cv2.imwrite(path, cleaned)
    return n_total, n_changed, n_flagged


def main():
    if not os.path.isdir(LIVE_DIR):
        print(f"Nothing to clean: {LIVE_DIR} does not exist.")
        return
    letters = sorted(unicodedata.normalize("NFC", d) for d in os.listdir(LIVE_DIR) if os.path.isdir(os.path.join(LIVE_DIR, d)))

    tot_total = tot_changed = tot_flagged = 0
    for letter in letters:
        if letter not in POLISH_ALPHABET:
            continue
        n_total, n_changed, n_flagged = clean_letter(letter)
        if n_changed:
            print(f"{letter}: {n_changed}/{n_total} cleaned, {n_flagged} flagged back to staging (unclear even after cleaning)")
        tot_total += n_total
        tot_changed += n_changed
        tot_flagged += n_flagged

    print(f"\n{tot_changed}/{tot_total} templates had a stray component removed ({tot_flagged} flagged for re-review).")
    print(f"Originals backed up under {BACKUP_DIR}")
    if tot_flagged:
        print("Run scripts/review_templates.py to look at the flagged ones.")


if __name__ == "__main__":
    main()
