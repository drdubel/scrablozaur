"""Interactively review harvested real-tile glyph templates before they're
used for anything -- harvest_templates.py's output (data/real_templates_staging/)
is only known to be correctly POSITIONED, not correctly EXTRACTED: a crop can
still be chopped in half, or lose a diacritic (Z on a photo with the
accent shadowed/mis-cropped will look exactly like Z rather than Z), and
nothing upstream of this tool checks for that.

Single window, everything driven by mouse -- no keyboard shortcuts to
memorise. The sidebar has two columns: "To review" lists letters with
unreviewed glyphs still in staging, "Reviewed" lists letters already
committed at least once (pulled back from real_templates/ and
real_templates_rejected/ so a past mistake can be corrected, not just new
crops). Click any letter in either column to jump straight to it. Click a
thumbnail to mark it rejected (red tint); click again to un-reject. The
footer bar has explicit Prev/Next Page and Prev/Next Letter buttons plus a
Quit & Save button, and a legend is drawn in the header so the controls are
visible without reading this docstring. Keyboard keys ([ ] for page, n/p
for letter, q/Esc to quit) still work as shortcuts, operating on whichever
column the current letter came from.

Moving to another letter (sidebar click, Next/Prev Letter, or quitting)
commits the current letter's decisions: accepted glyphs move to
data/real_templates/<LETTER>/ (the live directory the template matcher
reads and generate_synthetic_dataset.py mixes into CNN training data),
rejected ones move to data/real_templates_rejected/<LETTER>/ -- nothing is
ever deleted, so a misclick is always recoverable. A letter re-opened from
the "Reviewed" column shows its previous accepted+rejected split again
(still red-tinted where previously rejected) so flipping a past decision
just re-files that one glyph; already-correct files aren't touched.

Pass --digits to review harvest_digit_templates.py's output instead
(data/real_digit_templates_staging/ -> real_digit_templates/ /
real_digit_templates_rejected/) -- same tool, same UI, just a different
set of directories; "letter" below then means a point-value digit ("1",
"2", ... "9").

Usage:
    python scripts/review_templates.py             # letter glyphs
    python scripts/review_templates.py --digits     # point-value digit crops
"""

import argparse
import os
import shutil
import unicodedata

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGING_DIR = os.path.join(SCRIPT_DIR, "..", "src", "data", "real_templates_staging")
ACCEPTED_DIR = os.path.join(SCRIPT_DIR, "..", "src", "data", "real_templates")
REJECTED_DIR = os.path.join(SCRIPT_DIR, "..", "src", "data", "real_templates_rejected")
DIGIT_STAGING_DIR = os.path.join(SCRIPT_DIR, "..", "src", "data", "real_digit_templates_staging")
DIGIT_ACCEPTED_DIR = os.path.join(SCRIPT_DIR, "..", "src", "data", "real_digit_templates")
DIGIT_REJECTED_DIR = os.path.join(SCRIPT_DIR, "..", "src", "data", "real_digit_templates_rejected")

THUMB = 96  # displayed size per glyph cell (source glyphs are already 64x64)
PAD = 8
COLS = 10
ROWS = 6
PER_PAGE = COLS * ROWS
HEADER_H = 56
FOOTER_H = 48
SIDEBAR_W = 300  # two letter-list columns, see COL_W
COL_W = SIDEBAR_W // 2
SIDEBAR_ROW_H = 19
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "Template Review"

GRID_W = COLS * (THUMB + PAD) + PAD
GRID_H = ROWS * (THUMB + PAD) + PAD
CANVAS_W = SIDEBAR_W + GRID_W
CANVAS_H = HEADER_H + GRID_H + FOOTER_H
GRID_X0 = SIDEBAR_W
GRID_Y0 = HEADER_H

BTN_LABELS = ["<< Letter", "< Page", "Page >", "Letter >>", "Quit & Save"]
BTN_W = 150
BTN_H = 32
BTN_GAP = 12


def _button_rects():
    y0 = HEADER_H + GRID_H + (FOOTER_H - BTN_H) // 2
    rects = []
    x = SIDEBAR_W + 10
    for label in BTN_LABELS:
        rects.append((label, x, y0, x + BTN_W, y0 + BTN_H))
        x += BTN_W + BTN_GAP
    return rects


def _letters():
    if not os.path.isdir(STAGING_DIR):
        return []
    return sorted(unicodedata.normalize("NFC", d) for d in os.listdir(STAGING_DIR) if os.path.isdir(os.path.join(STAGING_DIR, d)))


def _files(letter):
    d = os.path.join(STAGING_DIR, letter)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))


def _dir_pngs(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")) if os.path.isdir(d) else []


def _reviewed_letters():
    """Letters with at least one glyph already committed, from either
    outcome -- these are re-openable even though staging no longer has
    anything for them."""
    seen = set()
    for base in (ACCEPTED_DIR, REJECTED_DIR):
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            if _dir_pngs(os.path.join(base, d)):
                seen.add(unicodedata.normalize("NFC", d))
    return sorted(seen)


def _reviewed_files(letter):
    """(accepted_paths, rejected_paths) as they currently sit on disk."""
    return _dir_pngs(os.path.join(ACCEPTED_DIR, letter)), _dir_pngs(os.path.join(REJECTED_DIR, letter))


def _reviewed_counts(letter):
    acc, rej = _reviewed_files(letter)
    return len(acc), len(rej)


def _unique_name(dest_dir, filename):
    if not os.path.exists(os.path.join(dest_dir, filename)):
        return filename
    base, ext = os.path.splitext(filename)
    i = 1
    while os.path.exists(os.path.join(dest_dir, f"{base}_{i}{ext}")):
        i += 1
    return f"{base}_{i}{ext}"


def _commit(letter, files, rejected, source):
    """File every glyph where its current decision says it belongs.
    Files already sitting in the right directory (untouched decisions on a
    re-review pass) are left alone rather than re-moved. Never deletes."""
    if not files:
        return 0, 0
    acc_dir = os.path.join(ACCEPTED_DIR, letter)
    rej_dir = os.path.join(REJECTED_DIR, letter)
    n_acc = n_rej = 0
    for i, path in enumerate(files):
        is_rejected = i in rejected
        n_rej += is_rejected
        n_acc += not is_rejected
        target_dir = rej_dir if is_rejected else acc_dir
        if os.path.normpath(os.path.dirname(path)) == os.path.normpath(target_dir):
            continue
        os.makedirs(target_dir, exist_ok=True)
        # A re-review can swap files between acc_dir and rej_dir in the same
        # commit; if their basenames ever collide (harvest_templates.py's
        # own img{n}_{row}_{col}.png names won't, but nothing guarantees
        # every file here came from it), blindly reusing the basename could
        # silently overwrite whatever's already at the destination.
        shutil.move(path, os.path.join(target_dir, _unique_name(target_dir, os.path.basename(path))))

    if source == "staging":
        staged_dir = os.path.join(STAGING_DIR, letter)
        if os.path.isdir(staged_dir) and not os.listdir(staged_dir):
            os.rmdir(staged_dir)
    for d in (acc_dir, rej_dir):
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    return n_acc, n_rej


class ReviewSession:
    """All mutable UI state for the single event loop below -- one object
    instead of nested closures so sidebar/footer/grid clicks can all
    reach the same navigation logic. `source` tracks whether the current
    letter came from the "to review" (staging) or "reviewed" (already
    committed) column, since committing files back differs slightly."""

    def __init__(self):
        self.staging_letters = []
        self.staging_counts = {}
        self.reviewed_letters = []
        self.reviewed_counts = {}
        self.totals = {"accepted": 0, "rejected": 0}
        self.quit = False
        self.source = None
        self.cur_letter = None
        self.files = []
        self.rejected = set()
        self.page = 0
        self._refresh_lists()
        if self.staging_letters:
            self._load(self.staging_letters[0], "staging")
        elif self.reviewed_letters:
            self._load(self.reviewed_letters[0], "reviewed")

    @property
    def n_pages(self):
        return max(1, (len(self.files) + PER_PAGE - 1) // PER_PAGE)

    def _refresh_lists(self):
        self.staging_letters = _letters()
        self.staging_counts = {letter: len(_files(letter)) for letter in self.staging_letters}
        self.reviewed_letters = _reviewed_letters()
        self.reviewed_counts = {letter: _reviewed_counts(letter) for letter in self.reviewed_letters}

    def _load(self, letter, source):
        self.cur_letter = letter
        self.source = source
        self.page = 0
        if source == "staging":
            self.files = _files(letter)
            self.rejected = set()
        else:
            acc, rej = _reviewed_files(letter)
            self.files = acc + rej
            self.rejected = set(range(len(acc), len(acc) + len(rej)))

    def commit_current(self):
        if self.cur_letter is None:
            return
        n_acc, n_rej = _commit(self.cur_letter, self.files, self.rejected, self.source)
        self.totals["accepted"] += n_acc
        self.totals["rejected"] += n_rej
        print(f"  {self.cur_letter}: {n_acc} accepted, {n_rej} rejected")
        self._refresh_lists()

    def goto_letter(self, letter, source):
        if letter == self.cur_letter and source == self.source:
            return
        self.commit_current()
        active = self.staging_letters if source == "staging" else self.reviewed_letters
        if letter not in active:
            return  # committing may have just emptied it out from under us
        self._load(letter, source)

    def step_letter(self, direction):
        active = self.staging_letters if self.source == "staging" else self.reviewed_letters
        if self.cur_letter in active:
            j = active.index(self.cur_letter) + direction
            if 0 <= j < len(active):
                self.goto_letter(active[j], self.source)
                return

        # Ran off the end of the current column (or committing dropped the
        # current letter from it) -- commit, then pick up wherever there's
        # still something to review, preferring to stay in the same column.
        self.commit_current()
        same = self.staging_letters if self.source == "staging" else self.reviewed_letters
        if same:
            self._load(same[0], self.source)
            return
        other_source = "reviewed" if self.source == "staging" else "staging"
        other = self.reviewed_letters if self.source == "staging" else self.staging_letters
        if other:
            self._load(other[0], other_source)
        else:
            self.cur_letter = None
            self.quit = True

    def step_page(self, direction):
        self.page = max(0, min(self.n_pages - 1, self.page + direction))

    def toggle(self, global_i):
        start = self.page * PER_PAGE
        if not (start <= global_i < start + min(PER_PAGE, len(self.files) - start)):
            return
        if global_i in self.rejected:
            self.rejected.discard(global_i)
        else:
            self.rejected.add(global_i)


def _draw_letter_column(canvas, letters, label_fn, source, sess, x0, x1, sidebar_rows):
    y = 34
    for letter in letters:
        is_cur = letter == sess.cur_letter and source == sess.source
        if is_cur:
            cv2.rectangle(canvas, (x0, y - 14), (x1, y + 5), (90, 60, 20), -1)
        color = (255, 255, 255) if is_cur else (170, 170, 170)
        cv2.putText(canvas, f"{letter}  {label_fn(letter)}", (x0 + 6, y), FONT, 0.46, color, 1)
        sidebar_rows.append((letter, source, x0, y - 14, x1, y + 5))
        y += SIDEBAR_ROW_H
        if y > CANVAS_H - 10:
            break


def _draw(sess):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 40, np.uint8)

    mode = "reviewing new" if sess.source == "staging" else "re-checking"
    header = f"{sess.cur_letter}  [{mode}]   page {sess.page + 1}/{sess.n_pages}   {len(sess.files)} glyphs, {len(sess.rejected)} rejected"
    cv2.putText(canvas, header, (SIDEBAR_W + 10, 24), FONT, 0.65, (255, 255, 255), 1)
    legend = "click a glyph = toggle reject   |   click a letter on the left = jump to it   |   buttons below move around"
    cv2.putText(canvas, legend, (SIDEBAR_W + 10, 46), FONT, 0.42, (160, 160, 160), 1)

    cv2.line(canvas, (COL_W, 0), (COL_W, CANVAS_H), (70, 70, 70), 1)
    cv2.putText(canvas, "To review", (10, 20), FONT, 0.48, (200, 200, 200), 1)
    cv2.putText(canvas, "Reviewed", (COL_W + 10, 20), FONT, 0.48, (200, 200, 200), 1)

    sidebar_rows = []
    _draw_letter_column(
        canvas, sess.staging_letters, lambda ltr: f"({sess.staging_counts.get(ltr, 0)})", "staging", sess, 0, COL_W, sidebar_rows
    )
    _draw_letter_column(
        canvas,
        sess.reviewed_letters,
        lambda ltr: "({}/{})".format(*sess.reviewed_counts.get(ltr, (0, 0))),
        "reviewed",
        sess,
        COL_W,
        SIDEBAR_W,
        sidebar_rows,
    )

    # Glyph grid for the current letter/page.
    start = sess.page * PER_PAGE
    page_files = sess.files[start : start + PER_PAGE]
    for i, path in enumerate(page_files):
        global_i = start + i
        row, col = divmod(i, COLS)
        x0 = GRID_X0 + PAD + col * (THUMB + PAD)
        y0 = GRID_Y0 + PAD + row * (THUMB + PAD)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        thumb = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        if global_i in sess.rejected:
            red = np.zeros_like(thumb_bgr)
            red[..., 2] = 255
            thumb_bgr = cv2.addWeighted(thumb_bgr, 0.55, red, 0.45, 0)
        border = (0, 0, 220) if global_i in sess.rejected else (90, 90, 90)
        canvas[y0 : y0 + THUMB, x0 : x0 + THUMB] = thumb_bgr
        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + THUMB, y0 + THUMB), border, 2)
        cv2.putText(canvas, str(global_i), (x0 + 2, y0 + 14), FONT, 0.4, (0, 255, 255), 1)

    for label, bx0, by0, bx1, by1 in _button_rects():
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (80, 80, 80), -1)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (140, 140, 140), 1)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.putText(canvas, label, (bx0 + (BTN_W - tw) // 2, by0 + (BTN_H + th) // 2), FONT, 0.5, (255, 255, 255), 1)

    return canvas, sidebar_rows


def _handle_click(sess, x, y, sidebar_rows):
    for letter, source, rx0, ry0, rx1, ry1 in sidebar_rows:
        if rx0 <= x < rx1 and ry0 <= y <= ry1:
            sess.goto_letter(letter, source)
            return

    for label, bx0, by0, bx1, by1 in _button_rects():
        if bx0 <= x <= bx1 and by0 <= y <= by1:
            {
                "<< Letter": lambda: sess.step_letter(-1),
                "< Page": lambda: sess.step_page(-1),
                "Page >": lambda: sess.step_page(1),
                "Letter >>": lambda: sess.step_letter(1),
                "Quit & Save": lambda: (sess.commit_current(), setattr(sess, "quit", True)),
            }[label]()
            return

    if GRID_X0 <= x < GRID_X0 + GRID_W and GRID_Y0 <= y < GRID_Y0 + GRID_H:
        col = (x - GRID_X0 - PAD) // (THUMB + PAD)
        row = (y - GRID_Y0 - PAD) // (THUMB + PAD)
        if 0 <= col < COLS and 0 <= row < ROWS:
            global_i = sess.page * PER_PAGE + row * COLS + col
            sess.toggle(global_i)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--digits", action="store_true", help="review point-value digit crops instead of letter glyphs")
    args = parser.parse_args()

    if args.digits:
        global STAGING_DIR, ACCEPTED_DIR, REJECTED_DIR
        STAGING_DIR, ACCEPTED_DIR, REJECTED_DIR = DIGIT_STAGING_DIR, DIGIT_ACCEPTED_DIR, DIGIT_REJECTED_DIR

    sess = ReviewSession()
    if sess.cur_letter is None:
        print(f"Nothing to review in {STAGING_DIR}, and nothing already reviewed in {ACCEPTED_DIR}/{REJECTED_DIR}.")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    sidebar_rows_holder = []

    def on_click(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        _handle_click(sess, x, y, sidebar_rows_holder)

    cv2.setMouseCallback(WINDOW, on_click)

    while not sess.quit:
        cv2.setWindowTitle(WINDOW, f"{WINDOW} -- {sess.cur_letter} ({len(sess.files)} glyphs, {sess.source})")
        canvas, rows = _draw(sess)
        sidebar_rows_holder[:] = rows
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            sess.commit_current()
            break
        elif key == ord("n"):
            sess.step_letter(1)
        elif key == ord("p"):
            sess.step_letter(-1)
        elif key == ord("]"):
            sess.step_page(1)
        elif key == ord("["):
            sess.step_page(-1)

    cv2.destroyAllWindows()
    print(f"Done. {sess.totals['accepted']} accepted, {sess.totals['rejected']} rejected this session.")
    print("Remaining unreviewed letters (if any) stay in the staging directory for next time.")


if __name__ == "__main__":
    main()
