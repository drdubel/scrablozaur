"""Stage 3: per-cell occupancy check and letter classification.

A self-contained classifier -- deliberately simpler than
ocr/scrabble_reader/'s CNN + template + OCR-fallback fusion (that package
already solves this problem more robustly; board_reader builds its own
independent version instead of wiring into it, per an explicit choice to
keep this stage simple and standalone). Recognition itself, though, is
Tesseract OCR (ocr_classify_glyph()) with template matching only as a
fallback -- see that function's docstring for the empirical comparison
that made OCR the default.

classify_cell() takes a *color* cell crop (a slice of the grid-aligned
warp, e.g. grid_reader.extract_cells(grid_warp, ...)). It first checks
is_tile_present() directly on that color crop: a physical tile (letter or
blank) is wood/cream-colored, while a bare board square -- normal or
premium, whatever its printed label says -- is noticeably more saturated,
so this rejects empty squares (and premium labels) before any binarization
or contour work runs. Only a cell with a tile actually present is then
binarized, per cell, via local_binarize() -- deliberately NOT a slice of
grid_reader.binarize_tiles()'s output, which applies one global HSV
threshold across the whole board. A single global cutoff doesn't fit every
region of a real photo (confirmed: it under-inked an entire row on one
test photo whose lighting varied across the frame), and no fixed preset
would either, since the same problem recurs *within* one photo, not just
across photos. Otsu's method recomputes its own cutoff from each cell's
own pixel histogram, so it adapts automatically with no tunable threshold
and no manual preset selection -- see local_binarize()'s docstring for
the empirical comparison against the global-threshold approach it replaced.
The isolated glyph is then read by ocr_classify_glyph() first, falling
back to template matching (against refs from render_reference_glyphs(),
rendered once and passed into classify_cell() rather than re-rendered per
call) only when OCR is unavailable or unconfident.

Each Polish Scrabble tile also prints its point value in the bottom-right
corner, and the point scale is a fixed, known mapping (LETTER_POINTS,
matching src/lib.rs's calculate_word_points -- the actual game engine's
scoring table, so this is guaranteed consistent with real gameplay). When
the top template-match candidates for a glyph are close (genuinely
ambiguous, e.g. L vs Ł), classify_cell() reads the tile's own point digit
and prefers whichever close candidate's point value actually matches it --
a deterministic tie-breaker the classifier wouldn't otherwise have. This
only applies on the template-matching fallback path (see classify_cell()).
"""

import concurrent.futures

import cv2
import numpy as np
from hsv_config import load_params
from PIL import Image, ImageDraw, ImageFont

try:
    import pytesseract

    _HAS_TESSERACT = True
except ImportError:  # pragma: no cover -- OCR is an optional enhancement
    _HAS_TESSERACT = False

# Polish Scrabble tile letters -- no Q/V/X on the physical tile set (see
# test/out/board*.txt ground truth and the photos in test/in/ for the
# actual alphabet in use).
ALPHABET = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ"
DIGITS = "0123456789"

# Point value printed on every tile of that letter, matching src/lib.rs's
# Board::calculate_word_points -- the game engine's own scoring table, sow
# this is guaranteed to match what's actually printed on the physical tiles.
LETTER_POINTS = {
    **{c: 1 for c in "AEIOZWNSR"},
    **{c: 2 for c in "DYCKLMPT"},
    **{c: 3 for c in "BGHJŁU"},
    **{c: 4 for c in "ĄĘFÓŚŻ"},
    "Ć": 6,
    "Ń": 7,
    "Ź": 9,
}

GLYPH_SIZE = 48

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
]

# Tunable knobs, overridden by a preset saved to hsv_config.json (name
# "letter_params") if one exists.
PARAM_DEFAULTS = {
    "tile_sat_ratio": 0.75,  # is_tile_present()'s real occupied/empty signal: a
    # physical tile (letter or blank) is wood/cream-colored
    # (low HSV saturation), while a bare board square --
    # normal *or* premium, regardless of its printed label
    # color -- is always noticeably more saturated. A *fixed*
    # saturation cutoff doesn't generalize, though: measured
    # real letters at median saturation ~30 under one photo's
    # lighting and ~115 under another's warmer cast -- same
    # relationship (letter clearly less saturated than the
    # board), wildly different absolute values, mirroring why
    # local_binarize() had to move off a single global
    # brightness cutoff. So this is a *ratio* against
    # board_saturation_reference()'s per-photo median (most
    # cells are empty board at any point in a game, so the
    # board's own color reliably anchors "board-colored" for
    # that photo): a cell counts as tile-present if its
    # saturation is below this fraction of that reference.
    # Swept across 5 test images including the problem photo
    # above: 0/193 real letters missed at ratio 0.75 (vs.
    # missing an entire image's worth at a fixed cutoff). On
    # the full em test set (validated with evaluate()), this
    # gate is a strict improvement over the old white-fraction
    # + dominance-ratio pre-filter it replaced: occupied-cell
    # accuracy unchanged (495/1495, 33.1%), overall cell
    # accuracy 82.2% -> 83.8% (fewer premium-square false
    # positives).
    "min_hole_area_frac": 0.033,  # the *real* letter/empty signal (see extract_glyph):
    # a premium square's printed white label ("PODWÓJNA
    # PREMIA SŁOWNA" etc.) is a large solid white blob
    # too, but its ink holes are
    # 2-3 similarly-small text fragments (observed ~0.03-
    # 0.04 of cell area of an *unexpanded* cell); a real
    # tile's letter is one dominant hole (observed ~0.06-
    # 0.10 unexpanded) -- so requiring a hole above this
    # threshold rejects decorative label text without
    # rejecting real letters. This value is scaled down
    # from that observed range to match expand_frac=0.08's
    # larger cell area (area grows by (1+2*0.08)^2 ~= 1.35x,
    # so a real letter's hole-to-cell-area ratio shrinks by
    # the same factor) -- confirmed empirically: raising
    # expand_frac without this compensation was a net
    # *regression* (29.8% -> 28.4% occupied-cell accuracy
    # on the em test set), while compensating it properly
    # was a real improvement (29.8% -> 34.7%). Retune
    # together if you change expand_frac.
    "max_hole_area_frac": 0.55,  # holes larger than this aren't a single letter
    "min_dominance_ratio": 4.5,  # best hole must be at least this many times bigger
    # than the runner-up (both already past
    # min_hole_area_frac) -- observed ~5-14x for real
    # letters vs ~1.1-3.4x for decorative premium-square
    # content (an icon or multi-word label), so this
    # catches false positives min_hole_area_frac alone
    # can't: under per-cell Otsu, some premium-square
    # holes are as big as or bigger than a real letter's
    # (observed up to ~0.11 of cell area), so no area
    # threshold alone separates them cleanly. Swept
    # empirically on the em test set: accuracy plateaus
    # at ratio >= ~4.5 (overall cell accuracy 72.0% ->
    # 82.2%, occupied-cell 34.8% -> 33.1% -- a small,
    # worthwhile trade for far fewer false positives).
    "digit_corner_frac": 0.35,  # bottom-right corner fraction of the *tile's own*
    # bounding box (not the whole, possibly padded/
    # shifted crop -- see extract_glyph()) reserved for
    # the score digit.
    "min_digit_area_frac": 0.003,  # digit holes are much smaller than letter holes
    "max_digit_area_frac": 0.04,
    "diacritic_max_area_frac": 0.0,  # DISABLED (0 = no hole ever qualifies): the
    # idea -- draw a small, horizontally-close secondary
    # hole onto the glyph mask as a diacritic mark (Ż's
    # dot, Ź/Ń/Ć's accent), instead of extract_glyph()
    # silently dropping every hole but the single biggest
    # -- is sound (real observed failure: Ź misread as Z)
    # but empirically net-harmful as implemented: swept
    # 0.12/0.15 (tight) through 0.35/0.3 (loose) on the
    # easy test set, and *every* setting, including the
    # tightest, reduced occupied-cell accuracy (63.9% ->
    # 61.5-62.6%) rather than improving it -- the false-
    # positive "diacritic" holes it draws (binarization
    # noise, stroke gaps) corrupt otherwise-correct OCR
    # input more than real diacritics get recovered. Left
    # in place (not deleted) in case a smarter shape-aware
    # trigger is worth revisiting later.
    "diacritic_max_offset_frac": 0.3,  # unused while diacritic_max_area_frac=0;
    # see its comment. Would be the max horizontal offset
    # (fraction of tile width) between a candidate
    # diacritic hole and the main hole if re-enabled.
    "premium_min_speck_frac": 0.02,  # much lower than min_hole_area_frac: the
    # row-clustering premium check (see extract_glyph())
    # needs to see individual small text-fragment holes
    # from a multi-line label, not just letter-sized ones.
    "premium_row_gap_frac": 0.02,  # vertical gap (as a fraction of tile height)
    # that starts a new row when clustering hole
    # y-centers -- separates one text line's holes from
    # the next. The original guess (0.25) made row-
    # clustering an accidental no-op: traced a concrete
    # false-positive double-word-premium cell and found its
    # 3 real text lines' hole-centers only ~0.03-0.11 apart,
    # far below that gap, so they all collapsed into one
    # band instead of 3 and nothing ever rejected it. Swept
    # properly on the easy test set (using read_board()'s
    # **param_overrides, not PARAM_DEFAULTS mutation -- see
    # its docstring for why that matters when hsv_config.json
    # already has a saved value for the key being swept):
    # 0.01-0.02 is a clean, real win over the 0.25 no-op
    # (overall cell accuracy 92.3% -> 92.8-92.9%, occupied-
    # cell accuracy flat within 1-2 cells either way) --
    # tighter than 0.01 or looser than 0.02 both give back
    # some of the gain.
    "premium_min_rows": 3,  # holes clustering into at least this many distinct
    # rows means multi-line label text, not a single
    # letter -- rejected regardless of min_dominance_ratio.
    # Every premium square's label was observed wrapping
    # across exactly 3 lines ("POTRÓJNA" / "PREMIA" /
    # "SŁOWNA" etc.) across every test photo checked.
    "ambiguity_margin": 0.08,  # candidates within this correlation score of the
    # best match are "close" -- only among those is the
    # point-value digit used to break the tie
    "expand_frac": 0.08,  # grid_reader.extract_cells()'s cell-crop margin: a
    # tile isn't always perfectly flush with its ideal
    # cell (residual keystone a single grid quad can't
    # correct, or the physical tile sitting slightly
    # off-centre), so crops grow by this fraction of a
    # cell's size per side to avoid clipping a shifted
    # or perspective-stretched tile. NOTE: growing the
    # crop shrinks a real letter's hole-to-cell-area
    # ratio too (same hole, bigger cell), so a large
    # expand_frac may need min_hole_area_frac lowered
    # to compensate -- tune them together.
    "binarize_open_kernel": 7,  # local_binarize()'s morphological-open kernel size,
    # strips thin grid-line/border remnants after Otsu.
    # No separate threshold parameter is needed -- that's
    # the point of using Otsu instead of a fixed cutoff.
}


def _params(overrides=None):
    """Merge hsv_config.json's saved "letter_params" preset with any
    explicit overrides, matching detect_board.py/grid_reader.py's pattern."""
    merged = load_params("letter_params", PARAM_DEFAULTS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def _load_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_glyphs(chars, size):
    font = _load_font(int(size * 0.8))
    refs = {}
    for ch in chars:
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        draw.text((size / 2, size / 2), ch, fill=255, font=font, anchor="mm")
        refs[ch] = np.array(img)
    return refs


def render_reference_glyphs(size=GLYPH_SIZE):
    """Render each alphabet letter centered on a size x size canvas, ink =
    255 on black = 0 (matching binarize_tiles()'s tile=bright convention).
    Returns {letter: np.ndarray}, meant to be computed once and reused."""
    return _render_glyphs(ALPHABET, size)


def render_digit_glyphs(size=GLYPH_SIZE):
    """Same as render_reference_glyphs(), for the tile's printed point-value
    digit (0-9) instead of the letter."""
    return _render_glyphs(DIGITS, size)


def local_binarize(cell_bgr, **param_overrides):
    """Per-cell Otsu threshold on this cell's own grayscale pixels, then
    the same morphological open used everywhere else to strip thin grid-
    line/border remnants. Tile = 255 (including a letter-ink hole, which
    stays 0), background = 0 -- same convention grid_reader.binarize_tiles()
    used, but computed fresh per cell instead of once globally.

    Empirically compared against the global-threshold approach on a known
    problem case (img13_e.jpg row 10, a whole row under-inked because that
    region of the photo is lit differently than the rest): per-cell Otsu
    recovered roughly 2-3x more of the true letter-ink area there. It's
    not a full fix for every cell -- on that same test, CLAHE local-
    contrast boosting on top of Otsu barely helped further, meaning some
    residual failures are genuine source blur (warp-interpolation
    softness / focus falloff), not a threshold-calibration problem. Also
    confirmed: this doesn't just help letters -- some premium-square icons
    (e.g. a "d" double-letter square, not just the word-premium label text
    fixed earlier) can produce a spurious hole comparable in size to a
    real letter's, so min_hole_area_frac may need retuning against these
    per-cell statistics rather than the old global-mask ones.
    """
    p = _params(param_overrides)
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    open_size = max(1, int(p["binarize_open_kernel"]))
    return cv2.morphologyEx(otsu, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)))


# Quadrant-symmetric premium-square layout (letter_mult, word_mult) per
# board position, indexed [min(r, 14-r)][min(c, 14-c)] -- copied from
# src/lib.rs's BONUS_TABLE (the real game engine's own scoring table), so
# this pipeline's premium-class grouping is guaranteed to match the
# actual physical board, not a separately-maintained guess.
_BONUS_TABLE = [
    [(1, 3), (1, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 3)],
    [(1, 1), (1, 2), (1, 1), (1, 1), (1, 1), (3, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (1, 2), (1, 1), (1, 1), (1, 1), (2, 1), (1, 1)],
    [(2, 1), (1, 1), (1, 1), (1, 2), (1, 1), (1, 1), (1, 1), (2, 1)],
    [(1, 1), (1, 1), (1, 1), (1, 1), (1, 2), (1, 1), (1, 1), (1, 1)],
    [(1, 1), (3, 1), (1, 1), (1, 1), (1, 1), (3, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1), (2, 1), (1, 1)],
    [(1, 3), (1, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 2)],
]


def premium_class(r, c, n=15):
    """(letter_mult, word_mult) for board position (r, c) -- (1, 1) for an
    unmarked/normal square. See _BONUS_TABLE."""
    return _BONUS_TABLE[min(r, n - 1 - r)][min(c, n - 1 - c)]


def board_saturation_reference(cells):
    """Per-premium-class median HSV saturation, for is_tile_present() to
    compare each cell against -- not one whole-board median (the original
    approach). A single global reference conflates every premium color
    together, but the classes print very differently: measured on empty
    ground-truth cells across 4 photos, "double word" squares (pale pink)
    have median saturation ~33 -- essentially indistinguishable from a
    real tile's ~11-35 -- while every other class sits at ~114-242. A
    global median (dominated by the much more common, much more
    saturated normal/triple-word squares) put that threshold nowhere near
    33, so an *empty* double-word square was reliably misread as tile-
    present. Grouping by the board's fixed, known layout (same rationale
    ocr/scrabble_reader/tile_detector.py already uses) fixes this without
    any new per-photo calibration: most cells of any one class are still
    empty at any point in a game, so that class's own median reliably
    reflects its printed color under this photo's lighting.

    Returns {(letter_mult, word_mult): saturation}. A class with too few
    cells to trust its own median (e.g. all 17 double-word squares
    occupied at once) falls back to the whole-board median -- and so does
    a class whose own median is itself too tile-like to use (double-word
    again: trusting its ~33 verbatim, before this floor existed, dropped
    is_tile_present()'s false-negative rate on *real* double-word tiles
    to 35%, since real tiles measure in that same ~11-35 range -- there's
    no threshold based on double-word's own color that can both accept
    real tiles on it and reject the bare square, so falling back to the
    (much higher, easily clearing real tiles) whole-board median is the
    better trade, even though it reintroduces a smaller false-positive
    rate on *empty* double-word squares specifically). Swept empirically:
    flooring any class's median at 50% of the whole-board median fixed
    double-word's false-negative rate (35% -> 8%) with zero effect on
    every other class (none of them were anywhere near that floor).
    """
    n = len(cells)
    sats = [[float(np.median(cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)[..., 1])) for cell in row] for row in cells]
    overall = float(np.median([s for row in sats for s in row]))
    by_class = {}
    for r in range(n):
        for c in range(n):
            by_class.setdefault(premium_class(r, c, n), []).append(sats[r][c])
    return {
        cls: max(float(np.median(vals)) if len(vals) >= 4 else overall, overall * 0.5) for cls, vals in by_class.items()
    }


def is_tile_present(cell_bgr, board_sat_ref, **param_overrides):
    """The real occupied/empty decision, checked on the raw *color* cell
    before any binarization: a physical tile (letter or blank) is wood/
    cream-colored (low HSV saturation) *relative to this photo's own
    board_sat_ref* (see board_saturation_reference() and tile_sat_ratio's
    PARAM_DEFAULTS comment for why a fixed cutoff doesn't work), while a
    bare board square -- normal or premium, regardless of its printed
    label's color -- is always noticeably more saturated than that. This
    replaces the old post-binarization white-fraction + dominant-hole-vs-
    label-text heuristic for telling a tile apart from a premium square:
    checking color up front rejects a premium square's printed label
    (short or long) before any hole-counting logic even runs."""
    p = _params(param_overrides)
    sat = float(np.median(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)[..., 1]))
    return bool(sat < board_sat_ref * p["tile_sat_ratio"])


def _tile_contours(cell):
    """(contours, hierarchy, outer_idx): the tile's own outer contour index
    (largest top-level contour) and the full hierarchy needed to find its
    holes. outer_idx is None if no top-level contour exists at all.

    cv2.RETR_TREE, not cv2.RETR_CCOMP: a letter's ink-hole can itself
    enclose a further "island" of tile-color -- O/Ó's counter, R's bowl,
    digit 0/6/8/9's counters -- which is a *third* nesting level (tile ->
    ink hole -> counter). RETR_CCOMP only tracks two levels and flattens
    anything deeper to top-level, so the counter silently lost its
    parent-child link to the hole it's actually inside; extract_glyph()
    then filled the whole hole solid with no carve-out for it, extracting
    O as a solid disc instead of a ring (confirmed: 0.4-0.5 white-pixel
    fraction on a 48x48 canvas, consistent with a filled disc, not a
    hollow one) -- which Tesseract/template matching, expecting a ring,
    couldn't read. RETR_TREE preserves the true nesting at any depth, so
    the counter now correctly appears as best_hole's own child and gets
    punched back out when drawing the mask (see extract_glyph()/
    extract_digit()). Every existing direct-child-of-outer_idx lookup in
    this module is unaffected: RETR_TREE agrees with RETR_CCOMP on level-1
    parentage, it just also keeps what CCOMP used to discard.
    """
    contours, hierarchy = cv2.findContours(cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return contours, None, None
    hierarchy = hierarchy[0]
    outer_idx, outer_area = None, 0
    for i, hier in enumerate(hierarchy):
        if hier[3] != -1:
            continue  # has a parent -> not a top-level (tile) contour
        area = cv2.contourArea(contours[i])
        if area > outer_area:
            outer_idx, outer_area = i, area
    return contours, hierarchy, outer_idx


def _draw_hole(mask, contours, hierarchy, idx):
    """Fill contour `idx` (a hole of the tile, ink=255) solid, then punch
    back out any of its own direct children -- an "island" nested inside
    it, e.g. O/Ó's counter, R's bowl, digit 0/6/8/9's counters -- so a
    ring-shaped hole renders as a ring, not a solid disc. Requires
    _tile_contours()'s RETR_TREE (see its docstring) for that nesting to
    be visible in the hierarchy at all."""
    cv2.drawContours(mask, contours, idx, 255, thickness=cv2.FILLED)
    for i, hier in enumerate(hierarchy):
        if hier[3] == idx:
            cv2.drawContours(mask, contours, i, 0, thickness=cv2.FILLED)


def _center_on_canvas(mask, size):
    """Crop `mask` (ink=255) to its bounding box and center it on a
    size x size canvas, preserving aspect ratio."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    box = int(size * 0.8)
    gh, gw = crop.shape
    s = box / max(gh, gw)
    nh, nw = max(1, int(gh * s)), max(1, int(gw * s))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def _digit_corner(tx, ty, tw, th, digit_corner_frac):
    """Bottom-right digit-corner region as absolute (x0, y0) bounds,
    relative to the *tile's own* bounding box (tx, ty, tw, th) -- not the
    whole cell crop, which may be padded well beyond the tile itself
    (grid_reader.extract_cells()'s expand_frac, needed to avoid clipping a
    parallax-shifted tile). Computing this against the crop's own w/h
    instead used to silently break as soon as expand_frac grew large
    enough that the tile stopped filling most of the crop -- the
    'corner' fraction then pointed at empty background, not the digit."""
    return tx + (1 - digit_corner_frac) * tw, ty + (1 - digit_corner_frac) * th


def extract_glyph(cell, size=GLYPH_SIZE, **param_overrides):
    """Isolate the letter-ink hole(s) within an occupied cell's tile blob
    and center them on a size x size canvas (ink=255), or None if no
    plausible letter is found -- which doubles as this pipeline's real
    occupied/empty decision (see classify_cell()), not just glyph
    isolation.

    cv2.RETR_CCOMP gives a two-level hierarchy: the tile's own outer
    contour (no parent) and its holes (parent = the tile). The largest
    hole is the letter; the small score digit printed in the bottom-right
    corner is excluded by position (see extract_digit(), which looks
    *only* there, and _digit_corner() for why that region is tile-
    relative), since a single-digit score and a narrow letter (e.g. I, Ł)
    can be similar in area. `min_hole_area_frac` is what actually rejects
    a premium square's printed label text here: real letters produce one
    dominant hole (observed ~6-10% of cell area) while label text
    fragments into several similarly-small holes (observed ~3-4%) with no
    dominant one, so a threshold between those ranges keeps letters and
    drops labels.

    Two more checks run once a dominant hole is found:
    - `premium_min_rows`: every hole in the tile (down to a much smaller
      `premium_min_speck_frac` floor, not just letter-sized ones) is
      grouped into vertical row-bands; a multi-line premium label (every
      one observed wraps across exactly 3 lines, e.g. "POTRÓJNA" /
      "PREMIA" / "SŁOWNA") produces several bands where a single letter
      produces one, so this rejects it directly, regardless of what the
      dominance-ratio check below would have said -- more direct than
      inferring "not one letter" from hole-size ratios alone.
    - Otherwise, any additional hole close to and much smaller than the
      dominant one is drawn onto the same mask as a diacritic mark (Ż's
      dot, Ź/Ń/Ć's accent are separate contours from the main stroke) --
      previously silently discarded since only the single biggest hole
      was ever kept, which plausibly explains real observed misreads
      (e.g. Ź read as Z).
    """
    p = _params(param_overrides)
    contours, hierarchy, outer_idx = _tile_contours(cell)
    if outer_idx is None:
        return None
    tx, ty, tw, th = cv2.boundingRect(contours[outer_idx])
    if tw == 0 or th == 0:
        return None
    corner_x, corner_y = _digit_corner(tx, ty, tw, th, p["digit_corner_frac"])

    cell_area = tw * th
    best_hole, best_area, second_area = None, 0, 0
    row_ys = []
    for i, hier in enumerate(hierarchy):
        if hier[3] != outer_idx:
            continue  # not a direct hole of the tile
        area = cv2.contourArea(contours[i])
        x, y, bw, bh = cv2.boundingRect(contours[i])
        cx, cy = x + bw / 2, y + bh / 2
        in_digit_corner = cx > corner_x and cy > corner_y
        if area >= p["premium_min_speck_frac"] * cell_area and not in_digit_corner:
            row_ys.append((cy - ty) / th)
        if area < p["min_hole_area_frac"] * cell_area or area > p["max_hole_area_frac"] * cell_area:
            continue
        if in_digit_corner:
            continue  # score digit corner
        if area > best_area:
            best_hole, best_area, second_area = i, area, best_area
        elif area > second_area:
            second_area = area
    if best_hole is None:
        return None

    # Multi-line label text: cluster hole y-centers into row-bands by a
    # gap threshold and count them, independent of any single hole's size.
    row_ys.sort()
    bands = 1 if row_ys else 0
    for prev, y in zip(row_ys, row_ys[1:]):
        if y - prev >= p["premium_row_gap_frac"]:
            bands += 1
    if bands >= p["premium_min_rows"]:
        return None

    # A letter produces one hole clearly bigger than any runner-up
    # (observed ~5-14x); decorative premium-square content -- an icon or
    # multi-word label -- produces holes closer in size to each other
    # (observed ~1.1-3.4x), so a second, comparably-sized hole nearby is a
    # sign this isn't a single letter even when its own area alone passed.
    if second_area and best_area / second_area < p["min_dominance_ratio"]:
        return None

    mask = np.zeros_like(cell)
    _draw_hole(mask, contours, hierarchy, best_hole)
    bx, _, bbw, _ = cv2.boundingRect(contours[best_hole])
    best_cx = bx + bbw / 2
    for i, hier in enumerate(hierarchy):
        if i == best_hole or hier[3] != outer_idx:
            continue
        area = cv2.contourArea(contours[i])
        if area <= 0 or area > p["diacritic_max_area_frac"] * best_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contours[i])
        cx, cy = x + bw / 2, y + bh / 2
        if cx > corner_x and cy > corner_y:
            continue  # score digit corner
        if abs(cx - best_cx) > p["diacritic_max_offset_frac"] * tw:
            continue  # too far sideways to be this letter's own diacritic
        _draw_hole(mask, contours, hierarchy, i)
    return _center_on_canvas(mask, size)


def extract_digit(cell, size=GLYPH_SIZE, **param_overrides):
    """Isolate the tile's printed point-value digit, the mirror image of
    extract_glyph(): looks *only* in the bottom-right corner region
    extract_glyph() excludes (see _digit_corner()), for a much smaller
    hole (a single digit is far smaller than a letter). Returns None if
    no plausible digit hole is found there."""
    p = _params(param_overrides)
    contours, hierarchy, outer_idx = _tile_contours(cell)
    if outer_idx is None:
        return None
    tx, ty, tw, th = cv2.boundingRect(contours[outer_idx])
    if tw == 0 or th == 0:
        return None
    corner_x, corner_y = _digit_corner(tx, ty, tw, th, p["digit_corner_frac"])

    cell_area = tw * th
    best_hole, best_area = None, 0
    for i, hier in enumerate(hierarchy):
        if hier[3] != outer_idx:
            continue
        area = cv2.contourArea(contours[i])
        if area < p["min_digit_area_frac"] * cell_area or area > p["max_digit_area_frac"] * cell_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contours[i])
        cx, cy = x + bw / 2, y + bh / 2
        if not (cx > corner_x and cy > corner_y):
            continue  # must be in the score-digit corner
        if area > best_area:
            best_hole, best_area = i, area
    if best_hole is None:
        return None

    mask = np.zeros_like(cell)
    _draw_hole(mask, contours, hierarchy, best_hole)
    return _center_on_canvas(mask, size)


def _match_all(glyph, refs):
    """Every (label, score) pair against `refs`, best first. score is a
    normalized correlation, roughly [-1, 1]."""
    glyph_f = glyph.astype(np.float32)
    scores = [
        (label, float(cv2.matchTemplate(glyph_f, ref.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0]))
        for label, ref in refs.items()
    ]
    scores.sort(key=lambda t: -t[1])
    return scores


def classify_glyph(glyph, refs):
    """Best-matching label for `glyph` against reference glyphs (letters or
    digits). Returns (label, score)."""
    return _match_all(glyph, refs)[0]


def _tesseract_read(padded, psm):
    text = pytesseract.image_to_string(
        padded, lang="pol", config=f"--psm {psm} --oem 3 -c tessedit_char_whitelist={ALPHABET}"
    ).strip()
    return text if len(text) == 1 and text in ALPHABET else None


def ocr_classify_glyph(glyph):
    """Read extract_glyph()'s isolated ink-hole mask (ink=255 on black=0)
    via Tesseract OCR, restricted to a single character from the known
    tile alphabet (`-c tessedit_char_whitelist`, `--psm 10` = "treat image
    as a single character"). Tesseract expects dark text on a light
    background, the opposite of the mask's convention, so it's inverted
    first; a border is added since Tesseract expects some quiet margin
    around the glyph. Returns the recognized letter, or None if Tesseract
    isn't installed, or it didn't return exactly one whitelisted character
    -- garbage, multi-character, and empty reads are all treated as "no
    answer" and left to fall back to template matching, rather than risk
    trusting a bad OCR guess.

    `--psm 10` combined with `tessedit_char_whitelist` has a real, narrow
    bug in this Tesseract build: confirmed directly that it returns empty
    on a cleanly-shaped, unambiguous "O" ring that the *exact same*
    whitelist reads correctly at 92% confidence under `--psm 8` ("single
    word"), and that `--psm 10` reads it fine with the whitelist removed
    entirely -- so it's specifically the whitelist/psm-10 combination
    that's broken there, not the image. Tempting fix: fall back to
    `--psm 8` whenever `--psm 10` returns nothing. Tried it, and it's a
    net *regression* through the full pipeline -- validated on the same
    44-image easy set: psm-10-only scores 79.3% occupied-cell accuracy,
    the psm-10-then-8 cascade scores 76.6%, a real loss of ~48 cells,
    despite an isolated OCR-only sample (bypassing classify_cell()'s
    other logic) showing the opposite (77.5% -> 82.0%). The reason: when
    `--psm 10` returns nothing, classify_cell() falls back to template
    matching + digit-value disambiguation, which turns out to be *more*
    reliable than `--psm 8`'s guess on the specific cells where `--psm
    10` is genuinely unsure -- adding the cascade pre-empts that better
    fallback with a worse one. So this stays `--psm 10` only; the O bug
    is real but doesn't have a cheap fix here that doesn't cost more
    elsewhere.

    By a wide empirical margin this is still the strongest signal
    available here: on the *same* isolated glyphs, across the full em
    test set, Tesseract got 851/1152 (73.9%) right vs. template
    matching's 489/1152 (42.4%) -- a trained OCR engine generalizes to
    real font/blur/rotation variation in a way correlation against one
    rendered font glyph per letter can't. Requires `brew install
    tesseract` plus Polish trained data (`pol.traineddata` in
    tesseract's tessdata dir -- the Homebrew formula only ships "eng",
    "osd", "snum") and `pip install pytesseract`.
    """
    if not _HAS_TESSERACT:
        return None
    inv = 255 - glyph
    padded = cv2.copyMakeBorder(inv, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return _tesseract_read(padded, 10)


def classify_cell(cell_bgr, refs, board_sat_ref, digit_refs=None, **param_overrides):
    """'-' for empty (score 1.0) or the best-matching letter for occupied
    cells. `cell_bgr` is a *color* cell crop (e.g. from
    grid_reader.extract_cells(grid_warp, ...)). `board_sat_ref` is
    board_saturation_reference()'s output for the *whole board this cell
    came from*, computed once per image and passed in rather than
    recomputed per cell.

    is_tile_present() is a cheap pre-filter, checked on the raw color cell
    before any binarization (see its docstring): a bare board square --
    normal or premium, including its printed label -- is rejected here
    without ever reaching the binarize/hole-search step below. It's
    deliberately tuned loose (favors recall, i.e. false positives over
    ever missing a real tile), so a cell passing it isn't yet certain to
    hold a tile -- "occupied" is only really decided by extract_glyph()
    finding a dominant ink hole once binarized (local_binarize()); a cell
    that passes the color pre-filter but has no dominant hole is reported
    as empty rather than as an unreadable occupied cell (tried reporting
    these as an "unresolved tile" marker instead, to also flag blank
    tiles -- regressed accuracy badly, since per-cell Otsu reliably
    manufactures letter-plausible-looking holes on ordinary board squares
    that slip past the deliberately loose color pre-filter, and no
    ground-truth photo has a blank tile in play to weigh that against
    anyway).

    Once a glyph is isolated, ocr_classify_glyph() (Tesseract OCR) is tried
    first -- see its docstring for why it's the default over template
    matching (a large, empirically validated accuracy win). Template
    matching is only a fallback for when OCR is unavailable or doesn't
    return a confident single-character answer.

    When `digit_refs` is given (render_digit_glyphs()'s output) and the
    top letter candidates are within `ambiguity_margin` of each other,
    the tile's own printed point-value digit (LETTER_POINTS) is used to
    pick among them -- e.g. L (2 points) vs Ł (3 points) are easy to
    confuse by shape alone, but the printed digit disambiguates exactly.
    Only intervenes when precisely one close candidate's point value
    matches the read digit, so a misread digit can't override a
    confident, unambiguous letter match. This tie-break only runs on the
    template-matching fallback path -- OCR returns a single answer, not a
    ranked list of candidates to disambiguate between.
    """
    if not is_tile_present(cell_bgr, board_sat_ref, **param_overrides):
        return "-", 1.0
    cell = local_binarize(cell_bgr, **param_overrides)
    glyph = extract_glyph(cell, **param_overrides)
    if glyph is None:
        return "-", 1.0

    ocr_letter = ocr_classify_glyph(glyph)
    if ocr_letter is not None:
        return ocr_letter, 1.0

    ranked = _match_all(glyph, refs)
    best_letter, best_score = ranked[0]

    if digit_refs:
        p = _params(param_overrides)
        close = [(letter, score) for letter, score in ranked if best_score - score < p["ambiguity_margin"]]
        if len(close) > 1:
            digit_glyph = extract_digit(cell, **param_overrides)
            if digit_glyph is not None:
                digit_char, _ = classify_glyph(digit_glyph, digit_refs)
                digit = int(digit_char)
                matches = [letter for letter, _ in close if LETTER_POINTS.get(letter) == digit]
                if len(matches) == 1:
                    return matches[0], dict(close)[matches[0]]

    return best_letter, best_score


def classify_board(cells, refs, digit_refs, sat_ref_by_class, max_workers=16, **param_overrides):
    """classify_cell() for every cell of an n x n `cells` grid, in
    parallel via a thread pool. classify_cell() mostly waits on a
    Tesseract subprocess per occupied cell (ocr_classify_glyph()) rather
    than doing CPU-bound work in Python itself -- subprocess calls
    release the GIL while waiting, so plain threads (not multiprocessing,
    and not a GPU -- there's no GPU-acceleratable compute step here, the
    cost is per-cell subprocess-spawn overhead) give a large real
    speedup for very little complexity. `sat_ref_by_class` is board_
    saturation_reference()'s output; each cell's own reference is looked
    up by its premium_class(). Returns (board, scores), both n x n
    nested lists, matching classify_cell()'s (letter, score) convention
    per cell."""
    n = len(cells)
    positions = [(r, c) for r in range(n) for c in range(len(cells[r]))]

    def _classify(pos):
        r, c = pos
        board_sat_ref = sat_ref_by_class[premium_class(r, c, n)]
        return classify_cell(cells[r][c], refs, board_sat_ref, digit_refs, **param_overrides)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_classify, positions))

    board = [["-"] * len(cells[r]) for r in range(n)]
    scores = [[1.0] * len(cells[r]) for r in range(n)]
    for (r, c), (letter, score) in zip(positions, results):
        board[r][c] = letter
        scores[r][c] = score
    return board, scores
