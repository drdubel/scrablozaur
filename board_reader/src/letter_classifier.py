"""Stage 3: per-cell occupancy check and letter classification.

A lightweight, self-contained template-matching classifier -- deliberately
simpler than ocr/scrabble_reader/'s CNN + template + OCR-fallback fusion
(that package already solves this problem more robustly; board_reader
builds its own independent version instead of wiring into it, per an
explicit choice to keep this stage simple and standalone).

classify_cell() takes a *color* cell crop (a slice of the grid-aligned
warp, e.g. grid_reader.extract_cells(grid_warp, ...)) and binarizes it
itself, per cell, via local_binarize() -- deliberately NOT a slice of
grid_reader.binarize_tiles()'s output, which applies one global HSV
threshold across the whole board. A single global cutoff doesn't fit every
region of a real photo (confirmed: it under-inked an entire row on one
test photo whose lighting varied across the frame), and no fixed preset
would either, since the same problem recurs *within* one photo, not just
across photos. Otsu's method recomputes its own cutoff from each cell's
own pixel histogram, so it adapts automatically with no tunable threshold
and no manual preset selection -- see local_binarize()'s docstring for
the empirical comparison against the global-threshold approach it replaced.
Reference glyphs are rendered once (render_reference_glyphs()) / digit
glyphs (render_digit_glyphs()) and passed into classify_cell() rather than
re-rendered per call.

Each Polish Scrabble tile also prints its point value in the bottom-right
corner, and the point scale is a fixed, known mapping (LETTER_POINTS,
matching src/lib.rs's calculate_word_points -- the actual game engine's
scoring table, so this is guaranteed consistent with real gameplay). When
the top template-match candidates for a glyph are close (genuinely
ambiguous, e.g. L vs Ł), classify_cell() reads the tile's own point digit
and prefers whichever close candidate's point value actually matches it --
a deterministic tie-breaker the classifier wouldn't otherwise have.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hsv_config import load_params

# Polish Scrabble tile letters -- no Q/V/X on the physical tile set (see
# test/out/board*.txt ground truth and the photos in test/in/ for the
# actual alphabet in use).
ALPHABET = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ"
DIGITS = "0123456789"

# Point value printed on every tile of that letter, matching src/lib.rs's
# Board::calculate_word_points -- the game engine's own scoring table, so
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
    "min_white_frac": 0.05,      # cheap pre-filter: skip near-blank cells entirely
    "min_hole_area_frac": 0.033, # the *real* occupancy signal (see extract_glyph):
                                  # a premium square's printed white label ("PODWÓJNA
                                  # PREMIA SŁOWNA" etc.) is a large solid white blob
                                  # too, easily >min_white_frac, but its ink holes are
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
    "digit_corner_frac": 0.35,   # bottom-right corner fraction reserved for the score digit
    "min_digit_area_frac": 0.003,  # digit holes are much smaller than letter holes
    "max_digit_area_frac": 0.04,
    "ambiguity_margin": 0.08,    # candidates within this correlation score of the
                                  # best match are "close" -- only among those is the
                                  # point-value digit used to break the tie
    "expand_frac": 0.08,         # grid_reader.extract_cells()'s cell-crop margin: a
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
    "binarize_open_kernel": 7,   # local_binarize()'s morphological-open kernel size,
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


def is_occupied(cell, **param_overrides):
    """Cheap pre-filter only: skip cells with negligible white content
    before doing any contour work. This is *not* the real occupied/empty
    decision -- a premium square's printed white label ("PODWÓJNA PREMIA
    SŁOWNA" etc.) is a large solid white blob too, easily clearing a
    white-fraction threshold, so raw fraction alone can't tell it apart
    from an actual tile. extract_glyph() finding (or not finding) a
    dominant ink hole is the real signal; see its docstring."""
    p = _params(param_overrides)
    return bool((cell > 0).mean() >= p["min_white_frac"])


def _tile_contours(cell):
    """(contours, hierarchy, outer_idx): the tile's own outer contour index
    (largest top-level contour) and the full hierarchy needed to find its
    holes. outer_idx is None if no top-level contour exists at all."""
    contours, hierarchy = cv2.findContours(cell, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
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
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def extract_glyph(cell, size=GLYPH_SIZE, **param_overrides):
    """Isolate the letter-ink hole within an occupied cell's tile blob and
    center it on a size x size canvas (ink=255), or None if no plausible
    ink hole is found -- which doubles as this pipeline's real occupied/
    empty decision (see classify_cell()), not just glyph isolation.

    cv2.RETR_CCOMP gives a two-level hierarchy: the tile's own outer
    contour (no parent) and its holes (parent = the tile). The largest
    hole is the letter; the small score digit printed in the bottom-right
    corner is excluded by position (see extract_digit(), which looks
    *only* there), since a single-digit score and a narrow letter (e.g.
    I, Ł) can be similar in area. `min_hole_area_frac` is what actually
    rejects a premium square's printed label text here: real letters
    produce one dominant hole (observed ~6-10% of cell area) while label
    text fragments into several similarly-small holes (observed ~3-4%)
    with no dominant one, so a threshold between those ranges keeps
    letters and drops labels.
    """
    p = _params(param_overrides)
    h, w = cell.shape[:2]
    contours, hierarchy, outer_idx = _tile_contours(cell)
    if outer_idx is None:
        return None

    cell_area = h * w
    best_hole, best_area, second_area = None, 0, 0
    for i, hier in enumerate(hierarchy):
        if hier[3] != outer_idx:
            continue  # not a direct hole of the tile
        area = cv2.contourArea(contours[i])
        if area < p["min_hole_area_frac"] * cell_area or area > p["max_hole_area_frac"] * cell_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contours[i])
        cx, cy = x + bw / 2, y + bh / 2
        if cx > (1 - p["digit_corner_frac"]) * w and cy > (1 - p["digit_corner_frac"]) * h:
            continue  # score digit corner
        if area > best_area:
            best_hole, best_area, second_area = i, area, best_area
        elif area > second_area:
            second_area = area
    if best_hole is None:
        return None
    # A letter produces one hole clearly bigger than any runner-up
    # (observed ~5-14x); decorative premium-square content -- an icon or
    # multi-word label -- produces holes closer in size to each other
    # (observed ~1.1-3.4x), so a second, comparably-sized hole nearby is a
    # sign this isn't a single letter even when its own area alone passed.
    if second_area and best_area / second_area < p["min_dominance_ratio"]:
        return None

    mask = np.zeros_like(cell)
    cv2.drawContours(mask, contours, best_hole, 255, thickness=cv2.FILLED)
    return _center_on_canvas(mask, size)


def extract_digit(cell, size=GLYPH_SIZE, **param_overrides):
    """Isolate the tile's printed point-value digit, the mirror image of
    extract_glyph(): looks *only* in the bottom-right corner region
    extract_glyph() excludes, for a much smaller hole (a single digit is
    far smaller than a letter). Returns None if no plausible digit hole
    is found there."""
    p = _params(param_overrides)
    h, w = cell.shape[:2]
    contours, hierarchy, outer_idx = _tile_contours(cell)
    if outer_idx is None:
        return None

    cell_area = h * w
    best_hole, best_area = None, 0
    for i, hier in enumerate(hierarchy):
        if hier[3] != outer_idx:
            continue
        area = cv2.contourArea(contours[i])
        if area < p["min_digit_area_frac"] * cell_area or area > p["max_digit_area_frac"] * cell_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contours[i])
        cx, cy = x + bw / 2, y + bh / 2
        if not (cx > (1 - p["digit_corner_frac"]) * w and cy > (1 - p["digit_corner_frac"]) * h):
            continue  # must be in the score-digit corner
        if area > best_area:
            best_hole, best_area = i, area
    if best_hole is None:
        return None

    mask = np.zeros_like(cell)
    cv2.drawContours(mask, contours, best_hole, 255, thickness=cv2.FILLED)
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


def classify_cell(cell_bgr, refs, digit_refs=None, **param_overrides):
    """'-' for empty (score 1.0) or the best-matching letter for occupied
    cells. `cell_bgr` is a *color* cell crop (e.g. from
    grid_reader.extract_cells(grid_warp, ...)) -- binarized here, once,
    per cell via local_binarize() rather than assuming a pre-binarized
    input; see that function's and the module docstring for why.

    "Occupied" means extract_glyph() found a dominant ink hole --
    is_occupied()'s white-fraction check is only a cheap pre-filter, not
    the real decision (see both docstrings): a bright cell with no
    dominant hole (e.g. a premium square's printed label) is reported as
    empty rather than as an unreadable occupied cell, since raw brightness
    can't distinguish the two here.

    When `digit_refs` is given (render_digit_glyphs()'s output) and the
    top letter candidates are within `ambiguity_margin` of each other,
    the tile's own printed point-value digit (LETTER_POINTS) is used to
    pick among them -- e.g. L (2 points) vs Ł (3 points) are easy to
    confuse by shape alone, but the printed digit disambiguates exactly.
    Only intervenes when precisely one close candidate's point value
    matches the read digit, so a misread digit can't override a
    confident, unambiguous letter match.
    """
    cell = local_binarize(cell_bgr, **param_overrides)
    if not is_occupied(cell, **param_overrides):
        return "-", 1.0
    glyph = extract_glyph(cell, **param_overrides)
    if glyph is None:
        return "-", 1.0

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
