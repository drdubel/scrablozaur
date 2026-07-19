"""Turn a detected tile's raw patch into a clean, normalised glyph image.

Steps per tile:
1.  (optional) rotate the patch by k*90 degrees -- used by read_letters.py's
    rotation search;
2.  locate the tile inside the (expanded) cell patch by its brightness,
    deskew it with the minimum-area-rectangle angle (tiles are often placed
    a few degrees crooked);
3.  flatten illumination and boost local contrast;
4.  binarise the dark ink and select the components that belong to the
    letter: the main body, plus diacritics (dot of Z, acute of O, ogonek of
    A/E), while *discarding the score digit* printed in the bottom-right
    corner and bevel shadows along the tile border;
5.  centre the glyph in a square canvas with margin, preserving aspect
    ratio, and emit both a grayscale crop and a clean binary mask at
    GLYPH_SIZE (64px).

The output feeds letter_classifier.py's CNN and template matcher. Ported
from ocr/scrabble_reader/tile_normalizer.py, adapted to this project's
house style: a namedtuple instead of a dataclass, no Config object.
"""

from collections import namedtuple

import cv2
import numpy as np

GLYPH_SIZE = 64  # bound to the trained CNN's input layer and the pre-harvested
# template PNGs -- not a meaningful tunable, same category as premium_layout.py's GRID.

# Fraction of the (expanded, EXPAND_FRACTION=0.12) cell patch that the tile
# itself is expected to span -- 1 / (1 + 2*0.12).
EXPECTED_TILE_FRAC = 1.0 / 1.24

NormalizedGlyph = namedtuple(
    "NormalizedGlyph",
    ["gray", "mask", "tile_gray", "has_glyph", "ink_fraction", "quality", "digit_score", "digit_mask", "digit_gray"],
)
DIGIT_SIZE = 32  # canvas size for the composed score-digit crop -- digits are simpler shapes, need less resolution


def flatten_illumination(gray, sigma_frac=0.25):
    """Remove smooth illumination gradients (shadows, vignetting) by
    dividing by a heavily blurred copy. Output is uint8, mean ~128."""
    g = gray.astype(np.float32) + 1.0
    h, w = gray.shape[:2]
    small = cv2.resize(g, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    sigma = max(2.0, sigma_frac * min(small.shape[:2]))
    bg = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    flat = g / bg
    flat = flat / max(np.mean(flat), 1e-6) * 128.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def _center_crop(patch_bgr, expected_frac):
    """Geometric tile crop: the mesh placed the tile at the patch centre,
    so take the expected tile extent minus the bevelled edge."""
    h, w = patch_bgr.shape[:2]
    side = expected_frac * min(h, w) * 0.84  # 8% bevel shrink per side
    x0 = int((w - side) / 2)
    y0 = int((h - side) / 2)
    return patch_bgr[y0 : y0 + int(side), x0 : x0 + int(side)]


def _locate_tile(patch_bgr, expected_frac=0.8):
    """Find the bright tile inside the expanded cell patch and deskew it.

    The brightness blob is only trusted when it is tile-sized and centred:
    on densely tiled boards the neighbours are equally bright, the blob
    spans the whole patch and its minAreaRect drifts onto a neighbouring
    tile -- in that case the mesh-aligned geometric centre crop is the
    reliable choice (tiles sit in the grid, so deskew is minor anyway).
    """
    h, w = patch_bgr.shape[:2]
    exp_side = expected_frac * min(h, w)
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    _, bright = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)

    best, best_area = None, 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        # The tile fills most of the cell; require a substantial region.
        if a > best_area and a > 0.25 * h * w:
            best, best_area = i, a
    if best is None:
        return _center_crop(patch_bgr, expected_frac), 0.0

    ys_pts, xs_pts = np.where(labels == best)
    # Every 4th pixel is ample for a bounding rectangle of a blob this size.
    pts = np.column_stack([xs_pts[::4], ys_pts[::4]]).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    if rw < rh:
        rw, rh, ang = rh, rw, ang - 90.0
    while ang > 45:
        ang -= 90
    while ang < -45:
        ang += 90
    if abs(ang) < 1.0 or abs(ang) > 15.0:
        ang = 0.0  # not worth rotating / clearly a bogus estimate

    side = float(min(rw, rh))
    # Off-centre placement alone isn't evidence of a fused neighbour blob --
    # a fused blob is *oversized* (this tile plus part of another), so the
    # size bounds above are what actually guards against that; a tile that
    # merely sits a bit off from the assumed cell centre (grid/mesh slop)
    # is still a genuine, trustworthy single-tile detection, and rejecting
    # it here just forces a blind centre_crop that chops the real letter.
    trustworthy = (
        0.70 * exp_side <= side
        and max(rw, rh) <= 1.25 * exp_side
        and abs(cx - w / 2) < 0.30 * w
        and abs(cy - h / 2) < 0.30 * h
    )
    if not trustworthy:
        return _center_crop(patch_bgr, expected_frac), 0.0

    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(patch_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Crop the tile interior; shrink to cut the bevelled edge and its shadow.
    shrink = 0.10
    half = side * (0.5 - shrink)
    # A diacritic (acute, dot) sits right at the top of the letter, so a
    # detected `side` that's a touch smaller than the true tile (the accent
    # isn't part of the bright-tile blob the size estimate comes from) costs
    # the top edge disproportionately -- give it a little extra headroom
    # that the other three edges, which only ever risk bevel shadow, don't
    # need.
    top_extra = side * 0.04
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half - top_extra)), int(min(h, cy + half))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return _center_crop(patch_bgr, expected_frac), 0.0
    return rot[y0:y1, x0:x1], float(ang)


# Fraction of the tile crop, measured from its own top-left, where the
# printed point-value digit sits -- a fixed corner of the physical tile's
# own layout, the same regardless of which letter is printed. Unlike the
# old relative-to-main-letter heuristic, searching this fixed window does
# not depend on having already correctly identified the main letter --
# which matters because on a bad crop the main-letter search can itself
# lock onto the wrong blob (see _select_glyph_components), and a
# relative-position digit test would then misfire right along with it.
DIGIT_WINDOW_X0 = 0.64
DIGIT_WINDOW_Y0 = 0.46
# A digit candidate must have at least this fraction of its own ink inside
# that window -- a genuine digit is small and sits entirely in the corner,
# while a letter stroke that merely reaches toward it (Z's diagonal, L's
# foot) has most of its ink well outside.
DIGIT_CONTAINMENT = 0.80


def _extract_digit(ink):
    """Isolate the tile's own printed point-value digit from its fixed
    bottom-right window, independent of letter/diacritic selection.

    Digits are always a single continuous shape (unlike letters, which
    can carry a detached diacritic), so no bridging or "which blob is the
    real glyph" reasoning is needed here: any component mostly contained
    in the window *is* the digit.

    Returns (digit_mask, digit_score, ink_without_digit) -- digit_mask/
    digit_score are None/0.0 when no plausible digit-shaped component sits
    in the window; ink_without_digit is `ink` with that component's pixels
    (if any) removed, so it can't leak into or get mistaken for letter ink.
    """
    h, w = ink.shape
    wx0, wy0 = DIGIT_WINDOW_X0 * w, DIGIT_WINDOW_Y0 * h
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    if n <= 1:
        return None, 0.0, ink

    best_label, best_frac, best_area = None, 0.0, 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        # Too small to be real ink, or too big to plausibly be one digit
        # (a single digit is a small mark; anything larger is letter ink
        # that happens to reach into the window -- Z's diagonal, A's own
        # leg -- and must be left as letter ink, not stolen as "digit").
        if a < 0.0015 * h * w or a > 0.045 * h * w:
            continue
        ys, xs = np.where(labels == i)
        frac = float(np.count_nonzero((xs >= wx0) & (ys >= wy0))) / len(xs)
        if frac < DIGIT_CONTAINMENT:
            continue
        if a > best_area:
            best_label, best_frac, best_area = i, frac, a
    if best_label is None:
        return None, 0.0, ink

    digit_mask = np.zeros_like(ink)
    digit_mask[labels == best_label] = 255
    remaining = ink.copy()
    remaining[labels == best_label] = 0
    return digit_mask, float(best_frac), remaining


def _select_glyph_components(ink, gray):
    """Keep letter body + diacritics; drop the score digit and border junk.

    Diacritics sit above (dot/acute) or below within the letter's columns
    (ogonek), bounded in area *and* width/height so a long, narrow bevel-
    shadow line (small area, but spanning most of the tile) can't pass as
    one. The score digit is handled separately by _extract_digit() before
    this runs.

    Returns (mask, digit_score, digit_mask) -- digit_score/digit_mask are
    just threaded through from _extract_digit() for read_letters.py's
    point-value disambiguation.
    """
    h, w = ink.shape
    digit_mask, digit_score, ink = _extract_digit(ink)

    # Bridge accents (acute of O/C/Z, dot of Z) to their base letter so they
    # cannot be lost as separate small components; the kernel is tall and
    # narrow, so it only bridges vertically-stacked marks, never sideways.
    bridged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(5, h // 16))))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bridged)
    if n <= 1:
        return None, digit_score, digit_mask

    def is_ring(i):
        """Tile bevel edges binarise into a thin frame around the glyph:
        a huge bounding box with almost no filled area."""
        x, y, bw, bh, a = stats[i]
        return bw > 0.72 * w and bh > 0.72 * h and a < 0.30 * bw * bh

    def is_viable_main(i):
        x, y, bw, bh, a = stats[i]
        # Border-hugging elongated components are bevel shadows.
        touches = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
        if touches >= 2 and (bw > 0.9 * w or bh > 0.9 * h):
            return False
        if is_ring(i):
            return False
        if a < 0.01 * h * w or a > 0.55 * h * w:
            return False
        # A printed letter is always tall enough to read -- a short, wide
        # strip (a bevel/shadow line running along one edge) can otherwise
        # out-area and even out-darken the real letter (a shadow can be
        # genuinely dark too, not just faint) and win main-candidacy.
        if bh < 0.23 * h:
            return False
        cx, cy = cents[i]
        return 0.13 * w < cx < 0.87 * w and 0.08 * h < cy < 0.92 * h

    candidates = [i for i in range(1, n) if is_viable_main(i)]
    if not candidates:
        return None, digit_score, digit_mask
    # A shadow/bevel artefact can pass every size-and-position filter above
    # (it commonly does -- see glyph_normalizer's design notes) and even
    # out-*area* the real letter, but it is only ever mildly darker than
    # the paper tone; genuine ink is dark. So among candidates plausibly
    # in contention by size, prefer the darkest instead of the largest.
    max_area = max(stats[i, cv2.CC_STAT_AREA] for i in candidates)
    contenders = [i for i in candidates if stats[i, cv2.CC_STAT_AREA] >= 0.35 * max_area]
    main = min(contenders, key=lambda i: float(gray[labels == i].mean()))

    mx, my, mw, mh, ma = stats[main]
    keep = np.zeros_like(ink)
    keep[labels == main] = 255
    for i in range(1, n):
        if i == main:
            continue
        x, y, bw, bh, a = stats[i]
        cx, cy = cents[i]
        if a < 0.02 * ma or a > 0.45 * ma or is_ring(i):
            continue
        overlaps_cols = (cx > mx - 0.15 * mw) and (cx < mx + 1.15 * mw)
        above = cy < my + 0.15 * mh
        below = cy > my + 0.85 * mh
        # A genuine diacritic (acute, dot, ogonek) is a small mark, bounded
        # in *every* dimension, not just area -- a long, narrow bevel-shadow
        # line can have a small area yet span most of the tile's width or
        # height, so area alone can't rule it out the way a width/height
        # cap does.
        small_enough = bw < 0.55 * mw and bh < 0.35 * mh
        diacritic = small_enough and overlaps_cols and (above or (below and bw < 0.6 * mw))
        if diacritic:
            keep[labels == i] = 255
    # Selection ran on the accent-bridged image; restrict the final mask to
    # actual ink so the bridging never thickens the glyph itself.
    keep = cv2.bitwise_and(keep, ink)
    if keep.sum() == 0:
        return None, digit_score, digit_mask
    return keep, digit_score, digit_mask


def _compose(gray_tile, mask, out_size, margin=0.12):
    """Centre the glyph bbox in a square canvas, preserving aspect ratio."""
    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop_m = (255 - mask)[y0:y1, x0:x1]  # ink black on white
    crop_g = gray_tile[y0:y1, x0:x1].copy()
    crop_g[mask[y0:y1, x0:x1] == 0] = 255  # erase background clutter

    box = int(out_size * (1 - 2 * margin))
    h, w = crop_g.shape
    s = box / max(h, w)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    rg = cv2.resize(crop_g, (nw, nh), interpolation=interp)
    rm = cv2.resize(crop_m, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas_g = np.full((out_size, out_size), 255, np.uint8)
    canvas_m = np.full((out_size, out_size), 255, np.uint8)
    ox, oy = (out_size - nw) // 2, (out_size - nh) // 2
    canvas_g[oy : oy + nh, ox : ox + nw] = rg
    canvas_m[oy : oy + nh, ox : ox + nw] = rm
    return canvas_g, 255 - canvas_m  # mask: ink=255


def _binarise_and_select(tile_bgr, variant):
    """Grayscale prep + ink binarisation + component selection for one
    candidate tile crop. Split out of normalize() so a failed extraction
    can be retried against a different crop (see normalize())."""
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gray = flatten_illumination(gray, sigma_frac=0.5)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)

    h, w = gray.shape
    if variant == 0:
        block = max(15, ((min(h, w) // 2) | 1))
        ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10)
    elif variant == 1:
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        thr = np.percentile(gray, 25)
        ink = ((gray < thr).astype(np.uint8)) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    keep, digit_score, digit_ink = _select_glyph_components(ink, gray)
    return gray, keep, digit_score, digit_ink


def normalize(tile_patch_bgr, rotation_k=0, variant=0):
    """`variant` selects alternative binarisation for reprocessing weak reads."""
    patch = np.rot90(tile_patch_bgr, -rotation_k).copy() if rotation_k else tile_patch_bgr

    tile, _ = _locate_tile(patch, EXPECTED_TILE_FRAC)
    gray, keep, digit_score, digit_ink = _binarise_and_select(tile, variant)

    if keep is None or keep.sum() == 0:
        # _locate_tile's brightness blob can fuse with a neighbouring tile
        # on densely packed boards (their corners are equally bright, and
        # MORPH_CLOSE bridges the thin gap between them), producing a crop
        # that omits part of the letter even though its own size/centring
        # narrowly read as trustworthy. Retry once against the mesh-aligned
        # geometric centre crop before giving up on this tile.
        center = _center_crop(patch, EXPECTED_TILE_FRAC)
        gray2, keep2, digit_score2, digit_ink2 = _binarise_and_select(center, variant)
        if keep2 is not None and keep2.sum() > 0:
            gray, keep, digit_score, digit_ink = gray2, keep2, digit_score2, digit_ink2

    digit_gray = digit_mask = None
    if digit_ink is not None:
        digit_gray, digit_mask = _compose(gray, digit_ink, DIGIT_SIZE)

    if keep is None or keep.sum() == 0:
        blank = np.full((GLYPH_SIZE, GLYPH_SIZE), 255, np.uint8)
        return NormalizedGlyph(
            gray=blank,
            mask=np.zeros_like(blank),
            tile_gray=gray,
            has_glyph=False,
            ink_fraction=0.0,
            quality=0.0,
            digit_score=digit_score,
            digit_mask=digit_mask,
            digit_gray=digit_gray,
        )

    glyph_gray, glyph_mask = _compose(gray, keep, GLYPH_SIZE)
    ink_frac = float((keep > 0).mean())
    # Quality: penalise extreme ink fractions (speckle or blobs).
    quality = float(np.clip(1.0 - abs(ink_frac - 0.16) / 0.16, 0.05, 1.0))
    return NormalizedGlyph(
        gray=glyph_gray,
        mask=glyph_mask,
        tile_gray=gray,
        has_glyph=True,
        ink_fraction=ink_frac,
        quality=quality,
        digit_score=digit_score,
        digit_mask=digit_mask,
        digit_gray=digit_gray,
    )
