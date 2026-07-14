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


def _centre_crop(patch_bgr, expected_frac):
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
        return _centre_crop(patch_bgr, expected_frac), 0.0

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
    trustworthy = (
        0.70 * exp_side <= side
        and max(rw, rh) <= 1.25 * exp_side
        and abs(cx - w / 2) < 0.15 * w
        and abs(cy - h / 2) < 0.15 * h
    )
    if not trustworthy:
        return _centre_crop(patch_bgr, expected_frac), 0.0

    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(patch_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Crop the tile interior; shrink to cut the bevelled edge and its shadow.
    shrink = 0.10
    half = side * (0.5 - shrink)
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return _centre_crop(patch_bgr, expected_frac), 0.0
    return rot[y0:y1, x0:x1], float(ang)


def _select_glyph_components(ink):
    """Keep letter body + diacritics; drop the score digit and border junk.

    The score digit is small and sits in the bottom-right corner, outside
    the main glyph's column span; diacritics sit above (dot/acute) or below
    within the letter's columns (ogonek).

    Returns (mask, digit_score, digit_mask) where digit_score is evidence
    that a score digit was found bottom-right of the letter, and
    digit_mask is that component's own ink (restricted to the un-bridged
    `ink`, same as the main mask) for read_letters.py's point-value
    disambiguation -- or None if no digit-like component was found.
    """
    h, w = ink.shape
    # Bridge accents (acute of O/C/Z, dot of Z) to their base letter so they
    # cannot be lost as separate small components; the kernel is tall and
    # narrow, so the score digit (offset horizontally) stays separate.
    bridged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(5, h // 16))))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bridged)
    if n <= 1:
        return None, 0.0, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1] + 1

    def is_ring(i):
        """Tile bevel edges binarise into a thin frame around the glyph:
        a huge bounding box with almost no filled area."""
        x, y, bw, bh, a = stats[i]
        return bw > 0.72 * w and bh > 0.72 * h and a < 0.30 * bw * bh

    main = None
    for i in order:
        x, y, bw, bh, a = stats[i]
        # Border-hugging elongated components are bevel shadows.
        touches = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
        if touches >= 2 and (bw > 0.9 * w or bh > 0.9 * h):
            continue
        if is_ring(i):
            continue
        if a < 0.01 * h * w or a > 0.55 * h * w:
            continue
        cx, cy = cents[i]
        if not (0.15 * w < cx < 0.85 * w and 0.10 * h < cy < 0.90 * h):
            continue
        main = i
        break
    if main is None:
        return None, 0.0, None

    mx, my, mw, mh, ma = stats[main]
    keep = np.zeros_like(ink)
    keep[labels == main] = 255
    digit_score = 0.0
    digit_label = None
    for i in range(1, n):
        if i == main:
            continue
        x, y, bw, bh, a = stats[i]
        cx, cy = cents[i]
        if a < 0.02 * ma or a > 1.2 * ma or is_ring(i):
            continue
        overlaps_cols = (cx > mx - 0.15 * mw) and (cx < mx + 1.15 * mw)
        above = cy < my + 0.15 * mh
        below = cy > my + 0.85 * mh
        big_and_central = a > 0.30 * ma and overlaps_cols and not below
        # A detached ogonek below the letter is narrow; wide "below"
        # components are tile-edge bars and must not ride this rule.
        diacritic = (above or (below and bw < 0.6 * mw)) and overlaps_cols and a < 0.45 * ma
        # Score digit heuristic: small, in the lower-right, right of the
        # glyph's midline -> excluded even if it slips the column test.
        digit_like = a < 0.35 * ma and cx > mx + 0.60 * mw and cy > my + 0.55 * mh
        if digit_like:
            score = min(1.0, a / (0.10 * ma))
            if score > digit_score:
                digit_score, digit_label = score, i
        if (big_and_central or diacritic) and not digit_like:
            keep[labels == i] = 255
    # Selection ran on the accent-bridged image; restrict the final mask to
    # actual ink so the bridging never thickens the glyph itself.
    keep = cv2.bitwise_and(keep, ink)
    digit_mask = None
    if digit_label is not None:
        digit_mask = np.zeros_like(ink)
        digit_mask[labels == digit_label] = 255
        digit_mask = cv2.bitwise_and(digit_mask, ink)
        if digit_mask.sum() == 0:
            digit_mask = None
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

    keep, digit_score, digit_ink = _select_glyph_components(ink)
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
        centre = _centre_crop(patch, EXPECTED_TILE_FRAC)
        gray2, keep2, digit_score2, digit_ink2 = _binarise_and_select(centre, variant)
        if keep2 is not None and keep2.sum() > 0:
            gray, keep, digit_score, digit_ink = gray2, keep2, digit_score2, digit_ink2

    digit_gray = digit_mask = None
    if digit_ink is not None:
        digit_gray, digit_mask = _compose(gray, digit_ink, DIGIT_SIZE)

    if keep is None or keep.sum() == 0:
        blank = np.full((GLYPH_SIZE, GLYPH_SIZE), 255, np.uint8)
        return NormalizedGlyph(
            gray=blank, mask=np.zeros_like(blank), tile_gray=gray, has_glyph=False, ink_fraction=0.0, quality=0.0,
            digit_score=digit_score, digit_mask=digit_mask, digit_gray=digit_gray,
        )

    glyph_gray, glyph_mask = _compose(gray, keep, GLYPH_SIZE)
    ink_frac = float((keep > 0).mean())
    # Quality: penalise extreme ink fractions (speckle or blobs).
    quality = float(np.clip(1.0 - abs(ink_frac - 0.16) / 0.16, 0.05, 1.0))
    return NormalizedGlyph(
        gray=glyph_gray, mask=glyph_mask, tile_gray=gray, has_glyph=True, ink_fraction=ink_frac, quality=quality,
        digit_score=digit_score, digit_mask=digit_mask, digit_gray=digit_gray,
    )
