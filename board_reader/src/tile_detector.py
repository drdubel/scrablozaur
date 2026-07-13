"""Decide, per square, whether a tile is present (no OCR involved).

Tiles differ from the printed board in colour (ivory plastic/wood vs. the
board's saturated print), in texture (a dark glyph creates strong compact
edges) and in brightness. But absolute thresholds fail across photos --
lighting and board editions vary, and (this project's original bug) a
single fixed brightness cutoff can't tell an empty *light-coloured* premium
square (double letter, double word) from a tile sitting on one. So this
detector is self-calibrating for every photo instead:

1. Every square gets a feature vector: CIELAB colour statistics of its
   central region, chroma, edge density, and a "glyph score" (presence of a
   compact dark blob of letter-like size).
2. Squares are grouped by premium class using the standard (rotation
   invariant) premium layout (premium_layout.py). Within each group the
   *empty* squares share the printed colour, so robust statistics
   (median/MAD) model "empty" per class even without knowing the edition's
   palette. Tiles show up as colour outliers w.r.t. their own class.
3. A cross-class check exploits that tiles look the *same* everywhere: the
   candidate outliers from all classes must agree in colour. Their robust
   centre defines the photo's "tile colour"; the final score combines
   distance-from-empty, distance-to-tile-colour, and the glyph score. This
   is what catches tiles on light premium squares, whose per-class z-score
   can collapse (light blue is close to ivory, and the class MAD is
   inflated by the tiles themselves).

Ported from ocr/scrabble_reader/tile_detector.py (validated there at
97.6% letter / 98.6% cell accuracy on this project's own test photos),
adapted to this project's conventions: no dataclass Config, tunables read
from hsv_config.json via _params()/PARAM_DEFAULTS like detect_board.py and
rotate_board.py.
"""

from collections import namedtuple

import cv2
import numpy as np
from hsv_config import load_params
from premium_layout import premium_class

Cell = namedtuple("Cell", ["row", "col", "patch"])
TileVerdict = namedtuple(
    "TileVerdict", ["row", "col", "is_tile", "confidence", "glyph_score", "z_score", "tile_dist"]
)

PARAM_DEFAULTS = {
    "color_z_threshold": 3.5,  # per-class robust z-score above which a cell is a colour outlier
    "seed_glyph_high": 0.45,  # glyph bar for a confident pass-2 tile-colour seed
    "seed_glyph_low": 0.30,  # relaxed glyph bar used when too few confident seeds exist
    "seed_glyph_fallback": 0.55,  # glyph-only bar when colour alone can't seed the model at all (soft focus)
    "seed_max": 8,  # max candidates used to build the photo's tile-colour model
    "seed_min": 3,  # below this many seeds, fall back to the generic ivory prior
    "same_tile_dist": 0.9,  # distance-to-tile-colour accepted outright (own-class z need not agree)
    "d_accept": 3.0,  # permissive distance-accept bound (non-strict)
    "z_accept": 2.2,  # permissive class-z accept bound (non-strict)
    "d_accept_strict": 2.2,  # distance-accept bound when strict=True
    "z_accept_strict": 3.0,  # class-z accept bound when strict=True
    "definitely_empty_d": 5.0,  # distance above which a cell is confidently empty
    "definitely_empty_z": 1.5,  # class-z below which a cell is confidently empty
    "ambiguous_glyph_min": 0.40,  # glyph bar arbitrating the band between "empty" and "accept"
    "strict_glyph_min": 0.35,  # glyph floor that rejects outright when strict=True
    "permissive_glyph_min": 0.05,  # glyph floor for the permissive d_accept/z_accept branch --
    # without it, a glare/specular highlight (bright, moderately colour-outlier, but with zero
    # glyph evidence) can pass on colour alone; the strict same_tile_dist branch above is
    # untouched, so a genuine blank/worn tile (no glyph, but a tight colour match) is still caught.
}


def _params(overrides=None):
    """Merge hsv_config.json's saved "tile_detector_params" preset with any
    explicit overrides -- used by tuner.py to preview live, not-yet-saved
    slider values without detect_tiles() needing its own long parameter
    list."""
    merged = load_params("tile_detector_params", PARAM_DEFAULTS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def _center(img, frac=0.70):
    h, w = img.shape[:2]
    mh, mw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return img[mh : h - mh, mw : w - mw]


def glyph_score(patch_gray):
    """Evidence that the square's centre contains a single letter-like blob.

    A real tile prints ONE dominant glyph (plus a small score digit); a
    premium square prints several lines of small text. The score therefore
    combines the best letter-sized component with a dominance factor that
    collapses when many similar-sized text components are present.
    """
    g = _center(patch_gray, 0.80)
    h, w = g.shape
    area = h * w
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, max(9, (h // 3) | 1), 7
    )
    n, _labels, stats, cents = cv2.connectedComponentsWithStats(th)
    best = 0.0
    best_area = 0.0
    sizes = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a > 0.004 * area:
            sizes.append(float(a))
        if not (0.015 * area < a < 0.40 * area):
            continue
        cx, cy = cents[i]
        # Letter ink sits near the centre of the tile.
        dist = np.hypot(cx - w / 2, cy - h / 2) / (0.5 * np.hypot(h, w))
        if dist > 0.55:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH] / w
        bh = stats[i, cv2.CC_STAT_HEIGHT] / h
        if bw > 0.9 or bh > 0.9:  # border artefacts / grid line remnants
            continue
        size_score = min(1.0, a / (0.06 * area))
        s = size_score * (1.0 - dist)
        if s > best:
            best, best_area = s, float(a)
    if best == 0.0:
        return 0.0
    # Dominance: fraction of the total ink held by the winning component.
    # One letter + digit -> ~0.8; several lines of premium text -> ~0.2.
    rest = sum(a for a in sizes if a != best_area)
    dominance = best_area / (best_area + rest + 1e-9)
    return float(min(1.0, best) * np.clip(dominance * 1.6, 0.15, 1.0))


def features_batch(cells):
    """Per-square feature vectors: L, a, b medians, chroma, edge density.

    All centre crops are stacked into one tall image so colour conversion
    and edge detection run once instead of 225 times.
    """
    crops = np.stack([_center(c.patch, 0.70) for c in cells])
    n, h, w, _ = crops.shape
    flat = crops.reshape(n * h, w, 3)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = lab.reshape(n, h * w, 3)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(A - 128, B - 128)
    edges = cv2.Canny(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), 60, 160)
    edge_density = edges.reshape(n, h * w).mean(axis=1) / 255.0
    return np.stack(
        [np.median(L, axis=1), np.median(A, axis=1), np.median(B, axis=1), np.median(chroma, axis=1), edge_density],
        axis=1,
    )


def detect_tiles(cells, strict=False, **param_overrides):
    """Self-calibrating per-photo tile-presence model -- see module
    docstring for the algorithm. `cells` must cover the whole board (or at
    least a large, class-diverse sample of it): the per-class and
    cross-class colour models need enough squares of each premium class to
    be statistically meaningful, so this can't be called one cell at a
    time.

    `strict` tightens acceptance -- meant for when the grid registration
    itself is uncertain, where a permissive detector floods the board with
    false tiles. board_reader has no grid-confidence signal yet, so nothing
    currently passes strict=True; kept for forward compatibility with ocr/'s
    version, which does.
    """
    p = _params(param_overrides)
    feats = features_batch(cells)
    patches = np.stack([c.patch for c in cells])
    grays = cv2.cvtColor(patches.reshape(-1, patches.shape[2], 3), cv2.COLOR_BGR2GRAY).reshape(
        len(cells), patches.shape[1], patches.shape[2]
    )
    glyphs = np.array([glyph_score(g) for g in grays])

    color = feats[:, :4]  # L, a, b, chroma
    # The centre star counts as a double-word square; on its own it would
    # form a one-member class with degenerate statistics.
    classes = np.array([premium_class(c.row, c.col).replace("*", "D") for c in cells])

    # --- pass 1: per-class robust colour model of "empty" -------------------
    z = np.zeros(len(cells))
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        sub = color[idx]
        med = np.median(sub, axis=0)
        mad = np.median(np.abs(sub - med), axis=0) * 1.4826 + 2.0
        z[idx] = np.sqrt((((sub - med) / mad) ** 2).mean(axis=1))

    # --- pass 2: global tile-colour model ------------------------------------
    # Only the highest-quality candidates seed the tile-colour model: a
    # handful of clean seeds beats many contaminated ones (glare patches and
    # washed-out premium squares also register as colour outliers).
    cand = np.where((z > p["color_z_threshold"]) & (glyphs > p["seed_glyph_high"]))[0]
    if len(cand) < p["seed_min"]:
        # Washed-out photos flatten glyph contrast; relax the glyph bar (the
        # z requirement still keeps premium squares out).
        cand = np.where((z > p["color_z_threshold"]) & (glyphs > p["seed_glyph_low"]))[0]
    if len(cand) < p["seed_min"]:
        # A soft-focus or low-contrast photo can flatten colour differences
        # enough that literally nothing clears the z bar even on a crowded
        # board full of real tiles (a heavily-occupied premium class also
        # inflates its own "empty" MAD, compounding this). Glyph shape
        # survives blur far better than colour z-score does, so fall back to
        # strong glyph evidence alone -- sorted by glyph score, since z is
        # not trustworthy enough here to help rank candidates.
        cand = np.where(glyphs > p["seed_glyph_fallback"])[0]
        cand = cand[np.argsort(-glyphs[cand])]
    else:
        cand = cand[np.argsort(-(np.minimum(z[cand], 8.0) * glyphs[cand]))]
    seeds = cand[: int(p["seed_max"])]
    model_ok = len(seeds) >= p["seed_min"]
    if model_ok:
        tile_med = np.median(color[seeds], axis=0)
        tile_mad = np.median(np.abs(color[seeds] - tile_med), axis=0) * 1.4826 + 4.0
        d_tile = np.sqrt((((color - tile_med) / tile_mad) ** 2).mean(axis=1))
    else:
        # No confident seeds (empty board, or all tiles blank): a generic
        # ivory prior -- bright, weak colour -- stands in for the model.
        bright = np.clip((color[:, 0] - np.median(color[:, 0])) / 25.0, 0, 3)
        d_tile = 5.0 - 1.5 * bright + np.clip((color[:, 3] - 40) / 10.0, 0, 3)

    # --- decide ---------------------------------------------------------------
    # A tile must both stand out from its own premium class (z) and look
    # like the other tiles on this photo (d_tile). The glyph arbitrates only
    # the ambiguous band, so blank tiles (no glyph, right colour) and busy
    # premium squares (glyph-ish text, wrong colour) both land on the
    # correct side.
    d_accept, z_accept = (p["d_accept_strict"], p["z_accept_strict"]) if strict else (p["d_accept"], p["z_accept"])
    verdicts = []
    for k, cell in enumerate(cells):
        zk, dk, gk = float(z[k]), float(d_tile[k]), float(glyphs[k])
        margin = min(zk, 8.0) - dk
        if strict and gk < p["strict_glyph_min"]:
            verdicts.append(
                TileVerdict(row=cell.row, col=cell.col, is_tile=False, confidence=0.5, glyph_score=gk, z_score=zk, tile_dist=dk)
            )
            continue
        if model_ok and dk < p["same_tile_dist"]:
            # Indistinguishable from the photo's own tiles in colour. This is
            # what catches tiles sitting on light premium squares, whose
            # per-class z collapses (light blue is close to ivory and the
            # class MAD is inflated by the tiles).
            is_tile = True
            conf = float(np.clip(0.75 + 0.15 * gk, 0.5, 1.0))
        elif dk < d_accept and zk > z_accept and gk > p["permissive_glyph_min"]:
            # Colour alone isn't enough here (that's the same_tile_dist
            # branch above) -- a specular highlight is also a bright,
            # moderately colour-outlier blob, so this weaker match needs at
            # least a trace of glyph evidence to rule out glare.
            is_tile = True
            conf = float(np.clip(0.6 + margin / 10.0 + 0.2 * gk, 0.5, 1.0))
        elif dk > p["definitely_empty_d"] or zk < p["definitely_empty_z"]:
            is_tile = False
            conf = float(
                np.clip(
                    0.6
                    + (dk - p["definitely_empty_d"]) / 8.0
                    + (p["definitely_empty_z"] - min(zk, p["definitely_empty_z"])) / 3.0,
                    0.5,
                    1.0,
                )
            )
        else:
            is_tile = gk > p["ambiguous_glyph_min"] and zk > p["color_z_threshold"] / 2
            conf = 0.5
        verdicts.append(
            TileVerdict(row=cell.row, col=cell.col, is_tile=is_tile, confidence=conf, glyph_score=gk, z_score=zk, tile_dist=dk)
        )
    return verdicts
