"""Locate the 15x15 playing grid inside the warped+rotated board.

warp_board()'s crop is never pixel-identical between photos -- the outer
quad it finds shifts a little with lighting/angle, and the warp itself
carries a dark border, the tile-distribution legend, the SCRABBLE logo, and
sometimes a sliver of background around an imperfect detection. A single
fixed offset/tile-size formula (this module's predecessor) only lines up
with cells when a photo happens to resemble whatever photo those constants
were tuned against; anything else silently shifts every cell box off the
real grid. So nothing here is assumed about where the grid starts, and
equal spacing is *not* assumed either -- every line is located individually.

Algorithm (ported from ocr/scrabble_reader/grid_detector.py, itself
validated on this project's own test photos)
--------------------------------------------
1. Line-response images: white grid lines are thin bright ridges on darker
   cells; a morphological top-hat with an elongated kernel responds to them
   while ignoring broad bright areas such as tiles.
2. Deskew: the four detected board corners never rectify perfectly, and a
   residual rotation of even 2 degrees smears projection profiles. A sweep
   over candidate rotations picks the angle whose comb fit (step 4) scores
   best.
3. Coverage profiles: for every column, the fraction of rows whose response
   is strong. A genuine grid line is near-continuous over the whole playing
   area, whereas table edges, wood grain and border artwork produce short
   segments with low coverage -- this is what makes the fit robust to
   clutter inside the warp.
4. Global comb fit: exhaustive (pitch, offset) search for the 16-tooth comb
   maximising total coverage, with a mild prior for combs centred in the
   warp.
5. Per-line refinement via dynamic programming over profile peaks
   (tolerates pitch drift from residual keystone), then a shift search
   against the premium-square colour pattern resolves +/-1-cell ambiguity
   (a comb fit can lock one pitch off onto the board's own decorative outer
   frame).
6. Per-intersection 2D refinement with a smooth quadratic displacement
   field, correcting small residual warp while ignoring intersections whose
   lines are hidden by tiles.

Returns None (like find_board_quad()/find_red_rectangle()) rather than
raising when detection fails, so callers use the same "if X is None" style
throughout this pipeline.
"""

from collections import namedtuple

import cv2
import numpy as np
from hsv_config import load_params
from premium_layout import GRID, premium_class

GridDetection = namedtuple("GridDetection", ["mesh", "confidence", "xs", "ys"])

PARAM_DEFAULTS = {
    "min_pitch_fraction": 0.045,  # acceptable cell-pitch lower bound, as a fraction of the board's side
    "max_pitch_fraction": 0.0667,  # acceptable cell-pitch upper bound
    "line_search_fraction": 0.30,  # half-width of the per-line refinement search window, as a fraction of pitch
    "max_intersection_shift": 0.25,  # max per-intersection displacement from the line-grid prediction
    # (fraction of pitch) before it's treated as an outlier and replaced by the fitted smooth field
    "min_comb_score": 0.35,  # minimum comb-fit z-score (both x and y) below which grid detection gives up
}


def _params(overrides=None):
    """Merge hsv_config.json's saved "grid_detector_params" preset with any
    explicit overrides -- same convention as detect_board.py/rotate_board.py/
    tile_detector.py."""
    merged = load_params("grid_detector_params", PARAM_DEFAULTS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def _subpixel_peak(profile, idx):
    """Parabolic sub-pixel refinement of a 1D peak at integer index `idx`."""
    if idx <= 0 or idx >= len(profile) - 1:
        return float(idx)
    left, center, right = profile[idx - 1], profile[idx], profile[idx + 1]
    den = left - 2.0 * center + right
    if abs(den) < 1e-9:
        return float(idx)
    return idx + 0.5 * (left - right) / den


def _line_response(gray):
    """Top-hat responses for thin vertical / horizontal bright lines."""
    n = max(9, int(gray.shape[0] * 0.015) | 1)
    kx = cv2.getStructuringElement(cv2.MORPH_RECT, (n, 1))
    ky = cv2.getStructuringElement(cv2.MORPH_RECT, (1, n))
    resp_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kx)
    resp_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ky)
    return resp_v.astype(np.float32), resp_h.astype(np.float32)


def _coverage_profile(resp, axis):
    """Fraction of pixels along `axis` with a strong line response."""
    # The percentile is a global scalar -- a 4x4-subsampled view gives the
    # same value at 1/16 the cost.
    thr = max(4.0, float(np.percentile(resp[::4, ::4], 90)))
    return (resp > thr).mean(axis=axis).astype(np.float64)


def _comb_fit(profile, min_pitch, max_pitch):
    """(offset, pitch, z-score) of the best 16-tooth comb.

    Score is the mean coverage at the teeth in units of the profile's
    standard deviation above its mean, with a mild penalty for combs far
    from the image centre (the board was detected around the grid, so a
    wildly off-centre comb is almost certainly locked onto clutter).
    """
    p = profile
    mu, sd = p.mean(), p.std() + 1e-9
    n = len(p)
    box = np.convolve(p, np.ones(5) / 5.0, mode="same")

    best = (0.0, 0.0, -np.inf)
    pitch = min_pitch
    while pitch <= max_pitch:
        span = pitch * GRID
        max_off = int(n - span - 1)
        if max_off >= 1:
            taps = [box[int(round(k * pitch)) : int(round(k * pitch)) + max_off] for k in range(GRID + 1)]
            m = min(len(t) for t in taps)
            total = np.sum([t[:m] for t in taps], axis=0) / (GRID + 1)
            offs = np.arange(m)
            centre_dev = np.abs(offs + span / 2.0 - n / 2.0) / (n / 2.0)
            score = (total - mu) / sd - 1.5 * centre_dev
            off = int(np.argmax(score))
            if score[off] > best[2]:
                best = (float(off), float(pitch), float(score[off]))
        pitch += 0.25
    return best


def _estimate_skew(resp_v, resp_h, min_pitch_frac, max_pitch_frac):
    """Residual grid rotation, found by maximising comb-fit quality.

    Even +/-2 degrees of tilt spreads each grid line across dozens of
    profile columns, so the comb score as a function of test rotation has a
    sharp peak at the true skew. The sweep runs on quarter-resolution
    responses (coarse 1.5 degree pass, then 0.4 degree refinement) which is
    accurate to a fraction of a degree at a fraction of the full-resolution
    cost.
    """
    small_v = cv2.resize(resp_v, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small_h = cv2.resize(resp_h, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    size = small_v.shape[0]
    centre = (small_v.shape[1] / 2.0, size / 2.0)
    min_p, max_p = min_pitch_frac * size, max_pitch_frac * size

    def score(angle):
        if angle:
            M = cv2.getRotationMatrix2D(centre, angle, 1.0)
            v = cv2.warpAffine(small_v, M, small_v.shape[::-1])
            h = cv2.warpAffine(small_h, M, small_h.shape[::-1])
        else:
            v, h = small_v, small_h
        fx = _comb_fit(_coverage_profile(v, axis=0), min_p, max_p)
        fy = _comb_fit(_coverage_profile(h, axis=1), min_p, max_p)
        return fx[2] + fy[2]

    coarse = np.arange(-4.5, 4.51, 1.5)
    scores = [score(a) for a in coarse]
    best = float(coarse[int(np.argmax(scores))])
    fine = best + np.arange(-1.0, 1.01, 0.4)
    scores = [score(a) for a in fine]
    return float(fine[int(np.argmax(scores))])


def _refine_lines(profile, offset, pitch, search_frac):
    """Locate the 16 grid lines given the comb prior, tolerating pitch drift.

    A uniform comb is only an approximation: residual keystone makes the
    true pitch vary a few percent across the board, which accumulates to
    half a cell over 15 lines. Lines are therefore chosen by dynamic
    programming over profile peaks: consecutive lines must be 0.82-1.18
    pitches apart, the first line must lie near the comb's first tooth, and
    the total peak support is maximised. Falls back to the plain comb when
    peaks are too sparse.

    Returns 18 positions: teeth -1..16 (one extra line on each side for the
    premium-registration check), extrapolated at the local spacing.
    """
    smooth = np.convolve(profile, np.ones(5) / 5.0, "same")
    n = len(smooth)

    # Candidate peaks: local maxima at least 0.4 pitch apart.
    min_dist = max(3, int(0.4 * pitch))
    cand = []
    for idx in np.argsort(smooth)[::-1]:
        if smooth[idx] <= 0:
            break
        if all(abs(idx - c) >= min_dist for c in cand):
            cand.append(int(idx))
        if len(cand) >= 80:
            break
    cand.sort()
    pos_arr = np.array(cand, dtype=np.float64)
    val = smooth[cand] if cand else np.array([])

    lines = None
    if len(cand) >= GRID + 1:
        first_lo, first_hi = offset - 0.6 * pitch, offset + 0.6 * pitch
        lo_gap, hi_gap = 0.82 * pitch, 1.18 * pitch
        m = len(cand)
        NEG = -1e18
        score = np.full((GRID + 1, m), NEG)
        parent = np.full((GRID + 1, m), -1, dtype=np.int32)
        # Slight pull toward the comb prediction keeps the DP from wandering
        # onto tile edges when a genuine line is weak.
        pred = offset + np.arange(GRID + 1) * pitch
        for j in range(m):
            if first_lo <= pos_arr[j] <= first_hi:
                score[0, j] = val[j] - 0.1 * abs(pos_arr[j] - pred[0]) / pitch
        for k in range(1, GRID + 1):
            for j in range(m):
                gaps = pos_arr[j] - pos_arr
                ok = (gaps >= lo_gap) & (gaps <= hi_gap)
                if not ok.any():
                    continue
                prev = np.where(ok)[0]
                best_prev = prev[np.argmax(score[k - 1, prev])]
                if score[k - 1, best_prev] <= NEG / 2:
                    continue
                score[k, j] = score[k - 1, best_prev] + val[j] - 0.1 * abs(pos_arr[j] - pred[k]) / pitch
                parent[k, j] = best_prev
        end = int(np.argmax(score[GRID]))
        if score[GRID, end] > NEG / 2:
            path = [end]
            for k in range(GRID, 0, -1):
                path.append(int(parent[k, path[-1]]))
            path.reverse()
            lines = np.array([_subpixel_peak(smooth, cand[j]) for j in path])

    if lines is None:
        ks = np.arange(GRID + 1)
        lines = offset + ks * pitch
        half = max(2, int(search_frac * pitch))
        for j, c in enumerate(lines):
            lo, hi = max(0, int(c) - half), min(n, int(c) + half + 1)
            if hi - lo >= 3:
                idx = lo + int(np.argmax(smooth[lo:hi]))
                if abs(idx - c) <= 0.35 * pitch:
                    lines[j] = _subpixel_peak(smooth, idx)
        lines = np.sort(lines)

    # Extrapolate teeth -1 and 16 at the local spacing.
    ext = np.empty(GRID + 3)
    ext[1:-1] = lines
    ext[0] = lines[0] - (lines[1] - lines[0])
    ext[-1] = lines[-1] + (lines[-1] - lines[-2])
    return ext


def _premium_shift(warp, xs, ys):
    """Resolve +/-1-cell grid mis-registration with the premium pattern.

    A comb fit can lock one pitch off when an extra board line (the playing
    area's outer frame) mimics a grid line. The premium squares form a
    known, 4-fold-symmetric pattern, so the correct registration is the
    shift whose cell colours cluster tightest within premium classes. `xs`/
    `ys` carry one extra line on each side (teeth -1..16); the returned
    (sx, sy) selects which window of 16 consecutive lines is the grid, and
    the third element is the score margin over the runner-up hypothesis.
    """
    small = cv2.resize(warp, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    classes = np.array([[premium_class(r, c).replace("*", "D") for c in range(GRID)] for r in range(GRID)])

    def sample(cx, cy):
        x, y = int(cx * 0.25), int(cy * 0.25)
        if not (2 <= x < w - 2 and 2 <= y < h - 2):
            return None
        patch = lab[y - 2 : y + 3, x - 2 : x + 3].reshape(-1, 3)
        return np.median(patch, axis=0)

    results = []
    for sy in (0, -1, 1):
        for sx in (0, -1, 1):
            colors = {}
            n_valid = 0
            for r in range(GRID):
                for c in range(GRID):
                    x0, x1 = xs[c + 1 + sx], xs[c + 2 + sx]
                    y0, y1 = ys[r + 1 + sy], ys[r + 2 + sy]
                    v = sample((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                    if v is None:
                        continue
                    n_valid += 1
                    colors.setdefault(classes[r, c], []).append(v)
            if n_valid < GRID * GRID * 0.9 or "." not in colors:
                continue
            # Within-class tightness alone cannot detect a shifted grid: the
            # premium pattern is sparse, so shifted classes land almost
            # entirely on plain squares and stay perfectly homogeneous. The
            # decisive signal is between-class separation -- only the
            # correct registration puts the red/blue premium squares into
            # their own classes, far away from the plain-square colour.
            within = 0.0
            plain_med = np.median(np.array(colors["."]), axis=0)
            sep, n_premium = 0.0, 0
            sep_min = np.inf
            for cls, vals in colors.items():
                arr = np.array(vals)
                med = np.median(arr, axis=0)
                within += np.median(np.abs(arr - med).sum(axis=1)) * len(vals)
                if cls != ".":
                    d = float(np.abs(med - plain_med).sum())
                    sep += d * len(vals)
                    sep_min = min(sep_min, d)
                    n_premium += len(vals)
            within /= n_valid
            sep /= max(1, n_premium)
            score = within - 0.5 * sep
            # Prefer no shift on near-ties (measurement noise).
            if (sx, sy) != (0, 0):
                score += 0.5
            results.append((score, sep, sep_min, (sx, sy)))
    if not results:
        return 0, 0, 0.0
    results.sort(key=lambda t: t[0])
    best_score, best_sep, best_sep_min, best = results[0]
    second = results[1][0] if len(results) > 1 else best_score
    # Normalised margin over the runner-up doubles as grid confidence: a
    # correct grid on a real Scrabble board separates premium colours by
    # tens of Lab units, dwarfing every shifted hypothesis. A mesh that only
    # partially overlaps the board can fake the aggregate separation, but
    # never separates EVERY premium class -- hence the min-class gate.
    margin = (second - best_score) / (1.0 + 0.5 * best_sep)
    margin *= float(np.clip(best_sep_min / 15.0, 0.0, 1.0))
    return best[0], best[1], float(margin)


def _refine_intersections(resp_v, resp_h, xs, ys, pitch, max_shift_frac):
    """Per-intersection 2D refinement + smooth-field regularisation.

    Displacements from the line grid are re-measured locally, then a
    quadratic surface is fit per axis with outlier rejection: genuine
    lens/warp residuals vary smoothly across the board, whereas errors from
    tiles hiding lines are isolated and get replaced by the fitted value.
    """
    h, w = resp_v.shape
    band = int(pitch * 0.35)
    win = int(pitch * 0.28)
    max_shift = max_shift_frac * pitch

    raw = np.zeros((GRID + 1, GRID + 1, 2), dtype=np.float64)
    weight = np.zeros((GRID + 1, GRID + 1), dtype=np.float64)

    for j, y in enumerate(ys):
        y0, y1 = max(0, int(y - band)), min(h, int(y + band) + 1)
        row_v = resp_v[y0:y1]
        for i, x in enumerate(xs):
            x0, x1 = max(0, int(x - win)), min(w, int(x + win) + 1)
            prof_x = row_v[:, x0:x1].sum(axis=0)
            xx0, xx1 = max(0, int(x - band)), min(w, int(x + band) + 1)
            yy0, yy1 = max(0, int(y - win)), min(h, int(y + win) + 1)
            prof_y = resp_h[yy0:yy1, xx0:xx1].sum(axis=1)
            if len(prof_x) < 5 or len(prof_y) < 5:
                raw[j, i] = (x, y)
                continue
            px = x0 + _subpixel_peak(prof_x, int(np.argmax(prof_x)))
            py = yy0 + _subpixel_peak(prof_y, int(np.argmax(prof_y)))
            raw[j, i] = (px, py)
            weight[j, i] = (prof_x.max() - prof_x.mean()) + (prof_y.max() - prof_y.mean())

    base = np.stack(np.meshgrid(xs, ys), axis=-1)
    disp = raw - base
    ii, jj = np.meshgrid(np.arange(GRID + 1), np.arange(GRID + 1))
    A = np.stack([np.ones_like(ii), ii, jj, ii * jj, ii**2, jj**2], axis=-1).reshape(-1, 6).astype(np.float64)
    out = base.copy()
    for axis in range(2):
        d = disp[..., axis].ravel()
        w_flat = weight.ravel().copy()
        w_flat[np.abs(d) > max_shift] = 0.0
        coef = None
        for _ in range(2):
            if (w_flat > 0).sum() < 12:
                coef = None
                break
            sw = np.sqrt(w_flat)
            coef, *_ = np.linalg.lstsq(A * sw[:, None], d * sw, rcond=None)
            resid = np.abs(d - A @ coef)
            w_flat[resid > max(2.0, 2.5 * np.median(resid[w_flat > 0]))] = 0.0
        if coef is not None:
            out[..., axis] = base[..., axis] + (A @ coef).reshape(GRID + 1, GRID + 1)
    return out


def detect_grid(warp, **param_overrides):
    """Locate the 16x16 grid-line intersections in `warp` (the warped+
    rotated board). Returns a GridDetection, or None if no confident grid
    was found (deliberately permissive at the comb-fit stage -- worn boards
    with busy artwork dilute the z-score badly -- so the premium-pattern
    registration in _premium_shift() is the real accept/reject signal, via
    its contribution to `confidence`)."""
    p = _params(param_overrides)
    size = warp.shape[0]
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    # Equalise illumination so a shadowed corner contributes to the
    # projections as much as a sunlit one.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    resp_v0, resp_h0 = _line_response(gray)

    centre = (size / 2.0, size / 2.0)
    min_p = p["min_pitch_fraction"] * size
    max_p = p["max_pitch_fraction"] * size

    skew = _estimate_skew(resp_v0, resp_h0, p["min_pitch_fraction"], p["max_pitch_fraction"])
    if abs(skew) > 0.05:
        M = cv2.getRotationMatrix2D(centre, skew, 1.0)
        resp_v = cv2.warpAffine(resp_v0, M, (size, size))
        resp_h = cv2.warpAffine(resp_h0, M, (size, size))
        M_inv = cv2.invertAffineTransform(M)
    else:
        resp_v, resp_h, M_inv = resp_v0, resp_h0, None

    prof_x = _coverage_profile(resp_v, axis=0)
    prof_y = _coverage_profile(resp_h, axis=1)
    off_x, pitch_x, score_x = _comb_fit(prof_x, min_p, max_p)
    off_y, pitch_y, score_y = _comb_fit(prof_y, min_p, max_p)

    if min(score_x, score_y) < p["min_comb_score"]:
        return None

    # One extra tooth on each side; the premium-square pattern then decides
    # which window of 16 lines is the actual grid -- comb fits occasionally
    # lock one pitch off onto the playing area's decorative outer frame.
    xs_ext = _refine_lines(prof_x, off_x, pitch_x, p["line_search_fraction"])
    ys_ext = _refine_lines(prof_y, off_y, pitch_y, p["line_search_fraction"])

    # The premium check runs in the deskewed frame, so rotate the warp
    # accordingly when a deskew was applied.
    if M_inv is not None:
        M = cv2.getRotationMatrix2D(centre, skew, 1.0)
        warp_deskewed = cv2.warpAffine(warp, M, (size, size))
    else:
        warp_deskewed = warp
    sx, sy, premium_margin = _premium_shift(warp_deskewed, xs_ext, ys_ext)
    xs = xs_ext[1 + sx : 2 + sx + GRID]
    ys = ys_ext[1 + sy : 2 + sy + GRID]

    mesh = _refine_intersections(resp_v, resp_h, xs, ys, (pitch_x + pitch_y) / 2.0, p["max_intersection_shift"])

    # Map intersections back into the (non-deskewed) warp frame.
    if M_inv is not None:
        flat = mesh.reshape(-1, 2)
        flat = np.hstack([flat, np.ones((len(flat), 1))]) @ M_inv.T
        mesh = flat.reshape(GRID + 1, GRID + 1, 2)

    # Confidence: mostly the premium-registration margin (how much better
    # the chosen alignment separates the premium colour classes than any
    # 1-cell shift), topped up by comb strength.
    conf = float(np.clip(0.7 * min(1.0, premium_margin * 2.5) + 0.3 * min(1.0, (score_x + score_y) / 12.0), 0.0, 1.0))
    return GridDetection(mesh=mesh, confidence=conf, xs=xs, ys=ys)
