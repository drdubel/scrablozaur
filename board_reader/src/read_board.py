import glob

import cv2
import numpy as np
from cv_utils import show_images
from detect_board import find_board_quad, warp_board
from grid_detector import detect_grid
from premium_layout import GRID, premium_class
from rotate_board import rotate_board
from tile_detector import Cell, detect_tiles, features_batch, glyph_score

CELL_SIZE = 96  # resolution of the sampled per-cell patch fed to tile_detector
TILE_SIZE = 160  # resolution of the higher-res patch fed to read_letters.py's recognition stage
# Cells are sampled with this expansion so tiles that sit slightly off
# centre / overhang the printed square are fully captured.
EXPAND_FRACTION = 0.12

# A physical tile sits a few mm above the board's flat surface, so a photo
# taken at a steep angle projects it shifted away from its printed square
# once the (flat-plane) perspective correction is applied, and a window
# sized to the cell itself then samples mostly background. Two correction
# strategies were tried first and both made the test set worse overall, not
# better: (1) an independent per-cell wide-window search for the nearest
# tile-like blob -- on a dense word row it just as often latches onto a
# neighbouring tile, a premium square's own printed text, or a grid-line
# fragment (97.7% -> 92.8% cell accuracy); (2) measuring a single photo-wide
# shift from cells the detector is already confident about, then
# re-centring every cell by that amount -- fails precisely on the
# worst-parallax photos, where nothing is confident enough in the first
# pass to bootstrap a measurement from (97.7% -> 97.3%).
#
# What actually works (find_parallax_shift() below): search a small grid of
# candidate photo-wide shifts directly, scoring each by aggregate evidence
# across ALL 225 cells jointly (tile_detector's own per-class colour z-score
# combined with glyph evidence) rather than trusting any single cell's
# verdict. This needs neither a confident bootstrap detection (score is
# summed over the whole board, so it doesn't matter that no individual cell
# clears the threshold yet) nor an independent per-cell decision (it's one
# global choice, so it can't be dragged off by one cell's neighbour/text/
# grid-line noise).
SHIFT_SEARCH_FRACS = (-0.4, -0.2, 0.0, 0.2, 0.4)  # candidate shift, as a fraction of the cell pitch, per axis
SHIFT_SCORE_SIZE = 48  # cheap low-res patch size used only to rank candidate shifts
SHIFT_NO_SHIFT_BONUS = (
    1.15  # mild preference for zero shift on a near-tie, so well-behaved photos aren't nudged by noise
)
SHIFT_Z_FLOOR = 2.0  # per-cell z-scores below this contribute nothing to a candidate shift's score
# Measured on the test set: a broken (0, 0) alignment (no real tiles found) scores under 1;
# a working one scores in the tens, even on a photo too soft-focus to seed its own tile-colour
# model. Comfortably splits the two -- see find_parallax_shift()'s docstring for why searching
# at all when the baseline is already healthy is actively risky, not just wasted work.
SHIFT_SEARCH_TRIGGER = 5.0


def _cell_quad(mesh, row, col):
    """The four warp-space corners (TL, TR, BR, BL) of cell (row, col)."""
    return np.array([mesh[row, col], mesh[row, col + 1], mesh[row + 1, col + 1], mesh[row + 1, col]], dtype=np.float32)


def _sample_quad(rotated, quad, size):
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(rotated, H, (size, size), flags=cv2.INTER_AREA)


def _cells_at(rotated, mesh, shift, size):
    cells = []
    for i in range(GRID):
        for j in range(GRID):
            quad = _cell_quad(mesh, i, j)
            center = quad.mean(axis=0) + shift
            expanded = (center + (quad - quad.mean(axis=0)) * (1.0 + 2 * EXPAND_FRACTION)).astype(np.float32)
            cells.append(Cell(row=i, col=j, patch=_sample_quad(rotated, expanded, size)))
    return cells


def extract_cells(rotated, mesh, global_shift=None):
    """Slice all 225 cell patches, each sampled with its own perspective
    transform off the four surrounding mesh intersections -- cells stay
    accurate even where the grid isn't perfectly uniform (residual keystone,
    lens distortion). `global_shift` (from find_parallax_shift()) re-centres
    every cell by the same photo-wide amount, correcting a consistent tile
    parallax offset. detect_tiles() is a per-photo batch model (its
    per-class and cross-class colour calibration needs every cell's
    features before any single verdict can be computed), so every patch
    must exist before detection runs."""
    shift = np.zeros(2, dtype=np.float32) if global_shift is None else global_shift
    return _cells_at(rotated, mesh, shift, CELL_SIZE)


TILE_EXPAND_FRACTION = 0.18  # margin for extract_tile_patches()'s recognition samples -- see its docstring


def extract_tile_patches(rotated, mesh, verdicts, global_shift=None):
    """Second, higher-res (TILE_SIZE) sample per cell -- only for cells
    `verdicts` flags is_tile, since only those ever reach read_letters.py's
    recognition stage. Resampling all 225 cells at this size unconditionally
    would waste several times the perspective-warp work on a typical
    20-70-tile board for no benefit. Returns {(row, col): patch}, using the
    same global_shift alignment as extract_cells() so both patch sizes stay
    registered to the same tile position.

    Uses its own margin, TILE_EXPAND_FRACTION, separate from tile_detector's
    EXPAND_FRACTION -- but a bigger one is a wash, not a clear win: on one
    tightly-packed board, a single mis-cropped glyph (the true tile sat
    close enough to the window edge that glyph_normalizer's re-centring
    search gave up and fell back to a plain geometric crop landing on the
    tile seam instead of the letter) was fully fixed by widening to 0.35.
    But swept against the whole test set, that same widening cost letter
    accuracy elsewhere (94.4% -> 87.5% at 0.35) -- the extra margin pulls
    neighbouring tiles' ink into frame often enough to outweigh the odd
    rescued crop, the same shape of regression EXPAND_FRACTION's own tuning
    already hit once (see the comment above SHIFT_SEARCH_FRACS). 0.18 is
    the measured local optimum, and it is a small one (94.36% -> 94.40%) --
    most of this failure mode is not actually fixable by margin alone.
    """
    shift = np.zeros(2, dtype=np.float32) if global_shift is None else global_shift
    out = {}
    for v in verdicts:
        if not v.is_tile:
            continue
        quad = _cell_quad(mesh, v.row, v.col)
        center = quad.mean(axis=0) + shift
        expanded = (center + (quad - quad.mean(axis=0)) * (1.0 + 2 * TILE_EXPAND_FRACTION)).astype(np.float32)
        out[(v.row, v.col)] = _sample_quad(rotated, expanded, TILE_SIZE)
    return out


def _shift_score(rotated, mesh, shift):
    """Aggregate evidence that `shift` is where this photo's tiles actually
    are: tile_detector's own pass-1 per-class colour z-score, combined with
    glyph evidence, summed across every cell that clears a modest floor. A
    correct shift lines tiles up with their sampling windows and lights up a
    cluster of strong, glyph-backed outliers; a wrong one doesn't. See the
    comment above SHIFT_SEARCH_FRACS for why this beats per-cell search."""
    cells = _cells_at(rotated, mesh, shift, SHIFT_SCORE_SIZE)
    color = features_batch(cells)[:, :4]
    patches = np.stack([c.patch for c in cells])
    grays = cv2.cvtColor(patches.reshape(-1, SHIFT_SCORE_SIZE, 3), cv2.COLOR_BGR2GRAY).reshape(
        len(cells), SHIFT_SCORE_SIZE, SHIFT_SCORE_SIZE
    )
    glyphs = np.array([glyph_score(g) for g in grays])
    classes = np.array([premium_class(c.row, c.col).replace("*", "D") for c in cells])

    z = np.zeros(len(cells))
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        sub = color[idx]
        med = np.median(sub, axis=0)
        mad = np.median(np.abs(sub - med), axis=0) * 1.4826 + 2.0
        z[idx] = np.sqrt((((sub - med) / mad) ** 2).mean(axis=1))
    return float(np.sum(np.clip(z - SHIFT_Z_FLOOR, 0, None) * glyphs))


def find_parallax_shift(rotated, mesh):
    """Search SHIFT_SEARCH_FRACS x SHIFT_SEARCH_FRACS candidate photo-wide
    shifts and return whichever scores best -- but only bother when the
    default (0, 0) alignment already looks badly broken (score below
    SHIFT_SEARCH_TRIGGER). A board the detector is already reading well
    scores far above that trigger (tens, not single digits, even on a
    photo too soft-focus to seed its own tile-colour model -- see
    tile_detector.py's seed_glyph_fallback for that case), and searching
    anyway risks a wrong-but-internally-coherent shift outscoring the
    correct one: e.g. one that happens to line every cell's sample window
    up with its premium square's own printed text, which lights up nearly
    as much aggregate colour+glyph evidence as real tiles would, board-wide
    (measured on the test set: a working 68/69 photo, score 22.7, was
    nearly wrecked to 0/69 by a shift that scored 26 for exactly this
    reason, before this guard was added).

    The up-to-24 remaining candidates are independent _shift_score() calls,
    which looks like an obvious win to hand to the shared thread pool --
    measured instead (isolated before/after timing on the 6 test-set photos
    that actually trigger this search) and it was consistently *slower*
    threaded (0.56x-0.96x) than the plain sequential loop below, never
    faster. Each _shift_score() call is itself dominated by a 225-iteration
    Python loop of small (48px) cv2 calls (see its own docstring), so the
    unit of work hitting the pool is too fine-grained for thread-dispatch
    overhead to pay for itself -- unlike this module's other per-tile loops,
    which parallelize over coarser, torch-batch-sized or gn.normalize-sized
    work. Left sequential.
    """
    baseline_score = _shift_score(rotated, mesh, np.zeros(2, dtype=np.float32))
    if baseline_score >= SHIFT_SEARCH_TRIGGER:
        return np.zeros(2, dtype=np.float32)

    pitch = float(np.linalg.norm(mesh[0, 1] - mesh[0, 0]))
    offsets = np.array(SHIFT_SEARCH_FRACS) * pitch
    best_shift, best_score = np.zeros(2, dtype=np.float32), baseline_score * SHIFT_NO_SHIFT_BONUS
    for dx in offsets:
        for dy in offsets:
            if dx == 0 and dy == 0:
                continue
            shift = np.array([dx, dy], dtype=np.float32)
            score = _shift_score(rotated, mesh, shift)
            if score > best_score:
                best_score, best_shift = score, shift
    return best_shift


def draw_tile_overlay(rotated, mesh, verdicts):
    """One debug image: green quad = tile, red = empty, confidence label on
    tiles -- a single overview instead of a popup per cell."""
    out = rotated.copy()
    for v in verdicts:
        quad = _cell_quad(mesh, v.row, v.col).astype(np.int32)
        color = (0, 200, 0) if v.is_tile else (0, 0, 200)
        cv2.polylines(out, [quad], True, color, 5)
        if v.is_tile:
            x1, _ = quad.min(axis=0)
            _, y2 = quad.max(axis=0)
            cv2.putText(out, f"{v.confidence:.2f}", (x1 + 2, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return out


def read_board(path, show=True):
    """Detect the board in the given image, warp+rotate it, locate the
    15x15 grid, and run tile detection on every square. Returns (rotated,
    mesh, cells, verdicts, shift), or (None, None, None, None, None) if no
    board or no grid was found. `mesh` and `shift` are exposed (not just
    used internally) so callers needing the raw grid geometry -- e.g.
    read_letters.py's recognition stage, sampling its own higher-res patches
    -- don't have to re-run the whole pipeline themselves the way
    tuner.py's tile_detector subcommand previously had to. `show=False`
    skips the debug popup -- used by eval scripts to batch over the test
    set."""
    image = cv2.imread(path)
    corners = find_board_quad(image)
    if corners is None:
        print(f"No board found in {path}")
        return None, None, None, None, None

    board = warp_board(image, corners)
    rotated = rotate_board(board)

    grid = detect_grid(rotated)
    if grid is None:
        print(f"No grid found in {path}")
        return None, None, None, None, None

    shift = find_parallax_shift(rotated, grid.mesh)
    cells = extract_cells(rotated, grid.mesh, global_shift=shift)
    verdicts = detect_tiles(cells)

    if show:
        show_images(
            [rotated, draw_tile_overlay(rotated, grid.mesh, verdicts)], ["Board", "Tile detection"], height=1200
        )

    return rotated, grid.mesh, cells, verdicts, shift


def main():
    path = "test/in/*_e.jpg"
    paths = glob.glob(path)
    print(paths)

    for path in paths:
        _, _, _, verdicts, _ = read_board(path)
        if verdicts is not None:
            print(f"{path}: {sum(v.is_tile for v in verdicts)} tiles")


if __name__ == "__main__":
    main()
