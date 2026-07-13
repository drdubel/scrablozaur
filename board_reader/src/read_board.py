import glob

import cv2
from cv_utils import show_images
from detect_board import find_board_quad, warp_board
from premium_layout import GRID
from rotate_board import rotate_board
from tile_detector import Cell, detect_tiles

# Tuned against a 4080px-wide photo (the majority of test/in/*.jpg); kept as
# fractions of the warped board's side (== the photo's own width -- see
# warp_board()) so cell boxes still land correctly on the smaller
# (4000px-wide) photos in the set instead of drifting by a fixed offset.
X_OFFSET_MUL = 0.0583
Y_OFFSET_MUL = 0.0287
TILE_SIZE_MUL = 0.0588
TILE_MARGIN_MUL = 0.005


def _cell_bbox(side, row, col):
    """Pixel box for cell (row, col) in a `side`x`side` warped+rotated
    board -- pure proportional math, no grid-line detection. The one source
    of truth for cell geometry, shared by extract_cells() and
    draw_tile_overlay() so they can never drift apart."""
    tile_size = int(side * TILE_SIZE_MUL)
    margin = int(side * TILE_MARGIN_MUL)
    x_offset = int(side * X_OFFSET_MUL)
    y_offset = int(side * Y_OFFSET_MUL)
    x1 = col * tile_size + x_offset - margin
    y1 = row * tile_size + y_offset - margin
    return x1, y1, x1 + tile_size + 2 * margin, y1 + tile_size + 2 * margin


def extract_cells(rotated):
    """Slice all 225 cell patches up front. detect_tiles() is a per-photo
    batch model (its per-class and cross-class colour calibration needs
    every cell's features before any single verdict can be computed), so
    every patch must exist before detection runs."""
    side = rotated.shape[0]
    cells = []
    for i in range(GRID):
        for j in range(GRID):
            x1, y1, x2, y2 = _cell_bbox(side, i, j)
            cells.append(Cell(row=i, col=j, patch=rotated[y1:y2, x1:x2]))
    return cells


def draw_tile_overlay(rotated, verdicts):
    """One debug image: green box = tile, red = empty, confidence label on
    tiles -- a single overview instead of a popup per cell."""
    side = rotated.shape[0]
    out = rotated.copy()
    for v in verdicts:
        x1, y1, x2, y2 = _cell_bbox(side, v.row, v.col)
        color = (0, 200, 0) if v.is_tile else (0, 0, 200)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 5)
        if v.is_tile:
            cv2.putText(out, f"{v.confidence:.2f}", (x1 + 2, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return out


def read_board(path, show=True):
    """Detect the board in the given image, warp+rotate it, and run tile
    detection on every square. Returns (rotated, cells, verdicts), or
    (None, None, None) if no board was found. `show=False` skips the debug
    popup -- used by eval_tile_detection.py to batch over the test set."""
    image = cv2.imread(path)
    corners = find_board_quad(image)
    if corners is None:
        print(f"No board found in {path}")
        return None, None, None

    board = warp_board(image, corners)
    rotated = rotate_board(board)
    cells = extract_cells(rotated)
    verdicts = detect_tiles(cells)

    if show:
        show_images([rotated, draw_tile_overlay(rotated, verdicts)], ["Board", "Tile detection"], height=1200)

    return rotated, cells, verdicts


def main():
    path = "test/in/*_e.jpg"
    paths = glob.glob(path)
    print(paths)

    for path in paths:
        _, _, verdicts = read_board(path)
        if verdicts is not None:
            print(f"{path}: {sum(v.is_tile for v in verdicts)} tiles")


if __name__ == "__main__":
    main()
