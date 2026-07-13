import glob

import cv2
import numpy as np
from cv_utils import show_images
from detect_board import find_board_quad, warp_board
from rotate_board import rotate_board

X_OFFSET = 134
Y_OFFSET = 66
TILE_SIZE_MUL = 0.0589
TILE_MARGIN_MUL = 0.006


def tile_mask(tile):
    """Create a mask for the given tile to detect if it's empty or not."""
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return mask


def is_tile_empty(tile):
    """Check if a tile is empty (i.e., mostly white)."""
    mask = tile_mask(tile)
    white_ratio = np.sum(mask == 255) / (tile.shape[0] * tile.shape[1])
    return white_ratio > 0.9


def read_board(path):
    """Detect the board in the given image and return a warped, square version
    of it. Returns None if no board was found."""
    image = cv2.imread(path)
    corners = find_board_quad(image)
    if corners is None:
        print(f"No board found in {path}")
        return None

    board = warp_board(image, corners)
    rotated = rotate_board(board)
    tile_size = int(rotated.shape[0] * TILE_SIZE_MUL)
    tile_margin = int(rotated.shape[0] * TILE_MARGIN_MUL)

    for i in range(15):
        for j in range(15):
            x1 = j * tile_size + X_OFFSET - tile_margin
            y1 = i * tile_size + Y_OFFSET - tile_margin
            x2 = (j + 1) * tile_size + X_OFFSET + tile_margin
            y2 = (i + 1) * tile_size + Y_OFFSET + tile_margin
            tile = rotated[y1:y2, x1:x2]
            mask = tile_mask(tile)
            is_empty = is_tile_empty(tile)
            show_images(
                [tile, mask],
                ["Empty " if is_empty else "Not empty " + f"Tile {i},{j}", f"Tile {i},{j} Mask"],
                height=500,
            )

    show_images([rotated], ["Board"], height=1200)

    return rotated


def main():
    path = "test/in/*_e.jpg"
    paths = glob.glob(path)
    print(paths)

    for path in paths:
        read_board(path)


if __name__ == "__main__":
    main()
