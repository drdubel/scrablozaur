import glob

import cv2
from cv_utils import show_images
from detect_board import find_board_quad, warp_board
from rotate_board import rotate_board


def detect_grid(path):
    """Detect the board in the given image and return a warped, square version
    of it. Returns None if no board was found."""
    image = cv2.imread(path)
    corners = find_board_quad(image)
    if corners is None:
        print(f"No board found in {path}")
        return None

    board = warp_board(image, corners)
    rotated = rotate_board(board)
    show_images([image, rotated], ["Original", "Warped", "Rotated"])

    return rotated


def main():
    path = "test/in/*_e.jpg"
    paths = glob.glob(path)
    print(paths)

    for path in paths:
        detect_grid(path)


if __name__ == "__main__":
    main()
