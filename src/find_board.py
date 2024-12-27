import argparse
import glob
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from helper import mix_filter

debug = False


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# noise removal
def remove_noise(image, size=5):
    return cv2.medianBlur(image, size)


# smoothing without removing edges
def bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)


# blur
def blur(image, size=5, sigmaX=0):
    return cv2.GaussianBlur(image, (size, size), sigmaX)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image, size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image, size=5):
    kernel = np.ones((size, size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def make_opening(image, size=5):
    kernel = np.ones((size, size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def make_canny(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2, 1)


# get contours
def get_contours(image):
    # Ensure the image is in grayscale
    if len(image.shape) == 3:  # Check if the image has 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the image is 8-bit
    if image.dtype != "uint8":
        image = cv2.convertScaleAbs(image)

    # Find contours
    return cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


# contour center calculation
def contour_center(contour):
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def remove_white(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the threshold range for "almost white"
    threshold_min = 90  # Adjust this value for sensitivity
    _, mask = cv2.threshold(gray, threshold_min, 255, cv2.THRESH_BINARY)

    # Apply the mask to make "almost white" pixels pure white
    result = cv2.bitwise_or(image, image, mask=mask)

    return result


def preprocess_image(image):
    without_glare = mix_filter(image)
    blurred = blur(without_glare, 9, 10)
    without_background = remove_white(blurred)
    gray = get_grayscale(without_background)
    canny = make_canny(gray)

    return canny


def transform_perspective(image, approx):
    corners = approx.reshape(-1, 2)

    rhombus_corners = np.float32(corners)
    new_corners = np.array(
        [
            [0, 0],
            [image.shape[1], 0],
            [image.shape[1], image.shape[0]],
            [0, image.shape[0]],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rhombus_corners, new_corners)
    warped_image = cv2.warpPerspective(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
    )

    resized_image = cv2.resize(warped_image, (warped_image.shape[1], warped_image.shape[1]))

    return resized_image


def check_contour(approx):
    if len(approx) != 4:
        return False

    distances = [cv2.norm(approx[i] - approx[j]) for i in range(4) for j in range(i + 1, 4)]
    avg_distance = sum(distances) / len(distances)
    tolerance_percent = 50
    tolerance = avg_distance * tolerance_percent / 100
    equal_distances = all(abs(distance - avg_distance) < tolerance for distance in distances)

    angles = []
    for i in range(4):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % 4][0]
        p3 = approx[(i + 2) % 4][0]

        v1 = p1 - p2
        v2 = p3 - p2

        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(np.degrees(angle))

    for angle in angles:
        if not (70 <= angle <= 110):
            return False

    if not equal_distances:
        return False

    contour_area = cv2.contourArea(approx)

    if contour_area < 50000:
        return False

    return True


def find_board(image):
    board = None
    original_image = image.copy()

    preprocessed_image = preprocess_image(image)

    if debug:
        plt.imshow(preprocessed_image)
        plt.show()

    contours = get_contours(preprocessed_image)

    for contour in contours:
        if debug:
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 3)

        hull = cv2.convexHull(contour)
        epsilon = 0.15 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if not check_contour(approx):
            continue

        if debug:
            cv2.drawContours(image, [approx], 0, (0, 255, 255), 3)

        board = transform_perspective(original_image, approx)
        board = cv2.resize(board, (1000, 1000))
        board = mix_filter(board)

        break

    if debug:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    return board


def read_board(image):
    board = find_board(image)

    if debug:
        if board is not None:
            plt.imshow(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
            plt.show()

        else:
            print("No board found")


def from_files():
    for filename in glob.glob("images/camera*.jpg"):
        image = cv2.imread(filename)
        read_board(image)


def from_camera():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    for _ in range(10):
        start = time.time()
        ret, frame = cap.read()

        if ret:
            if debug:
                plt.imshow(frame)
                plt.show()

            read_board(frame)
        else:
            print("No frame")

        end = time.time()
        print(f"Time: {end - start}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    global debug

    parser = argparse.ArgumentParser(description="A program with a --debug flag.")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed logging.",
    )

    args = parser.parse_args()

    try:
        debug = args.debug

        from_camera()
        # from_files(args.debug)

    except KeyboardInterrupt:
        print("\rBreak!")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
