import glob
import signal
import sys

import cv2
import numpy as np


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def make_opening(image):
    kernel = np.ones((7, 7), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def make_canny(image):
    return cv2.Canny(image, 100, 200)


def contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def signal_handler(sig, frame):
    """Handle Ctrl+C interruption gracefully"""
    print("\nClosing...")
    cv2.destroyAllWindows()
    sys.exit(0)


# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


def detect_board(image_path):
    image = cv2.imread(image_path)

    gray = get_grayscale(image)

    blurred = cv2.GaussianBlur(gray, (9, 9), 5)
    edged = cv2.Canny(blurred, 10, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    dilate = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    contours_img = image.copy()

    for i in range(len(contours)):
        cv2.drawContours(contours_img, contours, i, (0, 255, 0), 3)
        hull = cv2.convexHull(contours[i])
        perimeter = cv2.arcLength(hull, True)

        approx = None
        for epsilon_fraction in (0.02, 0.05, 0.08, 0.10):
            candidate = cv2.approxPolyDP(hull, epsilon_fraction * perimeter, True)
            if len(candidate) == 4 and cv2.isContourConvex(candidate):
                approx = candidate
                break

        if approx is None:
            continue

        distances = [cv2.norm(approx[i] - approx[j]) for i in range(4) for j in range(i + 1, 4)]

        avg_distance = sum(distances) / len(distances)

        tolerance_percent = 50
        tolerance = avg_distance * tolerance_percent / 100

        equal_distances = all(abs(distance - avg_distance) < tolerance for distance in distances)
        if not equal_distances:
            continue

        img_area = image_copy.shape[0] * image_copy.shape[1]
        contour_area = cv2.contourArea(approx)
        if contour_area < img_area * 0.2:
            continue

        corners = approx.reshape(-1, 2)

        rhombus_corners = np.float32(corners)
        new_corners = np.array(
            [
                [0, 0],
                [image_copy.shape[1], 0],
                [image_copy.shape[1], image_copy.shape[0]],
                [0, image_copy.shape[0]],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(rhombus_corners, new_corners)
        warped_image = cv2.warpPerspective(
            image_copy,
            matrix,
            (image_copy.shape[1], image_copy.shape[0]),
            flags=cv2.INTER_LINEAR,
        )

        resized_image = cv2.resize(warped_image, (warped_image.shape[1], warped_image.shape[1]))

        cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
        cv2.imshow("Board", resized_image)

        # Use shorter wait time to be responsive to Ctrl+C
        while True:
            key = cv2.waitKey(100)  # Wait 100ms
            if key != -1:  # If a key was pressed
                break

    cv2.destroyAllWindows()

    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
    cv2.imshow("Contours", contours_img)

    # Use shorter wait time to be responsive to Ctrl+C
    while True:
        key = cv2.waitKey(100)  # Wait 100ms
        if key != -1:  # If a key was pressed
            break
    cv2.destroyAllWindows()


def main():
    path = "test/in/*_e.jpg"
    paths = glob.glob(path)
    print(paths)

    for path in paths:
        detect_board(path)


if __name__ == "__main__":
    main()
