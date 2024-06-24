import glob

import cv2
import numpy as np
import pytesseract


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


def main():
    alphabet = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ"

    paths = ["images/all_letters8.jpg", "images/image6.jpg", *glob.glob("images/*.jpg")]
    print(paths)

    for path in paths:
        image = cv2.imread(path)

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.imshow("Original", image)

        gray = get_grayscale(image)

        blurred = cv2.GaussianBlur(gray, (7, 7), 5)
        edged = cv2.Canny(blurred, 10, 100)

        cv2.namedWindow("Original2", cv2.WINDOW_NORMAL)
        cv2.imshow("Original2", edged)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        dilate = cv2.dilate(edged, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_copy = image.copy()
        contours_img = image.copy()

        for i in range(len(contours)):
            cv2.drawContours(contours_img, contours, i, (0, 255, 0), 3)
            hull = cv2.convexHull(contours[i])
            epsilon = 0.15 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) == 4:
                distances = [
                    cv2.norm(approx[i] - approx[j])
                    for i in range(4)
                    for j in range(i + 1, 4)
                ]

                avg_distance = sum(distances) / len(distances)

                tolerance_percent = 50
                tolerance = avg_distance * tolerance_percent / 100

                equal_distances = all(
                    abs(distance - avg_distance) < tolerance for distance in distances
                )

                if equal_distances:
                    contour_area = cv2.contourArea(approx)
                    if contour_area > 100000:
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

                        matrix = cv2.getPerspectiveTransform(
                            rhombus_corners, new_corners
                        )
                        warped_image = cv2.warpPerspective(
                            image_copy,
                            matrix,
                            (image_copy.shape[1], image_copy.shape[0]),
                            flags=cv2.INTER_LINEAR,
                        )

                        resized_image = cv2.resize(
                            warped_image, (warped_image.shape[1], warped_image.shape[1])
                        )

                        cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
                        cv2.imshow("Board", resized_image)

                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                        target_color = (160, 200, 225)
                        threshold = 30
                        board = np.zeros((15, 15), dtype=str)
                        board.fill("-")

                        for i in range(15):
                            for j in range(15):
                                tile = resized_image[
                                    75 + i * 133 : 50 + (i + 1) * 133,
                                    175 + j * 132 : 150 + (j + 1) * 132,
                                ]

                                average_color = np.mean(tile, axis=(0, 1))
                                print(average_color)

                                if all(
                                    abs(c1 - c2) < threshold
                                    for c1, c2 in zip(average_color, target_color)
                                ):
                                    print(f"Tile {i}, {j} is a target tile")

                                    gray = get_grayscale(tile)
                                    thresh = thresholding(gray)
                                    opening = make_opening(thresh)

                                    blurred = cv2.GaussianBlur(thresh, (7, 7), 3)
                                    edged = cv2.Canny(blurred, 10, 100)
                                    kernel = cv2.getStructuringElement(
                                        cv2.MORPH_RECT, (3, 3)
                                    )

                                    dilate = cv2.dilate(edged, kernel, iterations=1)
                                    new_contours, _ = cv2.findContours(
                                        dilate,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )

                                    image_center = (
                                        tile.shape[1] // 2,
                                        tile.shape[0] // 2,
                                    )

                                    most_centered_contour = min(
                                        new_contours,
                                        key=lambda c: np.linalg.norm(
                                            np.array(contour_center(c))
                                            - np.array(image_center)
                                        ),
                                    )

                                    tile = thresh

                                    (x, y, w, h) = cv2.boundingRect(
                                        most_centered_contour
                                    )
                                    add = 4
                                    cut_tile = tile[
                                        max(0, y - 2 * add) : min(
                                            tile.shape[0], y + h + add
                                        ),
                                        max(0, x - add) : min(
                                            tile.shape[1], x + w + add
                                        ),
                                    ]

                                    ratio = w / h
                                    tolerance = 1.5

                                    if ratio > tolerance:
                                        print(f"This is not a tile")

                                        continue

                                    cv2.namedWindow("Cut Tile", cv2.WINDOW_NORMAL)
                                    cv2.imshow("Cut Tile", cut_tile)
                                    cv2.namedWindow("Tile", cv2.WINDOW_NORMAL)
                                    cv2.imshow("Tile", tile)
                                    cv2.waitKey(0)

                                    letter = input("Enter letter: ")

                                    if letter == "":
                                        continue

                                    board[i, j] = letter

                                    cv2.imwrite(f"font/{letter}.jpg", cut_tile)

                        print(board)

        cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
        cv2.imshow("Contours", contours_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
