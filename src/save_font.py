import glob

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


def custom_threshold(image, middle, threshold):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if abs(image[i, j] - middle) < threshold:
                image[i, j] = 255
            else:
                image[i, j] = 0


def main():
    layout = [
        [*["A"] * 8, "Ą", *["B"] * 2, *["C"] * 3, "Ć"],
        [*["D"] * 3, *["E"] * 7, "Ę", "F", *["G"] * 2, "H"],
        ["H", *["I"] * 8, *["J"] * 2, *["K"] * 3, "L"],
        ["A", "L", *["Ł"] * 2, *["M"] * 3, *["N"] * 5, "Ń", *["O"] * 2],
        [*["O"] * 4, "Ó", *["P"] * 3, *["R"] * 4, *["S"] * 3],
        ["S", "Ś", *["T"] * 3, *["U"] * 2, *["W"] * 4, *["Y"] * 4],
        [*["Z"] * 5, "Ź", "Ż", "", ""],
    ]
    alphabet = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ"
    ids = {letter: 0 for letter in alphabet}

    paths = [
        *glob.glob("images/all_letters*.jpg"),
        "images/image6.jpg",
        *glob.glob("images/*.jpg"),
    ]
    print(paths)

    for path in paths:
        image = cv2.imread(path)

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.imshow("Original", image)

        gray = get_grayscale(image)

        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edged = cv2.Canny(blurred, 120, 255, 1)

        cv2.namedWindow("Original2", cv2.WINDOW_NORMAL)
        cv2.imshow("Original2", edged)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        dilate = cv2.dilate(edged, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_copy = image.copy()
        contours_img = image.copy()

        for k in range(len(contours)):
            cv2.drawContours(contours_img, contours, k, (0, 255, 0), 3)
            hull = cv2.convexHull(contours[k])
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

                        resized_image = cv2.resize(warped_image, (2296, 2296))
                        print(resized_image.shape)
                        image = resized_image

                        cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
                        cv2.imshow("Board", image)
                        key = cv2.waitKey(0) & 0xFF

                        if key == ord("q"):
                            cv2.destroyAllWindows()

                            continue

                        while key == ord("r"):
                            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                            cv2.imshow("Board", image)
                            key = cv2.waitKey(0) & 0xFF

                        cv2.destroyAllWindows()

                        target_color = (160, 200, 225)
                        threshold = 100
                        board = np.zeros((15, 15), dtype=str)
                        board.fill("-")

                        for i in range(15):
                            for j in range(15):
                                tile = image[
                                    30 + i * 133 : 100 + (i + 1) * 133,
                                    130 + j * 132 : 180 + (j + 1) * 132,
                                ].copy()

                                average_color = np.mean(tile, axis=(0, 1))
                                print(average_color)

                                if all(
                                    abs(c1 - c2) < threshold
                                    for c1, c2 in zip(average_color, target_color)
                                ):
                                    print(f"Tile {i}, {j} is a target tile")

                                    tile_copy = tile.copy()
                                    resized_tile = tile.copy()
                                    tile_copy = cv2.GaussianBlur(tile_copy, (3, 3), 1)
                                    tile_copy[
                                        np.where(
                                            (tile_copy > [210, 220, 220]).all(axis=2)
                                        )
                                    ] = [0, 0, 0]
                                    tile_copy[
                                        np.where(
                                            (tile_copy < [175, 255, 255]).all(axis=2)
                                        )
                                    ] = [0, 0, 0]

                                    gray = get_grayscale(tile_copy)

                                    blurred = cv2.GaussianBlur(gray, (5, 5), 3)
                                    custom_threshold(blurred, 230, 15)
                                    # thresh = thresholding(blurred)
                                    edged = cv2.Canny(blurred, 120, 150)
                                    kernel = cv2.getStructuringElement(
                                        cv2.MORPH_RECT, (1, 1)
                                    )

                                    dilate = cv2.dilate(blurred, kernel, iterations=1)
                                    new_contours, _ = cv2.findContours(
                                        dilate,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )
                                    boxes = []

                                    for new_contour in new_contours:
                                        cv2.drawContours(
                                            tile, [new_contour], 0, (0, 255, 0), 3
                                        )
                                        # rectangle = cv2.minAreaRect(new_contour)
                                        # box = cv2.boxPoints(rectangle)
                                        # box = np.intp(box)
                                        hull = cv2.convexHull(new_contour)
                                        epsilon = 0.15 * cv2.arcLength(hull, True)
                                        approx = cv2.approxPolyDP(hull, epsilon, True)

                                        if len(approx) == 4:
                                            distances = [
                                                cv2.norm(approx[x] - approx[y])
                                                for x in range(4)
                                                for y in range(x + 1, 4)
                                            ]

                                            avg_distance = sum(distances) / len(
                                                distances
                                            )

                                            tolerance_percent = 50
                                            tolerance = (
                                                avg_distance * tolerance_percent / 100
                                            )

                                            equal_distances = all(
                                                abs(distance - avg_distance) < tolerance
                                                for distance in distances
                                            )

                                            if equal_distances:
                                                contour_area = cv2.contourArea(approx)
                                                if contour_area > 5000:
                                                    # rectangle = cv2.boundingRect(approx)
                                                    # x, y, w, h = rectangle
                                                    # box = np.array(
                                                    #    [
                                                    #        [x, y + h],
                                                    #        [x, y],
                                                    #        [x + w, y],
                                                    #        [x + w, y + h],
                                                    #    ]
                                                    # )
                                                    rectangle = cv2.minAreaRect(approx)
                                                    box = cv2.boxPoints(rectangle)
                                                    box = np.intp(box)
                                                    cv2.drawContours(
                                                        tile,
                                                        [box],
                                                        0,
                                                        (255, 0, 0),
                                                        3,
                                                    )
                                                    boxes.append(
                                                        (
                                                            max(
                                                                [
                                                                    distance
                                                                    / avg_distance
                                                                    for distance in distances
                                                                ]
                                                            ),
                                                            box,
                                                        )
                                                    )

                                    if boxes:
                                        box = min(boxes, key=lambda x: x[0])[1]

                                        corners = box.reshape(-1, 2)

                                        rhombus_corners = np.float32(corners)
                                        new_corners = np.array(
                                            [
                                                [0, 0],
                                                [resized_tile.shape[1], 0],
                                                [
                                                    resized_tile.shape[1],
                                                    resized_tile.shape[0],
                                                ],
                                                [0, resized_tile.shape[0]],
                                            ],
                                            dtype=np.float32,
                                        )

                                        print(rhombus_corners, new_corners)

                                        matrix = cv2.getPerspectiveTransform(
                                            rhombus_corners, new_corners
                                        )
                                        warped_tile = cv2.warpPerspective(
                                            resized_tile,
                                            matrix,
                                            (
                                                resized_tile.shape[1],
                                                resized_tile.shape[0],
                                            ),
                                            flags=cv2.INTER_LINEAR,
                                        )

                                        resized_tile = cv2.resize(
                                            warped_tile, (112, 112)
                                        )

                                        gray_resized_tile = get_grayscale(resized_tile)

                                        # Draw a 30 by 30 rectangle in the bottom right corner
                                        # Calculate the sum of pixel values in each corner
                                        top_left_sum = np.sum(
                                            gray_resized_tile[4:27, 4:27]
                                        )
                                        top_right_sum = np.sum(
                                            gray_resized_tile[4:27, -27:-4]
                                        )
                                        bottom_left_sum = np.sum(
                                            gray_resized_tile[-27:-4, 4:27]
                                        )
                                        bottom_right_sum = np.sum(
                                            gray_resized_tile[-27:-4, -27:-4]
                                        )

                                        # Draw rectangles in the corners
                                        # cv2.rectangle(
                                        #    resized_tile,
                                        #    (4, 4),
                                        #    (27, 27),
                                        #    (0, 0, 255),
                                        #    2,
                                        # )  # Top left corner
                                        # cv2.rectangle(
                                        #    resized_tile,
                                        #    (resized_tile.shape[1] - 27, 4),
                                        #    (resized_tile.shape[1] - 4, 27),
                                        #    (0, 0, 255),
                                        #    2,
                                        # )  # Top right corner
                                        # cv2.rectangle(
                                        #    resized_tile,
                                        #    (4, resized_tile.shape[0] - 27),
                                        #    (27, resized_tile.shape[0] - 4),
                                        #    (0, 0, 255),
                                        #    2,
                                        # )  # Bottom left corner
                                        # cv2.rectangle(
                                        #    resized_tile,
                                        #    (
                                        #        resized_tile.shape[1] - 27,
                                        #        resized_tile.shape[0] - 27,
                                        #    ),
                                        #    (
                                        #        resized_tile.shape[1] - 4,
                                        #        resized_tile.shape[0] - 4,
                                        #    ),
                                        #    (0, 0, 255),
                                        #    2,
                                        # )  # Bottom right corner

                                        # Find the corner with the maximum sum
                                        corner_sums = [
                                            top_left_sum,
                                            top_right_sum,
                                            bottom_left_sum,
                                            bottom_right_sum,
                                        ]
                                        print(corner_sums)
                                        max_sum_corner = np.argmin(corner_sums)

                                        # Rotate the image so that the corner with the maximum sum is in the right bottom corner
                                        if max_sum_corner == 1:  # Top right corner
                                            resized_tile = cv2.rotate(
                                                resized_tile,
                                                cv2.ROTATE_90_CLOCKWISE,
                                            )
                                        elif max_sum_corner == 2:  # Bottom left corner
                                            resized_tile = cv2.rotate(
                                                resized_tile,
                                                cv2.ROTATE_90_COUNTERCLOCKWISE,
                                            )

                                    # tile = gray

                                    # (x, y, w, h) = cv2.boundingRect(
                                    #    most_centered_contour
                                    # )
                                    # add = 4
                                    # cut_tile = tile[
                                    #    max(0, y - 2 * add) : min(
                                    #        tile.shape[0], y + h + add
                                    #    ),
                                    #    max(0, x - add) : min(
                                    #        tile.shape[1], x + w + add
                                    #    ),
                                    # ]
                                    #
                                    # ratio = w / h
                                    # tolerance = 1.5
                                    #
                                    # if ratio > tolerance:
                                    #    print(f"This is not a tile")
                                    #
                                    #    continue
                                    #
                                    # cv2.namedWindow("Cut Tile", cv2.WINDOW_NORMAL)
                                    # cv2.imshow("Cut Tile", cut_tile)
                                    gray_resized_tile = get_grayscale(resized_tile)

                                    cv2.namedWindow("Tile", cv2.WINDOW_NORMAL)
                                    cv2.imshow("Tile", tile)
                                    cv2.namedWindow("Til3", cv2.WINDOW_NORMAL)
                                    cv2.imshow("Tile3", tile_copy)
                                    cv2.namedWindow("Tile4", cv2.WINDOW_NORMAL)
                                    cv2.imshow("Tile4", blurred)
                                    cv2.namedWindow("Tile2", cv2.WINDOW_NORMAL)
                                    cv2.imshow("Tile2", gray_resized_tile)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()

                                    # letter = input("Enter letter: ")
                                    try:
                                        letter = layout[i][j]
                                    except IndexError:
                                        print("za daleko...")
                                        break

                                    if letter == "":
                                        continue

                                    board[i, j] = letter

                                    # cv2.imwrite(f"font/{letter}.jpg", cut_tile)
                                    cv2.imwrite(
                                        f"font/{letter}{ids[letter]}.jpg",
                                        gray_resized_tile,
                                    )
                                    ids[letter] += 1

                        print(board)

        cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
        cv2.imshow("Contours", contours_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
