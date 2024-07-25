import glob

import cv2
import numpy as np
import pytesseract
from requests import get


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
    image = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if abs(image[i, j] - middle) < threshold:
                image[i, j] = 255
            else:
                image[i, j] = 0

    return image


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
    alphabet = "AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻl"
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

                                    image_center = (
                                        tile.shape[1] // 2,
                                        tile.shape[0] // 2,
                                    )
                                    original_tile = tile.copy()

                                    gray = get_grayscale(original_tile)

                                    blurred = cv2.GaussianBlur(gray, (5, 5), 3)
                                    thresh = custom_threshold(blurred, 100, 75)
                                    edged = cv2.Canny(thresh, 120, 255)
                                    kernel = cv2.getStructuringElement(
                                        cv2.MORPH_RECT, (3, 3)
                                    )

                                    dilate = cv2.dilate(edged, kernel, iterations=1)
                                    new_contours, _ = cv2.findContours(
                                        dilate,
                                        cv2.RETR_CCOMP,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )

                                    new_contours = [
                                        c
                                        for c in new_contours
                                        if cv2.contourArea(c) > 1000
                                    ]

                                    for new_contour in new_contours:
                                        cv2.drawContours(
                                            tile, [new_contour], -1, (0, 255, 0), 3
                                        )

                                    bounding_boxes = [
                                        cv2.boundingRect(c) for c in new_contours
                                    ]

                                    bounding_boxes_contours = [
                                        np.array(
                                            [
                                                [[x, y]],
                                                [[x + w, y]],
                                                [[x + w, y + h]],
                                                [[x, y + h]],
                                            ]
                                        )
                                        for (x, y, w, h) in bounding_boxes
                                    ]

                                    filtered_bounding_boxes = [
                                        bb
                                        for bb in bounding_boxes_contours
                                        if cv2.pointPolygonTest(bb, image_center, False)
                                        >= 0
                                        and cv2.contourArea(bb) < 10000
                                    ]

                                    if not filtered_bounding_boxes:
                                        continue

                                    for (
                                        filtered_bounding_box
                                    ) in filtered_bounding_boxes:
                                        cv2.drawContours(
                                            tile,
                                            [filtered_bounding_box],
                                            -1,
                                            (255, 255, 0),
                                            2,
                                        )

                                    smallest_bounding_box = max(
                                        filtered_bounding_boxes, key=cv2.contourArea
                                    )

                                    cv2.drawContours(
                                        tile,
                                        [smallest_bounding_box],
                                        -1,
                                        (0, 0, 0),
                                        1,
                                    )

                                    # tile = dilate

                                    (x, y, w, h) = cv2.boundingRect(
                                        smallest_bounding_box
                                    )
                                    add = 2

                                    gray = get_grayscale(original_tile)
                                    blurred = cv2.GaussianBlur(gray, (3, 3), 3)
                                    thresh = custom_threshold(gray, 100, 80)

                                    letter = thresh[
                                        max(0, y - add) : min(
                                            tile.shape[0], y + h + add
                                        ),
                                        max(0, x - add) : min(
                                            tile.shape[1], x + w + add
                                        ),
                                    ]
                                    letter ^= 255

                                    ratio = w / h
                                    tolerance = 1.5

                                    if ratio > tolerance:
                                        print("This is not a tile")

                                        cv2.namedWindow("Letter", cv2.WINDOW_NORMAL)
                                        cv2.imshow("Letter", letter)
                                        cv2.waitKey(0)

                                        continue

                                    # FOR VISUALIZATION PURPOSES
                                    scale_percent = 300
                                    width = int(letter.shape[1] * scale_percent / 100)
                                    height = int(letter.shape[0] * scale_percent / 100)
                                    dim = (width, height)
                                    letter = cv2.resize(
                                        letter, dim, interpolation=cv2.INTER_AREA
                                    )

                                    custom_config = rf"--oem 3 --psm 10 -l pol -c tessedit_char_whitelist={alphabet}"
                                    text = pytesseract.image_to_string(
                                        letter,
                                        config=custom_config,
                                    )

                                    custom_config2 = rf"--oem 3 --psm 7 -l pol -c tessedit_char_whitelist={alphabet}"
                                    normal_text = pytesseract.image_to_string(
                                        letter,
                                        config=custom_config2,
                                    )

                                    print(text, normal_text)

                                    if (
                                        len([x for x in normal_text if x in alphabet])
                                        == 1
                                        and text
                                    ):
                                        if "l" in text:
                                            text = "I"

                                        board[i][j] = text
                                        print(f"The text is: '{text}'")

                                    else:
                                        print(f"The text is too long: '{text}'")

                                    cv2.namedWindow("Letter", cv2.WINDOW_NORMAL)
                                    cv2.imshow("Letter", letter)
                                    cv2.namedWindow("Tile", cv2.WINDOW_NORMAL)
                                    cv2.imshow("Tile", tile)
                                    cv2.waitKey(0)

                                    # letter = input("Enter letter: ")
                                    # try:
                                    #    letter = layout[i][j]
                                    # except IndexError:
                                    #    print("za daleko...")
                                    #    break
                                    #
                                    # if letter == "":
                                    #    continue
                                    #
                                    # board[i, j] = letter

                                    # cv2.imwrite(f"font/{letter}.jpg", cut_tile)
                                    # cv2.imwrite(
                                    #    f"font/{letter}{ids[letter]}.jpg",
                                    #    tile,
                                    # )
                                    # ids[letter] += 1
                        cv2.destroyAllWindows()

                        print(board)

        cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
        cv2.imshow("Contours", contours_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
