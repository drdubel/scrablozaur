import glob

import cv2
import numpy as np


def main():
    paths = glob.glob("images/*.jpg")
    print(paths)

    for path in paths:
        image = cv2.imread(path)

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.imshow("Original", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (9, 9), 5)
        edged = cv2.Canny(blurred, 10, 100)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        dilate = cv2.dilate(edged, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_copy = image.copy()

        for i in range(len(contours)):
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.intp(box)
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
                        cv2.drawContours(image_copy, [approx], 0, (0, 255, 0), 4)
                        cv2.namedWindow("Board Detection", cv2.WINDOW_NORMAL)
                        cv2.imshow("Board Detection", image_copy)
                        image_copy = image.copy()

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

                        rotated_image = cv2.rotate(
                            resized_image, cv2.ROTATE_90_CLOCKWISE
                        )

                        cv2.namedWindow("Board", cv2.WINDOW_NORMAL)
                        cv2.imshow("Board", rotated_image)

                        # rotated_image[100:120, 0:] = [0, 0, 255]

                        # cv2.namedWindow("Tiles", cv2.WINDOW_NORMAL)
                        # cv2.imshow("Tiles", rotated_image)

                        blurred = cv2.GaussianBlur(rotated_image, (15, 15), 9)

                        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

                        lower_red = np.array([0, 175, 100])
                        upper_red = np.array([70, 255, 255])

                        # Threshold the HSV image to get only red colors
                        mask = cv2.inRange(hsv, lower_red, upper_red)

                        # Find contours
                        contours2, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        # Loop over the contours
                        squares = []

                        for contour in contours2:
                            # Approximate contour
                            hull = cv2.convexHull(contour)
                            epsilon = 0.02 * cv2.arcLength(hull, True)
                            approx = cv2.approxPolyDP(hull, epsilon, True)

                            # Check if contour has 4 vertices (a square)
                            if len(approx) == 4:
                                distances = [
                                    cv2.norm(approx[i] - approx[j])
                                    for i in range(4)
                                    for j in range(i + 1, 4)
                                ]
                                # Draw contour
                                print(approx)
                                avg_distance = sum(distances) / len(distances)

                                tolerance_percent = 50
                                tolerance = avg_distance * tolerance_percent / 100

                                equal_distances = all(
                                    abs(distance - avg_distance) < tolerance
                                    for distance in distances
                                )

                                if equal_distances:
                                    contour_area = cv2.contourArea(approx)
                                    if contour_area > 1000:
                                        rect = cv2.minAreaRect(contour)
                                        box = cv2.boxPoints(rect)
                                        box = np.intp(box)
                                        squares.extend([*box])

                                        cv2.drawContours(
                                            rotated_image, [box], 0, (0, 255, 0), 4
                                        )

                        if squares:
                            squares = np.array(squares)

                            rect = cv2.minAreaRect(squares)
                            box = cv2.boxPoints(rect)
                            box = np.intp(box)
                            cv2.drawContours(rotated_image, [box], 0, (255, 255, 0), 15)

                            cv2.namedWindow("Square Detection", cv2.WINDOW_NORMAL)
                            cv2.imshow("Square Detection", rotated_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
