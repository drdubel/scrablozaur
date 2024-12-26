import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def remove_glare(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    grayimage1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.threshold(grayimage1, 150, 255, cv2.THRESH_BINARY)[1]
    result2 = cv2.inpaint(image, mask2, 0.1, cv2.INPAINT_TELEA)

    cv2.namedWindow("float image", cv2.WINDOW_NORMAL)
    cv2.imshow("float image", image)
    cv2.namedWindow("float glare", cv2.WINDOW_NORMAL)
    cv2.imshow("float glare", result2)

    while cv2.waitKey(0) & 0xFF != ord("q"):
        pass

    cv2.destroyAllWindows()

    return result2


def remove_background(image):
    # image = remove_glare(image)

    green_mask = image[:, :, 1] > image[:, :, 2]
    green_mask = (green_mask.astype(np.uint8)) * 255
    green_mask = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    green3_mask = (green_mask > 0).astype(np.uint8) * 255
    image_green = cv2.bitwise_and(green3_mask, image)

    return image_green


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

    if not equal_distances:
        return False

    contour_area = cv2.contourArea(approx)

    if contour_area < 40000:
        return False

    return True


def find_board(image):
    board = None
    without_background = remove_background(image.copy())

    plt.imshow(cv2.cvtColor(without_background, cv2.COLOR_BGR2RGB))
    plt.show()

    contours = get_contours(without_background)

    for contour in contours:
        # cv2.drawContours(image, [contour], 0, (0, 255, 0), 3)

        hull = cv2.convexHull(contour)
        epsilon = 0.15 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if not check_contour(approx):
            continue

        cv2.drawContours(image, [approx], 0, (0, 255, 255), 3)

        board = transform_perspective(image, approx)

        break

    # cv2.namedWindow("float contours", cv2.WINDOW_NORMAL)
    # cv2.imshow("float contours", image)
    #
    # while cv2.waitKey(0) & 0xFF != ord("q"):
    #    pass
    #
    # cv2.destroyAllWindows()

    return board


def read_board(image):
    board = find_board(image)

    if board is not None:
        plt.imshow(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
        plt.show()

    else:
        print("No board found")


def read_files():
    for filename in glob.glob("images/all_letters*.jpg"):
        image = cv2.imread(filename)
        read_board(image)


def from_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            plt.imshow(frame)
            plt.show()
            read_board(frame)
        else:
            print("No frame")

    cap.release()
    cv2.destroyAllWindows()


def main():
    try:
        from_camera()

    except KeyboardInterrupt:
        print("\rBreak!")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
