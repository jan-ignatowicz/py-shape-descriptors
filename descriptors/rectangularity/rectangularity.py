import math
import cv2 as cv
import numpy as np
import imutils

from descriptors.utils.moments import moments


def rectangularity(image, method='mbr'):
    """
    There are 5 methods to define rectangularity:
    Minimum bounding rectangle: MBR
    Rectangular discrepancy: R'_D
    Robust MBR: R_R
    Agreement method: R_A
    Moment method: R_M
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.160.506&rep=rep1&type=pdf
    https://www.researchgate.net/profile/Paul_Rosin/publication/227273599_Measuring_rectangularity/links/555e1cef08ae6f4dcc8dd1a9/Measuring-rectangularity.pdf

    :param image: np.ndarray: Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary.
    :param method: method of calculating the descriptor
    :return: float in [0, 1]
    """

    if method == "r_b" or method == 'Minimum Bounding Rectangle (MBR)' or method == "mbr":

        region_area, mbr_area = get_mbr_areas(image)
        R_B = region_area / mbr_area
        return R_B

    elif method == "r_d" or method == 'Rectangular Discrepancy':
        """
        formula taken from https://www.researchgate.net/profile/Paul_Rosin/publication/227273599_Measuring_rectangularity/links/555e1cef08ae6f4dcc8dd1a9/Measuring-rectangularity.pdf
        """
        rotated_image = imutils.rotate(image, 45)

        straight_image_descriptor = get_rectangular_discrepancy_descriptor(image)
        rotated_image_descriptor = get_rectangular_discrepancy_descriptor(rotated_image)

        if straight_image_descriptor > rotated_image_descriptor:
            return straight_image_descriptor
        return rotated_image_descriptor

    elif method == "r_r" or method == "Robust MBR":
        rotated_image = imutils.rotate(image, 45)

        straight_image_descriptor = get_robust_mbr_discrepancy(image)
        rotated_image_descriptor = get_robust_mbr_discrepancy(rotated_image)

        if straight_image_descriptor > rotated_image_descriptor:
            return straight_image_descriptor
        return rotated_image_descriptor

    elif method == "r_a" or method == "Agreement method":
        """
        based on rectangle sides calculated in two different ways
        """
        cnt = get_contour(image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

        # first sides
        M = cv.moments(cnt)
        a1, b1 = get_sides_based_on_moments(M)

        # second sides
        A = np.sum(image)  # region area
        P = cv.arcLength(cnt, True)  # True means the shape is a closed figure

        # wrong formula provided in article for side a2
        # https://www.researchgate.net/profile/Paul_Rosin/publication/227273599_Measuring_rectangularity/links/555e1cef08ae6f4dcc8dd1a9/Measuring-rectangularity.pdf
        a2_1 = (P + math.sqrt(abs(P ** 2 - 16 * A))) / 4  # absolute to prevent math domain error
        a2_2 = (P - math.sqrt(abs(P ** 2 - 16 * A))) / 4
        if a2_1 < 0 or a2_2 < 0:
            return 0  # shape is definitely not a rectangle in that case
        elif a2_1 > a2_2:
            a2 = a2_1
        else:
            a2 = a2_2
        b2 = A / a2

        R = abs(a1 - a2) / (a1 + a2) + abs(b1 - b2) / (b1 + b2)
        R_A = 1 - R / 2
        return R_A

    elif method == "r_m" or method == "Moments method":
        m = moments(image)

        # change to the formula given in the article; not m22 but mu22
        # https://www.researchgate.net/profile/Paul_Rosin/publication/227273599_Measuring_rectangularity/links/555e1cef08ae6f4dcc8dd1a9/Measuring-rectangularity.pdf
        R_M = 144 * m['mu22'] / (m['m00'] ** 3)

        if R_M <= 1:
            return R_M
        else:
            return 1 / R_M

    else:
        print("Wrong method.")


def get_mbr_areas(image):
    """
    Method used in mbr and robust mbr methods

    :param image
    :return region_area, mbr_area: calculated areas
    """
    cnt = get_contour(image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    # https://docs.opencv.org/3.4.1/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9
    rect = cv.minAreaRect(cnt)  # output ( center (x,y), (width, height), angle of rotation )
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gaf78d467e024b4d7936cf9397185d2f5c
    box = cv.boxPoints(rect)
    box = np.int0(box)

    region_area = np.sum(image)

    image = image.copy()
    cv.fillConvexPoly(image, box, 1)
    mbr_area = np.sum(image)

    return region_area, mbr_area


def get_rectangular_discrepancy_descriptor(image):
    """
    Method called twice in due to rotation of an image

    :param image
    :return R_D: descriptor
    """
    cnt = get_contour(image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    # sides
    M = cv.moments(cnt)
    a, b = get_sides_based_on_moments(M)

    x, y, w, h = cv.boundingRect(cnt)
    box_image = image[x:w, y:h]

    A1 = np.sum(image)  # complete region
    A2 = np.sum(box_image)  # clipped region
    A3 = a * b  # rectangle

    R = A3 - A2  # difference between rectangle and clipped region
    D = A1 - A2  # difference between whole region and clipped region
    B = A3  # rectangle area

    if B == 0:
        return 0

    R_D = abs(1 - (R + D) / B)

    if R_D > 1:
        return 1

    return R_D


def get_robust_mbr_discrepancy(image):
    cnt = get_contour(image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    region_area, mbr_area = get_mbr_areas(image)

    # sides
    M = cv.moments(cnt)
    a, b = get_sides_based_on_moments(M)

    x, y, w, h = cv.boundingRect(cnt)
    box_image = image[x:w, y:h]

    A1 = region_area  # complete region
    A2 = np.sum(box_image)  # clipped region
    A3 = a * b  # rectangle

    R = A3 - A2  # difference between rectangle and clipped region
    D = A1 - A2  # difference between whole region and clipped region
    I = mbr_area  # intersection of rectangle area and the region

    R_R = abs(1 - (R + D) / I)

    if R_R > 1:
        return 1
    return R_R


def get_sides_based_on_moments(M):
    """
    :param M: moments of shape calculated in OpenCv
    :return tuple a,b: sides of a rectangle
    """

    if M["m00"] != 0:
        a = math.sqrt((6 * (M['mu20'] + M['mu02'] + math.sqrt(
            (M['mu20'] - M['mu02']) ** 2 + 4 * M['mu11'] ** 2))) / (M['m00']))
        b = math.sqrt((6 * (M['mu20'] + M['mu02'] - math.sqrt(
            (M['mu20'] - M['mu02']) ** 2 + 4 * M['mu11'] ** 2))) / (M['m00']))
    else:
        a, b = 0, 0

    return a, b


def get_contour(image, mode, method):
    """
    mode: https://docs.opencv.org/3.4.0/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
    method: https://docs.opencv.org/3.4.0/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
    """

    contours, _ = cv.findContours(image, mode=mode, method=method)

    if len(contours) > 1:
        print("Found more than one blob.")

    cnt = contours[0]

    return cnt
