import cv2 as cv
import numpy as np

COLOR_LETTER = {
    "white": "W",
    "yellow": "Y",
    "green": "G",
    "blue": "B",
    "orange": "O",
    "red": "R",
}
RUBIK_HSV_RANGES = {
    "white": [(np.array([0, 0, 160]), np.array([179, 60, 255]),)],
    "yellow": [(np.array([20, 80, 120]), np.array([35, 255, 255]))],
    "green": [(np.array([40, 80, 60]), np.array([85, 255, 255]))],
    "blue": [(np.array([90, 80, 40]), np.array([130, 255, 255]))],
    "orange": [(np.array([0, 120, 60]), np.array([7, 255, 255]))],
    "red": [(np.array([170, 120, 60]), np.array([179, 255, 255]))],
}
def classify_cell_hsv(cell_bgr, ranges_dict=RUBIK_HSV_RANGES):
    hsv = cv.cvtColor(cell_bgr, cv.COLOR_BGR2HSV)
    best_name = None
    best_score = (
        -1
    )  # start at -1 so any real score will be higher. if score is higher than best_score, we update best_name and best_score.

    for name, ranges in ranges_dict.items():
        mask_total = None
        for (lo,hi,) in (ranges):  # range is RUBIK_HSV_RANGES[name], which is a list of (lo, hi) tuples.
            m = cv.inRange(
                hsv, lo, hi
            )  # inRang arguments: hsv image, lower bound, upper bound.
            # so m is black and white image where white pixels are those that fall within the specified HSV range.
            if mask_total is None:
                mask_total = m
            else:
                mask_total = cv.bitwise_or(mask_total, m)
            # red has 2 ranges, so we need to combine the masks for both ranges using bitwise OR. For other colors with only one range, mask_total will just be that single mask.
            # bitwise_or is used to combine the masks for multiple ranges of the same color(red)
        score = cv.countNonZero(
            mask_total
        )  # countnonzero Count how many pixels match this color.
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score