import cv2 as cv
import numpy as np
from grid import draw_3x3_grid


print("OpenCV version:", cv.__version__)
cap = cv.VideoCapture(0)

#I just know that the origin of cordinate is at the top-left corner. lol :O
#ROI = Region of Interest

RUBIK_HSV_RANGES = {
    "white": [
        (np.array([0,   0, 160]), np.array([179,  60, 255])) #np is used to create arrays for the lower and upper bounds of the HSV values for white color. The lower bound is [0, 0, 160] and the upper bound is [179, 60, 255]. This means that any pixel with a hue between 0 and 179, saturation between 0 and 60, and value between 160 and 255 will be classified as white.
    ],
    "yellow": [
        (np.array([20,  80, 120]), np.array([35, 255, 255]))
    ],
    "green": [
        (np.array([40,  80,  60]), np.array([85, 255, 255]))
    ],
    "blue": [
        (np.array([90,  80,  40]), np.array([130, 255, 255]))
    ],
    "orange": [
        (np.array([8,  100,  80]), np.array([19, 255, 255]))
    ],
    "red": [
        (np.array([0,  120,  60]), np.array([7, 255, 255])),
        (np.array([170,120,  60]), np.array([179,255,255]))
    ],
}


def classify_cell_hsv(cell_bgr, ranges_dict):
    hsv = cv.cvtColor(cell_bgr, cv.COLOR_BGR2HSV)
    best_name = None 
    best_score = -1 # start at -1 so any real score will be higher. if score is higher than best_score, we update best_name and best_score. 

    for name, ranges in ranges_dict.items(): 
        mask_total = None
        for (lo, hi) in ranges: #range is RUBIK_HSV_RANGES[name], which is a list of (lo, hi) tuples.
            m = cv.inRange(hsv, lo, hi) #inRang arguments: hsv image, lower bound, upper bound. 
            #so m is black and white image where white pixels are those that fall within the specified HSV range.
            if mask_total is None:
                mask_total = m
            else:
                mask_total = cv.bitwise_or(mask_total, m)
            #red has 2 ranges, so we need to combine the masks for both ranges using bitwise OR. For other colors with only one range, mask_total will just be that single mask.
            #bitwise_or is used to combine the masks for multiple ranges of the same color(red)
        score = cv.countNonZero(mask_total)#countnonzero Count how many pixels match this color.
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        # print(ret)#     Boolean (True / False) •	True → frame successfully captured , False → failed to capture frame
        # print(frame) # (height, width, channels)
        break
    # Flip the frame horizontally
    frame = cv.flip(frame, 1)
    frame, boxes = draw_3x3_grid(frame)  # <-- overlay grid 
    # (frame) can stack modifier like cv.flip, cv.resize, etc. to modify the frame before displaying it.

    detected_colors = []

    for idx, (x1, y1, x2, y2) in enumerate(boxes):

        # take center region only (avoid borders)
        margin_x = int((x2 - x1) * 0.25)
        margin_y = int((y2 - y1) * 0.25)

        cell = frame[
            y1 + margin_y : y2 - margin_y,
            x1 + margin_x : x2 - margin_x
        ]

        color_name, score = classify_cell_hsv(cell, RUBIK_HSV_RANGES)

        detected_colors.append(color_name)

        print(f"Grid {idx} → {color_name}")
    
    print(boxes[0])
    print(boxes[2])
    print(boxes[6])
    print(boxes[8])    #I dump the cordinates of the 4 corners

    # (650, 230, 850, 430) -> (x1, y1, x2, y2) of top-left
    # (1070, 230, 1270, 430) --> (x1, y1, x2, y2) of top-right
    # (650, 650, 850, 850) --> (x1, y1, x2, y2) of bottom-left
    # (1070, 650, 1270, 850) --> (x1, y1, x2, y2) of bottom-right


    left   = boxes[0][0]   # x1 of top-left (650)
    top    = boxes[0][1]   # y1 of top-left (230)
    right  = boxes[8][2]   # x2 of bottom-right (1270)
    bottom = boxes[8][3]   # y2 of bottom-right (850)

    roi = frame[top:bottom, left:right]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    #hsvFrame = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blurred, 10, 50)

    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow("Camera", frame)
    

    if cv.waitKey(1) & 0xFF == ord("q"):  # Stop the camera
        break



