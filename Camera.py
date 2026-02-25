import cv2 as cv
import numpy as np
from grid import draw_3x3_grid


print("OpenCV version:", cv.__version__)
cap = cv.VideoCapture(0)

#I just know that the origin of cordinate is at the top-left corner. lol :O
#ROI = Region of Interest

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
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blurred, 10, 50)

    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    if cv.waitKey(1) & 0xFF == ord("q"):  # Stop the camera
        break


