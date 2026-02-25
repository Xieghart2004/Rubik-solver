import cv2 as cv
import numpy
from grid import draw_3x3_grid


print("OpenCV version:", cv.__version__)


cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        # print(ret)#     Boolean (True / False) •	True → frame successfully captured , False → failed to capture frame
        # print(frame) # (height, width, channels)
        break
    frame, boxes = draw_3x3_grid(frame)  # <-- overlay grid
    # Flip the frame horizontally
    frame = cv.flip(frame, 1)
    # (frame) can stack modifier like cv.flip, cv.resize, etc. to modify the frame before displaying it.


    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blurred, 10, 50)

    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):  # Stop the camera
        break


