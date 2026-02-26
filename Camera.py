import cv2 as cv
import numpy as np
from grid import draw_3x3_grid
import kociemba
from collections import Counter

print("OpenCV version:", cv.__version__)
print("press u r f d l b → store faces")
print("FACE Green to the camera ")

cap = cv.VideoCapture(0)
#I just know that the origin of cordinate is at the top-left corner. lol :O
#ROI = Region of Interest
COLOR_LETTER = {
    "white": "W",
    "yellow": "Y",
    "green": "G",
    "blue": "B",
    "orange": "O",
    "red": "R",
}

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
        (np.array([0,120,60]), np.array([7,255,255]))    ],
    "red": [
        (np.array([170,120,60]), np.array([179,255,255]))
    ],
}

faces = {"U": None, "R": None, "F": None, "D": None, "L": None, "B": None}
# white  → U up
# red    → R right
# blue   → B back
# orange → L left
# green  → F front
# yellow → D down

COLOR_TO_FACE = {
    "white": "U",
    "red": "R",
    "green": "F",
    "yellow": "D",
    "orange": "L",
    "blue": "B",
}
FACE_TO_COLOR = {v: k for k, v in COLOR_TO_FACE.items()} # this is the reverse mapping of COLOR_TO_FACE
#This is called a dictionary comprehension.
# For every (key, value) pair inside COLOR_TO_FACE,
# create a new dictionary where:
# new_key = value
# new_value = key
def face_to_str(face9):
    return "".join(COLOR_TO_FACE[c] for c in face9)

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

def show_face(face_key):
    s = face_to_str(faces[face_key])
    print(face_key)
    print(s[0:3])
    print(s[3:6])
    print(s[6:9])
    print()

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
        #(0, boxes[0]). (1, boxes[1]). (2, boxes[2])
        # take center region only (avoid borders)
        margin_x = int((x2 - x1) * 0.25)
        margin_y = int((y2 - y1) * 0.25)
        cell = frame[
            y1 + margin_y : y2 - margin_y,
            x1 + margin_x : x2 - margin_x
        ]

        color_name, score = classify_cell_hsv(cell, RUBIK_HSV_RANGES)
        letter = COLOR_LETTER.get(color_name, "?") #.get is used to retrieve the letter corresponding to the detected color name from the COLOR_LETTER dictionary. If the color name is not found in the dictionary, it returns "?" as a default value.
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv.putText(
            frame,
            letter,
            (cx - 10, cy + 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (248, 255, 44),                      # text color (BGR)
            2,
            cv.LINE_AA
        )
        detected_colors.append(color_name)
        #print(f"Grid {idx} → {color_name}")
    #---------------
    # my red and orage ranges are not good enough, so I want to see the average hue of the cell to adjust the ranges.
    #---------------
    # hsv_cell = cv.cvtColor(cell, cv.COLOR_BGR2HSV)
    # avg_h = np.mean(hsv_cell[:,:,0]) 
    # print("Average Hue:", avg_h)

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
    key = cv.waitKey(1) & 0xFF

    # --- store faces ---
    if key == ord('u'):
        faces["U"] = detected_colors.copy()
        print("Stored U face")
        print("White side's value:", faces["U"])
    elif key == ord('r'):
        faces["R"] = detected_colors.copy()
        print("Stored R face")
        print("Red side's value:", faces["R"])
    elif key == ord('f'):
        faces["F"] = detected_colors.copy()
        print("Stored F face")
        print("Green side's value:", faces["F"])
    elif key == ord('d'):
        faces["D"] = detected_colors.copy()
        print("Stored D face")
        print("Yellow side's value:", faces["D"])
    elif key == ord('l'):
        faces["L"] = detected_colors.copy()
        print("Stored L face")
        print("Orange side's value:", faces["L"])
    elif key == ord('b'):
        faces["B"] = detected_colors.copy()
        print("Stored B face")
        print("Blue side's value:", faces["B"])

    elif key == ord('q'):
        break
    
# --- solve when all faces exist (optional: trigger on 's' instead) ---
    elif key == ord('s'):
        
        if not all(v is not None for v in faces.values()):
            print("Not all faces captured yet!")
            missing = [k for k, v in faces.items() if v is None]
            if missing:
                missing_colors = [FACE_TO_COLOR[m] for m in missing]
                print("Missing faces:", missing_colors)
        else:
            cube_str = (
                face_to_str(faces["U"]) +
                face_to_str(faces["R"]) +
                face_to_str(faces["F"]) +
                face_to_str(faces["D"]) +
                face_to_str(faces["L"]) +
                face_to_str(faces["B"])
            )
            counts = Counter(cube_str)
            print("Counts:", counts)
            print("cube_str length:", len(cube_str))
            print("cube_str:", cube_str)

            try:
                print("solution:", kociemba.solve(cube_str))
            except Exception as e:
                print("Solve error:", e)

            

