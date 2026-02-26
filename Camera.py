import cv2 as cv
import numpy as np
from grid import draw_3x3_grid
import kociemba
from collections import Counter

print("OpenCV version:", cv.__version__ + "\n")
print("=== RUBIK'S CUBE CAPTURE GUIDE ===\n")
print("Press U R F D L B to store each face")


print("1) Hold the GREEN face toward the camera.")
print("   Make sure the WHITE face is on TOP.")
print("   Press 'F' to store.\n")

print("2) Rotate the cube to the RIGHT.")
print("   The RED face should now face the camera.")
print("   Keep the WHITE face on TOP.")
print("   Press 'R' to store.\n")

print("3) Rotate the cube to the RIGHT again.")
print("   The BLUE face should now face the camera.")
print("   Keep the WHITE face on TOP.")
print("   Press 'B' to store.\n")

print("4) Rotate the cube to the RIGHT again.")
print("   The ORANGE face should now face the camera.")
print("   Keep the WHITE face on TOP.")
print("   Press 'L' to store.\n")

print("5) Turn The GREEN side faces the camera again.")
print("   Then rotate the cube UP so WHITE faces the camera.")
print("   Press 'U' to store.\n")

print("6) Rotate the cube DOWN so YELLOW faces the camera.")
print("   Press 'D' to store.\n")

print("After all 6 faces are stored, press 'S' to solve. \n")
print("==================================== GOOD LUCK! ====================================\n")


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

def unmirror_face(face9):
    # face9 is [0..8] row-major.
    # Mirror fix = reverse each row: (0,1,2)->(2,1,0), (3,4,5)->(5,4,3), (6,7,8)->(8,7,6)
    return [
        face9[2], face9[1], face9[0],
        face9[5], face9[4], face9[3],
        face9[8], face9[7], face9[6],
    ]
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
        break
    display = cv.flip(frame, 1) # flip the frame horizontally for a mirror-like display
    
    frame, boxes = draw_3x3_grid(display)  # <-- overlay grid 
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
            display,
            letter,
            (cx - 10, cy + 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (248, 255, 44),                      # text color (BGR)
            2,
            cv.LINE_AA
        )
        detected_colors.append(color_name)

    left   = boxes[0][0]   # x1 of top-left (650)
    top    = boxes[0][1]   # y1 of top-left (230)
    right  = boxes[8][2]   # x2 of bottom-right (1270)
    bottom = boxes[8][3]   # y2 of bottom-right (850)

    cv.imshow("Camera", display)


    roi = frame[top:bottom, left:right]
    
    key = cv.waitKey(1) & 0xFF

    # --- store faces ---
    if key == ord('u'):
        faces["U"] = unmirror_face(detected_colors)
        print("Stored Upper face\n")
        print("Rotate the cube to down so the YELLOW center faces the camera make sure that the orange side is on the right and then press 'D'.")
        #print("White side's value:", faces["U"])
    elif key == ord('r'):
        faces["R"] = unmirror_face(detected_colors)
        print("Stored Right face\n")
        print("Rotate the cube to the right so the BLUE center faces the camera make sure that WHITE stays on top, then press 'B'.")
        #print("Red side's value:", faces["R"])
    elif key == ord('f'):
        faces["F"] = unmirror_face(detected_colors)
        print("Stored Front face\n")
        print("Rotate the cube to the right so the RED center faces the camera and WHITE stays on top, then press 'R'.")
        
        #print("Green side's value:", faces["F"])
    elif key == ord('d'):
        faces["D"] = unmirror_face(detected_colors)
        print("Stored Down face\n")
        print("Press 'S' to solve the cube.")
        #print("Stored D face")
        #print("Yellow side's value:", faces["D"])
    elif key == ord('l'):
        faces["L"] = unmirror_face(detected_colors)
        print("Stored Left face\n")
        print("Rotate the cube to the right so the GREEN center faces the camera and make sure that white face is still on top, Then rotate the cube up so WHITE faces the camera then press 'U'.")
        #print("Stored L face")
        #print("Orange side's value:", faces["L"])
    elif key == ord('b'):
        faces["B"] = unmirror_face(detected_colors)
        print("Stored Back face\n")
        print("Rotate the cube to the right so the ORANGE center faces the camera and WHITE stays on top, then press 'L'.")
        #print("Stored B face")
        #print("Blue side's value:", faces["B"])

    elif key == ord('q'):
        break
    
# --- solve when all faces exist (optional: trigger on 's' instead) ---
    elif key == ord('s'):
        for k in ["U","R","F","D","L","B"]:
            show_face(k)
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

            

