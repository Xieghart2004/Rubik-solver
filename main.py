from Camera import mirror_frame, open_camera, read_frame
from Guide import print_guide, print_move_manual
from cube import create_faces, unmirror_face,missing_faces, build_cube_string, FACE_TO_COLOR
from grid import draw_3x3_grid
from color import COLOR_LETTER, classify_cell_hsv, RUBIK_HSV_RANGES
import kociemba
from collections import Counter
import cv2 as cv

#main.py has 5 jobs mixed together: camera, grid drawing, color detection, cube data handling, solving / printing instructions

def solve_cube(cube_str):
    counts = Counter(cube_str)
    return counts, kociemba.solve(cube_str)

print_guide()

faces = create_faces()
cap = open_camera(0)

while True:
    frame = read_frame(cap)
    if frame is None:
        break

    display = mirror_frame(frame)
    display, boxes = draw_3x3_grid(display)

    detected_colors = []

    for (x1, y1, x2, y2) in boxes:
        margin_x = int((x2 - x1) * 0.25)
        margin_y = int((y2 - y1) * 0.25)
        cell = display[y1 + margin_y : y2 - margin_y, x1 + margin_x : x2 - margin_x]

        color_name, score = classify_cell_hsv(cell)
        letter = COLOR_LETTER.get(color_name, "?")

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv.putText(
            display,
            letter,
            (cx - 10, cy + 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (248, 255, 44),
            2,
            cv.LINE_AA,
        )

        detected_colors.append(color_name)

    cv.imshow("Camera", display)
    key = cv.waitKey(1) & 0xFF

    if key == ord("f"):
        faces["F"] = unmirror_face(detected_colors)
        print("Stored Front face")
    elif key == ord("r"):
        faces["R"] = unmirror_face(detected_colors)
        print("Stored Right face")
    elif key == ord("b"):
        faces["B"] = unmirror_face(detected_colors)
        print("Stored Back face")
    elif key == ord("l"):
        faces["L"] = unmirror_face(detected_colors)
        print("Stored Left face")
    elif key == ord("u"):
        faces["U"] = unmirror_face(detected_colors)
        print("Stored Upper face")
    elif key == ord("d"):
        faces["D"] = unmirror_face(detected_colors)
        print("Stored Down face")
    elif key == ord("s"):
        missing = missing_faces(faces)
        if missing:
            missing_colors = [FACE_TO_COLOR[m] for m in missing]
            print("Not all faces captured yet!")
            print("Missing faces:", missing_colors)
        else:
            cube_str = build_cube_string(faces)
            print("cube_str:", cube_str)

            try:
                counts, solution = solve_cube(cube_str)
                print("Counts:", counts)
                print("solution:", solution)
                print_move_manual()
            except Exception as e:
                print("Solve error:", e)
    elif key == ord("q"):
        break

cap.release()