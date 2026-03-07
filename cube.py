
#handles cube data and conversions. cause I mirrored a camera. so the detected colorr are mirrored.

COLOR_TO_FACE = {
    "white": "U",
    "red": "R",
    "green": "F",
    "yellow": "D",
    "orange": "L",
    "blue": "B",
}
FACE_TO_COLOR = {
    v: k for k, v in COLOR_TO_FACE.items()
}  #dictionary comprehension. convert COLOR_TO_FACE to FACE_TO_COLOR by swapping key and value.
def missing_faces(faces):
    return [k for k, v in faces.items() if v is None]
def create_faces():
    return {"U": None, "R": None, "F": None, "D": None, "L": None, "B": None}
def face_to_str(face9):
    return "".join(COLOR_TO_FACE[c] for c in face9)
def unmirror_face(face9):
    # face9 is [0..8] row-major.
    # Mirror fix = reverse each row: (0,1,2)->(2,1,0), (3,4,5)->(5,4,3), (6,7,8)->(8,7,6)
    return [
        face9[2],
        face9[1],
        face9[0],
        face9[5],
        face9[4],
        face9[3],
        face9[8],
        face9[7],
        face9[6],
    ]
def build_cube_string(faces):
    return (
        face_to_str(faces["U"]) +
        face_to_str(faces["R"]) +
        face_to_str(faces["F"]) +
        face_to_str(faces["D"]) +
        face_to_str(faces["L"]) +
        face_to_str(faces["B"])
    )