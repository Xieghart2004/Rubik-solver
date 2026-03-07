import cv2 as cv

def open_camera(index=0):
    cap = cv.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def mirror_frame(frame):
    return cv.flip(frame, 1)