import cv2 as cv

def draw_3x3_grid(Grid_frame, cell=200, gap=10, thickness=2):
    """
    Draw a centered 3x3 grid on the frame.
    Returns: (frame, boxes)
    boxes = [(x1,y1,x2,y2), ...] 9 items, row-major order
    """
    h, w = Grid_frame.shape[:2]

    grid_size = 3 * cell + 2 * gap
    start_x = w // 2 - grid_size // 2
    start_y = h // 2 - grid_size // 2

    boxes = []
    for r in range(3):
        for c in range(3):
            x1 = start_x + c * (cell + gap)
            y1 = start_y + r * (cell + gap)
            x2 = x1 + cell
            y2 = y1 + cell

            cv.rectangle(Grid_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            boxes.append((x1, y1, x2, y2))

    return Grid_frame, boxes