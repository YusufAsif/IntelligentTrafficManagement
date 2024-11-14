import cv2
import numpy as np

def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_ranges = {"Red": ((0, 100, 100), (10, 255, 255)),
                    "Green": ((40, 100, 100), (80, 255, 255)),
                    "Blue": ((100, 100, 100), (140, 255, 255)),
                    "Yellow": ((20, 100, 100), (40, 255, 255)),
                    "White": ((0, 0, 200), (180, 25, 255)),
                    "Black": ((0, 0, 0), (180, 255, 50))}
    for color, (lower, upper) in color_ranges.items():
        if cv2.inRange(hsv, np.array(lower), np.array(upper)).any():
            return color
    return "Unknown"

def get_direction(prev_pos, cur_pos):
    dx, dy = cur_pos[0] - prev_pos[0], cur_pos[1] - prev_pos[1]
    if abs(dx) > abs(dy):
        return "Left" if dx < 0 else "Right"
    else:
        return "Up" if dy < 0 else "Down"