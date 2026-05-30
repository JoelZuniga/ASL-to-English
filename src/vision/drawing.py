import cv2 as cv
from src.config import HAND_CONNECTIONS

def draw_hand_landmarks(image, hand_landmarks):
    h, w, _ = image.shape
    points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))
        cv.circle(image, (x, y), 5, (0, 255, 0), -1)

    if len(points) == 21:
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv.line(image, points[start_idx], points[end_idx], (255, 0, 0), 2)

    return image