from pathlib import Path
from typing import Any
import cv2 as cv
import mediapipe as mp 
import time


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker 
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode 

latest_result: Any = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure the model is correctly placed.")

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]

def update_result(result, output_image, timestamp_ms): 
    global latest_result
    latest_result = result
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_result,
    num_hands=2
    )
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0) 
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    while cap.isOpened(): 
        success, image = cap.read() 
        if not success: 
            print('Ignoring empty camera frame.') 
            continue 
        image = cv.flip(image, 1)
        timestamp_ms = int(time.time() * 1000)
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image) 
        landmarker.detect_async(mp_image, timestamp_ms)
        if latest_result is not None and latest_result.hand_landmarks:
            h, w, _ = image.shape
            for hand_landmarks in latest_result.hand_landmarks:
                points = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * w) 
                    y = int(landmark.y * h) 
                    points.append((x, y))
                    cv.circle(image, (x, y), 5, (0, 255, 0), -1)
                if len(points) == 21:
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        cv.line(image, points[start_idx], points[end_idx], (255, 0, 0), 2)
        cv.imshow("Hand Landmarker", image) 
        if cv.waitKey(1) & 0xFF == 27: 
            break 
    cap.release()
    cv.destroyAllWindows()
        
         


