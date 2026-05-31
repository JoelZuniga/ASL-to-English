from typing import Any

import cv2 as cv
import mediapipe as mp  # type: ignore[import-untyped]

from src.config import NUM_HANDS
from src.utils.paths import HAND_LANDMARKER_MODEL

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result: Any = None


def make_mp_image(image):
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)


def update_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


def create_hand_landmarker():
    if not HAND_LANDMARKER_MODEL.exists():
        raise FileNotFoundError(f"Model file not found at {HAND_LANDMARKER_MODEL}.")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=update_result,
        num_hands=NUM_HANDS,
    )

    return HandLandmarker.create_from_options(options)


def get_latest_result():
    return latest_result
