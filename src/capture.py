import time

import cv2 as cv

from src.config import CAMERA_INDEX, WINDOW_NAME
from src.vision.drawing import draw_hand_landmarks
from src.vision.hand_tracker import (
    create_hand_landmarker,
    get_latest_result,
    make_mp_image,
)


def main():
    with create_hand_landmarker() as landmarker:
        cap = cv.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}.")

        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv.flip(image, 1)

            mp_image = make_mp_image(image)

            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            result = get_latest_result()

            if result is not None and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    image = draw_hand_landmarks(image, hand_landmarks)

            cv.imshow(WINDOW_NAME, image)

            if cv.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
