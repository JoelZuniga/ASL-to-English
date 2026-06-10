from __future__ import annotations

import time
from collections import Counter, deque

import cv2 as cv
import numpy as np

from src.config import (
    CAMERA_INDEX,
    MIN_CONFIDENCE,
    MOTION_THRESHOLD,
    SEQUENCE_WINDOW,
    SMOOTHING_WINDOW,
    WINDOW_NAME,
)
from src.ml.classifier import (
    load_model,
    mediapipe_landmarks_to_values,
    predict_landmarks,
)
from src.ml.motion_classifier import (
    MotionClassifier,
    load_motion_model,
    normalize_raw_sequence,
)
from src.utils.paths import ASL_CLASSIFIER_MODEL, ASL_MOTION_LABELS, ASL_MOTION_MODEL
from src.vision.drawing import draw_hand_landmarks, draw_prediction
from src.vision.hand_tracker import (
    create_hand_landmarker,
    get_latest_result,
    make_mp_image,
)


def _has_motion(buffer: deque[list[float]]) -> bool:
    wrist_x = [frame[0] for frame in buffer]
    return float(np.std(wrist_x)) > MOTION_THRESHOLD


def main() -> None:
    model = load_model(ASL_CLASSIFIER_MODEL)

    motion_model: MotionClassifier | None
    try:
        motion_model = load_motion_model(ASL_MOTION_MODEL, ASL_MOTION_LABELS)
    except FileNotFoundError:
        motion_model = None
        print("Motion model not found — J/Z disabled until trained.")

    landmark_buffer: deque[list[float]] = deque(maxlen=SEQUENCE_WINDOW)
    prediction_history: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    no_hand_frames = 0

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
                no_hand_frames = 0
                hand_landmarks = result.hand_landmarks[0]

                raw_values = mediapipe_landmarks_to_values(hand_landmarks)
                landmark_buffer.append(raw_values)

                if (
                    motion_model is not None
                    and len(landmark_buffer) == SEQUENCE_WINDOW
                    and _has_motion(landmark_buffer)
                ):
                    sequence = normalize_raw_sequence(list(landmark_buffer))
                    prediction = motion_model.predict_sequence(sequence)
                else:
                    prediction = predict_landmarks(model, hand_landmarks)

                prediction_history.append(prediction.label)
                smoothed_label = Counter(prediction_history).most_common(1)[0][0]

                image = draw_hand_landmarks(image, hand_landmarks)
                if prediction.confidence >= MIN_CONFIDENCE:
                    image = draw_prediction(
                        image, smoothed_label, prediction.confidence
                    )
            else:
                no_hand_frames += 1
                if no_hand_frames > SEQUENCE_WINDOW // 2:
                    landmark_buffer.clear()

            cv.imshow(WINDOW_NAME, image)

            if cv.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
