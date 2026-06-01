from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from src.config import (
    COORDINATE_COUNT,
    EPSILON,
    LANDMARK_COUNT,
    WRIST_INDEX,
)

FEATURE_COUNT = LANDMARK_COUNT * COORDINATE_COUNT
LandmarkArray: TypeAlias = npt.NDArray[np.float32]


# Return the expected CSV landmark column names in order.
def get_landmark_columns() -> list[str]:

    columns: list[str] = []

    for landmark_index in range(LANDMARK_COUNT):
        columns.extend(
            [
                f"x{landmark_index}",
                f"y{landmark_index}",
                f"z{landmark_index}",
            ]
        )

    return columns


# Convert a flat landmark row into 21x3 landmark array.
def row_to_landmarks(values: Iterable[float]) -> LandmarkArray:

    array = np.asarray(list(values), dtype=np.float32)

    if array.size != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} landmark values, got {array.size}.")

    return array.reshape(LANDMARK_COUNT, COORDINATE_COUNT)


# Center landmarks around the wrist.
def center_landmarks(landmarks: LandmarkArray) -> LandmarkArray:

    validate_landmarks_shape(landmarks)

    wrist = landmarks[WRIST_INDEX]

    return landmarks - wrist  # type: ignore[no-any-return]


# Calculate a scale factor based on the farthest landmark from wrist.
def get_scale_factor(centered_landmarks: LandmarkArray) -> float:

    validate_landmarks_shape(centered_landmarks)

    distances = np.linalg.norm(centered_landmarks, axis=1)
    scale = float(np.max(distances))

    if scale < EPSILON:
        return 1.0

    return scale


# Center and scale landmarks.
def normalize_landmarks(landmarks: LandmarkArray) -> LandmarkArray:

    validate_landmarks_shape(landmarks)

    centered_landmarks = center_landmarks(landmarks)
    scale = get_scale_factor(centered_landmarks)

    return centered_landmarks / scale


# Convert landmarks into a normalized flat feature vector.
def landmarks_to_feature_vector(landmarks: LandmarkArray) -> LandmarkArray:

    normalized_landmarks = normalize_landmarks(landmarks)

    return normalized_landmarks.flatten()


# Convert raw flat CSV landmark values into a normalized feature vector.
def values_to_feature_vector(values: Iterable[float]) -> LandmarkArray:

    landmarks = row_to_landmarks(values)

    return landmarks_to_feature_vector(landmarks)


# Validate that landmarks have the expected shape of 21x3.
def validate_landmarks_shape(landmarks: LandmarkArray) -> None:
    if landmarks.shape != (LANDMARK_COUNT, COORDINATE_COUNT):
        raise ValueError(
            "Expected landmarks with shape "
            f"({LANDMARK_COUNT}, {COORDINATE_COUNT}), got {landmarks.shape}."
        )
