from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.ml.features import values_to_feature_vector


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


# Load trained classifier from disk
def load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier model file not found: {model_path}")

    return joblib.load(model_path)


# Convert MediaPipe hand landmarks into the flat format
def mediapipe_landmarks_to_values(hand_landmarks: Any) -> list[float]:

    values: list[float] = []

    for landmark in hand_landmarks:
        values.extend(
            [
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
            ]
        )

    return values


# Predict an ASL letter from one detected hands landmarks
def predict_landmarks(model: Any, hand_landmarks: Any) -> Prediction:

    values = mediapipe_landmarks_to_values(hand_landmarks)
    feature_vector = values_to_feature_vector(values)

    prediction = model.predict([feature_vector.tolist()])[0]
    probabilities = model.predict_proba([feature_vector.tolist()])[0]

    confidence = float(max(probabilities))

    return Prediction(
        label=str(prediction),
        confidence=confidence,
    )
