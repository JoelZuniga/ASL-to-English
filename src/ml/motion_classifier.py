from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from src.config import SEQUENCE_WINDOW
from src.ml.classifier import Prediction
from src.ml.features import values_to_feature_vector


def softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    e = np.exp(logits - np.max(logits))
    return np.asarray(e / e.sum(), dtype=np.float32)


class MotionClassifier:
    def __init__(
        self,
        model_path: Path,
        label_to_int: dict[str, int],
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Motion model not found: {model_path}")
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._int_to_label: dict[int, str] = {v: k for k, v in label_to_int.items()}

    def predict_sequence(
        self,
        sequence: npt.NDArray[np.float32],
    ) -> Prediction:
        batch = sequence[np.newaxis].astype(np.float32)
        logits: npt.NDArray[np.float32] = self._session.run(
            ["logits"], {"sequence": batch}
        )[0][0]
        probs = softmax(logits)
        class_idx = int(np.argmax(probs))
        return Prediction(
            label=self._int_to_label[class_idx],
            confidence=float(probs[class_idx]),
        )


def normalize_raw_sequence(
    raw_frames: list[list[float]],
    window: int = SEQUENCE_WINDOW,
) -> npt.NDArray[np.float32]:
    frames = raw_frames[-window:]
    result = np.empty((len(frames), 63), dtype=np.float32)
    for i, frame in enumerate(frames):
        result[i] = values_to_feature_vector(frame)
    return result


def load_motion_model(model_path: Path, labels_path: Path) -> MotionClassifier:
    if not labels_path.exists():
        raise FileNotFoundError(f"Label map not found: {labels_path}")
    label_to_int: dict[str, int] = json.loads(labels_path.read_text())
    return MotionClassifier(model_path, label_to_int)
