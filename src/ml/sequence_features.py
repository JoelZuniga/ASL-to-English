from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from src.config import SEQUENCE_WINDOW
from src.ml.features import values_to_feature_vector

LandmarkSeq: npt.NDArray[np.float32]


# Normalize each frame
def normalize_sequence(
    raw_sequence: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    result = np.empty_like(raw_sequence)
    for i, row in enumerate(raw_sequence):
        result[i] = values_to_feature_vector(row)
    return result


# Crop to fixed frame window
def crop_sequence(
    sequence: npt.NDArray[np.float32],
    window: int = SEQUENCE_WINDOW,
    mode: Literal["random", "last"] = "random",
) -> npt.NDArray[np.float32] | None:
    t = len(sequence)
    if t < window:
        return None
    if mode == "last":
        return sequence[-window:]
    start = int(np.random.randint(0, t - window + 1))
    return sequence[start : start + window]


# Augment raw sequences and flip coordinates for prediction with both hands
def augment_sequence(
    raw_sequence: npt.NDArray[np.float32],
    flip: bool = False,
    noise_std: float = 0.002,
    speed_factor: float = 1.0,
) -> npt.NDArray[np.float32]:
    seq = raw_sequence.copy()

    if flip:
        seq[:, 0::3] = 1.0 - seq[:, 0::3]

    if noise_std > 0.0:
        seq += np.random.normal(0.0, noise_std, seq.shape).astype(np.float32)

    if speed_factor != 1.0:
        t = len(seq)
        new_len = max(1, int(t / speed_factor))
        indices = np.linspace(0, t - 1, new_len).astype(int)
        seq = seq[indices]

    return seq


# Load and preprocess sequences for lstm
def load_sequences_from_dir(
    sequences_dir: Path,
    labels: list[str],
    window: int = SEQUENCE_WINDOW,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    label_to_int: dict[str, int] = {label: i for i, label in enumerate(labels)}
    x_list: list[npt.NDArray[np.float32]] = []
    y_list: list[int] = []

    for label in labels:
        label_dir = sequences_dir / label
        if not label_dir.exists():
            continue

        npy_files = sorted(label_dir.glob("seq_*.npy"))
        class_idx = label_to_int[label]

        for npy_path in npy_files:
            raw = np.load(str(npy_path)).astype(np.float32)

            for flip in [False, True]:
                aug = augment_sequence(raw, flip=flip, noise_std=0.002)
                normalized = normalize_sequence(aug)
                cropped = crop_sequence(normalized, window=window, mode="random")
                if cropped is None:
                    continue
                x_list.append(cropped)
                y_list.append(class_idx)

    if not x_list:
        raise ValueError(
            f"No valid sequences found in {sequences_dir}. "
            "Run src.ml.record_sequences first."
        )

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    return x, y
