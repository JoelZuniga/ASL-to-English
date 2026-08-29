#   python -m src.ml.record_sequences --label J
#   python -m src.ml.record_sequences --label Z --num-sequences 200

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Any

import cv2 as cv
import mediapipe as mp
import numpy as np
import numpy.typing as npt

from src.ml.classifier import mediapipe_landmarks_to_values
from src.utils.paths import HAND_LANDMARKER_MODEL, SEQUENCES_DIR
from src.vision.hand_tracker import make_mp_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record ASL motion letter sequences from webcam."
    )
    parser.add_argument("--label", required=True, choices=["J", "Z"])
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=300,
        help="Total sequences to record.",
    )
    parser.add_argument(
        "--record-frames",
        type=int,
        default=60,
        help="Frames per sequence at ~30 fps (~2 s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SEQUENCES_DIR,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=HAND_LANDMARKER_MODEL,
    )
    return parser.parse_args()


def create_landmarker(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found: {model_path}\n"
            "Download hand_landmarker.task and place it at that path."
        )
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def countdown(
    cap: cv.VideoCapture,
    label: str,
    seq_num: int,
    total: int,
) -> bool:
    for count in [3, 2, 1]:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]
            cv.putText(
                frame,
                f"Sign '{label}'  ({seq_num}/{total})",
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv.putText(
                frame,
                str(count),
                (w // 2 - 25, h // 2 + 30),
                cv.FONT_HERSHEY_SIMPLEX,
                4.0,
                (0, 220, 0),
                5,
            )
            cv.imshow("Record Sequences", frame)
            if cv.waitKey(1) & 0xFF == 27:
                return False
    return True


def record_sequence(
    cap: cv.VideoCapture,
    landmarker: Any,
    num_frames: int,
) -> npt.NDArray[np.float32] | None:
    frames: list[list[float]] = []

    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv.flip(frame, 1)
        result = landmarker.detect_for_video(
            make_mp_image(frame), int(time.time() * 1000)
        )

        if not result.hand_landmarks:
            cv.putText(
                frame,
                "No hand — move closer",
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 220),
                2,
            )
            cv.imshow("Record Sequences", frame)
            cv.waitKey(1)
            return None

        frames.append(mediapipe_landmarks_to_values(result.hand_landmarks[0]))

        progress = len(frames) / num_frames
        bar_len = int(progress * 200)
        cv.rectangle(frame, (20, 45), (220, 70), (50, 50, 50), -1)
        cv.rectangle(frame, (20, 45), (20 + bar_len, 70), (0, 220, 0), -1)
        cv.putText(
            frame,
            f"REC  {len(frames)}/{num_frames}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 220),
            2,
        )
        cv.imshow("Record Sequences", frame)
        if cv.waitKey(1) & 0xFF == 27:
            return None

    return np.array(frames, dtype=np.float32)


def save_sequence(
    sequence: npt.NDArray[np.float32],
    output_dir: Path,
    label: str,
) -> Path:
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    path = label_dir / f"seq_{uuid.uuid4().hex[:8]}.npy"
    np.save(str(path), sequence)
    return path


def run_recording(
    label: str,
    output_dir: Path,
    num_sequences: int,
    record_frames: int,
    model_path: Path,
) -> None:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam at index 0.")

    saved = 0
    print(f"Recording {num_sequences} sequences for '{label}'. ESC to stop early.")
    print(f"Output → {output_dir / label}\n")

    with create_landmarker(model_path) as landmarker:
        while saved < num_sequences:
            if not countdown(cap, label, saved + 1, num_sequences):
                break

            sequence = record_sequence(cap, landmarker, record_frames)

            if sequence is None:
                print("  Hand lost — retrying.")
                continue

            path = save_sequence(sequence, output_dir, label)
            saved += 1
            print(f"  [{saved}/{num_sequences}]  {path.name}  {sequence.shape}")

            # Pause between recordings
            rest_end = time.time() + 0.4
            stop = False
            while time.time() < rest_end:
                ok, frame = cap.read()
                if ok:
                    frame = cv.flip(frame, 1)
                    cv.putText(
                        frame,
                        f"Saved {saved}/{num_sequences}",
                        (20, 35),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 220, 0),
                        2,
                    )
                    cv.imshow("Record Sequences", frame)
                if cv.waitKey(1) & 0xFF == 27:
                    stop = True
                    break
            if stop:
                break

    cap.release()
    cv.destroyAllWindows()
    print(f"\nDone. {saved} sequences saved to {output_dir / label}")


def main() -> None:
    args = parse_args()
    run_recording(
        label=args.label,
        output_dir=args.output_dir,
        num_sequences=args.num_sequences,
        record_frames=args.record_frames,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
