from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2 as cv
import mediapipe as mp  # type: ignore[import-untyped]
from mediapipe.tasks.python import BaseOptions  # type: ignore[import-untyped]
from mediapipe.tasks.python.vision import (  # type: ignore[import-untyped]
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDED_CLASSES = {"J", "Z", "del", "nothing", "space"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from labeled ASL image folders."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/SigNN"),
        help="Path to the raw dataset folder containing class subfolders.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/landmarks.csv"),
        help="Path where the landmark CSV file will be saved.",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="Path to the MediaPipe hand landmarker .task model file.",
    )

    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Optional limit for how many images to process per class.",
    )

    return parser.parse_args()


def get_image_paths(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_class_dirs(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_dir() and path.name not in EXCLUDED_CLASSES
    )


def build_csv_header() -> list[str]:
    header = ["label"]

    for landmark_index in range(21):
        header.extend(
            [
                f"x{landmark_index}",
                f"y{landmark_index}",
                f"z{landmark_index}",
            ]
        )

    return header


def load_image_as_mp_image(image_path: Path) -> mp.Image | None:
    image = cv.imread(str(image_path))

    if image is None:
        return None

    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image,
    )


def extract_landmark_row(
    image_path: Path,
    label: str,
    landmarker: Any,
) -> list[str] | None:
    mp_image = load_image_as_mp_image(image_path)

    if mp_image is None:
        return None

    result = landmarker.detect(mp_image)

    if len(result.hand_landmarks) != 1:
        return None

    hand_landmarks = result.hand_landmarks[0]

    row = [label]

    for landmark in hand_landmarks:
        row.extend(
            [
                f"{landmark.x:.8f}",
                f"{landmark.y:.8f}",
                f"{landmark.z:.8f}",
            ]
        )

    return row


def create_hand_landmarker(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model file not found: {model_path}\n"
            "Download hand_landmarker.task and place it at this path, "
            "or pass a custom path using --model-path."
        )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return HandLandmarker.create_from_options(options)


def extract_landmarks(
    input_dir: Path,
    output_file: Path,
    model_path: Path,
    max_images_per_class: int | None,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_images = 0
    successful_images = 0
    failed_images = 0

    class_dirs = get_class_dirs(input_dir)

    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_csv_header())

        with create_hand_landmarker(model_path) as landmarker:
            for class_dir in class_dirs:
                label = class_dir.name
                image_paths = get_image_paths(class_dir)

                if max_images_per_class is not None:
                    image_paths = image_paths[:max_images_per_class]

                class_success_count = 0

                for image_path in image_paths:
                    total_images += 1

                    row = extract_landmark_row(
                        image_path=image_path,
                        label=label,
                        landmarker=landmarker,
                    )

                    if row is None:
                        failed_images += 1
                        continue

                    writer.writerow(row)

                    successful_images += 1
                    class_success_count += 1

                print(f"{label}: {class_success_count} samples extracted")

    print()
    print("Extraction complete")
    print(f"Images processed: {total_images}")
    print(f"Successful detections: {successful_images}")
    print(f"Failed detections: {failed_images}")
    print(f"Output file: {output_file}")


def main() -> None:
    args = parse_args()

    extract_landmarks(
        input_dir=args.input_dir,
        output_file=args.output_file,
        model_path=args.model_path,
        max_images_per_class=args.max_images_per_class,
    )


if __name__ == "__main__":
    main()
