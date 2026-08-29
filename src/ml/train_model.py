from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    RF_ESTIMATORS,
    TEST_SIZE,
)
from src.ml.features import (
    flip_landmarks,
    get_landmark_columns,
    landmarks_to_feature_vector,
    row_to_landmarks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline ASL static letter classifier."
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/processed/landmarks.csv"),
        help="Path to the processed landmark CSV file.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("models/asl_static_rf.pkl"),
        help="Path where the trained model will be saved.",
    )

    return parser.parse_args()


def load_dataset(input_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    data = pd.read_csv(input_file)

    if "label" not in data.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    landmark_columns = get_landmark_columns()
    missing_columns = [
        column for column in landmark_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Input CSV is missing landmark columns: {missing_columns}")

    features = data.loc[:, landmark_columns]
    labels = data["label"]

    return features, labels


def build_augmented_matrix(
    features: pd.DataFrame,
    labels: pd.Series,
    augment_flipped: bool = True,
) -> tuple[list[list[float]], list[str]]:
    feature_vectors: list[list[float]] = []
    expanded_labels: list[str] = []

    for row, label in zip(features.itertuples(index=False, name=None), labels):
        landmarks = row_to_landmarks(row)

        feature_vector = landmarks_to_feature_vector(landmarks)
        feature_vectors.append(feature_vector.tolist())
        expanded_labels.append(str(label))

        if augment_flipped:
            flipped_landmarks = flip_landmarks(landmarks)
            flipped_feature_vector = landmarks_to_feature_vector(flipped_landmarks)
            feature_vectors.append(flipped_feature_vector.tolist())
            expanded_labels.append(str(label))

    return feature_vectors, expanded_labels


def train_model(
    feature_matrix: list[list[float]],
    labels: list[str],
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(feature_matrix, labels)

    return model


def save_model(model: RandomForestClassifier, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_file)


def main() -> None:
    args = parse_args()

    raw_features, labels = load_dataset(args.input_file)

    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
        raw_features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    x_train, y_train = build_augmented_matrix(x_train_raw, y_train_raw)
    x_test, y_test = build_augmented_matrix(
        x_test_raw, y_test_raw, augment_flipped=False
    )

    model = train_model(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, predictions))

    save_model(model, args.output_file)

    print()
    print(f"Model saved to: {args.output_file}")


if __name__ == "__main__":
    main()
