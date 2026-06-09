from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Project directories
MODELS_DIR = PROJECT_ROOT / "models"

# Landmarkers
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"

# ASL Classifier
ASL_CLASSIFIER_MODEL = MODELS_DIR / "asl_static_rf.pkl"

# Motion classifier artifacts
ASL_MOTION_MODEL = MODELS_DIR / "asl_motion_lstm.onnx"
ASL_MOTION_LABELS = MODELS_DIR / "asl_motion_lstm_labels.json"

# Sequence training data
SEQUENCES_DIR = PROJECT_ROOT / "data" / "raw" / "sequences"
