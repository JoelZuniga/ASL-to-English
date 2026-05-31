from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Project directories
MODELS_DIR = PROJECT_ROOT / "models"

# Landmarkers
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"
