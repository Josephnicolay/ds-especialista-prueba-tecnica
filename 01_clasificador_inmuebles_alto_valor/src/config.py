"""Project-wide configuration constants."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Housing.txt"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_FINAL_DIR = PROCESSED_DIR / "final"
WITH_SCORE_PATH = PROCESSED_DIR / "housing_with_score.csv"
ENGINEERED_FEATURES_PATH = PROCESSED_DIR / "housing_engineered_features.csv"

MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_RANDOM_STATE = 42
TEST_SIZE = 0.2

X_TRAIN_PATH = PROCESSED_FINAL_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_FINAL_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_FINAL_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_FINAL_DIR / "y_test.csv"
FEATURE_NAMES_PATH = PROCESSED_FINAL_DIR / "feature_names.csv"
SCALER_PATH = PROCESSED_FINAL_DIR / "scaler.pkl"
