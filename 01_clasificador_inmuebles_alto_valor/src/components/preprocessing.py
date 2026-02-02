"""Dataset preparation utilities for modeling."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

from src import config

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset fails integrity checks."""


def load_engineered_dataset(path: Path | str = config.ENGINEERED_FEATURES_PATH) -> pd.DataFrame:
    logger.info("Loading engineered dataset from %s", path)
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> None:
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        raise DataValidationError("Dataset contains infinite values")
    logger.info("Dataset passed validation checks")


def split_features_target(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=["High_Value"])
    y = df["High_Value"]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, PowerTransformer]:
    scaler = PowerTransformer()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    logger.info("Applied PowerTransformer scaling to %d features", X_train.shape[1])
    return X_train_scaled, X_test_scaled, scaler


def save_prepared_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: PowerTransformer,
) -> Dict[str, Path]:
    config.PROCESSED_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "X_train": config.X_TRAIN_PATH,
        "X_test": config.X_TEST_PATH,
        "y_train": config.Y_TRAIN_PATH,
        "y_test": config.Y_TEST_PATH,
        "feature_names": config.FEATURE_NAMES_PATH,
        "scaler": config.SCALER_PATH,
    }

    X_train.to_csv(paths["X_train"], index=False)
    X_test.to_csv(paths["X_test"], index=False)
    y_train.to_csv(paths["y_train"], index=False, header=True)
    y_test.to_csv(paths["y_test"], index=False, header=True)
    pd.DataFrame({"feature": X_train.columns}).to_csv(paths["feature_names"], index=False)
    joblib.dump(scaler, paths["scaler"])
    logger.info("Saved prepared datasets to %s", config.PROCESSED_FINAL_DIR)
    return paths


def run(
    input_path: Path | str | None = None,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> Dict[str, Path]:
    dataset_path = Path(input_path) if input_path else config.ENGINEERED_FEATURES_PATH
    df = load_engineered_dataset(dataset_path)
    validate_dataset(df)
    X_train, X_test, y_train, y_test = split_features_target(df, test_size, random_state)
    _, _, scaler = scale_features(X_train, X_test)
    return save_prepared_datasets(X_train, X_test, y_train, y_test, scaler)
