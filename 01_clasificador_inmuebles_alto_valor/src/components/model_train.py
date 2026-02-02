"""Model training and evaluation utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier

from src import config

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    estimator: Pipeline
    best_params: Dict[str, object]
    metrics: Dict[str, object]


DEFAULT_MODELS: Dict[str, Tuple[object, Dict[str, Iterable[object]]]] = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"classifier__C": [0.01, 0.1, 1, 10], "classifier__penalty": ["l2"]},
    ),
    "Random Forest": (
        RandomForestClassifier(),
        {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None, 10, 20]},
    ),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__max_depth": [3, 6],
        },
    ),
}


def load_training_data(
    X_train_path: Path | str = config.X_TRAIN_PATH,
    X_test_path: Path | str = config.X_TEST_PATH,
    y_train_path: Path | str = config.Y_TRAIN_PATH,
    y_test_path: Path | str = config.Y_TEST_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()
    logger.info("Loaded training and testing splits")
    return X_train, X_test, y_train, y_test


def build_pipeline(model: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", PowerTransformer()),
            ("feature_selector", SelectKBest(score_func=f_classif, k=20)),
            ("classifier", model),
        ]
    )


def train_model(
    model: object,
    param_grid: Dict[str, Iterable[object]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[Pipeline, Dict[str, object]]:
    pipeline = build_pipeline(model)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.DEFAULT_RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    logger.info("Finished training for %s", model.__class__.__name__)
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, object]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-y_proba))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    logger.info("Evaluation metrics: accuracy=%.3f, roc_auc=%.3f", metrics["accuracy"], metrics["roc_auc"])
    return metrics


def train_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    models: Dict[str, Tuple[object, Dict[str, Iterable[object]]]] = DEFAULT_MODELS,
) -> Dict[str, ModelResult]:
    results: Dict[str, ModelResult] = {}
    for name, (model, params) in models.items():
        logger.info("Training %s", name)
        best_estimator, best_params = train_model(model, params, X_train, y_train)
        metrics = evaluate_model(best_estimator, X_test, y_test)
        results[name] = ModelResult(name=name, estimator=best_estimator, best_params=best_params, metrics=metrics)
    return results


def select_best_model(results: Dict[str, ModelResult]) -> ModelResult:
    if not results:
        raise ValueError("No model results to select from")
    best = max(results.values(), key=lambda result: result.metrics["roc_auc"])
    logger.info("Best model: %s (ROC AUC=%.3f)", best.name, best.metrics["roc_auc"])
    return best


def save_model(result: ModelResult, directory: Path | str = config.MODELS_DIR) -> Path:
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"best_model_{result.name.replace(' ', '_').lower()}.joblib"
    joblib.dump(result.estimator, filename)
    logger.info("Persisted %s to %s", result.name, filename)
    return filename


def run(
    models: Dict[str, Tuple[object, Dict[str, Iterable[object]]]] | None = None,
) -> ModelResult:
    X_train, X_test, y_train, y_test = load_training_data()
    results = train_models(X_train, y_train, X_test, y_test, models or DEFAULT_MODELS)
    best = select_best_model(results)
    save_model(best)
    return best
