"""End-to-end pipeline orchestrating the full modeling workflow."""
from __future__ import annotations

import logging
from typing import Dict, Any

from src.components import data_engineering, feature_engineering, model_train, preprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline() -> Dict[str, Any]:
    logger.info("Starting data engineering stage")
    df_scored, threshold = data_engineering.run()

    logger.info("Starting feature engineering stage")
    engineered_df = feature_engineering.engineer_features(df_scored)
    feature_engineering.save_engineered_features(engineered_df)

    logger.info("Starting preprocessing stage")
    prep_paths = preprocessing.run()

    logger.info("Starting model training stage")
    best_model = model_train.run()

    summary = {
        "score_threshold": threshold,
        "prepared_files": {name: str(path) for name, path in prep_paths.items()},
        "best_model": best_model.name,
        "best_model_metrics": best_model.metrics,
    }
    logger.info("Pipeline finished successfully")
    return summary


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
