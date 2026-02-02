"""Data engineering utilities to generate the High Value target."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreWeights:
    quality: float = 0.35
    size: float = 0.30
    location: float = 0.20
    premium: float = 0.15


@dataclass(frozen=True)
class QualityWeights:
    overall_quality: float = 0.70
    overall_condition: float = 0.30


@dataclass(frozen=True)
class SizeWeights:
    living_area: float = 0.40
    basement: float = 0.25
    garage: float = 0.20
    lot: float = 0.15


def load_raw_dataset(path: str | Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw housing data from the provided path."""
    logger.info("Loading raw dataset from %s", path)
    return pd.read_csv(path, sep="\t")


def _ordinal_to_numeric(series: pd.Series, mapping: dict[str, int], default: int = 0) -> pd.Series:
    return series.map(mapping).fillna(default)


def _identify_premium_neighborhoods(df: pd.DataFrame, top_fraction: float = 0.25) -> Iterable[str]:
    neighborhood_means = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)
    top_n = max(1, int(len(neighborhood_means) * top_fraction))
    return neighborhood_means.head(top_n).index


def _scale_columns(values: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(values)


def _build_quality_score(df: pd.DataFrame, weights: QualityWeights) -> np.ndarray:
    cols = ["Overall Qual", "Overall Cond"]
    quality_df = df[cols].copy()
    quality_df = quality_df.fillna(quality_df.median())
    scaled = _scale_columns(quality_df)
    return scaled[:, 0] * weights.overall_quality + scaled[:, 1] * weights.overall_condition


def _build_size_score(df: pd.DataFrame, weights: SizeWeights) -> np.ndarray:
    cols = ["Gr Liv Area", "Total Bsmt SF", "Garage Area", "Lot Area"]
    size_df = df[cols].copy().fillna(0)
    scaled = _scale_columns(size_df)
    return (
        scaled[:, 0] * weights.living_area
        + scaled[:, 1] * weights.basement
        + scaled[:, 2] * weights.garage
        + scaled[:, 3] * weights.lot
    )


def _build_location_score(df: pd.DataFrame, neighborhoods: Iterable[str]) -> np.ndarray:
    return df["Neighborhood"].isin(neighborhoods).astype(float).values


def _build_premium_score(df: pd.DataFrame) -> np.ndarray:
    premium_num = ["Garage Cars", "Fireplaces", "Full Bath"]
    premium_ord = ["Kitchen Qual", "Exter Qual", "Bsmt Qual"]

    num_scaled = _scale_columns(df[premium_num].fillna(0))
    quality_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "NA": 0}
    ord_numeric = df[premium_ord].apply(lambda s: _ordinal_to_numeric(s, quality_map))
    ord_scaled = _scale_columns(ord_numeric)

    return 0.5 * num_scaled.mean(axis=1) + 0.5 * ord_scaled.mean(axis=1)


def build_composite_score(df: pd.DataFrame, percentile: float = 75.0) -> Tuple[pd.DataFrame, float]:
    df_score = df.copy()

    quality_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    bsmt_map = {**quality_map, "NA": 0}
    df_score["Kitchen Qual_num"] = _ordinal_to_numeric(df_score["Kitchen Qual"], quality_map)
    df_score["Exter Qual_num"] = _ordinal_to_numeric(df_score["Exter Qual"], quality_map)
    df_score["Bsmt Qual_num"] = _ordinal_to_numeric(df_score["Bsmt Qual"], bsmt_map)

    premium_neighborhoods = _identify_premium_neighborhoods(df_score)
    df_score["Premium_Neighborhood"] = _build_location_score(df_score, premium_neighborhoods)

    quality_score = _build_quality_score(df_score, QualityWeights())
    size_score = _build_size_score(df_score, SizeWeights())
    location_score = df_score["Premium_Neighborhood"].values
    premium_score = _build_premium_score(df_score)

    weights = ScoreWeights()
    df_score["Composite_Score"] = (
        quality_score * weights.quality
        + size_score * weights.size
        + location_score * weights.location
        + premium_score * weights.premium
    )

    threshold = np.percentile(df_score["Composite_Score"], percentile)
    df_score["High_Value"] = (df_score["Composite_Score"] >= threshold).astype(int)

    logger.info(
        "Composite score built with percentile %.1f (threshold %.3f) and class balance %.2f%%",
        percentile,
        threshold,
        df_score["High_Value"].mean() * 100,
    )

    return df_score, float(threshold)


def generate_high_value_dataset(
    raw_path: Path | str = config.RAW_DATA_PATH,
    output_path: Path | str = config.WITH_SCORE_PATH,
    percentile: float = 75.0,
) -> Tuple[pd.DataFrame, float]:
    df_raw = load_raw_dataset(raw_path)
    df_scored, threshold = build_composite_score(df_raw, percentile)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(output, index=False)
    logger.info("Saved scored dataset to %s", output)
    return df_scored, threshold


def run(raw_path: str | Path | None = None, output_path: str | Path | None = None) -> Tuple[pd.DataFrame, float]:
    """Convenience wrapper for CLI/automation usage."""
    raw = raw_path or config.RAW_DATA_PATH
    output = output_path or config.WITH_SCORE_PATH
    return generate_high_value_dataset(raw, output)
