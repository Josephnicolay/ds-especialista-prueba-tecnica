"""Feature engineering transformations for housing dataset."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

ENGINEERED_FEATURES: List[str] = [
    "House_Age",
    "Years_Since_Remod",
    "Was_Remodeled",
    "Garage_Age",
    "Is_New_House",
    "Total_Living_Area",
    "Total_Area_Including_Garage",
    "Basement_Ratio",
    "Garage_Ratio",
    "First_Second_Ratio",
    "Area_Per_Room",
    "Building_Density",
    "Total_Full_Baths",
    "Total_Half_Baths",
    "Total_Bathrooms",
    "Bath_Bedroom_Ratio",
    "Has_Multiple_Baths",
    "Total_Rooms_Baths",
    "Has_Pool",
    "Has_Fireplace",
    "Has_Garage",
    "Has_Basement",
    "Has_Deck_Porch",
    "Total_Outdoor_SF",
    "Has_Large_Garage",
    "Luxury_Score",
    "Total_Quality_Score",
    "Avg_Quality_Score",
    "Is_High_Quality",
    "Is_Excellent_Quality",
    "Quality_Condition_Score",
    "Is_Single_Family",
    "Is_Two_Story",
    "Is_One_Story",
    "Is_Residential_Low_Density",
    "Quality_x_Size",
    "Premium_Neighborhood_x_Quality",
    "New_x_Quality",
    "Area_x_Bathrooms",
    "Is_Normal_Sale",
    "Is_Partial_Sale",
    "Is_Abnormal_Sale",
    "Is_Peak_Season",
]


def _add_temporal_features(df: pd.DataFrame) -> None:
    df["House_Age"] = df["Yr Sold"] - df["Year Built"]
    df["Years_Since_Remod"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["Was_Remodeled"] = (df["Year Built"] != df["Year Remod/Add"]).astype(int)
    df["Garage_Age"] = (df["Yr Sold"] - df["Garage Yr Blt"]).fillna(0)
    df["Is_New_House"] = (df["House_Age"] <= 1).astype(int)


def _add_area_features(df: pd.DataFrame) -> None:
    df["Total_Living_Area"] = df["Gr Liv Area"] + df["Total Bsmt SF"]
    df["Total_Area_Including_Garage"] = df["Total_Living_Area"] + df["Garage Area"]
    df["Basement_Ratio"] = df["Total Bsmt SF"] / (df["Gr Liv Area"] + 1)
    df["Garage_Ratio"] = df["Garage Area"] / (df["Gr Liv Area"] + 1)
    df["First_Second_Ratio"] = df["1st Flr SF"] / (df["2nd Flr SF"] + 1)
    df["Area_Per_Room"] = df["Gr Liv Area"] / (df["TotRms AbvGrd"] + 1)
    df["Building_Density"] = df["Gr Liv Area"] / (df["Lot Area"] + 1)


def _add_bathroom_features(df: pd.DataFrame) -> None:
    df["Total_Full_Baths"] = df["Bsmt Full Bath"] + df["Full Bath"]
    df["Total_Half_Baths"] = df["Bsmt Half Bath"] + df["Half Bath"]
    df["Total_Bathrooms"] = df["Total_Full_Baths"] + df["Total_Half_Baths"] * 0.5
    df["Bath_Bedroom_Ratio"] = df["Total_Bathrooms"] / (df["Bedroom AbvGr"] + 1)
    df["Has_Multiple_Baths"] = (df["Total_Full_Baths"] >= 2).astype(int)
    df["Total_Rooms_Baths"] = df["TotRms AbvGrd"] + df["Total_Bathrooms"]


def _add_luxury_features(df: pd.DataFrame) -> None:
    df["Has_Pool"] = (df["Pool Area"] > 0).astype(int)
    df["Has_Fireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["Has_Garage"] = (df["Garage Cars"] > 0).astype(int)
    df["Has_Basement"] = (df["Total Bsmt SF"] > 0).astype(int)
    df["Has_Deck_Porch"] = (
        (df["Wood Deck SF"] > 0)
        | (df["Open Porch SF"] > 0)
        | (df["Enclosed Porch"] > 0)
        | (df["Screen Porch"] > 0)
    ).astype(int)
    df["Total_Outdoor_SF"] = (
        df["Wood Deck SF"]
        + df["Open Porch SF"]
        + df["Enclosed Porch"]
        + df["3Ssn Porch"]
        + df["Screen Porch"]
    )
    df["Has_Large_Garage"] = (df["Garage Cars"] >= 3).astype(int)
    df["Luxury_Score"] = (
        df["Has_Pool"]
        + df["Has_Fireplace"]
        + df["Has_Large_Garage"]
        + df["Has_Deck_Porch"]
        + (df["Total_Full_Baths"] >= 3).astype(int)
    )


def _add_quality_features(df: pd.DataFrame) -> None:
    quality_vars = ["Kitchen Qual_num", "Exter Qual_num", "Bsmt Qual_num"]
    df["Total_Quality_Score"] = df[quality_vars].sum(axis=1)
    df["Avg_Quality_Score"] = df["Total_Quality_Score"] / len(quality_vars)
    df["Is_High_Quality"] = (df["Overall Qual"] >= 7).astype(int)
    df["Is_Excellent_Quality"] = (df["Overall Qual"] >= 8).astype(int)
    df["Quality_Condition_Score"] = df["Overall Qual"] + df["Overall Cond"]


def _add_property_type_features(df: pd.DataFrame) -> None:
    df["Is_Single_Family"] = (df["Bldg Type"] == "1Fam").astype(int)
    df["Is_Two_Story"] = df["House Style"].isin(["2Story", "2.5Fin", "2.5Unf"]).astype(int)
    df["Is_One_Story"] = df["House Style"].isin(["1Story", "1.5Fin", "1.5Unf"]).astype(int)
    df["Is_Residential_Low_Density"] = (df["MS Zoning"] == "RL").astype(int)


def _add_interaction_features(df: pd.DataFrame) -> None:
    df["Quality_x_Size"] = df["Overall Qual"] * df["Gr Liv Area"]
    df["Premium_Neighborhood_x_Quality"] = df["Premium_Neighborhood"] * df["Overall Qual"]
    df["New_x_Quality"] = df["Is_New_House"] * df["Overall Qual"]
    df["Area_x_Bathrooms"] = df["Gr Liv Area"] * df["Total_Bathrooms"]


def _add_sale_condition_features(df: pd.DataFrame) -> None:
    df["Is_Normal_Sale"] = (df["Sale Condition"] == "Normal").astype(int)
    df["Is_Partial_Sale"] = (df["Sale Condition"] == "Partial").astype(int)
    df["Is_Abnormal_Sale"] = (df["Sale Condition"] == "Abnorml").astype(int)
    df["Is_Peak_Season"] = df["Mo Sold"].isin([5, 6, 7]).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    _add_temporal_features(data)
    _add_area_features(data)
    _add_bathroom_features(data)
    _add_luxury_features(data)
    _add_quality_features(data)
    _add_property_type_features(data)
    _add_interaction_features(data)
    _add_sale_condition_features(data)

    required_columns = ENGINEERED_FEATURES + ["High_Value"]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns required for engineered dataset: {missing}")

    logger.info("Generated %d engineered features", len(ENGINEERED_FEATURES))
    return data[required_columns]


def save_engineered_features(
    engineered_df: pd.DataFrame,
    output_path: Path | str = config.ENGINEERED_FEATURES_PATH,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_csv(output, index=False)
    logger.info("Saved engineered features to %s", output)
    return output


def run(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    source = Path(input_path) if input_path else config.WITH_SCORE_PATH
    df = pd.read_csv(source)
    engineered = engineer_features(df)
    save_engineered_features(engineered, output_path or config.ENGINEERED_FEATURES_PATH)
    return engineered
