from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add compact, high-signal features for Ames house price prediction."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "HouseFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()

        data["TotalSF"] = (
            data["TotalBsmtSF"].fillna(0)
            + data["1stFlrSF"].fillna(0)
            + data["2ndFlrSF"].fillna(0)
        )
        data["TotalBathrooms"] = (
            data["FullBath"].fillna(0)
            + 0.5 * data["HalfBath"].fillna(0)
            + data["BsmtFullBath"].fillna(0)
            + 0.5 * data["BsmtHalfBath"].fillna(0)
        )
        data["TotalPorchSF"] = (
            data["OpenPorchSF"].fillna(0)
            + data["EnclosedPorch"].fillna(0)
            + data["3SsnPorch"].fillna(0)
            + data["ScreenPorch"].fillna(0)
            + data["WoodDeckSF"].fillna(0)
        )
        data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
        data["RemodAge"] = data["YrSold"] - data["YearRemodAdd"]
        data["GarageAge"] = data["YrSold"] - data["GarageYrBlt"]
        data["GarageAge"] = data["GarageAge"].where(data["GarageYrBlt"].notna(), np.nan)

        data["HasBasement"] = (data["TotalBsmtSF"].fillna(0) > 0).astype(int)
        data["HasGarage"] = (data["GarageArea"].fillna(0) > 0).astype(int)
        data["HasFireplace"] = (data["Fireplaces"].fillna(0) > 0).astype(int)
        data["HasPool"] = (data["PoolArea"].fillna(0) > 0).astype(int)
        data["WasRemodeled"] = (data["YearBuilt"] != data["YearRemodAdd"]).astype(int)

        if "Id" in data.columns:
            data = data.drop(columns=["Id"])

        return data


def build_preprocessor() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical_transformer, make_column_selector(dtype_include=object)),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("features", HouseFeatureEngineer()),
            ("columns", column_transformer),
        ]
    )
