from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"


def ensure_output_dirs() -> None:
    for directory in (MODELS_DIR, METRICS_DIR, SUBMISSIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_train_data(path: Path = TRAIN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def load_test_data(path: Path = TEST_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def load_sample_submission(path: Path = SAMPLE_SUBMISSION_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(
    data: pd.DataFrame,
    target_column: str = "SalePrice",
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    return data.drop(columns=[target_column]), data[target_column].astype(float)
