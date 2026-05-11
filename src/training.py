from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data import (
    METRICS_DIR,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    ensure_output_dirs,
    load_sample_submission,
    load_test_data,
    load_train_data,
    split_features_target,
)
from src.models import RANDOM_STATE, get_classic_models
from src.preprocessing import build_preprocessor


@dataclass(frozen=True)
class RegressionReport:
    model: str
    cv_rmsle_mean: float
    cv_rmsle_std: float
    cv_rmse_mean: float
    holdout_rmsle: float
    holdout_rmse: float
    holdout_mae: float
    holdout_r2: float


def rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_pred = np.maximum(np.asarray(y_pred), 0)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_pipeline(model: object) -> TransformedTargetRegressor:
    regressor = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def evaluate_model(
    name: str,
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[RegressionReport, TransformedTargetRegressor]:
    pipeline = make_pipeline(model)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "neg_root_mean_squared_log_error": "neg_root_mean_squared_log_error",
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
        },
        n_jobs=1,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    holdout_pipeline = make_pipeline(clone(model))
    holdout_pipeline.fit(X_train, y_train)
    predictions = holdout_pipeline.predict(X_valid)

    report = RegressionReport(
        model=name,
        cv_rmsle_mean=float(-cv_scores["test_neg_root_mean_squared_log_error"].mean()),
        cv_rmsle_std=float(cv_scores["test_neg_root_mean_squared_log_error"].std()),
        cv_rmse_mean=float(-cv_scores["test_neg_root_mean_squared_error"].mean()),
        holdout_rmsle=rmsle(y_valid, predictions),
        holdout_rmse=rmse(y_valid, predictions),
        holdout_mae=float(mean_absolute_error(y_valid, predictions)),
        holdout_r2=float(r2_score(y_valid, predictions)),
    )
    return report, holdout_pipeline


def train_and_compare_models() -> tuple[pd.DataFrame, str, TransformedTargetRegressor]:
    ensure_output_dirs()
    train_data = load_train_data()
    X, y = split_features_target(train_data)

    reports: list[RegressionReport] = []
    for name, model in get_classic_models().items():
        report, _ = evaluate_model(name, model, X, y)
        reports.append(report)

    metrics = pd.DataFrame(asdict(report) for report in reports).sort_values(
        by=["cv_rmsle_mean", "holdout_rmsle"],
        ascending=True,
    )
    best_model_name = str(metrics.iloc[0]["model"])

    final_pipeline = make_pipeline(get_classic_models()[best_model_name])
    final_pipeline.fit(X, y)

    metrics.to_csv(METRICS_DIR / "classic_ml_metrics.csv", index=False)
    with (METRICS_DIR / "classic_ml_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics.to_dict(orient="records"), file, indent=2)

    joblib.dump(final_pipeline, MODELS_DIR / "best_classic_model.joblib")
    return metrics, best_model_name, final_pipeline


def create_submission(
    model: TransformedTargetRegressor,
    output_path: Path | None = None,
) -> Path:
    ensure_output_dirs()
    test_data = load_test_data()
    sample_submission = load_sample_submission()

    predictions = np.maximum(model.predict(test_data), 0)
    submission = sample_submission.copy()
    submission["SalePrice"] = predictions

    output_path = output_path or SUBMISSIONS_DIR / "classic_ml_submission.csv"
    submission.to_csv(output_path, index=False)
    return output_path


def run_classic_ml_pipeline() -> tuple[pd.DataFrame, str, Path]:
    metrics, best_model_name, best_model = train_and_compare_models()
    submission_path = create_submission(best_model)
    return metrics, best_model_name, submission_path
