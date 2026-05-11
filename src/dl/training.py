from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
from src.dl.model import HousePriceMLP
from src.models import RANDOM_STATE
from src.preprocessing import build_preprocessor
from src.training import rmsle, rmse


@dataclass(frozen=True)
class DLConfig:
    epochs: int = 250
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    dropout: float = 0.2
    validation_size: float = 0.2
    patience: int = 35


@dataclass(frozen=True)
class DLReport:
    model: str
    best_epoch: int
    train_loss: float
    validation_loss: float
    validation_rmsle: float
    validation_rmse: float
    validation_mae: float
    validation_r2: float


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_dense_float32(features: object) -> np.ndarray:
    if hasattr(features, "toarray"):
        features = features.toarray()
    return np.asarray(features, dtype=np.float32)


def make_loader(
    features: np.ndarray,
    target_log: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    if target_log is None:
        dataset = TensorDataset(feature_tensor)
    else:
        target_tensor = torch.tensor(target_log, dtype=torch.float32)
        dataset = TensorDataset(feature_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: HousePriceMLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []

    for features, target in loader:
        features = features.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(
    model: HousePriceMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for features, target in loader:
        features = features.to(device)
        target = target.to(device)

        output = model(features)
        loss = criterion(output, target)

        losses.append(float(loss.detach().cpu()))
        predictions.append(output.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

    return float(np.mean(losses)), np.concatenate(predictions), np.concatenate(targets)


@torch.no_grad()
def predict_log_prices(model: HousePriceMLP, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []

    for (features,) in loader:
        features = features.to(device)
        predictions.append(model(features).detach().cpu().numpy())

    return np.concatenate(predictions)


def train_validation_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: DLConfig,
    device: torch.device,
) -> tuple[HousePriceMLP, DLReport, object, float, float]:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=config.validation_size,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    X_train_processed = to_dense_float32(preprocessor.fit_transform(X_train))
    X_valid_processed = to_dense_float32(preprocessor.transform(X_valid))
    y_train_log = np.log1p(y_train.to_numpy(dtype=np.float32))
    y_valid_log = np.log1p(y_valid.to_numpy(dtype=np.float32))
    target_mean = float(y_train_log.mean())
    target_std = float(y_train_log.std())
    y_train_scaled = (y_train_log - target_mean) / target_std
    y_valid_scaled = (y_valid_log - target_mean) / target_std

    train_loader = make_loader(X_train_processed, y_train_scaled, config.batch_size, shuffle=True)
    valid_loader = make_loader(X_valid_processed, y_valid_scaled, config.batch_size, shuffle=False)

    model = HousePriceMLP(
        input_dim=X_train_processed.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    best_train_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, _, _ = evaluate(model, valid_loader, criterion, device)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_loss, validation_predictions_log, validation_targets_log = evaluate(
        model, valid_loader, criterion, device
    )
    validation_predictions_log = validation_predictions_log * target_std + target_mean
    validation_targets_log = validation_targets_log * target_std + target_mean
    validation_predictions = np.maximum(np.expm1(validation_predictions_log), 0)
    validation_targets = np.expm1(validation_targets_log)

    report = DLReport(
        model="house_price_mlp",
        best_epoch=best_epoch,
        train_loss=best_train_loss,
        validation_loss=validation_loss,
        validation_rmsle=rmsle(validation_targets, validation_predictions),
        validation_rmse=rmse(validation_targets, validation_predictions),
        validation_mae=float(mean_absolute_error(validation_targets, validation_predictions)),
        validation_r2=float(r2_score(validation_targets, validation_predictions)),
    )
    return model, report, preprocessor, target_mean, target_std


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: DLConfig,
    device: torch.device,
) -> tuple[HousePriceMLP, object, float, float]:
    preprocessor = build_preprocessor()
    X_processed = to_dense_float32(preprocessor.fit_transform(X))
    y_log = np.log1p(y.to_numpy(dtype=np.float32))
    target_mean = float(y_log.mean())
    target_std = float(y_log.std())
    y_scaled = (y_log - target_mean) / target_std

    loader = make_loader(X_processed, y_scaled, config.batch_size, shuffle=True)
    model = HousePriceMLP(X_processed.shape[1], config.hidden_dim, config.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for _ in range(config.epochs):
        train_one_epoch(model, loader, criterion, optimizer, device)

    return model, preprocessor, target_mean, target_std


def save_report(report: DLReport, config: DLConfig, device: torch.device) -> None:
    report_dict = asdict(report)
    report_dict["device"] = str(device)
    report_dict["config"] = asdict(config)

    pd.DataFrame([report_dict]).to_csv(METRICS_DIR / "dl_metrics.csv", index=False)
    with (METRICS_DIR / "dl_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report_dict, file, indent=2)


def save_model(
    model: HousePriceMLP,
    preprocessor: object,
    input_dim: int,
    config: DLConfig,
    target_mean: float,
    target_std: float,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    torch.save(checkpoint, MODELS_DIR / "house_price_mlp.pt")
    joblib.dump(preprocessor, MODELS_DIR / "dl_preprocessor.joblib")


def create_dl_submission(
    model: HousePriceMLP,
    preprocessor: object,
    config: DLConfig,
    device: torch.device,
    target_mean: float,
    target_std: float,
    output_path: Path | None = None,
) -> Path:
    test_data = load_test_data()
    sample_submission = load_sample_submission()
    test_processed = to_dense_float32(preprocessor.transform(test_data))
    test_loader = make_loader(test_processed, batch_size=config.batch_size, shuffle=False)

    predictions_log = predict_log_prices(model, test_loader, device) * target_std + target_mean
    predictions = np.maximum(np.expm1(predictions_log), 0)
    submission = sample_submission.copy()
    submission["SalePrice"] = predictions

    output_path = output_path or SUBMISSIONS_DIR / "dl_submission.csv"
    submission.to_csv(output_path, index=False)
    return output_path


def run_dl_pipeline(config: DLConfig | None = None) -> tuple[DLReport, Path]:
    ensure_output_dirs()
    set_seed()
    config = config or DLConfig()
    device = get_device()

    train_data = load_train_data()
    X, y = split_features_target(train_data)

    _, report, _, _, _ = train_validation_model(X, y, config, device)
    final_model, final_preprocessor, target_mean, target_std = train_final_model(X, y, config, device)

    input_dim = to_dense_float32(final_preprocessor.transform(X.head(1))).shape[1]
    save_report(report, config, device)
    save_model(final_model, final_preprocessor, input_dim, config, target_mean, target_std)
    submission_path = create_dl_submission(
        final_model,
        final_preprocessor,
        config,
        device,
        target_mean,
        target_std,
    )

    return report, submission_path
