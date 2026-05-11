from __future__ import annotations

import argparse

from src.training import run_classic_ml_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train House Prices regression models and create Kaggle submissions."
    )
    parser.add_argument(
        "--mode",
        choices=["classic", "dl", "all"],
        default="classic",
        help="Which pipeline to run.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=7,
        help="How many rows of the classic ML metrics table to print.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of epochs for the DL pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in {"classic", "all"}:
        metrics, best_model_name, submission_path = run_classic_ml_pipeline()

        print("\nClassic ML leaderboard:")
        print(metrics.head(args.top).to_string(index=False))
        print(f"\nBest classic model: {best_model_name}")
        print(f"Classic submission saved to: {submission_path}")

    if args.mode in {"dl", "all"}:
        from src.dl.training import DLConfig, run_dl_pipeline

        report, submission_path = run_dl_pipeline(DLConfig(epochs=args.epochs))

        print("\nDeep Learning report:")
        print(f"Model: {report.model}")
        print(f"Best epoch: {report.best_epoch}")
        print(f"Validation RMSLE: {report.validation_rmsle:.5f}")
        print(f"Validation RMSE: {report.validation_rmse:.2f}")
        print(f"Validation R2: {report.validation_r2:.4f}")
        print(f"DL submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
