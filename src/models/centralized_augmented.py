"""Centralized shared-linear METR-LA experiment with train-only noise jitter."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.noise_augmentation import NoiseAugmentationConfig, augment_training_windows
from src.models.centralized_baseline import TrainingConfig, _load_split, _metrics, train_model_with_parts


def run_experiment(
    data_dir: Path,
    output_dir: Path,
    training: TrainingConfig,
    augmentation: NoiseAugmentationConfig | None = None,
) -> dict[str, Any]:
    """Fit/select using train and validation only; open test after selection."""
    augmentation = augmentation or NoiseAugmentationConfig(seed=training.seed)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["lookback_steps"] != training.lookback_steps or metadata["forecast_horizon_steps"] != training.horizon_steps:
        raise ValueError("Training configuration does not match prepared-data lookback/horizon")

    # Data access order is deliberate: no test data is opened until selection finishes.
    train_X, train_y = _load_split(data_dir, "train")
    validation_X, validation_y = _load_split(data_dir, "validation")
    synthetic_X, synthetic_y, augmentation_details = augment_training_windows(
        train_X, train_y, augmentation, seed=augmentation.seed
    )
    model, history, best_epoch, training_seconds = train_model_with_parts(
        [(train_X, train_y), (synthetic_X, synthetic_y)],
        validation_X,
        validation_y,
        training,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = "shared_linear_forecaster.npz"
    np.savez_compressed(output_dir / model_file, **model.state())
    del train_X, train_y, validation_X, validation_y, synthetic_X, synthetic_y
    test_X, test_y = _load_split(data_dir, "test")
    test_metrics = _metrics(model, test_X, test_y, training.batch_windows)

    config_payload = asdict(training)
    config_payload.update({
        "model": "shared per-sensor linear autoregressive forecaster",
        "input_shape": [None, training.lookback_steps, None],
        "output_shape": [None, training.horizon_steps, None],
        "training_objective": "mean squared error across samples, sensors, and forecast steps",
        "optimizer": "NumPy Adam; state retained across centralized batches and epochs",
        "selection_split": "validation",
        "selection_metric": "MSE",
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "augmentation": asdict(augmentation),
        "augmentation_rule": "5% per-sensor standard deviation fitted from central X_train; independent Gaussian jitter to inputs; targets retained; clip at 0",
        "augmentation_fit_source": "train split only",
        "original_training_windows": int(augmentation_details["original_window_count"]),
        "synthetic_training_windows": int(augmentation_details["synthetic_window_count"]),
        "test_opened_after_model_selection": True,
        "federated_comparison_note": "same noise rule and base seed; central scale is estimated over all central train sensor data, while FedAvg estimates scales independently per client and uses seeds 42-45",
    })
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "augmentation_manifest.json").write_text(json.dumps({
        "augmentation": asdict(augmentation),
        "details": augmentation_details,
        "validation_used": False,
        "test_used": False,
        "synthetic_arrays_saved": False,
        "reproducibility": "Regenerate from the prepared train.npz using the recorded configuration and seed.",
    }, indent=2) + "\n", encoding="utf-8")
    result = {
        "dataset": "METR-LA",
        "model_file": model_file,
        "selection": {"split": "validation", "metric": "MSE", "best_epoch": best_epoch},
        "augmentation_generation_seconds": float(augmentation_details.get("generation_seconds", 0.0)),
        "training_seconds": training_seconds,
        "test": test_metrics,
        "test_shape": {"X": list(test_X.shape), "y": list(test_y.shape)},
        "test_opened_after_model_selection": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "validation_mse", "validation_mae"])
        writer.writeheader()
        writer.writerows(history)
    epochs = [int(row["epoch"]) for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_mse"] for row in history], label="Train MSE (original + jittered)")
    plt.plot(epochs, [row["validation_mse"] for row in history], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error (dataset speed units²)")
    plt.title("Centralized forecaster with input-noise augmentation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close()
    leads = np.arange(1, training.horizon_steps + 1) * 5
    plt.figure(figsize=(7, 4))
    plt.plot(leads, test_metrics["mae_by_horizon"], marker="o", label="MAE")
    plt.plot(leads, test_metrics["rmse_by_horizon"], marker="o", label="RMSE")
    plt.xlabel("Forecast lead time (minutes)")
    plt.ylabel("Error (dataset speed units)")
    plt.title("Centralized augmented test error by forecast lead")
    plt.xticks(leads)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_error_by_horizon.png", dpi=150)
    plt.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/metr_la"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/centralized_augmented"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_experiment(
        args.data_dir,
        args.output_dir,
        TrainingConfig(seed=args.seed),
        NoiseAugmentationConfig(seed=args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
