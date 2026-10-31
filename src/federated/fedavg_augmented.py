"""Run the client-local noise augmentation and compare the three baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.noise_augmentation import NoiseAugmentationConfig, generate_synthetic_client_files
from src.federated.fedavg import FederatedConfig, run_federated_experiment


def _difference(augmented: float, reference: float) -> dict[str, float]:
    change = augmented - reference
    return {
        "augmented": float(augmented),
        "reference": float(reference),
        "difference_augmented_minus_reference": float(change),
        "percent_difference_vs_reference": float(100.0 * change / reference),
    }


def run_augmented_experiment(
    data_dir: Path,
    output_dir: Path,
    centralized_dir: Path,
    federated_dir: Path,
    augmentation_config: NoiseAugmentationConfig,
    federated_config: FederatedConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir = output_dir / "synthetic_training_data"
    manifest = generate_synthetic_client_files(data_dir, synthetic_dir, augmentation_config)
    result = run_federated_experiment(
        data_dir,
        output_dir,
        centralized_dir,
        federated_config,
        synthetic_dir=synthetic_dir,
    )
    centralized = json.loads((centralized_dir / "metrics.json").read_text(encoding="utf-8"))
    federated = json.loads((federated_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = result["test"]
    comparisons = {
        "centralized_baseline": {
            "mae": _difference(metrics["mae"], centralized["test"]["mae"]),
            "rmse": _difference(metrics["rmse"], centralized["test"]["rmse"]),
        },
        "fedavg_without_augmentation": {
            "mae": _difference(metrics["mae"], federated["test"]["mae"]),
            "rmse": _difference(metrics["rmse"], federated["test"]["rmse"]),
        },
    }
    result["three_model_comparison"] = {
        "centralized_baseline": centralized["test"],
        "fedavg_without_augmentation": federated["test"],
        "fedavg_with_augmentation": metrics,
        "augmented_vs_centralized_and_fedavg": comparisons,
        "same_test_split": True,
    }
    result["augmentation_generation_seconds"] = manifest["generation_seconds"]
    result["augmentation_counts"] = {
        "per_client": [
            {
                "client_id": client["client_id"],
                "original_windows": client["original_window_count"],
                "synthetic_windows": client["synthetic_window_count"],
                "original_sensor_window_examples": client["original_sensor_window_examples"],
                "synthetic_sensor_window_examples": client["synthetic_sensor_window_examples"],
            }
            for client in manifest["clients"]
        ],
        "total_original_windows_across_clients": int(sum(c["original_window_count"] for c in manifest["clients"])),
        "total_synthetic_windows_across_clients": int(sum(c["synthetic_window_count"] for c in manifest["clients"])),
        "total_original_sensor_window_examples": int(sum(c["original_sensor_window_examples"] for c in manifest["clients"])),
        "total_synthetic_sensor_window_examples": int(sum(c["synthetic_sensor_window_examples"] for c in manifest["clients"])),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    leads = np.arange(1, federated_config.horizon_steps + 1) * 5
    series = [
        ("Centralized", centralized["test"]),
        ("FedAvg", federated["test"]),
        ("FedAvg + noise", result["test"]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, model_metrics in series:
        axes[0].plot(leads, model_metrics["mae_by_horizon"], marker="o", label=label)
        axes[1].plot(leads, model_metrics["rmse_by_horizon"], marker="o", label=label)
    axes[0].set_title("MAE by forecast lead")
    axes[1].set_title("RMSE by forecast lead")
    for axis in axes:
        axis.set_xlabel("Forecast lead time (minutes)")
        axis.set_ylabel("Error (dataset speed units)")
        axis.set_xticks(leads)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "three_model_test_comparison.png", dpi=150)
    plt.close(figure)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/metr_la"))
    parser.add_argument("--centralized-dir", type=Path, default=Path("models/centralized_baseline"))
    parser.add_argument("--federated-dir", type=Path, default=Path("models/federated_baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/federated_augmented"))
    parser.add_argument("--noise-fraction", type=float, default=0.05)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-windows", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    augmentation_config = NoiseAugmentationConfig(
        noise_std_fraction=args.noise_fraction,
        synthetic_copies_per_window=args.copies,
        seed=args.seed,
    )
    federated_config = FederatedConfig(
        communication_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_windows=args.batch_windows,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    result = run_augmented_experiment(
        args.data_dir,
        args.output_dir,
        args.centralized_dir,
        args.federated_dir,
        augmentation_config,
        federated_config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
