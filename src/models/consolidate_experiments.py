"""Create a consistent comparison of the four saved METR-LA experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = {
    "Centralized baseline": Path("models/centralized_baseline"),
    "Centralized + noise": Path("models/centralized_augmented"),
    "FedAvg": Path("models/federated_baseline"),
    "FedAvg + noise": Path("models/federated_augmented"),
}


def consolidate(experiments: dict[str, Path] | None = None, output_dir: Path = Path("models/experiment_comparison")) -> dict[str, Any]:
    experiments = experiments or EXPERIMENTS
    rows: list[dict[str, Any]] = []
    per_horizon: dict[str, dict[str, list[float]]] = {}
    expected_shape = None
    expected_horizon = None
    for label, directory in experiments.items():
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        test = metrics["test"]
        shape = metrics.get("test_shape", {})
        if expected_shape is None:
            expected_shape = shape
            expected_horizon = config["horizon_steps"]
        if shape != expected_shape or config["horizon_steps"] != expected_horizon:
            raise ValueError(f"Experiment {label} does not use the same test shape and horizon")
        rows.append({"model": label, "mae": test["mae"], "rmse": test["rmse"]})
        per_horizon[label] = {
            "mae": test["mae_by_horizon"],
            "rmse": test["rmse_by_horizon"],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "dataset": "METR-LA",
        "evaluation_split": "test",
        "test_shape": expected_shape,
        "forecast_horizon_steps": expected_horizon,
        "metric_aggregation": "equal weight for every sample x forecast-step x sensor scalar prediction",
        "models": rows,
        "per_horizon": per_horizon,
    }
    (output_dir / "metrics.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, metric in zip(axes, ("mae", "rmse")):
        vals = [row[metric] for row in rows]
        bars = axis.bar(x, vals, color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"])
        axis.set_title(metric.upper())
        axis.set_ylabel("Dataset speed units")
        axis.set_xticks(x, [row["model"] for row in rows], rotation=20, ha="right")
        axis.bar_label(bars, fmt="%.3f", padding=3)
    fig.suptitle("Four model configurations on the common METR-LA test split")
    fig.tight_layout()
    fig.savefig(output_dir / "four_model_overall_comparison.png", dpi=160)
    plt.close(fig)

    leads = np.arange(1, expected_horizon + 1) * 5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label in per_horizon:
        axes[0].plot(leads, per_horizon[label]["mae"], marker="o", label=label)
        axes[1].plot(leads, per_horizon[label]["rmse"], marker="o", label=label)
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        axis.set_title(f"Test {metric} by forecast lead")
        axis.set_xlabel("Forecast lead time (minutes)")
        axis.set_ylabel("Dataset speed units")
        axis.set_xticks(leads)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "four_model_per_horizon_comparison.png", dpi=160)
    plt.close(fig)
    return combined


if __name__ == "__main__":
    print(json.dumps(consolidate(), indent=2))
