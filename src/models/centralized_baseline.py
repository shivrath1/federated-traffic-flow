"""Shared per-sensor linear forecaster and centralized METR-LA training CLI.

The model applies one learned linear mapping from the previous 12 readings to
the next 12 readings independently to each sensor. Parameters do not depend on
the number or identity of sensors, so the same model can later be trained on
the 51- or 52-sensor client partitions and combined with FedAvg.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingConfig:
    lookback_steps: int = 12
    horizon_steps: int = 12
    max_epochs: int = 30
    batch_windows: int = 256
    learning_rate: float = 0.001
    patience: int = 4
    min_delta: float = 1e-5
    seed: int = 42
    train_split: str = "train"
    selection_split: str = "validation"
    final_evaluation_split: str = "test"


class SharedLinearForecaster:
    """Linear temporal map shared across all sensor channels."""

    def __init__(self, lookback_steps: int = 12, horizon_steps: int = 12):
        if lookback_steps < 1 or horizon_steps < 1:
            raise ValueError("lookback_steps and horizon_steps must be positive")
        self.lookback_steps = int(lookback_steps)
        self.horizon_steps = int(horizon_steps)
        self.weights = np.zeros((self.lookback_steps, self.horizon_steps), dtype=np.float32)
        # A persistence-like initialization forecasts each horizon from the
        # latest observation. Training can learn adjustments from all history.
        self.weights[-1, :] = 1.0
        self.bias = np.zeros(self.horizon_steps, dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ``(..., horizon_steps, sensors)`` from ``(..., lookback, sensors)``."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, lookback_steps, sensors)")
        if X.shape[1] != self.lookback_steps:
            raise ValueError(f"Expected {self.lookback_steps} input steps, got {X.shape[1]}")
        if X.shape[2] < 1:
            raise ValueError("X must contain at least one sensor")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        per_sensor = np.transpose(X, (0, 2, 1))
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = per_sensor @ self.weights + self.bias
        if not np.isfinite(predicted).all():
            raise ValueError("Model prediction contains non-finite values")
        return np.transpose(predicted, (0, 2, 1)).astype(np.float32, copy=False)

    def state(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy(), "bias": self.bias.copy()}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        weights = np.asarray(state["weights"], dtype=np.float32)
        bias = np.asarray(state["bias"], dtype=np.float32)
        if weights.shape != (self.lookback_steps, self.horizon_steps):
            raise ValueError("Saved weight shape does not match model configuration")
        if bias.shape != (self.horizon_steps,):
            raise ValueError("Saved bias shape does not match model configuration")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError("Saved model parameters must contain only finite values")
        self.weights = weights.copy()
        self.bias = bias.copy()


def load_saved_model(model_dir: str | Path) -> tuple[SharedLinearForecaster, dict[str, Any]]:
    """Load a saved model and its configuration from an experiment directory."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = SharedLinearForecaster(config["lookback_steps"], config["horizon_steps"])
    with np.load(model_dir / "shared_linear_forecaster.npz") as state:
        model.load_state({"weights": state["weights"], "bias": state["bias"]})
    return model, config


def _flatten_windows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """View each sensor-time window as one training example."""
    if X.ndim != 3 or y.ndim != 3:
        raise ValueError("X and y must have shape (samples, steps, sensors)")
    if X.shape[0] != y.shape[0] or X.shape[2] != y.shape[2]:
        raise ValueError("X and y sample/sensor dimensions must match")
    x_flat = np.ascontiguousarray(np.transpose(X, (0, 2, 1))).reshape(-1, X.shape[1])
    y_flat = np.ascontiguousarray(np.transpose(y, (0, 2, 1))).reshape(-1, y.shape[1])
    return x_flat, y_flat


def _metrics(model: SharedLinearForecaster, X: np.ndarray, y: np.ndarray, batch_windows: int) -> dict[str, Any]:
    """Compute element-weighted MAE/RMSE over windows, horizons, and sensors."""
    if X.shape[0] != y.shape[0] or X.shape[2] != y.shape[2]:
        raise ValueError("X and y sample/sensor dimensions must match")
    abs_sum = np.zeros(model.horizon_steps, dtype=np.float64)
    squared_sum = np.zeros(model.horizon_steps, dtype=np.float64)
    element_count = np.zeros(model.horizon_steps, dtype=np.int64)
    for begin in range(0, X.shape[0], batch_windows):
        stop = min(begin + batch_windows, X.shape[0])
        prediction = model.predict(X[begin:stop])
        error = prediction.astype(np.float64) - y[begin:stop].astype(np.float64)
        abs_sum += np.abs(error).sum(axis=(0, 2))
        squared_sum += np.square(error).sum(axis=(0, 2))
        element_count += error.shape[0] * error.shape[2]
    mae_by_horizon = abs_sum / element_count
    rmse_by_horizon = np.sqrt(squared_sum / element_count)
    total = int(element_count.sum())
    return {
        "mae": float(abs_sum.sum() / total),
        "rmse": float(np.sqrt(squared_sum.sum() / total)),
        "mae_by_horizon": mae_by_horizon.tolist(),
        "rmse_by_horizon": rmse_by_horizon.tolist(),
        "aggregation": "all sample x forecast-step x sensor errors; each scalar prediction has equal weight",
        "evaluated_values": total,
    }


def _train_epoch(
    model: SharedLinearForecaster,
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
    rng: np.random.Generator,
    adam: dict[str, Any],
) -> float:
    order = rng.permutation(X.shape[0])
    loss_sum = 0.0
    value_count = 0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for begin in range(0, len(order), config.batch_windows):
        selected = order[begin : begin + config.batch_windows]
        xb, yb = _flatten_windows(X[selected], y[selected])
        prediction = xb @ model.weights + model.bias
        error = prediction - yb
        batch_values = error.size
        loss_sum += float(np.square(error, dtype=np.float64).sum())
        value_count += batch_values
        grad_w = (2.0 / batch_values) * (xb.T @ error)
        grad_b = (2.0 / batch_values) * error.sum(axis=0)

        adam["step"] += 1
        adam["mw"] = beta1 * adam["mw"] + (1 - beta1) * grad_w
        adam["vw"] = beta2 * adam["vw"] + (1 - beta2) * np.square(grad_w)
        adam["mb"] = beta1 * adam["mb"] + (1 - beta1) * grad_b
        adam["vb"] = beta2 * adam["vb"] + (1 - beta2) * np.square(grad_b)
        step = adam["step"]
        mw_hat = adam["mw"] / (1 - beta1**step)
        vw_hat = adam["vw"] / (1 - beta2**step)
        mb_hat = adam["mb"] / (1 - beta1**step)
        vb_hat = adam["vb"] / (1 - beta2**step)
        model.weights -= config.learning_rate * mw_hat / (np.sqrt(vw_hat) + epsilon)
        model.bias -= config.learning_rate * mb_hat / (np.sqrt(vb_hat) + epsilon)
    return loss_sum / value_count


def _train_epoch_from_parts(
    model: SharedLinearForecaster,
    train_parts: list[tuple[np.ndarray, np.ndarray]],
    config: TrainingConfig,
    rng: np.random.Generator,
    adam: dict[str, Any],
) -> float:
    """Train from distinct arrays without allocating a full concatenated copy."""
    lengths = [X.shape[0] for X, _ in train_parts]
    total_windows = sum(lengths)
    order = rng.permutation(total_windows)
    loss_sum = 0.0
    value_count = 0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for begin in range(0, len(order), config.batch_windows):
        selected = order[begin : begin + config.batch_windows]
        xb_parts: list[np.ndarray] = []
        yb_parts: list[np.ndarray] = []
        offset = 0
        for (X_part, y_part), part_length in zip(train_parts, lengths):
            mask = (selected >= offset) & (selected < offset + part_length)
            if np.any(mask):
                indices = selected[mask] - offset
                xb_parts.append(X_part[indices])
                yb_parts.append(y_part[indices])
            offset += part_length
        batch_X = xb_parts[0] if len(xb_parts) == 1 else np.concatenate(xb_parts, axis=0)
        batch_y = yb_parts[0] if len(yb_parts) == 1 else np.concatenate(yb_parts, axis=0)
        xb, yb = _flatten_windows(batch_X, batch_y)
        prediction = xb @ model.weights + model.bias
        error = prediction - yb
        batch_values = error.size
        loss_sum += float(np.square(error, dtype=np.float64).sum())
        value_count += batch_values
        grad_w = (2.0 / batch_values) * (xb.T @ error)
        grad_b = (2.0 / batch_values) * error.sum(axis=0)

        adam["step"] += 1
        adam["mw"] = beta1 * adam["mw"] + (1 - beta1) * grad_w
        adam["vw"] = beta2 * adam["vw"] + (1 - beta2) * np.square(grad_w)
        adam["mb"] = beta1 * adam["mb"] + (1 - beta1) * grad_b
        adam["vb"] = beta2 * adam["vb"] + (1 - beta2) * np.square(grad_b)
        step = adam["step"]
        model.weights -= config.learning_rate * (adam["mw"] / (1 - beta1**step)) / (
            np.sqrt(adam["vw"] / (1 - beta2**step)) + epsilon
        )
        model.bias -= config.learning_rate * (adam["mb"] / (1 - beta1**step)) / (
            np.sqrt(adam["vb"] / (1 - beta2**step)) + epsilon
        )
    return loss_sum / value_count


def train_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    config: TrainingConfig,
) -> tuple[SharedLinearForecaster, list[dict[str, float]], int, float]:
    """Train on train split and choose the epoch using validation MSE only."""
    for name, X, y in (("train", train_X, train_y), ("validation", validation_X, validation_y)):
        if X.ndim != 3 or y.ndim != 3 or X.shape[1] != config.lookback_steps or y.shape[1] != config.horizon_steps:
            raise ValueError(f"{name} arrays do not match configured input/output steps")
        if X.shape[0] != y.shape[0] or X.shape[2] != y.shape[2]:
            raise ValueError(f"{name} X/y sample or sensor dimensions do not match")

    model = SharedLinearForecaster(config.lookback_steps, config.horizon_steps)
    best_state = model.state()
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(config.seed)
    adam = {
        "step": 0,
        "mw": np.zeros_like(model.weights),
        "vw": np.zeros_like(model.weights),
        "mb": np.zeros_like(model.bias),
        "vb": np.zeros_like(model.bias),
    }
    start_time = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        train_mse = _train_epoch(model, train_X, train_y, config, rng, adam)
        validation_metrics = _metrics(model, validation_X, validation_y, config.batch_windows)
        validation_mse = validation_metrics["rmse"] ** 2
        history.append({
            "epoch": float(epoch),
            "train_mse": float(train_mse),
            "validation_mse": float(validation_mse),
            "validation_mae": float(validation_metrics["mae"]),
        })
        if validation_mse < best_validation_mse - config.min_delta:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = model.state()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    elapsed = time.perf_counter() - start_time
    model.load_state(best_state)
    return model, history, best_epoch, elapsed


def train_model_with_parts(
    train_parts: list[tuple[np.ndarray, np.ndarray]],
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    config: TrainingConfig,
) -> tuple[SharedLinearForecaster, list[dict[str, float]], int, float]:
    """Train on separately held training arrays and select using validation MSE."""
    if not train_parts:
        raise ValueError("At least one training data part is required")
    sensor_count: int | None = None
    for index, (X, y) in enumerate(train_parts):
        if X.ndim != 3 or y.ndim != 3:
            raise ValueError(f"Training part {index} must be 3D arrays")
        if X.shape[1] != config.lookback_steps or y.shape[1] != config.horizon_steps:
            raise ValueError(f"Training part {index} does not match configured input/output steps")
        if X.shape[0] == 0 or X.shape[0] != y.shape[0] or X.shape[2] != y.shape[2]:
            raise ValueError(f"Training part {index} has mismatched or empty dimensions")
        if sensor_count is None:
            sensor_count = X.shape[2]
        elif X.shape[2] != sensor_count:
            raise ValueError("All centralized training parts must have the same sensor count")
    if validation_X.shape[2] != sensor_count or validation_y.shape[2] != sensor_count:
        raise ValueError("Validation sensor count must match training parts")
    if validation_X.shape[1] != config.lookback_steps or validation_y.shape[1] != config.horizon_steps:
        raise ValueError("Validation arrays do not match configured input/output steps")
    if validation_X.shape[0] == 0 or validation_X.shape[0] != validation_y.shape[0]:
        raise ValueError("Validation arrays have mismatched or empty sample dimensions")

    model = SharedLinearForecaster(config.lookback_steps, config.horizon_steps)
    best_state = model.state()
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(config.seed)
    adam = {
        "step": 0,
        "mw": np.zeros_like(model.weights),
        "vw": np.zeros_like(model.weights),
        "mb": np.zeros_like(model.bias),
        "vb": np.zeros_like(model.bias),
    }
    start_time = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        train_mse = _train_epoch_from_parts(model, train_parts, config, rng, adam)
        validation_metrics = _metrics(model, validation_X, validation_y, config.batch_windows)
        validation_mse = validation_metrics["rmse"] ** 2
        history.append({
            "epoch": float(epoch),
            "train_mse": float(train_mse),
            "validation_mse": float(validation_mse),
            "validation_mae": float(validation_metrics["mae"]),
        })
        if validation_mse < best_validation_mse - config.min_delta:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = model.state()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break
    elapsed = time.perf_counter() - start_time
    model.load_state(best_state)
    return model, history, best_epoch, elapsed


def train_local_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    initial_state: dict[str, np.ndarray],
    config: TrainingConfig,
    *,
    local_epochs: int = 1,
    seed: int | None = None,
) -> tuple[SharedLinearForecaster, list[float], float]:
    """Train one simulated client's data locally from a broadcast model state.

    This function receives only one client's arrays. It does not aggregate data
    or consult validation/test arrays. Adam state resets for each invocation,
    corresponding to one client's local work in a communication round.
    """
    if local_epochs < 1:
        raise ValueError("local_epochs must be positive")
    if train_X.ndim != 3 or train_y.ndim != 3:
        raise ValueError("Client X/y must have shape (samples, steps, sensors)")
    if train_X.shape[1] != config.lookback_steps or train_y.shape[1] != config.horizon_steps:
        raise ValueError("Client arrays do not match configured input/output steps")
    if train_X.shape[0] != train_y.shape[0] or train_X.shape[2] != train_y.shape[2]:
        raise ValueError("Client X/y sample or sensor dimensions do not match")
    model = SharedLinearForecaster(config.lookback_steps, config.horizon_steps)
    model.load_state(initial_state)
    rng = np.random.default_rng(config.seed if seed is None else seed)
    adam = {
        "step": 0,
        "mw": np.zeros_like(model.weights),
        "vw": np.zeros_like(model.weights),
        "mb": np.zeros_like(model.bias),
        "vb": np.zeros_like(model.bias),
    }
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(local_epochs):
        losses.append(_train_epoch(model, train_X, train_y, config, rng, adam))
    return model, losses, time.perf_counter() - started


def _load_split(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(root / f"{name}.npz") as stored:
        return stored["X"], stored["y"]


def run_experiment(data_dir: Path, output_dir: Path, config: TrainingConfig) -> dict[str, Any]:
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["lookback_steps"] != config.lookback_steps or metadata["forecast_horizon_steps"] != config.horizon_steps:
        raise ValueError("Training configuration does not match prepared-data lookback/horizon")

    # Load only training and validation data until model selection has ended.
    train_X, train_y = _load_split(data_dir, "train")
    validation_X, validation_y = _load_split(data_dir, "validation")
    model, history, best_epoch, training_seconds = train_model(
        train_X, train_y, validation_X, validation_y, config
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "shared_linear_forecaster.npz"
    np.savez_compressed(model_path, weights=model.weights, bias=model.bias)
    # Only open the test split after validation-based epoch selection is complete.
    del train_X, train_y, validation_X, validation_y
    test_X, test_y = _load_split(data_dir, "test")
    test_metrics = _metrics(model, test_X, test_y, config.batch_windows)

    config_payload = asdict(config)
    config_payload.update({
        "model": "shared per-sensor linear autoregressive forecaster",
        "input_shape": [None, config.lookback_steps, None],
        "output_shape": [None, config.horizon_steps, None],
        "training_objective": "mean squared error across samples, sensors, and forecast steps",
        "optimizer": "Adam implemented with NumPy",
        "parameter_shapes": {"weights": list(model.weights.shape), "bias": list(model.bias.shape)},
        "client_compatibility": "parameters independent of sensor count; same parameter shapes for 51- and 52-sensor groups",
        "best_epoch": best_epoch,
        "epochs_run": len(history),
    })
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    result = {
        "dataset": "METR-LA",
        "data_dir": str(data_dir),
        "model_file": model_path.name,
        "selection": {"split": "validation", "metric": "MSE", "best_epoch": best_epoch},
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
    plt.plot(epochs, [row["train_mse"] for row in history], label="Train MSE")
    plt.plot(epochs, [row["validation_mse"] for row in history], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error (dataset speed units²)")
    plt.title("Centralized linear forecaster training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=150)
    plt.close()

    horizons = np.arange(1, config.horizon_steps + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(horizons * 5, test_metrics["mae_by_horizon"], marker="o", label="MAE")
    plt.plot(horizons * 5, test_metrics["rmse_by_horizon"], marker="o", label="RMSE")
    plt.xlabel("Forecast lead time (minutes)")
    plt.ylabel("Error (dataset speed units)")
    plt.title("Test error by forecast lead time")
    plt.xticks(horizons * 5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_error_by_horizon.png", dpi=150)
    plt.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/metr_la"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/centralized_baseline"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-windows", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = TrainingConfig(
        max_epochs=args.epochs,
        batch_windows=args.batch_windows,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    result = run_experiment(args.data_dir, args.output_dir, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
