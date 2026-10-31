"""Four-client FedAvg simulation for the shared linear METR-LA forecaster."""

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

from src.data.metr_la import load_client_training_data
from src.models.centralized_baseline import (
    SharedLinearForecaster,
    TrainingConfig,
    _metrics,
    load_saved_model,
    train_local_model,
)


@dataclass
class FederatedConfig:
    lookback_steps: int = 12
    horizon_steps: int = 12
    communication_rounds: int = 30
    local_epochs: int = 1
    batch_windows: int = 256
    learning_rate: float = 0.001
    patience: int = 5
    min_delta: float = 1e-5
    seed: int = 42
    weighting: str = "number of local sensor-window examples (windows x sensors)"
    optimizer_note: str = "Adam state resets for each client's local work in each round"


def load_federated_model(model_dir: str | Path) -> tuple[SharedLinearForecaster, dict[str, Any]]:
    """Load the selected global FedAvg model and its configuration."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = SharedLinearForecaster(config["lookback_steps"], config["horizon_steps"])
    with np.load(model_dir / "fedavg_global_model.npz") as state:
        model.load_state({"weights": state["weights"], "bias": state["bias"]})
    return model, config


def fedavg(states: list[dict[str, np.ndarray]], weights: list[int | float]) -> dict[str, np.ndarray]:
    """Average model tensors by client example counts; reject mismatched states."""
    if not states or len(states) != len(weights):
        raise ValueError("Provide one positive aggregation weight for every client state")
    numeric_weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(numeric_weights).all() or np.any(numeric_weights <= 0):
        raise ValueError("FedAvg weights must be finite and positive")
    total_weight = float(numeric_weights.sum())
    keys = set(states[0])
    if not keys or any(set(state) != keys for state in states):
        raise ValueError("Every client state must contain the same parameter names")
    aggregated: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        arrays = [np.asarray(state[key]) for state in states]
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError(f"Client parameter shape mismatch for {key}")
        accumulator = np.zeros(arrays[0].shape, dtype=np.float64)
        for weight, array in zip(numeric_weights, arrays):
            accumulator += weight * array.astype(np.float64)
        aggregated[key] = (accumulator / total_weight).astype(arrays[0].dtype, copy=False)
    return aggregated


def _load_split(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(root / f"{name}.npz") as stored:
        return stored["X"], stored["y"]


def _compare_metric(federated_value: float, centralized_value: float) -> dict[str, float]:
    difference = federated_value - centralized_value
    return {
        "federated": float(federated_value),
        "centralized": float(centralized_value),
        "difference_federated_minus_centralized": float(difference),
        "percent_difference_vs_centralized": float(100.0 * difference / centralized_value),
    }


def run_federated_experiment(
    data_dir: Path,
    output_dir: Path,
    centralized_dir: Path,
    config: FederatedConfig,
    synthetic_dir: Path | None = None,
) -> dict[str, Any]:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    clients = metadata["client_partition"]["clients"]
    if len(clients) != 4:
        raise ValueError(f"Expected exactly four client partitions; metadata lists {len(clients)}")
    if metadata["lookback_steps"] != config.lookback_steps or metadata["forecast_horizon_steps"] != config.horizon_steps:
        raise ValueError("Federated configuration does not match prepared-data lookback/horizon")
    sensor_counts = [client["sensor_count"] for client in clients]
    if sorted(sensor_counts) != [51, 52, 52, 52]:
        raise ValueError(f"Unexpected METR-LA client sensor counts: {sensor_counts}")
    synthetic_manifest: dict[str, Any] | None = None
    if synthetic_dir is not None:
        synthetic_dir = Path(synthetic_dir)
        synthetic_manifest = json.loads((synthetic_dir / "synthetic_manifest.json").read_text(encoding="utf-8"))
        if synthetic_manifest.get("validation_used") or synthetic_manifest.get("test_used"):
            raise ValueError("Synthetic-data manifest indicates validation/test data was used")
        if len(synthetic_manifest.get("clients", [])) != len(clients):
            raise ValueError("Synthetic files must be available for each client")

    # Central validation is used only to select global checkpoints. The test
    # arrays are intentionally not opened until all round selection is done.
    validation_X, validation_y = _load_split(data_dir, "validation")
    initial_model = SharedLinearForecaster(config.lookback_steps, config.horizon_steps)
    global_state = initial_model.state()
    best_state = {key: value.copy() for key, value in global_state.items()}
    best_validation_mse = float("inf")
    best_round = 0
    rounds_without_improvement = 0
    history: list[dict[str, float]] = []
    model_tensor_bytes = sum(value.nbytes for value in global_state.values())
    download_bytes = 0
    upload_bytes = 0
    local_train_seconds = 0.0
    started = time.perf_counter()

    local_config = TrainingConfig(
        lookback_steps=config.lookback_steps,
        horizon_steps=config.horizon_steps,
        max_epochs=config.local_epochs,
        batch_windows=config.batch_windows,
        learning_rate=config.learning_rate,
        patience=config.patience,
        min_delta=config.min_delta,
        seed=config.seed,
    )

    for communication_round in range(1, config.communication_rounds + 1):
        client_states: list[dict[str, np.ndarray]] = []
        aggregation_weights: list[int] = []
        client_train_losses: list[float] = []
        for client_index, client_metadata in enumerate(clients):
            client_id = client_metadata["client_id"]
            client_data = load_client_training_data(client_id, data_dir)
            original_X, original_y = client_data["X"], client_data["y"]
            client_X, client_y = original_X, original_y
            expected_sensors = int(client_metadata["sensor_count"])
            expected_windows = int(metadata["splits"]["train"]["samples"])
            if client_X.shape != (expected_windows, config.lookback_steps, expected_sensors):
                raise ValueError(f"Unexpected local X shape for {client_id}: {client_X.shape}")
            if client_y.shape != (client_X.shape[0], config.horizon_steps, expected_sensors):
                raise ValueError(f"Unexpected local y shape for {client_id}: {client_y.shape}")

            synthetic_windows = 0
            if synthetic_dir is not None:
                synthetic_path = synthetic_dir / f"{client_id}_synthetic_train.npz"
                with np.load(synthetic_path) as synthetic_file:
                    synthetic_X = synthetic_file["X"]
                    synthetic_y = synthetic_file["y"]
                if synthetic_X.shape[1:] != original_X.shape[1:] or synthetic_y.shape[1:] != original_y.shape[1:]:
                    raise ValueError(f"Synthetic array shapes do not match original data for {client_id}")
                if synthetic_X.shape[0] == 0 or synthetic_y.shape[0] != synthetic_X.shape[0]:
                    raise ValueError(f"Invalid synthetic window count for {client_id}")
                synthetic_windows = int(synthetic_X.shape[0])
                # Original and generated examples remain distinct in storage;
                # concatenate only in this client's local fit call.
                client_X = np.concatenate((original_X, synthetic_X), axis=0)
                client_y = np.concatenate((original_y, synthetic_y), axis=0)
                del synthetic_X, synthetic_y

            # A client receives only global parameters and its own local data.
            download_bytes += model_tensor_bytes
            local_model, losses, elapsed = train_local_model(
                client_X,
                client_y,
                global_state,
                local_config,
                local_epochs=config.local_epochs,
                seed=config.seed + communication_round * 100 + client_index,
            )
            local_train_seconds += elapsed
            state = local_model.state()
            if any(value.shape != global_state[key].shape for key, value in state.items()):
                raise ValueError(f"{client_id} returned model parameters incompatible with global model")
            client_states.append(state)
            aggregation_weights.append(int(client_X.shape[0] * client_X.shape[2]))
            client_train_losses.append(float(losses[-1]))
            upload_bytes += sum(value.nbytes for value in state.values())
            del client_X, client_y, original_X, original_y, client_data, local_model

        # Only model parameters and example counts are aggregated.
        global_state = fedavg(client_states, aggregation_weights)
        initial_model.load_state(global_state)
        validation_result = _metrics(initial_model, validation_X, validation_y, config.batch_windows)
        validation_mse = validation_result["rmse"] ** 2
        weighted_train_mse = float(np.average(client_train_losses, weights=aggregation_weights))
        history.append({
            "round": float(communication_round),
            "weighted_client_train_mse": weighted_train_mse,
            "validation_mse": float(validation_mse),
            "validation_mae": float(validation_result["mae"]),
        })
        if validation_mse < best_validation_mse - config.min_delta:
            best_validation_mse = validation_mse
            best_round = communication_round
            best_state = {key: value.copy() for key, value in global_state.items()}
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        if rounds_without_improvement >= config.patience:
            break

    total_training_seconds = time.perf_counter() - started
    initial_model.load_state(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "fedavg_global_model.npz"
    np.savez_compressed(model_path, **best_state)

    # Test data is first opened after validation-based round selection.
    del validation_X, validation_y
    test_X, test_y = _load_split(data_dir, "test")
    test_metrics = _metrics(initial_model, test_X, test_y, config.batch_windows)
    centralized_model, centralized_config = load_saved_model(centralized_dir)
    del centralized_model
    central_recorded = json.loads((centralized_dir / "metrics.json").read_text(encoding="utf-8"))
    if central_recorded["test_shape"] != {"X": list(test_X.shape), "y": list(test_y.shape)}:
        raise ValueError("Centralized and federated test arrays do not have matching shapes")
    if centralized_config["horizon_steps"] != config.horizon_steps:
        raise ValueError("Centralized and federated forecast horizons differ")

    actual_rounds = len(history)
    one_way_bytes_per_round = model_tensor_bytes * len(clients)
    total_communication_bytes = download_bytes + upload_bytes
    config_payload = asdict(config)
    config_payload.update({
        "model": "shared per-sensor linear autoregressive forecaster",
        "aggregation": "FedAvg weighted by client training sensor-window example count",
        "client_sensor_counts": {c["client_id"]: int(c["sensor_count"]) for c in clients},
        "client_example_counts": {
            c["client_id"]: int(
                (metadata["splits"]["train"]["samples"] +
                 (synthetic_manifest["clients"][index]["synthetic_window_count"] if synthetic_manifest else 0))
                * c["sensor_count"]
            ) for index, c in enumerate(clients)
        },
        "client_original_window_counts": {
            c["client_id"]: int(metadata["splits"]["train"]["samples"]) for c in clients
        },
        "client_synthetic_window_counts": {
            c["client_id"]: int(synthetic_manifest["clients"][index]["synthetic_window_count"] if synthetic_manifest else 0)
            for index, c in enumerate(clients)
        },
        "selection_split": "validation",
        "selection_metric": "global validation MSE",
        "best_round": best_round,
        "rounds_run": actual_rounds,
        "parameters": {"weights": list(best_state["weights"].shape), "bias": list(best_state["bias"].shape)},
        "communication_payload": {
            "parameters_per_client_bytes": int(model_tensor_bytes),
            "downlink_bytes_per_round": int(one_way_bytes_per_round),
            "uplink_bytes_per_round": int(one_way_bytes_per_round),
            "downlink_bytes_total": int(download_bytes),
            "uplink_bytes_total": int(upload_bytes),
            "total_bytes": int(total_communication_bytes),
            "basis": "float32 model arrays only; excludes transport/protocol/serialization metadata",
        },
        "training_time_includes": "local client fitting, parameter aggregation, and validation per round; excludes test evaluation and artifact/plot writing",
        "test_opened_after_model_selection": True,
        "synthetic_augmentation": None if synthetic_manifest is None else {
            "manifest_file": str(Path(synthetic_dir) / "synthetic_manifest.json"),
            "parameters": synthetic_manifest["augmentation"],
            "generation_seconds": synthetic_manifest["generation_seconds"],
            "validation_used": False,
            "test_used": False,
        },
        "privacy_note": "single-machine simulation; raw client arrays remain in local client files and only model parameters are aggregated; no formal differential privacy or secure aggregation",
    })
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    comparison = {
        "mae": _compare_metric(test_metrics["mae"], central_recorded["test"]["mae"]),
        "rmse": _compare_metric(test_metrics["rmse"], central_recorded["test"]["rmse"]),
        "centralized_training_seconds": float(central_recorded["training_seconds"]),
        "federated_training_seconds": float(total_training_seconds),
    }
    result = {
        "dataset": "METR-LA",
        "test": test_metrics,
        "comparison_with_centralized": comparison,
        "training_seconds": total_training_seconds,
        "local_client_training_seconds_sum": local_train_seconds,
        "communication": config_payload["communication_payload"],
        "rounds_run": actual_rounds,
        "best_round": best_round,
        "test_shape": {"X": list(test_X.shape), "y": list(test_y.shape)},
        "test_opened_after_model_selection": True,
        "model_file": model_path.name,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "round_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "weighted_client_train_mse", "validation_mse", "validation_mae"])
        writer.writeheader()
        writer.writerows(history)

    rounds = [int(row["round"]) for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(rounds, [row["weighted_client_train_mse"] for row in history], label="Weighted client train MSE")
    plt.plot(rounds, [row["validation_mse"] for row in history], label="Global validation MSE")
    plt.xlabel("Communication round")
    plt.ylabel("Mean squared error (dataset speed units²)")
    plt.title("FedAvg training and validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "round_learning_curve.png", dpi=150)
    plt.close()

    horizons = np.arange(1, config.horizon_steps + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(horizons * 5, test_metrics["mae_by_horizon"], marker="o", label="FedAvg MAE")
    plt.plot(horizons * 5, test_metrics["rmse_by_horizon"], marker="o", label="FedAvg RMSE")
    plt.plot(horizons * 5, central_recorded["test"]["mae_by_horizon"], marker="o", label="Centralized MAE")
    plt.plot(horizons * 5, central_recorded["test"]["rmse_by_horizon"], marker="o", label="Centralized RMSE")
    plt.xlabel("Forecast lead time (minutes)")
    plt.ylabel("Error (dataset speed units)")
    plt.title("Centralized and federated test errors by lead time")
    plt.xticks(horizons * 5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_error_comparison_by_horizon.png", dpi=150)
    plt.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/metr_la"))
    parser.add_argument("--centralized-dir", type=Path, default=Path("models/centralized_baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/federated_baseline"))
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-windows", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = FederatedConfig(
        communication_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_windows=args.batch_windows,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    result = run_federated_experiment(args.data_dir, args.output_dir, args.centralized_dir, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
