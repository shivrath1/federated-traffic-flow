"""Run the existing FedAvg experiment with AES-GCM-protected client updates.

This is a single-machine simulation. Each client update crosses an explicit
serialization/encryption boundary before the coordinator authenticates and
decrypts it. The baseline runner and its artifacts are not modified.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from src.data.metr_la import load_client_training_data
from src.federated.fedavg import (
    FederatedConfig,
    _load_split,
    fedavg,
)
from src.models.centralized_baseline import (
    SharedLinearForecaster,
    TrainingConfig,
    _metrics,
    load_saved_model,
    train_local_model,
)
from src.security.model_update_encryption import (
    ReplayStore,
    aggregate_encrypted_updates,
    encrypt_model_update,
    load_client_keys,
    serialize_model_update,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"Unable to read required experiment metadata: {path.name}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Experiment metadata must be an object: {path.name}")
    return value


def _config_from_baseline(baseline_dir: Path) -> FederatedConfig:
    saved = _read_json(baseline_dir / "config.json")
    required = {item.name for item in fields(FederatedConfig)}
    missing = required.difference(saved)
    if missing:
        raise ValueError(f"Baseline federated configuration is missing fields: {sorted(missing)}")
    return FederatedConfig(**{name: saved[name] for name in required})


def run_secure_federated_experiment(
    data_dir: Path,
    output_dir: Path,
    centralized_dir: Path,
    baseline_dir: Path,
) -> dict[str, Any]:
    """Re-run baseline FedAvg settings with authenticated encrypted updates."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    centralized_dir, baseline_dir = Path(centralized_dir), Path(baseline_dir)
    config = _config_from_baseline(baseline_dir)
    baseline_metrics = _read_json(baseline_dir / "metrics.json")
    baseline_saved_config = _read_json(baseline_dir / "config.json")
    if baseline_metrics.get("rounds_run") != config.communication_rounds:
        raise ValueError("Secure comparison requires baseline rounds to equal configured communication rounds")
    if baseline_saved_config.get("rounds_run") != config.communication_rounds:
        raise ValueError("Baseline artifact did not complete the configured number of rounds")

    metadata = _read_json(data_dir / "metadata.json")
    clients = metadata.get("client_partition", {}).get("clients")
    if not isinstance(clients, list) or len(clients) != 4:
        raise ValueError("Prepared data must define exactly four client partitions")
    client_ids = [client.get("client_id") for client in clients]
    expected_counts = {"client_1": 52, "client_2": 52, "client_3": 52, "client_4": 51}
    if {client.get("client_id"): client.get("sensor_count") for client in clients} != expected_counts:
        raise ValueError("Prepared client partitions do not match the existing baseline configuration")
    keys = load_client_keys(client_ids)
    if metadata.get("lookback_steps") != config.lookback_steps or metadata.get("forecast_horizon_steps") != config.horizon_steps:
        raise ValueError("Baseline model configuration does not match prepared data")
    if baseline_saved_config.get("client_sensor_counts") != expected_counts:
        raise ValueError("Baseline model configuration has different client partitions")

    # The durable SQLite ledger makes a process restart retire the prior
    # session and prevents session-ID and per-client nonce reuse.
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_store = ReplayStore(output_dir / "replay_state.sqlite3")
    session_id = replay_store.begin_session()

    validation_X, validation_y = _load_split(data_dir, "validation")
    initial_model = SharedLinearForecaster(config.lookback_steps, config.horizon_steps)
    global_state = initial_model.state()
    best_state = {name: array.copy() for name, array in global_state.items()}
    best_validation_mse = float("inf")
    best_round = 0
    rounds_without_improvement = 0
    model_bytes = sum(array.nbytes for array in global_state.values())
    local_training_seconds = 0.0
    encryption_seconds: list[float] = []
    decryption_seconds: list[float] = []
    secure_training_seconds = 0.0
    original_payload_bytes = 0
    encrypted_payload_bytes = 0
    round_records: list[dict[str, float]] = []

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

    experiment_started = time.perf_counter()
    for communication_round in range(1, config.communication_rounds + 1):
        round_started = time.perf_counter()
        messages: list[bytes] = []
        weights: list[int] = []
        client_train_losses: list[float] = []
        round_local_seconds = 0.0
        round_encrypt_seconds = 0.0
        for client_index, client_id in enumerate(client_ids):
            client_data = load_client_training_data(client_id, data_dir)
            client_X, client_y = client_data["X"], client_data["y"]
            expected_sensors = expected_counts[client_id]
            expected_windows = metadata["splits"]["train"]["samples"]
            if client_X.shape != (expected_windows, config.lookback_steps, expected_sensors):
                raise ValueError(f"Unexpected local X shape for {client_id}: {client_X.shape}")
            if client_y.shape != (expected_windows, config.horizon_steps, expected_sensors):
                raise ValueError(f"Unexpected local y shape for {client_id}: {client_y.shape}")

            local_model, losses, elapsed = train_local_model(
                client_X,
                client_y,
                global_state,
                local_config,
                local_epochs=config.local_epochs,
                seed=config.seed + communication_round * 100 + client_index,
            )
            local_training_seconds += elapsed
            round_local_seconds += elapsed
            state = local_model.state()
            if any(state[name].shape != global_state[name].shape for name in global_state):
                raise ValueError(f"{client_id} returned model parameters incompatible with global model")

            serialized_size = len(serialize_model_update(state))
            encrypt_started = time.perf_counter()
            encrypted = encrypt_model_update(
                client_id,
                state,
                communication_round,
                session_id,
                keys[client_id],
                replay_store,
            )
            encrypt_elapsed = time.perf_counter() - encrypt_started
            encryption_seconds.append(encrypt_elapsed)
            round_encrypt_seconds += encrypt_elapsed
            original_payload_bytes += serialized_size
            encrypted_payload_bytes += len(encrypted)
            messages.append(encrypted)
            weights.append(int(client_X.shape[0] * client_X.shape[2]))
            client_train_losses.append(float(losses[-1]))
            del client_X, client_y, client_data, local_model

        aggregated_state = aggregate_encrypted_updates(
            messages,
            client_ids,
            weights,
            communication_round,
            session_id,
            keys,
            replay_store,
            fedavg,
            decryption_seconds,
        )
        global_state = aggregated_state
        initial_model.load_state(global_state)
        validation_result = _metrics(initial_model, validation_X, validation_y, config.batch_windows)
        validation_mse = validation_result["rmse"] ** 2
        weighted_train_mse = float(np.average(client_train_losses, weights=weights))
        round_records.append({
            "round": float(communication_round),
            "weighted_client_train_mse": weighted_train_mse,
            "validation_mse": float(validation_mse),
            "validation_mae": float(validation_result["mae"]),
        })
        if validation_mse < best_validation_mse - config.min_delta:
            best_validation_mse = validation_mse
            best_round = communication_round
            best_state = {name: array.copy() for name, array in global_state.items()}
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        # The baseline training timer includes local fits, aggregation, and
        # validation. Subtract only measured encode/encrypt and decrypt work.
        round_elapsed = time.perf_counter() - round_started
        secure_training_seconds += max(
            0.0,
            round_elapsed - round_encrypt_seconds - sum(decryption_seconds[-len(client_ids):]),
        )
        if rounds_without_improvement >= config.patience:
            break

    total_wall_seconds = time.perf_counter() - experiment_started
    initial_model.load_state(best_state)
    del validation_X, validation_y
    test_X, test_y = _load_split(data_dir, "test")
    test_metrics = _metrics(initial_model, test_X, test_y, config.batch_windows)
    from src.federated.fedavg import load_federated_model

    baseline_model, _ = load_federated_model(baseline_dir)
    baseline_state = baseline_model.state()
    if any(not np.array_equal(best_state[name], baseline_state[name]) for name in best_state):
        raise RuntimeError("Secure run did not reproduce the complete baseline model state")
    baseline_prediction = baseline_model.predict(test_X[:32])
    secure_prediction = initial_model.predict(test_X[:32])
    if not np.array_equal(baseline_prediction, secure_prediction):
        raise RuntimeError("Secure run did not reproduce the baseline model under identical settings")
    _, centralized_config = load_saved_model(centralized_dir)
    if centralized_config.get("horizon_steps") != config.horizon_steps:
        raise ValueError("Centralized and federated forecast horizons differ")

    model_path = output_dir / "fedavg_global_model.npz"
    np.savez_compressed(model_path, **best_state)
    output_config = asdict(config)
    output_config.update({
        "model": baseline_saved_config["model"],
        "aggregation": baseline_saved_config["aggregation"],
        "client_sensor_counts": expected_counts,
        "best_round": best_round,
        "rounds_run": len(round_records),
        "selection_split": "validation",
        "selection_metric": "global validation MSE",
        "secure_transport": {
            "algorithm": "AES-256-GCM",
            "message_version": 1,
            "aad_fields": ["client_id", "round", "session_id", "version"],
            "session_id": session_id,
            "key_environment_variable_pattern": "FEDERATED_CLIENT_KEY_CLIENT_N (64 hexadecimal characters per client)",
            "replay_ledger": "replay_state.sqlite3",
            "identity_note": "authenticates possession of a shared client secret; not public-key identity or non-repudiation",
        },
        "comparison_with_baseline": {
            "baseline_directory": str(baseline_dir),
            "same_data_partitions": True,
            "same_random_seed": config.seed,
            "same_model_and_training_configuration": True,
            "same_completed_rounds": len(round_records),
            "baseline_completed_rounds": baseline_saved_config["rounds_run"],
            "first_32_test_predictions_bitwise_equal": True,
        },
        "timing_seconds": {
            "training_excluding_measured_crypto": float(secure_training_seconds),
            "encryption_including_serialization": float(sum(encryption_seconds)),
            "decryption_and_authentication_including_deserialization": float(sum(decryption_seconds)),
            "experiment_wall_before_test_evaluation": float(total_wall_seconds),
            "local_training_fit_sum": float(local_training_seconds),
            "encryption_message_count": len(encryption_seconds),
            "decryption_message_count": len(decryption_seconds),
        },
        "payload_bytes": {
            "serialized_update_payload_bytes": int(original_payload_bytes),
            "encrypted_envelope_bytes": int(encrypted_payload_bytes),
            "added_bytes": int(encrypted_payload_bytes - original_payload_bytes),
            "percentage_overhead": float(100 * (encrypted_payload_bytes - original_payload_bytes) / original_payload_bytes),
            "encrypted_messages": len(encryption_seconds),
            "basis": "serialized NPZ parameter tensors vs UTF-8 JSON envelope with base64 nonce/ciphertext; no network protocol overhead",
        },
        "test_opened_after_model_selection": True,
        "privacy_note": "single-machine simulation; AES-GCM authenticates shared-secret possession and protects message confidentiality/integrity; no secure aggregation, signatures, differential privacy, production key management, or real network",
    })
    (output_dir / "config.json").write_text(json.dumps(output_config, indent=2) + "\n", encoding="utf-8")
    result = {
        "dataset": "METR-LA",
        "test": test_metrics,
        "training_seconds_excluding_crypto": float(secure_training_seconds),
        "encryption_seconds": float(sum(encryption_seconds)),
        "decryption_verification_seconds": float(sum(decryption_seconds)),
        "experiment_wall_seconds_before_test_evaluation": float(total_wall_seconds),
        "payload_bytes": output_config["payload_bytes"],
        "rounds_run": len(round_records),
        "best_round": best_round,
        "baseline_test_mae": baseline_metrics["test"]["mae"],
        "baseline_test_rmse": baseline_metrics["test"]["rmse"],
        "secure_test_predictions_match_baseline": True,
        "model_file": model_path.name,
        "test_opened_after_model_selection": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "round_history.json").write_text(json.dumps(round_records, indent=2) + "\n", encoding="utf-8")
    replay_store.close_session(session_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/metr_la"))
    parser.add_argument("--centralized-dir", type=Path, default=Path("models/centralized_baseline"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("models/federated_baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/federated_secure_baseline"))
    args = parser.parse_args()
    result = run_secure_federated_experiment(
        args.data_dir, args.output_dir, args.centralized_dir, args.baseline_dir
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
