"""Deterministic, client-local Gaussian jitter for training windows only.

This is a simple robustness augmentation, not a generative model. The input
windows are perturbed while their original future targets are retained.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.metr_la import load_client_training_data


@dataclass(frozen=True)
class NoiseAugmentationConfig:
    noise_std_fraction: float = 0.05
    synthetic_copies_per_window: int = 1
    minimum_speed: float = 0.0
    seed: int = 42


def augment_training_windows(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: NoiseAugmentationConfig = NoiseAugmentationConfig(),
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create jittered input copies, retaining the original training targets.

    Per-sensor noise standard deviations are ``fraction * std(X_train)`` where
    the standard deviation is fitted only from this client's training inputs.
    The function takes no validation/test arguments and does not mutate inputs.
    """
    generation_started = time.perf_counter()
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if X_train.ndim != 3 or y_train.ndim != 3:
        raise ValueError("Training X/y must have shape (windows, steps, sensors)")
    if X_train.shape[0] == 0 or X_train.shape[2] == 0:
        raise ValueError("Training arrays must contain windows and sensors")
    if X_train.shape[0] != y_train.shape[0] or X_train.shape[2] != y_train.shape[2]:
        raise ValueError("Training X/y window and sensor dimensions must match")
    if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
        raise ValueError("Training arrays must contain only finite values")
    if config.noise_std_fraction < 0:
        raise ValueError("noise_std_fraction must be non-negative")
    if config.synthetic_copies_per_window < 1:
        raise ValueError("synthetic_copies_per_window must be positive")
    if np.any(X_train < config.minimum_speed) or np.any(y_train < config.minimum_speed):
        raise ValueError("Training values fall below configured minimum_speed")

    sensor_std = X_train.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    noise_scale = sensor_std * np.float32(config.noise_std_fraction)
    rng = np.random.default_rng(config.seed if seed is None else seed)
    X_copies: list[np.ndarray] = []
    y_copies: list[np.ndarray] = []
    for _ in range(config.synthetic_copies_per_window):
        noise = rng.normal(0.0, noise_scale[None, None, :], size=X_train.shape).astype(np.float32)
        noisy_inputs = np.maximum(X_train + noise, np.float32(config.minimum_speed))
        X_copies.append(noisy_inputs)
        # Keep labels unchanged: the task remains forecasting the observed
        # future values, while the model learns robustness to input jitter.
        y_copies.append(y_train.copy())
    synthetic_X = np.concatenate(X_copies, axis=0)
    synthetic_y = np.concatenate(y_copies, axis=0)
    details = {
        "method": "independent additive Gaussian jitter to input windows; targets copied unchanged",
        "noise_std_fraction_of_client_sensor_train_input_std": config.noise_std_fraction,
        "minimum_speed_clip": config.minimum_speed,
        "synthetic_copies_per_original_window": config.synthetic_copies_per_window,
        "seed": int(config.seed if seed is None else seed),
        "original_window_count": int(X_train.shape[0]),
        "synthetic_window_count": int(synthetic_X.shape[0]),
        "sensor_count": int(X_train.shape[2]),
        "noise_std_by_sensor": noise_scale.tolist(),
        "noise_std_summary": {
            "min": float(noise_scale.min()),
            "mean": float(noise_scale.mean()),
            "max": float(noise_scale.max()),
        },
        "fit_source": "this client's X_train only",
        "generation_seconds": time.perf_counter() - generation_started,
    }
    return synthetic_X, synthetic_y, details


def generate_synthetic_client_files(
    data_dir: str | Path,
    output_dir: str | Path,
    config: NoiseAugmentationConfig = NoiseAugmentationConfig(),
) -> dict[str, Any]:
    """Generate separate synthetic files using client training files only."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    clients = metadata["client_partition"]["clients"]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    client_records: list[dict[str, Any]] = []
    for client_index, client_metadata in enumerate(clients):
        client_id = client_metadata["client_id"]
        local_train = load_client_training_data(client_id, data_dir)
        synthetic_X, synthetic_y, details = augment_training_windows(
            local_train["X"],
            local_train["y"],
            config,
            seed=config.seed + client_index,
        )
        synthetic_path = output_dir / f"{client_id}_synthetic_train.npz"
        np.savez_compressed(synthetic_path, X=synthetic_X, y=synthetic_y)
        details.update({
            "client_id": client_id,
            "seed": config.seed + client_index,
            "source_file": client_metadata["train_file"],
            "synthetic_file": synthetic_path.name,
            "original_sensor_window_examples": int(local_train["X"].shape[0] * local_train["X"].shape[2]),
            "synthetic_sensor_window_examples": int(synthetic_X.shape[0] * synthetic_X.shape[2]),
        })
        client_records.append(details)
        del local_train, synthetic_X, synthetic_y
    manifest = {
        "augmentation": asdict(config),
        "client_count": len(clients),
        "source_data_scope": "per-client training files only",
        "validation_used": False,
        "test_used": False,
        "generation_seconds": time.perf_counter() - started,
        "clients": client_records,
    }
    (output_dir / "synthetic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
