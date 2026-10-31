"""Prepare METR-LA into chronological forecasting splits and client files.

Input and target arrays use ``float32`` with shapes ``(samples, steps, sensors)``.
The default lookback and horizon are each 12 five-minute readings (one hour).
Validation and test targets stay within their own chronological split; their
input context may use readings immediately before the split boundary, as it
would in real forecasting. No validation/test targets are included in training.

Clients represent non-overlapping groups of sensors, not real organizations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("data/raw/metr-la.h5")
DEFAULT_OUTPUT = Path("data/processed/metr_la")
DEFAULT_LOOKBACK = 12
DEFAULT_HORIZON = 12
EXPECTED_INTERVAL = pd.Timedelta(minutes=5)


def load_metr_la(path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load and validate the METR-LA pandas HDF5 DataFrame at ``/df``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"METR-LA HDF5 file not found: {path}")
    try:
        frame = pd.read_hdf(path, key="df")
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"Could not read a DataFrame from HDF5 key '/df': {exc}") from exc
    validate_frame(frame)
    return frame


def validate_frame(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` for invalid time index, sensors, values, or cadence."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Dataset must be a non-empty pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Dataset index must be a DatetimeIndex")
    if frame.index.hasnans:
        raise ValueError("Timestamp index contains invalid timestamps")
    if frame.index.has_duplicates:
        raise ValueError("Timestamp index contains duplicate timestamps")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be in strictly increasing order")
    if frame.columns.has_duplicates:
        raise ValueError("Sensor columns must be unique")
    if len(frame.columns) == 0:
        raise ValueError("Dataset must contain at least one sensor column")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("All sensor values must be numeric")
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Sensor values contain missing or non-finite values")
    if len(frame) < 2:
        raise ValueError("At least two timestamp rows are required to validate cadence")
    intervals = frame.index.to_series().diff().iloc[1:]
    if not intervals.eq(EXPECTED_INTERVAL).all():
        observed = sorted({str(value) for value in intervals.unique()})
        raise ValueError(
            "Timestamps must be regularly spaced at five-minute intervals; "
            f"observed intervals: {observed}"
        )


def _split_bounds(n_rows: int, train_fraction: float, validation_fraction: float) -> dict[str, tuple[int, int]]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Train and validation fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1")
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    if train_end == 0 or validation_end <= train_end or validation_end >= n_rows:
        raise ValueError("Split fractions produce an empty time partition")
    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, n_rows),
    }


def _make_split(values: np.ndarray, bounds: tuple[int, int], lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create windows whose target timesteps are wholly inside ``bounds``."""
    split_start, split_end = bounds
    first_target = max(lookback, split_start)
    target_starts = range(first_target, split_end - horizon + 1)
    if len(target_starts) == 0:
        raise ValueError("A split is too short for the requested lookback/horizon")
    samples = len(target_starts)
    sensor_count = values.shape[1]
    X = np.empty((samples, lookback, sensor_count), dtype=np.float32)
    y = np.empty((samples, horizon, sensor_count), dtype=np.float32)
    for row, target_start in enumerate(target_starts):
        X[row] = values[target_start - lookback : target_start]
        y[row] = values[target_start : target_start + horizon]
    return X, y, np.asarray([first_target, split_end - horizon], dtype=np.int64)


def _save_npz(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)


def prepare_dataset(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    client_count: int = 4,
) -> dict[str, Any]:
    """Write central splits, four sensor-group client training sets, and metadata."""
    if lookback < 1 or horizon < 1:
        raise ValueError("Lookback and horizon must be positive")
    if client_count != 4:
        raise ValueError("This project data contract uses exactly four simulated clients")
    frame = load_metr_la(input_path)
    values = frame.to_numpy(dtype=np.float32, copy=True)
    bounds = _split_bounds(len(frame), train_fraction, validation_fraction)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_metadata: dict[str, Any] = {}
    for split_name, row_bounds in bounds.items():
        X, y, target_range = _make_split(values, row_bounds, lookback, horizon)
        _save_npz(output_dir / f"{split_name}.npz", X, y)
        start, end = row_bounds
        target_first, target_last = map(int, target_range)
        split_metadata[split_name] = {
            "row_start_inclusive": start,
            "row_end_exclusive": end,
            "row_start_timestamp": frame.index[start].isoformat(),
            "row_end_timestamp_exclusive": frame.index[end].isoformat() if end < len(frame) else None,
            "target_start_row_first_sample": target_first,
            "target_start_row_last_sample": target_last,
            "samples": int(X.shape[0]),
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "dtype": "float32",
        }
        if split_name == "train":
            train_X, train_y = X, y
        del X, y

    sensor_groups = np.array_split(np.arange(len(frame.columns)), client_count)
    clients: list[dict[str, Any]] = []
    for client_number, sensor_indices in enumerate(sensor_groups, start=1):
        client_id = f"client_{client_number}"
        client_dir = output_dir / "clients"
        _save_npz(
            client_dir / f"{client_id}_train.npz",
            train_X[:, :, sensor_indices],
            train_y[:, :, sensor_indices],
        )
        clients.append(
            {
                "client_id": client_id,
                "partition_rule": "contiguous groups in original sensor-column order; balanced by sensor count",
                "sensor_count": int(len(sensor_indices)),
                "sensor_ids": [str(frame.columns[i]) for i in sensor_indices],
                "train_file": f"clients/{client_id}_train.npz",
                "X_shape": [int(train_X.shape[0]), lookback, int(len(sensor_indices))],
                "y_shape": [int(train_y.shape[0]), horizon, int(len(sensor_indices))],
            }
        )
    del train_X, train_y, values

    metadata: dict[str, Any] = {
        "dataset": "METR-LA",
        "source_file": str(Path(input_path)),
        "hdf5_key": "/df",
        "raw_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "raw_value_dtypes": sorted({str(dtype) for dtype in frame.dtypes}),
        "processed_dtype": "float32",
        "sensor_ids": [str(sensor) for sensor in frame.columns],
        "timestamp_start": frame.index[0].isoformat(),
        "timestamp_end": frame.index[-1].isoformat(),
        "sampling_interval_minutes": 5,
        "lookback_steps": lookback,
        "lookback_minutes": lookback * 5,
        "forecast_horizon_steps": horizon,
        "forecast_horizon_minutes": horizon * 5,
        "split_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1 - train_fraction - validation_fraction,
        },
        "split_method": "chronological row boundaries; no shuffling",
        "window_policy": "targets lie wholly within their split; validation/test inputs may use preceding context",
        "missing_values": int(frame.isna().to_numpy().sum()),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "interval_check": "all adjacent timestamps are five minutes apart",
        "splits": split_metadata,
        "client_partition": {
            "count": client_count,
            "meaning": "simulated sensor groups on one machine; not real organizations",
            "training_data_only": True,
            "clients": clients,
        },
        "test_data_usage": "test split is saved for evaluation; it is not included in client training files",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_prepared_data(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Load central train/validation/test arrays and metadata for Member 2."""
    root = Path(output_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    splits: dict[str, dict[str, np.ndarray]] = {}
    for name in ("train", "validation", "test"):
        with np.load(root / f"{name}.npz") as stored:
            splits[name] = {"X": stored["X"], "y": stored["y"]}
    return {"metadata": metadata, "splits": splits}


def load_client_training_data(
    client_id: str, output_dir: str | Path = DEFAULT_OUTPUT
) -> dict[str, np.ndarray]:
    """Load one simulated client's training arrays for Member 3."""
    if client_id not in {f"client_{i}" for i in range(1, 5)}:
        raise ValueError("client_id must be one of client_1, client_2, client_3, client_4")
    path = Path(output_dir) / "clients" / f"{client_id}_train.npz"
    with np.load(path) as stored:
        return {"X": stored["X"], "y": stored["y"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    metadata = prepare_dataset(
        args.input,
        args.output,
        lookback=args.lookback,
        horizon=args.horizon,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps({
        "output": str(args.output),
        "raw_shape": metadata["raw_shape"],
        "splits": {name: details["samples"] for name, details in metadata["splits"].items()},
        "clients": {client["client_id"]: client["sensor_count"] for client in metadata["client_partition"]["clients"]},
    }, indent=2))


if __name__ == "__main__":
    main()
