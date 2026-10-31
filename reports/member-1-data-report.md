# Member 1 Data Preparation Report — Draft 1

**Status:** Stage 1 data pipeline completed; ready for Member 2 and Member 3 review. No model training has been performed.

## Dataset and validation

The pipeline uses only `data/raw/metr-la.h5`, reading the pandas DataFrame stored at HDF5 key `/df`. Validation on this copy found 34,272 timestamp rows and 207 sensor columns. The values are stored as `float64`; processed windows are saved as `float32`. The index is a unique, strictly increasing `DatetimeIndex` covering 2012-03-01 00:00 through 2012-06-27 23:55. Every adjacent timestamp is exactly five minutes apart. There are no missing or non-finite sensor values, duplicate timestamps, or irregular intervals in this file.

The reusable loader checks the key conditions before processing: non-empty numeric DataFrame, datetime index, valid/unique/increasing timestamps, unique sensor names, finite values, and exact five-minute cadence. It raises a clear error if a condition fails.

## Window and forecast contract

The default input lookback is 12 readings (one hour). The forecast horizon is 12 future readings (one hour). Both defaults are configurable in the preparation command. Each `X` sample has shape `(12, sensor_count)` and its matching `y` sample has shape `(12, sensor_count)`. Files store batches as `(samples, steps, sensors)`.

The default split is chronological by raw timestamp rows: 70% train, 15% validation, 15% test. There is no random shuffling. A target sequence must be fully inside one split. Validation and test windows may use the immediately preceding 12 readings as input context, including readings before a split boundary; those earlier readings are context only, not target labels in the later split. This matches how a forecast is made at a time boundary while keeping training target labels out of validation/test and test targets out of training.

| Split | Raw row range (start inclusive, end exclusive) | Target timestamps covered | Window samples | Central `X` and `y` shapes |
|---|---:|---|---:|---|
| Train | `[0, 23990)` | 2012-03-01 01:00 through 2012-05-23 07:05 | 23,967 | `(23967, 12, 207)` |
| Validation | `[23990, 29130)` | 2012-05-23 07:10 through 2012-06-10 03:25 | 5,129 | `(5129, 12, 207)` |
| Test | `[29130, 34272)` | 2012-06-10 03:30 through 2012-06-27 23:55 | 5,131 | `(5131, 12, 207)` |

Each split's `X` and `y` have the same sample/time/sensor shape, with `float32` values. No scaling or synthetic augmentation is done at this stage. Any scaler or synthetic data fit/generation in Stage 2 should use training data only.

## Four simulated clients

Clients represent disjoint groups of METR-LA sensor columns, not organizations or real devices. Sensor columns are divided in their original order into four balanced groups: 52, 52, 52, and 51 sensors. Every sensor is assigned exactly once. For Member 3, each client's training file contains only that client's sensor channels, with shapes `(23967, 12, 52)` for clients 1–3 and `(23967, 12, 51)` for client 4; matching `y` has the same shape. Client files contain training data only. Validation and test are saved once as central evaluation splits.

Since client channel counts differ by one and each client has a different sensor set, Members 2 and 3 should choose a model parameterization that can be shared across sensor groups independently of a fixed 207-channel input layer (for example, a shared per-sensor temporal model). If they instead choose a model that binds parameters to sensor IDs or fixed channel positions, the client/model interface must be aligned before federated training.

## Output files and loading

Running `python3 -m src.data.metr_la` writes the reproducible outputs under `data/processed/metr_la/`:

- `train.npz`, `validation.npz`, `test.npz`: compressed arrays named `X` and `y` for centralized training/evaluation.
- `clients/client_1_train.npz` through `client_4_train.npz`: each client's training arrays, named `X` and `y`.
- `metadata.json`: source, dimensions, timestamp and cadence checks, split boundaries, shapes, horizon/lookback, sensor IDs, and client assignments.

Member 2 can use:

```python
from src.data.metr_la import load_prepared_data

data = load_prepared_data("data/processed/metr_la")
X_train, y_train = data["splits"]["train"]["X"], data["splits"]["train"]["y"]
X_val, y_val = data["splits"]["validation"]["X"], data["splits"]["validation"]["y"]
X_test, y_test = data["splits"]["test"]["X"], data["splits"]["test"]["y"]
```

Member 3 can load a client's local training data with:

```python
from src.data.metr_la import load_client_training_data

client = load_client_training_data("client_1", "data/processed/metr_la")
X_local, y_local = client["X"], client["y"]
```

Keep test arrays for final evaluation only; do not fit preprocessing or generate synthetic training data from validation or test targets. The pipeline does not start training.

## Tests and observed outcomes

`python3 -m pytest -q` completed with **3 passed**. Tests cover valid cadence, rejection of missing values, duplicate timestamps and irregular intervals, chronological boundaries and window counts, `float32` serialization/loading, disjoint full sensor assignment, client array shapes, and exclusion of validation/test targets from client training files.

The full-data command `python3 -m src.data.metr_la` completed successfully on the existing METR-LA HDF5 file. A separate artifact readback confirmed all central/client arrays load with expected shapes and contain only finite values. No prediction model, synthetic augmentation, or experiment was run.

## Limitations and handoff notes

- The client split is a simple sensor-column partition for a local simulation. It does not establish real-world institutional data silos or provide a privacy guarantee.
- The split boundary fractions and one-hour input lookback are documented defaults, not requirements explicitly specified by the research paper or assessment brief.
- Values are cast from `float64` to `float32`; no normalization is performed here.
- A future model should fit any normalization only on training data and must handle client tensors with 51 or 52 sensor channels.
