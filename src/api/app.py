"""FastAPI endpoints for held-out historical METR-LA forecasts and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data.metr_la import load_metr_la
from src.models.centralized_baseline import SharedLinearForecaster

ROOT = Path(__file__).resolve().parents[2]
MODEL_SPECS = {
    "centralized_baseline": {"name": "Centralized baseline", "directory": "centralized_baseline", "weights_file": "shared_linear_forecaster.npz", "training_mode": "centralized"},
    "centralized_augmented": {"name": "Centralized with noise augmentation", "directory": "centralized_augmented", "weights_file": "shared_linear_forecaster.npz", "training_mode": "centralized + train-only noise augmentation"},
    "fedavg": {"name": "FedAvg without augmentation", "directory": "federated_baseline", "weights_file": "fedavg_global_model.npz", "training_mode": "four-client simulated FedAvg"},
    "fedavg_augmented": {"name": "FedAvg with noise augmentation", "directory": "federated_augmented", "weights_file": "fedavg_global_model.npz", "training_mode": "four-client simulated FedAvg + train-only noise augmentation"},
}
ModelId = Literal["centralized_baseline", "centralized_augmented", "fedavg", "fedavg_augmented"]
LOOKBACK_STEPS = 12
HORIZON_STEPS = 12
INTERVAL_MINUTES = 5


class PredictionRequest(BaseModel):
    sensor_id: str = Field(min_length=1, max_length=32)
    model_id: ModelId
    # The timestamp is the first target/forecast reading, not the final input reading.
    forecast_origin: datetime | None = None


@dataclass
class LoadedModel:
    model: SharedLinearForecaster
    config: dict[str, Any]
    metrics: dict[str, Any]


def _error_summary(error: np.ndarray) -> dict[str, float]:
    error64 = np.asarray(error, dtype=np.float64)
    return {
        "mae": float(np.abs(error64).mean()),
        "rmse": float(np.sqrt(np.square(error64).mean())),
    }


class TrafficService:
    """Read-only dataset/model access; all forecast origins are held-out test targets."""

    def __init__(self, dataset_path: Path, model_root: Path, processed_root: Path):
        self.dataset_path = dataset_path
        self.model_root = model_root
        self.processed_root = processed_root
        try:
            self.frame = load_metr_la(dataset_path)
            self.metadata = json.loads((processed_root / "metadata.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("METR-LA historical data could not be initialized") from exc
        self.sensor_ids = [str(sensor) for sensor in self.frame.columns]
        self._models: dict[str, LoadedModel] = {}
        self._evaluation: dict[str, Any] | None = None
        test = self.metadata["splits"]["test"]
        self.test_start = int(test["row_start_inclusive"])
        self.test_end = int(test["row_end_exclusive"])
        self.default_origin, self.default_origin_method = self._choose_default_origin()

    def _choose_default_origin(self) -> tuple[pd.Timestamp, str]:
        """Choose the test 08:00 input nearest the training-period 08:00 median.

        Selection uses only the preceding one-hour inputs and a training-derived
        reference, never the selected test window's future target.
        """
        train_end = int(self.metadata["splits"]["train"]["row_end_exclusive"])
        train_means = [
            float(self.frame.iloc[position - LOOKBACK_STEPS:position].to_numpy().mean())
            for position in range(LOOKBACK_STEPS, train_end)
            if self.frame.index[position].hour == 8 and self.frame.index[position].minute == 0
        ]
        candidates = [
            position for position in range(self.test_start, self.test_end - HORIZON_STEPS + 1)
            if self.frame.index[position].hour == 8 and self.frame.index[position].minute == 0
        ]
        if not train_means or not candidates:
            raise RuntimeError("No valid representative held-out morning forecast window is available")
        reference = float(np.median(train_means))
        selected_position = min(
            candidates,
            key=lambda position: abs(
                float(self.frame.iloc[position - LOOKBACK_STEPS:position].to_numpy().mean()) - reference
            ),
        )
        selected = self.frame.index[selected_position]
        method = (
            "Held-out test 08:00 origin whose preceding 12-reading, all-sensor mean is closest "
            "to the median of training-period 08:00 input means; future targets were not used."
        )
        return selected, method

    def _validated_origin(self, value: datetime | pd.Timestamp | str | None) -> tuple[pd.Timestamp, int]:
        if value is None:
            origin = self.default_origin
        else:
            try:
                origin = pd.Timestamp(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail="Forecast origin must be a valid METR-LA timestamp") from None
        if origin.tzinfo is not None:
            raise HTTPException(status_code=422, detail="Forecast origin must use the dataset's timezone-naive timestamps")
        position = int(self.frame.index.get_indexer([origin])[0])
        if position < 0:
            raise HTTPException(status_code=422, detail="Forecast origin must match a five-minute METR-LA timestamp")
        if position < self.test_start or position + HORIZON_STEPS > self.test_end:
            start = self.frame.index[self.test_start].isoformat()
            last_origin = self.frame.index[self.test_end - HORIZON_STEPS].isoformat()
            raise HTTPException(
                status_code=422,
                detail=f"Choose a held-out test forecast origin from {start} through {last_origin}",
            )
        if position < LOOKBACK_STEPS:
            raise HTTPException(status_code=422, detail="The selected forecast origin has insufficient input history")
        return origin, position

    def get_model(self, model_id: str) -> LoadedModel:
        if model_id not in MODEL_SPECS:
            raise HTTPException(status_code=422, detail="Unknown model selection")
        if model_id in self._models:
            return self._models[model_id]
        spec = MODEL_SPECS[model_id]
        directory = self.model_root / spec["directory"]
        try:
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            if config.get("lookback_steps") != LOOKBACK_STEPS or config.get("horizon_steps") != HORIZON_STEPS:
                raise ValueError("Unexpected model dimensions")
            model = SharedLinearForecaster(config["lookback_steps"], config["horizon_steps"])
            with np.load(directory / spec["weights_file"]) as weights:
                model.load_state({"weights": weights["weights"], "bias": weights["bias"]})
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            raise HTTPException(
                status_code=503,
                detail=f"The saved artifact for '{spec['name']}' is unavailable or invalid",
            ) from None
        loaded = LoadedModel(model=model, config=config, metrics=metrics)
        self._models[model_id] = loaded
        return loaded

    def forecast_options(self) -> dict[str, Any]:
        return {
            "dataset": "METR-LA",
            "forecast_origin_definition": "Timestamp of the first predicted/target reading; the 12 inputs end five minutes before this timestamp.",
            "held_out_test_origin_start": self.frame.index[self.test_start].isoformat(),
            "held_out_test_origin_end": self.frame.index[self.test_end - HORIZON_STEPS].isoformat(),
            "default_forecast_origin": self.default_origin.isoformat(),
            "default_selection_method": self.default_origin_method,
            "forecast_horizon_steps": HORIZON_STEPS,
            "forecast_horizon_minutes": HORIZON_STEPS * INTERVAL_MINUTES,
            "sampling_interval_minutes": INTERVAL_MINUTES,
            "presets": [
                {"id": "morning_peak", "label": "Morning peak", "time": "08:00"},
                {"id": "midday", "label": "Midday", "time": "13:00"},
                {"id": "evening_peak", "label": "Evening peak", "time": "18:00"},
                {"id": "late_night", "label": "Late night", "time": "23:00"},
            ],
        }

    def history(self, sensor_id: str, limit: int, forecast_origin: datetime | None = None) -> dict[str, Any]:
        if sensor_id not in self.sensor_ids:
            raise HTTPException(status_code=404, detail="Sensor ID was not found in METR-LA")
        origin, position = self._validated_origin(forecast_origin)
        values = self.frame[sensor_id].iloc[max(0, position - limit):position]
        return {
            "dataset": "METR-LA",
            "sensor_id": sensor_id,
            "forecast_origin": origin.isoformat(),
            "sampling_interval_minutes": INTERVAL_MINUTES,
            "readings": [{"timestamp": stamp.isoformat(), "value": float(value)} for stamp, value in values.items()],
        }

    def predict(self, sensor_id: str, model_id: str, forecast_origin: datetime | None) -> dict[str, Any]:
        if sensor_id not in self.sensor_ids:
            raise HTTPException(status_code=404, detail="Sensor ID was not found in METR-LA")
        origin, position = self._validated_origin(forecast_origin)
        selected = self.get_model(model_id)
        input_history = self.frame[sensor_id].iloc[position - LOOKBACK_STEPS:position]
        context_history = self.frame[sensor_id].iloc[max(0, position - 36):position]
        actual_series = self.frame[sensor_id].iloc[position:position + HORIZON_STEPS]
        input_values = input_history.to_numpy(dtype=np.float32)
        actual = actual_series.to_numpy(dtype=np.float32)
        prediction = selected.model.predict(input_values.reshape(1, LOOKBACK_STEPS, 1))[0, :, 0]
        persistence = np.full(HORIZON_STEPS, input_values[-1], dtype=np.float32)
        model_error = prediction.astype(np.float64) - actual.astype(np.float64)
        persistence_error = persistence.astype(np.float64) - actual.astype(np.float64)
        model_by_step = np.abs(model_error)
        persistence_by_step = np.abs(persistence_error)

        def readings(series: pd.Series) -> list[dict[str, Any]]:
            return [{"timestamp": stamp.isoformat(), "value": float(value)} for stamp, value in series.items()]

        spec = MODEL_SPECS[model_id]
        return {
            "dataset": "METR-LA",
            "sensor_id": sensor_id,
            "model_id": model_id,
            "model_name": spec["name"],
            "training_mode": spec["training_mode"],
            "forecast_origin": origin.isoformat(),
            "last_observation_timestamp": input_history.index[-1].isoformat(),
            "forecast_horizon_steps": HORIZON_STEPS,
            "forecast_horizon_minutes": HORIZON_STEPS * INTERVAL_MINUTES,
            "sampling_interval_minutes": INTERVAL_MINUTES,
            "input_readings": readings(input_history),
            "history_readings": readings(context_history),
            "forecasts": [
                {"timestamp": stamp.isoformat(), "step": step, "minutes_ahead": step * INTERVAL_MINUTES, "value": float(value)}
                for step, (stamp, value) in enumerate(zip(actual_series.index, prediction), start=1)
            ],
            "actuals": [
                {"timestamp": stamp.isoformat(), "step": step, "minutes_ahead": step * INTERVAL_MINUTES, "value": float(value)}
                for step, (stamp, value) in enumerate(zip(actual_series.index, actual), start=1)
            ],
            "selected_window_metrics": {
                **_error_summary(model_error),
                "mae_by_step": model_by_step.tolist(),
                "rmse_by_step": np.abs(model_error).tolist(),
                "source": "new calculation on this held-out test window; not used for training or checkpoint selection",
            },
            "persistence_baseline": {
                "definition": "Repeat the final historical input value at all 12 forecast steps.",
                "predicted_value": float(persistence[0]),
                "forecasts": [
                    {"timestamp": stamp.isoformat(), "step": step, "value": float(value)}
                    for step, (stamp, value) in enumerate(zip(actual_series.index, persistence), start=1)
                ],
                "metrics": {
                    **_error_summary(persistence_error),
                    "mae_by_step": persistence_by_step.tolist(),
                    "rmse_by_step": np.abs(persistence_error).tolist(),
                    "source": "new calculation on the same held-out test window and targets",
                },
            },
            "context": "Historical METR-LA held-out test data; not a live traffic service.",
        }

    def model_list(self) -> dict[str, Any]:
        choices = []
        for model_id, spec in MODEL_SPECS.items():
            directory = self.model_root / spec["directory"]
            model_file = directory / spec["weights_file"]
            config_path = directory / "config.json"
            metrics_path = directory / "metrics.json"
            record: dict[str, Any] = {
                "model_id": model_id,
                "name": spec["name"],
                "training_mode": spec["training_mode"],
                "available": model_file.is_file() and config_path.is_file() and metrics_path.is_file(),
            }
            if record["available"]:
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    record["test_metrics"] = metrics["test"]
                    record["checkpoint"] = {key: config[key] for key in ("best_epoch", "best_round", "epochs_run", "rounds_run") if key in config}
                    record["configuration"] = {
                        key: config[key] for key in (
                            "lookback_steps", "horizon_steps", "max_epochs", "communication_rounds",
                            "local_epochs", "batch_windows", "learning_rate", "seed", "client_sensor_counts",
                            "aggregation", "communication_payload", "optimizer", "optimizer_note", "model",
                        ) if key in config
                    }
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    record["available"] = False
            choices.append(record)
        return {"models": choices}

    def _load_test_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            with np.load(self.processed_root / "test.npz") as stored:
                X, y = stored["X"], stored["y"]
        except (FileNotFoundError, OSError, KeyError, ValueError):
            raise HTTPException(status_code=503, detail="Prepared held-out evaluation data is unavailable") from None
        expected_samples = self.test_end - self.test_start - HORIZON_STEPS + 1
        if X.shape != (expected_samples, LOOKBACK_STEPS, len(self.sensor_ids)) or y.shape != (expected_samples, HORIZON_STEPS, len(self.sensor_ids)):
            raise HTTPException(status_code=503, detail="Prepared held-out evaluation data has an invalid shape")
        return X, y

    def evaluation(self) -> dict[str, Any]:
        """Reuse saved overall/per-horizon metrics and compute held-out diagnostics once."""
        if self._evaluation is not None:
            return self._evaluation
        X, y = self._load_test_arrays()
        sample_count, _, sensor_count = X.shape
        origin_positions = np.arange(self.test_start, self.test_start + sample_count)
        origin_hours = self.frame.index.take(origin_positions).hour.to_numpy()
        models: list[dict[str, Any]] = []

        for model_id, spec in MODEL_SPECS.items():
            loaded = self.get_model(model_id)
            sensor_abs = np.zeros(sensor_count, dtype=np.float64)
            sensor_sq = np.zeros(sensor_count, dtype=np.float64)
            hour_abs = np.zeros(24, dtype=np.float64)
            hour_sq = np.zeros(24, dtype=np.float64)
            hour_counts = np.zeros(24, dtype=np.int64)
            calculated_abs = 0.0
            calculated_sq = 0.0
            calculated_count = 0
            for begin in range(0, sample_count, 256):
                end = min(begin + 256, sample_count)
                error = loaded.model.predict(X[begin:end]).astype(np.float64) - y[begin:end].astype(np.float64)
                calculated_abs += float(np.abs(error).sum())
                calculated_sq += float(np.square(error).sum())
                calculated_count += int(error.size)
                sensor_abs += np.abs(error).sum(axis=(0, 1))
                sensor_sq += np.square(error).sum(axis=(0, 1))
                batch_hours = origin_hours[begin:end]
                for hour in np.unique(batch_hours):
                    mask = batch_hours == hour
                    hour_abs[hour] += float(np.abs(error[mask]).sum())
                    hour_sq[hour] += float(np.square(error[mask]).sum())
                    hour_counts[hour] += int(error[mask].size)

            calculated_mae = calculated_abs / calculated_count
            calculated_rmse = float(np.sqrt(calculated_sq / calculated_count))
            saved = loaded.metrics["test"]
            matches_saved = bool(
                np.isclose(calculated_mae, saved["mae"], rtol=1e-6, atol=1e-6)
                and np.isclose(calculated_rmse, saved["rmse"], rtol=1e-6, atol=1e-6)
            )
            models.append({
                "model_id": model_id,
                "name": spec["name"],
                "overall": {"mae": saved["mae"], "rmse": saved["rmse"]},
                "overall_source": "existing saved models/<experiment>/metrics.json",
                "mae_by_horizon": saved["mae_by_horizon"],
                "rmse_by_horizon": saved["rmse_by_horizon"],
                "by_hour": [
                    {"hour": hour, "label": f"{hour:02d}:00", "mae": float(hour_abs[hour] / hour_counts[hour]) if hour_counts[hour] else None,
                     "rmse": float(np.sqrt(hour_sq[hour] / hour_counts[hour])) if hour_counts[hour] else None,
                     "evaluated_values": int(hour_counts[hour])}
                    for hour in range(24)
                ],
                "by_sensor": [
                    {"sensor_id": sensor_id, "mae": float(sensor_abs[index] / (sample_count * HORIZON_STEPS)),
                     "rmse": float(np.sqrt(sensor_sq[index] / (sample_count * HORIZON_STEPS)))}
                    for index, sensor_id in enumerate(self.sensor_ids)
                ],
                "diagnostic_metrics_source": "new calculation from the existing chronological test.npz; no training, tuning, or model selection",
                "recomputed_overall_matches_saved": matches_saved,
                "recomputed_overall": {"mae": float(calculated_mae), "rmse": calculated_rmse},
            })

        persistence_abs = np.zeros(HORIZON_STEPS, dtype=np.float64)
        persistence_sq = np.zeros(HORIZON_STEPS, dtype=np.float64)
        persistence_sensor_abs = np.zeros(sensor_count, dtype=np.float64)
        persistence_sensor_sq = np.zeros(sensor_count, dtype=np.float64)
        persistence_hour_abs = np.zeros(24, dtype=np.float64)
        persistence_hour_sq = np.zeros(24, dtype=np.float64)
        persistence_hour_counts = np.zeros(24, dtype=np.int64)
        for begin in range(0, sample_count, 256):
            end = min(begin + 256, sample_count)
            prediction = np.repeat(X[begin:end, -1:, :], HORIZON_STEPS, axis=1)
            error = prediction.astype(np.float64) - y[begin:end].astype(np.float64)
            persistence_abs += np.abs(error).sum(axis=(0, 2))
            persistence_sq += np.square(error).sum(axis=(0, 2))
            persistence_sensor_abs += np.abs(error).sum(axis=(0, 1))
            persistence_sensor_sq += np.square(error).sum(axis=(0, 1))
            batch_hours = origin_hours[begin:end]
            for hour in np.unique(batch_hours):
                mask = batch_hours == hour
                persistence_hour_abs[hour] += float(np.abs(error[mask]).sum())
                persistence_hour_sq[hour] += float(np.square(error[mask]).sum())
                persistence_hour_counts[hour] += int(error[mask].size)

        total_values = sample_count * HORIZON_STEPS * sensor_count
        persistence = {
            "name": "Persistence baseline",
            "definition": "Repeat the final input reading at all 12 future steps.",
            "overall": {
                "mae": float(persistence_abs.sum() / total_values),
                "rmse": float(np.sqrt(persistence_sq.sum() / total_values)),
            },
            "mae_by_horizon": (persistence_abs / (sample_count * sensor_count)).tolist(),
            "rmse_by_horizon": np.sqrt(persistence_sq / (sample_count * sensor_count)).tolist(),
            "by_hour": [
                {"hour": hour, "label": f"{hour:02d}:00", "mae": float(persistence_hour_abs[hour] / persistence_hour_counts[hour]) if persistence_hour_counts[hour] else None,
                 "rmse": float(np.sqrt(persistence_hour_sq[hour] / persistence_hour_counts[hour])) if persistence_hour_counts[hour] else None,
                 "evaluated_values": int(persistence_hour_counts[hour])}
                for hour in range(24)
            ],
            "by_sensor": [
                {"sensor_id": sensor_id, "mae": float(persistence_sensor_abs[index] / (sample_count * HORIZON_STEPS)),
                 "rmse": float(np.sqrt(persistence_sensor_sq[index] / (sample_count * HORIZON_STEPS)))}
                for index, sensor_id in enumerate(self.sensor_ids)
            ],
            "source": "new calculation from the existing chronological held-out test.npz, same samples/targets as the four model variants",
        }
        self._evaluation = {
            "dataset": "METR-LA",
            "split": "chronological held-out test",
            "test_shape": {"X": list(X.shape), "y": list(y.shape)},
            "forecast_origin_hour_grouping": "hour of the first target timestamp for each rolling test window; windows and horizons follow the project's saved metric protocol",
            "evaluated_values": int(total_values),
            "models": models,
            "persistence_baseline": persistence,
            "note": "Saved four-model metrics are unchanged. Time-of-day, per-sensor, and persistence diagnostics are computed from the existing test split only; they were not used for model fitting, selection, or tuning.",
        }
        return self._evaluation


def create_app(
    dataset_path: str | Path = ROOT / "data/raw/metr-la.h5",
    model_root: str | Path = ROOT / "models",
    processed_root: str | Path = ROOT / "data/processed/metr_la",
) -> FastAPI:
    app = FastAPI(
        title="Federated Traffic Flow Prediction API",
        description="Local historical METR-LA research prototype; forecasts are evaluated on held-out test windows.",
        version="2.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.dataset_path = Path(dataset_path)
    app.state.model_root = Path(model_root)
    app.state.processed_root = Path(processed_root)

    @app.exception_handler(Exception)
    async def safe_unhandled_error(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "The request could not be completed"})

    def service(request: Request) -> TrafficService:
        if not hasattr(request.app.state, "traffic"):
            try:
                request.app.state.traffic = TrafficService(
                    request.app.state.dataset_path,
                    request.app.state.model_root,
                    request.app.state.processed_root,
                )
            except RuntimeError:
                request.app.state.initialization_error = "METR-LA data could not be initialized"
                raise HTTPException(status_code=503, detail=request.app.state.initialization_error) from None
        return request.app.state.traffic

    @app.get("/api/health")
    def health(request: Request) -> dict[str, str]:
        service(request)
        return {"status": "ok", "dataset": "METR-LA", "mode": "historical held-out research prototype"}

    @app.get("/api/sensors")
    def list_sensors(request: Request) -> dict[str, Any]:
        traffic = service(request)
        return {"dataset": "METR-LA", "count": len(traffic.sensor_ids), "sensor_ids": traffic.sensor_ids}

    @app.get("/api/models")
    def list_models(request: Request) -> dict[str, Any]:
        return service(request).model_list()

    @app.get("/api/forecast-options")
    def forecast_options(request: Request) -> dict[str, Any]:
        return service(request).forecast_options()

    @app.get("/api/evaluation")
    def evaluation(request: Request) -> dict[str, Any]:
        return service(request).evaluation()

    @app.get("/api/sensors/{sensor_id}/history")
    def recent_history(
        sensor_id: str,
        request: Request,
        limit: int = Query(default=36, ge=1, le=288),
        forecast_origin: datetime | None = None,
    ) -> dict[str, Any]:
        return service(request).history(sensor_id, limit, forecast_origin)

    @app.post("/api/predict")
    def predict(payload: PredictionRequest, request: Request) -> dict[str, Any]:
        return service(request).predict(payload.sensor_id, payload.model_id, payload.forecast_origin)

    return app


app = create_app()
