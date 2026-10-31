# Stage 6 — API and Dashboard Report

## Summary

Stage 6 adds a local FastAPI backend and a Streamlit dashboard. The app reads the existing METR-LA HDF5 dataset and loads the saved model artifacts from Stages 2–5. It does not train or change model weights. It is a research prototype over historical data, not a live traffic service or deployed system.

No frontend framework was already established in the project. Streamlit was selected because it is available in the project environment and provides a small Python-only dashboard. FastAPI was not installed initially, so it is declared with the other dependencies in [`requirements.txt`](/Users/shivrath/Documents/MSE806-Federated-Traffic/requirements.txt). For verification, it was installed to a temporary target after sandbox network restrictions blocked the first attempt; the installation did not change project files or model artifacts.

## API endpoints

The API module is [`src/api/app.py`](/Users/shivrath/Documents/MSE806-Federated-Traffic/src/api/app.py). It loads `data/raw/metr-la.h5` and uses the prepared, validated five-minute timestamp index. All APIs use JSON.

| Endpoint | Behavior |
|---|---|
| `GET /api/health` | Returns service status and identifies the historical METR-LA prototype. |
| `GET /api/sensors` | Returns the 207 available sensor IDs and count. |
| `GET /api/models` | Lists four model IDs, display names, availability, training mode, and recorded test metrics when artifacts exist. |
| `GET /api/forecast-options` | Returns the valid held-out origin range, selected default, 60-minute horizon, and four time presets. |
| `GET /api/evaluation` | Returns saved overall/per-horizon metrics for the four models and computes persistence, time-of-day, and per-sensor diagnostics from the existing test arrays. |
| `GET /api/sensors/{sensor_id}/history?limit=36&forecast_origin=...` | Returns the chosen sensor's recent readings before an origin. `limit` is validated from 1 to 288; the default is 36. If omitted, the origin defaults to the selected held-out window. |
| `POST /api/predict` | Accepts a sensor, model ID, and optional forecast origin. It returns 12 model inputs, recent context, 12 forecasts, aligned held-out actuals, selected-window errors, and a persistence comparison. |

Example request:

```json
{
  "sensor_id": "773869",
  "model_id": "fedavg_augmented",
  "forecast_origin": "2012-06-13T08:00:00"
}
```

The forecast origin is the timestamp of the first predicted reading; the final input is five minutes earlier. Requests must match the five-minute timestamp grid, fall within the held-out test period, and leave all 12 future actuals available. Invalid windows return a clear 422 response. The default held-out origin is `2012-06-13T08:00:00`: its preceding-hour, all-sensor mean (52.90197) is closest among valid held-out 08:00 windows to the training-period 08:00 median (52.71428). The selection rule uses only pre-origin inputs and a training-derived reference, never future targets.

The prediction response includes the selected model name and mode, dataset, exact 12 input readings, recent context, 12 forecasts, 12 held-out actuals at matching timestamps, selected-window errors, persistence forecasts/errors, and an explicit historical-data note. Values are in dataset speed units; no unit conversion is applied.

Unknown sensors return HTTP 404. Invalid model IDs or invalid request/query values return HTTP 422. Missing or invalid saved model artifacts return HTTP 503 with a short message. Unexpected server errors are returned without local paths or stack traces. CORS is limited to the local Streamlit origins used by the run instructions.

## Supported saved models

All four existing model directories were found and loaded without retraining:

| Model ID | Display name | Saved artifact |
|---|---|---|
| `centralized_baseline` | Centralized baseline | `models/centralized_baseline/shared_linear_forecaster.npz` |
| `centralized_augmented` | Centralized with noise augmentation | `models/centralized_augmented/shared_linear_forecaster.npz` |
| `fedavg` | FedAvg without augmentation | `models/federated_baseline/fedavg_global_model.npz` |
| `fedavg_augmented` | FedAvg with noise augmentation | `models/federated_augmented/fedavg_global_model.npz` |

The two model formats have different artifact filenames; the API maps both to the same shared linear forecaster class and validates the stored 12-step input/output configuration. Model parameters are sensor-count independent, so prediction can run for any one of the METR-LA sensor columns.

## Dashboard

The dashboard is [`app/dashboard.py`](/Users/shivrath/Documents/MSE806-Federated-Traffic/app/dashboard.py). It has a research overview, held-out date/time controls and morning/midday/evening/late-night presets. Its chart separates earlier history, the exact model-input hour, forecast, and actual held-out target, with a marked forecast origin. It compares selected-window MAE/RMSE with persistence and displays all four saved overall/per-horizon model metrics plus newly calculated time-of-day, sensor-specific, and persistence diagnostics. Expanders explain dataset, architecture, FedAvg, communication assumptions, augmentation, and evaluation. Loading/error states and historical/no-live/no-formal-privacy context remain visible.

The dashboard calls the shared Python `TrafficService` directly. It reads the HDF5 file and prepared metadata, loads saved model weights/configs/metrics, and uses the service's existing prediction and test-evaluation code. It does not issue HTTP requests or require a running FastAPI process. `app/requirements.txt` lists dependencies for the Streamlit entrypoint; because `TrafficService` is shared with the API module, FastAPI remains an import dependency, but no API server is started. The FastAPI app now initializes its service lazily on its first request.

## Run locally

From the project root, install the full project dependencies if you need both the API and training/test tools:

```bash
python3 -m pip install -r requirements.txt
```

Start the backend in one terminal:

```bash
python3 -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Start the dashboard in another terminal:

```bash
python3 -m streamlit run app/dashboard.py
```

For the standalone dashboard, install `app/requirements.txt` and start Streamlit from the project root. It needs no API address. The API's interactive schema is available at `/docs` only when you separately start the optional FastAPI service.

## Standalone dashboard / Community Cloud follow-up (2026-10-31)

The original deployed page was opened and visually confirmed to show the reported `127.0.0.1:8000` connection-refused error. The first rebuild after removing HTTP calls then exposed a repository-package import-path error on Community Cloud. The dashboard now inserts its entrypoint-derived repository root into `sys.path` before importing project modules. It has no local URL, `requests` import, or HTTP helper; it loads data and saved artifacts through `TrafficService` and calls `predict()` directly. Streamlit caches the service per app process, avoiding repeated HDF5/model/evaluation loads on widget reruns.

Local AppTest was run with no API server and with the process launched from `/tmp`, outside the repository: the page loaded all six charts and selected-window metrics, then updated its forecast after changing sensor and model. The full test suite passed after the lazy API initialization change. The pushed Git history includes the raw HDF5, processed train/validation/test arrays and client partitions, four model weights/config/metrics sets, comparison metrics, and related model artifacts. No substitute dataset/results were generated.

## Dashboard redesign verification (2026-10-31)

- Default origin: selected from valid held-out 08:00 windows by minimum distance between pre-origin all-sensor mean and training-period 08:00 median; selected `2012-06-13T08:00:00` without using its future target.
- The API now validates five-minute alignment, test-period membership, and full future target availability; it also returns same-window actuals and a last-value persistence baseline.
- Full project tests (final run): **22 passed in 5.66s**, one non-failing Starlette/httpx deprecation warning. `py_compile` completed successfully.
- Local API + Streamlit AppTest loaded the dashboard with six charts and selected-window metrics, then changed to sensor `767541` and the centralized noise-augmented model; both states rendered without Streamlit exceptions. The model selector stores readable names and maps them to saved API model records.

- `PYTHONPATH=/tmp/mse806-stage6-deps python3 -m pytest -q tests/test_api.py` — **4 passed in 5.70s**. Tests covered health/sensor/history responses, 404 invalid sensor, query validation, all four saved models, 12 future timestamps, invalid model selection, and a clean 503 when an artifact is missing.
- `PYTHONPATH=/tmp/mse806-stage6-deps python3 -m pytest -q` — **20 passed in 4.19s** across the project suite. One Starlette/httpx deprecation warning was emitted; no test failed.
- `PYTHONPATH=/tmp/mse806-stage6-deps python3 -m compileall -q src app tests` — completed successfully.
- `GET /api/models` using FastAPI TestClient — HTTP 200; all four model IDs reported available.
- Local Uvicorn server plus `streamlit.testing.v1.AppTest` — passed. The dashboard loaded the real sensor/model choices, returned a forecast for sensor `773869` with the centralized baseline, rendered its history/forecast chart without Streamlit exceptions, then changed to sensor `767541` with FedAvg plus augmentation and rendered the updated chart.

The first AppTest chart assertion queried the wrong Streamlit element key (`vega_lite_chart`); inspection showed this Streamlit version exposes it as `arrow_vega_lite_chart`. The check was corrected and the full sensor/model-to-chart flow passed. A sandbox attempt to bind loopback networking was blocked; after approval, the end-to-end check ran with loopback access enabled.

## Limitations

- Forecast dates are restricted to valid held-out test origins; the interface does not support arbitrary pre-test windows or live observations.
- The model receives one sensor channel at a time. There is no cross-sensor spatial context in this model.
- The dashboard presents model output but does not claim formal privacy guarantees. The federated model's privacy limitations remain as described in the federated report.
- This local prototype is not deployed and has no authentication, database, or production operational controls.
- FastAPI is a new project dependency. Install the requirements before launching the API in another environment.
