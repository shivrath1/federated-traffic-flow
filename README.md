# MSE806 Federated Traffic Flow Prediction

A one-machine research prototype that predicts the next hour of METR-LA traffic speed readings and compares a centralized shared-linear model with a four-client FedAvg simulation. The forecast uses 12 five-minute readings as input and returns 12 future five-minute readings. Clients are simulated sensor groups, not real organizations.

This application uses historical METR-LA data ending in June 2012. It is not a live traffic service. The project is a simplified research adaptation, not a reproduction of FedTPS. Noise augmentation is not a formal privacy mechanism; FedAvg alone does not guarantee privacy.

## Project contents

- `data/raw/metr-la.h5` — source HDF5 DataFrame.
- `data/processed/metr_la/` — prepared chronological train/validation/test arrays, metadata, and four client training partitions.
- `src/data/` — data loading, preparation, and train-only augmentation.
- `src/models/` — shared-linear model, centralized runs, and result consolidation.
- `src/federated/` — four-client FedAvg and augmented FedAvg runs.
- `src/api/` — FastAPI service for historical readings and forecasts.
- `app/dashboard.py` — Streamlit dashboard.
- `models/` — saved configurations, model weights, metrics, histories, and plots for all evaluated configurations.
- `reports/` — stage reports, evidence, final experiment comparison, and final review.

## Environment setup

Use Python 3.10 or later. From the project root, install the runtime and test dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The source METR-LA file must be present at `data/raw/metr-la.h5`. PeMS-Bay is not used in the project experiments.

## Reproduce the prepared data and experiments

Run the steps from the project root in order. Commands use their default project paths and will overwrite the corresponding processed files or experiment outputs when rerun.

1. Prepare and validate the METR-LA arrays and client partitions:

   ```bash
   python3 -m src.data.metr_la
   ```

2. Train and evaluate the centralized baseline:

   ```bash
   python3 -m src.models.centralized_baseline
   ```

3. Train and evaluate FedAvg without augmentation (depends on the centralized model artifact):

   ```bash
   python3 -m src.federated.fedavg
   ```

4. Train and evaluate the centralized model with training-window noise augmentation:

   ```bash
   python3 -m src.models.centralized_augmented
   ```

5. Generate client-local training-only noise copies and run augmented FedAvg (depends on the centralized baseline and FedAvg artifacts):

   ```bash
   python3 -m src.federated.fedavg_augmented
   ```

6. Rebuild the consolidated four-model metrics and plots from the saved results:

   ```bash
   python3 -m src.models.consolidate_experiments
   ```

The data pipeline uses chronological 70/15/15 splits. Model runs use train data for fitting, validation data for checkpoint selection, and test data only for final evaluation. Defaults, seeds, metrics, and details are recorded in each `models/<experiment>/config.json` and `metrics.json`. See the member reports and `reports/final-experiment-comparison.md` for methods and interpretation.

## Start the API and dashboard

Start the FastAPI service in the first terminal:

```bash
python3 -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Start the dashboard in a second terminal:

```bash
python3 -m streamlit run app/dashboard.py
```

Open the local Streamlit address printed in the second terminal. The dashboard uses `http://127.0.0.1:8000` by default. To point it elsewhere, set `TRAFFIC_API_URL` before launching Streamlit. FastAPI's interactive API schema is at `http://127.0.0.1:8000/docs`.

The dashboard lets a user choose a sensor, one of the four saved model configurations, and a date/time in the held-out test period. It defaults to a representative 08:00 test window selected by matching its preceding-hour, all-sensor mean to the training-period 08:00 median; selection does not use future targets. Morning peak, midday, evening peak, and late-night presets are available. The chart distinguishes earlier context, the exact 12 model-input readings, the next 12 predictions, and their held-out actual values. The API rejects timestamps without a full input and future target window.

The dashboard also shows selected-window MAE/RMSE beside a persistence baseline, saved full-test-set MAE/RMSE for all four models, and diagnostics by forecast horizon, start hour, and sensor. Overall saved metrics remain unchanged; the API computes persistence and diagnostic breakdowns from the existing test arrays. Values are shown in the dataset's speed units because the project did not independently verify a physical unit conversion. Every forecast is historical and retrospective; the interface is not a live traffic service.

## Tests and syntax checks

Run the project test suite and compilation check:

```bash
python3 -m pytest -q
python3 -m compileall -q src app tests
```

API tests use FastAPI's `TestClient`. The dashboard/API integration was checked using a local API and Streamlit `AppTest`; see `reports/report-evidence.md` for observed results. Local loopback access may be restricted in some managed or sandboxed environments.

## Further documentation

- [Project instructions](AGENTS.md)
- [Implementation stages](PLANS.md)
- [Task tracker](reports/task-tracker.md)
- [Data preparation report](reports/member-1-data-report.md)
- [Model report](reports/member-2-model-report.md)
- [Federated learning report](reports/member-3-federated-report.md)
- [Synthetic augmentation report](reports/synthetic-augmentation-report.md)
- [Four-model comparison](reports/final-experiment-comparison.md)
- [API and dashboard report](reports/api-dashboard-report.md)
- [Final integration and reproducibility review](reports/final-review.md)
- [Assessment brief](MSE806-Assessment_Brief.pdf)
