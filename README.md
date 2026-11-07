# Federated Traffic Flow Prediction

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

## Start the dashboard

The Streamlit app reads the tracked METR-LA data, saved models, and metrics directly. It does not need the FastAPI server. Install its entrypoint-specific dependencies and run it from the project root:

```bash
python3 -m pip install -r app/requirements.txt
python3 -m streamlit run app/dashboard.py
```

To run the optional FastAPI service separately for API development or testing, install the full project requirements and start it with `python3 -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000`. Its interactive schema is at `http://127.0.0.1:8000/docs`; this service is not a dashboard dependency.

The dashboard lets a user choose a sensor, one of the four saved model configurations, and a date/time in the held-out test period. It defaults to a representative 08:00 test window selected by matching its preceding-hour, all-sensor mean to the training-period 08:00 median; selection does not use future targets. Morning peak, midday, evening peak, and late-night presets are available. The chart distinguishes earlier context, the exact 12 model-input readings, the next 12 predictions, and their held-out actual values. Forecast validation is performed directly by the shared Python service.

The dashboard also shows selected-window MAE/RMSE beside a persistence baseline, saved full-test-set MAE/RMSE for all four models, and diagnostics by forecast horizon, start hour, and sensor. Overall saved metrics remain unchanged; persistence and diagnostic breakdowns are computed from the existing test arrays. Values are shown in the dataset's speed units because the project did not independently verify a physical unit conversion. Every forecast is historical and retrospective; the interface is not a live traffic service.

For Streamlit Community Cloud, `app/requirements.txt` declares the dashboard dependencies next to its entrypoint. The app resolves data and artifact locations from the repository root, so it does not need a local API URL, secrets, or a separate backend deployment. Keep `data/raw/metr-la.h5`, `data/processed/metr_la/`, and the required saved model folders in the repository checkout.

## Tests and syntax checks

Run the project test suite and compilation check:

```bash
python3 -m pytest -q
python3 -m compileall -q src app tests
```

API tests use FastAPI's `TestClient`. The dashboard was also checked with Streamlit `AppTest` loading the tracked data and artifacts directly, with no API server running; see `reports/report-evidence.md` for observed results.

## Secure FedAvg transport experiment

`src/federated/secure_fedavg.py` runs the existing four-client FedAvg settings with AES-256-GCM protecting each serialized client model update before the simulated client/coordinator boundary. The coordinator checks message version, client identifier, communication round, session identity, nonce, and authentication tag, then decrypts every update before calling the existing FedAvg aggregator. It rejects the complete round before aggregation if any update fails verification. The original FedAvg runner and its saved baseline artifacts are not changed.

AES-GCM provides message confidentiality and authentication/integrity when used with a secret key. Here it authenticates possession of a client-specific shared secret; it does not establish a public-key identity or provide non-repudiation. Set a distinct, cryptographically random 32-byte key for each client as a 64-character hexadecimal environment variable. The example below generates fresh keys in the current shell; keep them out of source control and do not reuse a value across clients:

```bash
export FEDERATED_CLIENT_KEY_CLIENT_1="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export FEDERATED_CLIENT_KEY_CLIENT_2="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export FEDERATED_CLIENT_KEY_CLIENT_3="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export FEDERATED_CLIENT_KEY_CLIENT_4="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

Install dependencies (including `cryptography`) with `python3 -m pip install -r requirements.txt`, then run the secure comparison:

```bash
python3 -m src.federated.secure_fedavg
```

The command reads the existing prepared METR-LA partitions and baseline configuration and writes a separate experiment under `models/federated_secure_baseline/`. It does not overwrite `models/federated_baseline/`. A persistent SQLite replay ledger in that output directory issues unique 96-bit nonces and retires the previous session on a new run or process restart. Keep the ledger with the experiment output; deleting it discards replay history. Run focused and full tests with:

```bash
python3 -m pytest -q tests/test_federated_security.py tests/test_federated.py
python3 -m pytest -q
```

This remains a single-process simulation, not a network deployment. The experiment does not provide secure aggregation, digital signatures, differential privacy, malicious-client defenses, production key management, or real multi-machine networking. AES-GCM protects each update in this modeled boundary; the coordinator decrypts updates and can inspect them before aggregation. Anyone who obtains a client's shared key can impersonate possession of that key. The persistent replay ledger is local coordinator state and must itself be protected and backed up in a real deployment. Encryption overhead measurements are local CPU and serialized-payload measurements, not network latency or throughput.

## Further documentation
- [Data preparation report](reports/member-1-data-report.md)
- [Model report](reports/member-2-model-report.md)
- [Federated learning report](reports/member-3-federated-report.md)
- [Synthetic augmentation report](reports/synthetic-augmentation-report.md)
- [Four-model comparison](reports/final-experiment-comparison.md)
- [API and dashboard report](reports/api-dashboard-report.md)
- [Final integration and reproducibility review](reports/final-review.md)
