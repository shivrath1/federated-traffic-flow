# Report Evidence Log

Use this file as a factual record for the final report and demo. Add dated entries as work occurs. Keep planned work separate from completed work. Enter only observed test and experiment results; do not estimate or invent metrics.

## Actual implementation work completed

| Date | Work completed | Member/source of confirmation | Evidence or notes |
|---|---|---|---|
| 2026-10-31 | Inspected the existing METR-LA HDF5 file with Python, pandas, and h5py. Confirmed `/df`, 34,272 × 207, five-minute timestamp spacing from 2012-03-01 00:00 to 2012-06-27 23:55, and zero missing values in the stored frame. | Project setup inspection | This validates the current file only; preprocessing and experiments remain to be done. |
| 2026-10-31 | Extracted the selected paper and assessment brief using the bundled Poppler `pdftotext`; reviewed the project plan DOCX via its OOXML text. | Project setup inspection | Paper describes FedTPS with federated diffusion synthetic generation and advanced spatio-temporal modeling. Assessment requirements recorded in `AGENTS.md`. |
| 2026-10-31 | Implemented and ran the Stage 1 reusable METR-LA pipeline. Wrote chronological central splits, metadata, and four sensor-group client training files. | Member 1 stage implementation | 70/15/15 row split; 12-step lookback and horizon; no test targets in client training files. Details and shapes in `reports/member-1-data-report.md`. No model training. |
| 2026-10-31 | Implemented and evaluated a centralized shared per-sensor linear forecaster using only the prepared METR-LA train split; selected its checkpoint using validation MSE, then evaluated once on the test split. | Member 2 stage implementation | Best checkpoint epoch 29 of 30. Test MAE 5.698667 and RMSE 11.969190, element-weighted over 12,745,404 predictions. No federated training or synthetic augmentation. |
| 2026-10-31 | Implemented and ran a four-client FedAvg simulation reusing the shared linear forecaster. Clients trained on their own sensor-group training files; only model arrays were aggregated. Selected the global checkpoint using central validation MSE, then evaluated it on test. | Member 3 stage implementation | 30 communication rounds, one local epoch per client per round; best round 30. Test MAE 5.522869 and RMSE 11.973663. Calculated model-array payload: 149,760 bytes total, excluding protocol overhead. No synthetic augmentation or API/dashboard work. |
| 2026-10-31 | Implemented seeded client-local Gaussian input jitter and generated one synthetic copy per client training window. Ran augmented FedAvg, selected by validation MSE, then evaluated on the same test split and compared all three requested models. | Member 2/3 Stage 4 implementation | Noise standard deviation 5% of client sensor training-input standard deviation; targets unchanged; client seeds 42–45. Test MAE 5.665383 and RMSE 11.883041. Versus FedAvg without augmentation, MAE worsened 2.5804% and RMSE improved 0.7568%. No API/dashboard work. |
| 2026-10-31 | Implemented and ran centralized training with the existing train-only Gaussian input jitter. Selected epoch 30 using validation MSE, then evaluated on the same held-out test split. Consolidated all four model configurations and per-horizon results. | Member 2 Stage 5 implementation | Test MAE 5.790541, RMSE 11.871505. Compared with central baseline, MAE increased 1.612% and RMSE decreased 0.816%; augmentation does not consistently improve both metrics. The four-model JSON and plots are under `models/experiment_comparison/`. |
| 2026-10-31 | Implemented FastAPI endpoints backed by the existing HDF5 dataset and four saved checkpoints, plus a Streamlit sensor/model selector and historical/forecast chart. Ran API tests and a live local API + Streamlit AppTest integration flow. | Member 3/4 Stage 6 implementation | All four saved model choices were available. AppTest fetched actual sensors and forecasts, rendered one history/forecast chart, then switched sensor and model successfully. No model was retrained. |
| 2026-10-31 | Completed Stage 7 integration/reproducibility review. Rebuilt data into a temporary directory and compared all split and client arrays; checked model/config/metric files against the API and consolidated comparison; read the five-page assessment brief; added README run instructions; corrected the stale Member 3 status and added the missing central augmented metric pair to the Stage 5 follow-up. | Member 4 Stage 7 review | Regenerated prepared arrays matched exactly. All four saved MAE/RMSE pairs matched the API and comparison; the review did not change metrics or models. Full suite: 20 passed in 3.93s; compile checks succeeded. Final academic report/presentation/package remain incomplete. |

## Files created or changed

| Date | File | Change |
|---|---|---|
| 2026-10-31 | `AGENTS.md` | Added project scope, dataset, responsibilities, constraints, testing/reporting rules, and evidence-update instructions. |
| 2026-10-31 | `PLANS.md` | Added approved stages, dependencies, parallel work, and one-week schedule. |
| 2026-10-31 | `reports/task-tracker.md` | Added assigned tasks, status, estimated effort, dependencies, blockers, and deliverables. Actual effort is unreported. |
| 2026-10-31 | `reports/report-evidence.md` | Created this evidence log and recorded setup inspection/document-reading evidence. |
| 2026-10-31 | `src/data/metr_la.py` | Added validated loader, chronological window generation, client partition writer, metadata, CLI, and central/client loading functions. |
| 2026-10-31 | `tests/test_data_pipeline.py` | Added tests for data validation, split/window contract, client partitioning, serialization, and leakage boundaries. |
| 2026-10-31 | `data/processed/metr_la/` | Generated central split NPZs, four client training NPZs, and metadata JSON from the existing METR-LA file. |
| 2026-10-31 | `reports/member-1-data-report.md` | Drafted Member 1 data preparation report and handoff instructions. |
| 2026-10-31 | `src/models/centralized_baseline.py` | Added sensor-count-independent linear forecaster, NumPy Adam training, validation checkpoint selection, delayed test evaluation, artifact loading, and CLI. |
| 2026-10-31 | `tests/test_centralized_baseline.py` | Added model shape, persistence initialization, training, and saved-model loading tests. |
| 2026-10-31 | `models/centralized_baseline/` | Saved trained weights, configuration, test metrics, training history, learning curve, and test error-by-horizon plot. |
| 2026-10-31 | `reports/member-2-model-report.md` | Drafted model design, data contract, training/evaluation method, actual results, and handoff notes. |
| 2026-10-31 | `src/models/centralized_baseline.py` | Added a local-client training utility that starts from broadcast weights and reuses the centralized model's optimizer/update implementation. |
| 2026-10-31 | `src/federated/fedavg.py` | Added weighted FedAvg, simulated coordinator/client round loop, validation-based global checkpoint selection, delayed test evaluation, communication accounting, artifact saving, plotting, model loader, and CLI. |
| 2026-10-31 | `tests/test_federated.py` | Added FedAvg arithmetic/validation tests, local training tests for 51/52 sensors, and global model loading test. |
| 2026-10-31 | `models/federated_baseline/` | Saved selected global weights, config, metrics, round history, and training/comparison plots. |
| 2026-10-31 | `reports/member-3-federated-report.md` | Drafted federated method, communication/privacy notes, actual metrics, centralized comparison, and limitations. |
| 2026-10-31 | `src/data/noise_augmentation.py` | Added deterministic client-local Gaussian input jitter from per-sensor training-input standard deviation, unchanged targets, separate synthetic files, and a train-only manifest. |
| 2026-10-31 | `src/federated/fedavg.py` | Extended coordinator to locally combine separately stored synthetic client windows for an augmented run and record original/synthetic counts. |
| 2026-10-31 | `src/federated/fedavg_augmented.py` | Added reproducible augmentation + FedAvg runner and three-model test comparison plot. |
| 2026-10-31 | `tests/test_augmentation.py` | Added seeded behavior, target preservation, no-mutation, model-compatibility, and train-only file-access tests. |
| 2026-10-31 | `models/federated_augmented/` | Saved augmented FedAvg model/config/metrics/history/plots plus separate synthetic client training files and manifest. |
| 2026-10-31 | `reports/member-2-model-report.md` | Added Stage 4 method and actual FedAvg-with-augmentation results. |
| 2026-10-31 | `reports/member-3-federated-report.md` | Added augmented FedAvg setup, result, timing, and limitations. |
| 2026-10-31 | `reports/synthetic-augmentation-report.md` | Created the Stage 4 augmentation experiment report. |
| 2026-10-31 | `src/models/centralized_baseline.py` | Added minibatch training over separate original and synthetic arrays, avoiding a full augmented-dataset concatenation. |
| 2026-10-31 | `src/data/noise_augmentation.py` | Added measured generation duration to reusable augmentation details. |
| 2026-10-31 | `src/models/centralized_augmented.py` | Added train-only centralized noise augmentation experiment, validation selection, delayed test evaluation, metrics/config/history/manifest and plots. |
| 2026-10-31 | `src/models/consolidate_experiments.py` | Added four-run common-test comparison JSON and overall/per-horizon plots from saved results. |
| 2026-10-31 | `tests/test_centralized_augmented.py` | Added multipart training compatibility and workflow-order/data-scope tests. |
| 2026-10-31 | `models/centralized_augmented/` | Saved centralized augmented weights, config, metrics, manifest, history, and plots. |
| 2026-10-31 | `models/experiment_comparison/` | Saved four-model aggregate metrics and overall/per-horizon comparison plots. |
| 2026-10-31 | `reports/member-2-model-report.md`, `reports/synthetic-augmentation-report.md` | Added centralized augmented experiment and consolidated comparison results. |
| 2026-10-31 | `reports/final-experiment-comparison.md` | Created a consolidated four-configuration results and interpretation report. |
| 2026-10-31 | `reports/synthetic-augmentation-report.md` | Added the Stage 5 centralized augmented metrics explicitly to the four-model follow-up. |
| 2026-10-31 | `src/api/app.py`, `src/api/__init__.py` | Added FastAPI service, sensor/history/model endpoints, 12-step forecast endpoint, request validation, CORS for local dashboard, and sanitized error responses. |
| 2026-10-31 | `app/dashboard.py` | Added Streamlit dashboard with sensor/model controls, loading/error states, historical/forecast chart, recorded metrics, and historical-data notice. |
| 2026-10-31 | `requirements.txt` | Declared project runtime and API/dashboard test dependencies. |
| 2026-10-31 | `tests/test_api.py` | Added API tests for health/sensor/history, invalid sensor/input, all four model choices, forecast horizon/timestamps, and missing-artifact error handling. |
| 2026-10-31 | `reports/api-dashboard-report.md` | Documented API contract, dashboard, setup, actual verification, and limitations. |
| 2026-10-31 | `README.md` | Added root-level environment, data preparation, training/augmentation, API/dashboard, and test run instructions. |
| 2026-10-31 | `reports/member-3-federated-report.md` | Corrected stale opening status that said API/dashboard work had not started; linked the Stage 6 report. |
| 2026-10-31 | `reports/task-tracker.md` | Marked README and Stage 7 review complete; separated remaining final academic report, proposal confirmation, presentation, Turnitin, and ZIP tasks. Actual effort remains unreported. |
| 2026-10-31 | `reports/final-review.md` | Recorded Stage 7 implementation/artifact audit, assessment requirements, test results, documentation corrections, limitations, and readiness assessment. |
| 2026-10-31 | `src/api/app.py` | Added held-out forecast origin validation, a training-reference-matched default morning window, actual future test targets for evaluation, persistence baseline, and test-set diagnostic breakdowns. Saved model metrics and weights were not changed. |
| 2026-10-31 | `app/dashboard.py` | Redesigned the interface with a research overview, date/time controls and four presets, distinct history/model-input/actual/prediction series, selected-window and persistence errors, overall/horizon/time-of-day/sensor comparisons, and technical detail expanders. |
| 2026-10-31 | `README.md`, `reports/api-dashboard-report.md`, `reports/task-tracker.md` | Updated run instructions, API/dashboard contracts, forecast origin rationale, limitations, and task deliverables. |

## Commands used

- `python3 -m src.data.metr_la` — completed successfully; output: raw shape `[34272, 207]`; window samples: train `23967`, validation `5129`, test `5131`; clients `52`, `52`, `52`, `51` sensors.
- `python3 -m pytest -q` — final run: `3 passed in 1.99s`.
- `python3 -m compileall -q src/data tests` — completed successfully (no syntax errors).
- `python3 -m src.models.centralized_baseline` — completed successfully; 30 epochs run, validation-selected best epoch 29, training time 28.5255 seconds; test MAE 5.698667, RMSE 11.969190.
- `python3 -m pytest -q` — after Stage 2 additions: `7 passed in 2.85s`.
- `python3 -m compileall -q src/models tests` — completed successfully.
- Artifact readback using `load_saved_model` — loaded weights `(12, 12)`, bias `(12,)`, best epoch 29, and saved test metrics.
- `python3 -m src.federated.fedavg` — completed successfully; four clients, 30 rounds, selected round 30; test MAE 5.522869, RMSE 11.973663; coordinator-round wall time 42.8984 seconds; local-fit time sum 16.9813 seconds; parameter payload 74,880 bytes down and 74,880 bytes up.
- `python3 -m pytest -q` — after Stage 3 additions: `11 passed in 3.36s`.
- `python3 -m compileall -q src tests` — completed successfully.
- `python3 -m pip install --target /tmp/mse806-stage6-deps 'fastapi>=0.110,<1.0'` — sandbox network attempt failed because DNS/network access was restricted; the same installation was approved and completed into a temporary directory for verification. Project dependencies are declared in `requirements.txt`; no package was installed into the project tree.
- `PYTHONPATH=/tmp/mse806-stage6-deps python3 -m pytest -q tests/test_api.py` — `4 passed in 5.70s`; one Starlette/httpx deprecation warning.
- Local Uvicorn API + `streamlit.testing.v1.AppTest` integration script — **PASS**: sensor/model choices loaded from API, first sensor/model produced a rendered chart without Streamlit exceptions, and changing to sensor `767541` plus FedAvg-with-augmentation updated the rendered forecast/chart.
- `PYTHONPATH=/tmp/mse806-stage6-deps python3 -m pytest -q` — Stage 6 full suite: `20 passed in 4.19s`; one Starlette/httpx deprecation warning.
- `PYTHONPATH=/tmp/mse806-stage6-deps python3 -m compileall -q src app tests` — completed successfully.
- `python3 -m py_compile src/api/app.py app/dashboard.py tests/test_api.py` — completed successfully after dashboard redesign; the first syntax check found a misindented forecast request block, which was corrected.
- `PYTHONPATH=/tmp/mse806-stage6-deps python3 -m pytest -q` — **22 passed in 5.66s**, one non-failing Starlette/httpx deprecation warning (final run after dashboard changes).
- Default test window selection audit — chose `2012-06-13T08:00:00`. Its preceding-hour all-sensor mean is 52.901969; it is the valid held-out 08:00 input nearest the training-period 08:00 median (52.714281). Selection used no future target values.
- Local API + Streamlit `AppTest` against saved project artifacts — **PASS**: no Streamlit exceptions; rendered six charts and selected-window metrics; changing selection to sensor `767541` and “Centralized with noise augmentation” updated the forecast display successfully.
- The first interaction check exposed a testing incompatibility when Streamlit's model selectbox stored full model dictionaries; the selector now stores readable model names and resolves them to API records. A repeat interaction passed. No model weights or saved evaluation results changed.
- `GET /api/models` via FastAPI TestClient — HTTP 200; all four configured model artifacts reported available.
- `python3 -m src.data.metr_la --output /tmp/mse806-stage7-reprepared` — completed; generated 34,272 × 207 raw contract, split samples 23,967/5,129/5,131, and client sizes 52/52/52/51. Element-wise comparisons confirmed exact equality against existing processed split arrays and four client train arrays.
- One-off Python artifact audit using FastAPI TestClient and saved configs/weights/metrics — confirmed all four weight/bias shapes `(12,12)`/`(12,)`, all API models available, exact MAE/RMSE agreement between model `metrics.json`, API `/api/models`, and `models/experiment_comparison/metrics.json`; all four pairs were present in the final comparison report. The first audit attempt used internal model names instead of comparison display labels and stopped at a script lookup; after correcting that audit mapping, all four checks passed. No project metric or model file was modified.
- Bundled PDF review: `/Users/shivrath/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfinfo MSE806-Assessment_Brief.pdf` reported 5 pages; the bundled-runtime Python `pypdf` read all pages and the Task A/Task 2 requirements and rubric were reviewed.
- `PYTHONPATH=/tmp/mse806-stage6-deps python3 -m pytest -q` — Stage 7 full suite: `20 passed in 3.93s`; one Starlette/httpx deprecation warning.
- `PYTHONPATH=/tmp/mse806-stage6-deps python3 -m compileall -q src app tests` — completed successfully.
- `python3 -m py_compile src/api/app.py app/dashboard.py tests/test_api.py` — completed successfully.
- Artifact readback using `load_federated_model` — loaded global weights `(12, 12)`, bias `(12,)`; verified 30 rounds, client sensor counts, saved metrics, and communication payload.
- `python3 -m src.federated.fedavg_augmented` — completed successfully; generated 95,868 synthetic multichannel windows (4,961,169 sensor-window examples), trained 30 rounds, best round 30; test MAE 5.665383, RMSE 11.883041; noise generation 17.6795 seconds, coordinator round-loop 123.6770 seconds, summed local-fit time 36.6516 seconds.
- Stage 4 communication payload remained 74,880 bytes down + 74,880 bytes up (149,760 total float32 model-array bytes; excludes protocol overhead).
- `python3 -m pytest -q` — final full suite after Stage 4: `14 passed in 3.34s`.
- `python3 -m compileall -q src tests` — completed successfully.
- Augmented artifact readback — loaded global weights `(12, 12)` and bias `(12,)`; verified 30 rounds, per-client counts, and manifest flags `validation_used: false`, `test_used: false`.
- `python3 -m src.models.centralized_augmented` — completed successfully; one synthetic copy per central training window, selected epoch 30 of 30, test MAE 5.790541 and RMSE 11.871505. Noise generation took 2.9680 s; training/validation took 49.0059 s.
- `python3 -m src.models.consolidate_experiments` — completed successfully; loaded all four saved experiment metrics, confirmed shared test shape `(5131, 12, 207)` and 12-step horizon, and wrote comparison JSON and plots.
- `python3 -m pytest -q tests/test_centralized_augmented.py` — `2 passed in 1.57s`.
- `python3 -m pytest -q` — Stage 5 full suite: `16 passed in 3.22s`.
- `python3 -m compileall -q src tests` — completed successfully.
- `python3` readback using `load_prepared_data` and `load_client_training_data` — central and client files loaded; all inspected arrays had expected shapes/dtypes and finite values.
- Python with `pandas` and `h5py` to inspect `data/raw/metr-la.h5`: shape `(34272, 207)`, 207 `float64` columns, no nulls, no duplicate timestamps, increasing index, all 34,271 intervals equal five minutes.
- Bundled Poppler `pdftotext -layout Research_Paper.pdf /tmp/research-paper.txt`.
- Bundled Poppler `pdftotext -layout MSE806-Assessment_Brief.pdf /tmp/assessment-brief.txt`.
- Python `zipfile` and `xml.etree.ElementTree` to extract readable paragraph/table text from the project plan DOCX because `python-docx` is unavailable in the default Python environment.

## Tests run and actual results

Stage 1 tests: `3 passed`. Stage 2 full suite: `7 passed`. Stage 3 full suite: `11 passed`. Stage 4 full suite: `14 passed in 3.34s`. Stage 5 full suite: `16 passed in 3.22s`. Stage 6 full suite: `20 passed in 4.19s`. Stage 7 full suite: `20 passed in 3.93s`; compileall and py_compile completed successfully. Stage 6/7 API tests reported one Starlette/httpx deprecation warning; all tests passed. The first Stage 1 test run had `1 failed, 2 passed` because the validation-window test expected 21 examples where a 24-row split with a 3-step horizon correctly yields 22; the assertion was corrected. All four saved model configurations are exposed by the API and were exercised. A live local API + Streamlit AppTest run fetched choices, rendered the historical/forecast chart, and updated after sensor/model changes. Stage 7 regenerated processed arrays exactly and confirmed artifact/API/comparison metric consistency. The initial sandbox attempt to install FastAPI and bind loopback networking was blocked; FastAPI was installed into a temporary verification target after approval, and loopback integration ran after approval. Matplotlib emitted non-fatal warnings about its default cache/fontconfig locations; saved plots were inspected. PyTorch was unavailable, so the shared model and optimizer use NumPy.

## Experiment metrics

Centralized, unaugmented FedAvg, and augmented FedAvg results (same real METR-LA test split):

| Experiment | Dataset/split | Augmentation | Training setup | MAE | RMSE | Training time | Run/evidence |
|---|---|---|---|---:|---:|---:|---|
| Centralized shared linear forecaster | METR-LA, chronological 70/15/15; test has 5,131 windows | No | NumPy Adam; best validation checkpoint epoch 29 | 5.698667 | 11.969190 | 28.5255 seconds training | `models/centralized_baseline/metrics.json` |
| Four-client FedAvg shared linear forecaster | Same METR-LA test split; 5,131 windows | No | 30 rounds × one local epoch, validation-selected round 30 | 5.522869 | 11.973663 | 42.8984 seconds coordinator wall time; 16.9813 seconds summed client-fit time | `models/federated_baseline/metrics.json` |
| Four-client FedAvg with noise augmentation | Same METR-LA test split; 5,131 windows | One 5%-std jittered input copy per client training window; target unchanged | 30 rounds × one local epoch, validation-selected round 30 | 5.665383 | 11.883041 | 123.6770 s coordinator wall; 36.6516 s local-fit sum; 17.6795 s generation | `models/federated_augmented/metrics.json` |
| Centralized shared linear forecaster with noise augmentation | METR-LA chronological test split; 5,131 windows | One 5%-std Gaussian-jittered input copy per central training window; targets retained | 30 maximum epochs, Adam state persists; validation-MSE checkpoint epoch 30 | 5.790541 | 11.871505 | 49.0059 s train/validation; 2.9680 s augmentation generation | `models/centralized_augmented/metrics.json` |

## Difficulties and resolutions

- Initial PDF extraction attempt could not find a system `pdftotext`; located and used the bundled Poppler executable successfully. No PDF-reading blocker remains.
- The default Python does not include `python-docx`; extracted the DOCX text directly from its XML with the standard library instead.

## Incomplete work

- Stage 1 data preparation, centralized baseline implementation/evaluation, and federated baseline implementation/evaluation are complete.
- Stages 1–6 data preparation, all four model experiments, API, dashboard, and local integration checks are complete. Stage 7 final report, presentation slides, recording, and ZIP remain incomplete; these were intentionally not started in this stage.
- Stage 1 chose and documented a 12-reading lookback, 12-reading forecast horizon, chronological 70/15/15 row split, and balanced original-column-order sensor partitions. These are practical defaults and can be revisited by the project team before Stage 2 if needed.
