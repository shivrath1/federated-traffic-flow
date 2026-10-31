# Stage 7 — Final Integration Review

**Review date:** 2026-10-31  
**Scope:** Implementation, artifacts, reproducibility, run instructions, report consistency, and submission preparation status. The final academic report and presentation were not created in this review.

## Verified project components

- **Data:** METR-LA only, source `data/raw/metr-la.h5`; validated HDF5/DataFrame loader and chronological data preparation at `src/data/metr_la.py`. The prepared contract is 12 input readings and 12 forecast readings at five-minute intervals, with chronological 70/15/15 splits and four balanced sensor partitions.
- **Centralized models:** shared per-sensor linear forecaster and training runner in `src/models/centralized_baseline.py`; centralized training-window noise-augmentation runner in `src/models/centralized_augmented.py`.
- **Federated models:** four simulated sensor-group clients, local training, and weighted FedAvg in `src/federated/fedavg.py`; client-local noise augmentation runner in `src/federated/fedavg_augmented.py`.
- **Augmentation and comparison:** reusable train-only noise jitter in `src/data/noise_augmentation.py`; consolidated actual metric reader/plots in `src/models/consolidate_experiments.py`.
- **Application:** FastAPI in `src/api/app.py`, Streamlit dashboard in `app/dashboard.py`. The API reads the latest historical METR-LA observations and the existing model artifacts; it does not retrain them.
- **Documentation:** project instructions, stage reports, comparison report, evidence log, task tracker, API/dashboard report, and the new project-root `README.md`.

## Data and artifact reproducibility checks

The data-preparation command was run into a temporary directory so that the reviewed project outputs were not overwritten:

```bash
python3 -m src.data.metr_la --output /tmp/mse806-stage7-reprepared
```

It completed with raw dimensions `(34272, 207)`, train/validation/test sample counts `23967/5129/5131`, and client sensor counts `52/52/52/51`. Element-by-element comparisons confirmed that the regenerated `X` and `y` arrays for all three splits and all four client training partitions exactly match the committed/prepared outputs under `data/processed/metr_la`.

All four saved configurations contain a 12-step lookback and 12-step horizon. Each expected weight file and config/metrics JSON exists; all weight matrices have shape `(12, 12)` and biases shape `(12,)`. The API model listing reported all four configurations available. For each configuration, the API's recorded test MAE/RMSE matched its saved `metrics.json` exactly, and the consolidated comparison matched the same saved values exactly. The final comparison report contains all four saved metric pairs.

| Configuration | Test MAE | Test RMSE |
|---|---:|---:|
| Centralized baseline | 5.698667 | 11.969190 |
| Centralized with noise augmentation | 5.790541 | 11.871505 |
| FedAvg without augmentation | 5.522869 | 11.973663 |
| FedAvg with noise augmentation | 5.665383 | 11.883041 |

The recorded interpretation remains accurate: augmentation raises MAE and lowers RMSE relative to each corresponding unaugmented run, so it does not consistently improve both metrics. No metric or training artifact was changed during the review.

## Run instructions and integration review

Before this review there was no root README. [`README.md`](../README.md) now documents environment setup, dependency installation, preprocessing, experiment order, API startup, dashboard startup, and test/compilation commands. The commands were checked against the current CLI defaults and module entry points. Dependencies are declared in `requirements.txt`.

Stage 6 API tests exercise sensor/history behavior, invalid sensors/inputs, each saved model, 12-step future timestamps, and a clean unavailable-artifact response. The recorded Stage 6 Streamlit AppTest used a live local API, rendered the history-plus-forecast chart, and confirmed that changing both sensor and model changed the rendered view. The API validates model IDs and sensor IDs and does not expose local paths or tracebacks. Dashboard and API copy describe the historical dataset; neither claims live prediction or formal privacy guarantees.

## Tests and compilation

The complete current test command was:

```bash
PYTHONPATH=/tmp/mse806-stage6-deps python3 -m pytest -q
```

**Actual result:** `20 passed in 3.93s`, with one Starlette/httpx TestClient deprecation warning and no test failures. FastAPI was installed in a temporary verification target because it was absent from the starting Python environment; `requirements.txt` includes FastAPI and the local API runtime dependencies.

Compilation checks:

```bash
PYTHONPATH=/tmp/mse806-stage6-deps python3 -m compileall -q src app tests
python3 -m py_compile src/api/app.py app/dashboard.py tests/test_api.py
```

Both completed successfully.

The assessment brief was read directly (five pages). It requires a proposal covering project title/objectives/innovation/ethics; an implementation report covering methodology, results, security/efficiency assessment, discussion and reflection; a 2,500–3,000-word report in APA style; student names and IDs; Turnitin report with the stated 20%-or-lower threshold; a single ZIP containing report/code/demo materials and the Turnitin report; and slides plus a recording whose combined length is at most 15 minutes. The assessment also grades software communication-system development and ITS/security analysis. See [`MSE806-Assessment_Brief.pdf`](../MSE806-Assessment_Brief.pdf).

## Documentation review

- No unsupported claim was found that the implementation reproduces FedTPS. Reports consistently identify the project as a simplified adaptation using Gaussian input jitter and a shared linear forecaster.
- Privacy text states that FedAvg is not a privacy guarantee and that the simulation has no formal differential privacy or secure aggregation.
- API and dashboard wording describes historical METR-LA observations, not live traffic.
- The four clients are described as sensor groups simulated on one machine, not real organizations.
- Augmentation discussion reports the actual mixed MAE/RMSE outcomes rather than claiming consistent improvement.
- Three documentation gaps were corrected: the absent root README was added, the opening status sentence in the Member 3 report was updated because it incorrectly said API/dashboard work had not started, and the Stage 5 augmentation follow-up now includes the centralized augmented metric pair explicitly. The tracker distinguishes completed review/run-instruction work from the still-unstarted final academic deliverables.
- Earlier member reports retain stage-specific statements such as “no federated training” or “the centralized augmented run had not yet been done”; those describe what was true at their earlier stage and are followed by later-stage reports with completed results.
- `requirements.txt` uses version ranges rather than a complete lockfile. The full requirements set was not installed into a clean environment during this review, and exact package-version reproduction is not fixed.

## Remaining limitations and incomplete items

- This is a small, single-seed experiment using a shared temporal linear map; it does not model spatial relationships between sensors. Results are descriptive and do not establish statistical significance.
- The dataset ends on 2012-06-27. Forecasts are anchored to its last observation; there is no live feed or arbitrary historical cut-off.
- The four clients are simulated on one machine. There is no formal privacy mechanism, secure aggregation, authentication, or production deployment.
- The assessment brief names a Task A proposal document. No separate proposal document is present in the reviewed project structure; the team should confirm whether its proposal requirements are already covered in another submitted artifact or prepare that deliverable.
- The submission ZIP has not been assembled. Confirm whether the METR-LA input file may be included; otherwise include the permitted dataset access/placement instructions needed to run the demo.
- Python bytecode and pytest cache directories are present in the working tree; exclude these generated caches when assembling the ZIP.
- The final 2,500–3,000-word academic report, final APA reference list, names/IDs, verified member contribution/reflection statements, presentation slides, recorded presentation, Turnitin report, and ZIP package have not been prepared. They remain required before submission.

## Readiness assessment

**Ready to begin final report and presentation preparation:** yes. The implementation, saved experiment evidence, run instructions, API/dashboard integration checks, and actual comparison metrics are available as inputs. **Ready for final assessment submission:** no. The academic deliverables and package listed above remain open, and the team must confirm or produce the proposal document requirement.
