# Federated Traffic Flow Prediction

## Project purpose and expected outcome

Build a small, explainable university project prototype that forecasts traffic speed one hour ahead from METR-LA sensor readings. Compare centralized training with a simulated federated setup using four clients on one machine. Include reproducible data preparation, prediction, a backend API, a dashboard, and recorded results. The project is a practical adaptation of the selected research paper, not an exact reproduction.

## Dataset

- Use METR-LA only: `data/raw/metr-la.h5`.
- Do not include PeMS-Bay in the main experiments.
- Inspection found a pandas HDF5 DataFrame at `/df`, shape 34,272 time steps × 207 sensors, spanning 2012-03-01 00:00 to 2012-06-27 23:55 at five-minute intervals, with no missing values in this copy.
- The one-hour forecast horizon is 12 future time steps at five-minute sampling. Preserve chronological train/validation/test splits to avoid leakage.
- Recheck and document the dataset during preprocessing; do not assume any metadata not present in the file.

## Member responsibilities

- **Member 1 — Data preparation:** inspect/validate METR-LA, define preprocessing and chronological splits, create four simulated client partitions, document the dataset and provide reusable data loading/preparation.
- **Member 2 — Prediction model and synthetic data:** implement a compact, explainable forecasting baseline/model, add simple synthetic time-series augmentation, and compare runs with and without augmentation.
- **Member 3 — Federated learning and backend API:** implement four-client federated simulation and aggregation, expose prediction and experiment-result endpoints, and provide example responses/contracts.
- **Member 4 — Dashboard, integration, and testing:** build the dashboard, integrate it with the API, run end-to-end checks, and prepare demo/report/presentation material.

Coordinate shared interfaces early. Member 4 may use clearly labelled mock responses while the API is under development.

## Implementation constraints

- Keep the scope suitable for a one-week university project and the implementation straightforward and explainable.
- Use only METR-LA for main experiments and four simulated clients on one machine.
- The core comparison is centralized without augmentation, centralized with augmentation, and federated without augmentation. Federated with augmentation is a stretch goal if time permits.
- Use the same chronological test split for comparisons. Report MAE, RMSE, training time, and the client data-handling setup.
- Do not build a diffusion generator, complex graph/attention architecture, live traffic feed, production service, or multi-machine deployment unless the core scope is complete and time remains.
- Do not describe the simplified augmentation or FL setup as a reproduction of FedTPS. State the differences and limitations.
- Federated learning does not by itself guarantee privacy. Discuss update leakage and other relevant security limitations; do not claim raw data is anonymous or fully protected.

## Coding and testing expectations

- Keep code modular, readable, and runnable from documented commands; use clear names and comments where they help explain decisions.
- Keep raw data in `data/raw/`; generated data, models, and metrics should be reproducible and stored separately.
- Add checks for data shape, timestamps, missing values, split ordering, forecast horizon, and API payloads as appropriate.
- Run focused checks after each stage and an end-to-end demo check before finalization. Record the exact command and actual result in `reports/report-evidence.md`.
- Never claim a test passed unless it was run and its result observed. Do not fabricate data, metrics, screenshots, or outputs.

## Reporting and academic deliverables

- Prepare a 2,500–3,000-word project report with introduction, literature review, methodology, results and analysis, discussion, conclusion/self-reflection, and APA-style references.
- Report actual experiment results and performance metrics, methodology, software functionality, efficiency, security/privacy considerations, limitations, each member's actual contribution, and lessons learned.
- Prepare presentation slides and a recorded presentation with the combined presentation no longer than 15 minutes. Prepare the required ZIP submission with project report, implementation, demonstration dependencies/files, slides/recording, and Turnitin report according to course instructions.
- Write in the team's own words and cite sources using APA. The brief states a Turnitin originality threshold of 20% or lower; follow the course's submission instructions.
- Cite the selected paper and explain that its FedTPS diffusion-based synthetic generation and spatio-temporal architecture are beyond this prototype's scope unless implemented.

## Evidence and progress tracking

- Update `reports/task-tracker.md` as tasks are assigned, started, completed, blocked, or estimated. Record actual effort only when a team member reports it; otherwise leave it `Not provided`.
- Update `reports/report-evidence.md` whenever implementation work, files, commands, tests, experiments, metrics, difficulties/resolutions, or incomplete work change.
- Keep evidence factual and distinguish completed work from planned work. Never invent results.
