# Approved Implementation Plan

## Scope

Build a one-week prototype using `data/raw/metr-la.h5` only. Predict one hour ahead, represented as 12 future five-minute steps. Compare a centralized baseline with a four-client federated simulation on one machine. Add simple synthetic time-series augmentation, an API, and a dashboard. Use actual results only. This is a practical, simplified adaptation of the selected FedTPS paper, not a reproduction of its federated diffusion generator or advanced spatio-temporal prediction architecture.

## Stages, tasks, and dependencies

| Stage | Tasks | Depends on | Owner(s) | Completion evidence |
|---|---|---|---|---|
| 1. Confirm data contract | Load `/df`, validate dimensions/timestamps/values, decide input window, target representation, split boundaries, and client partition rule | Nothing; do first | Member 1, with Members 2–3 | Documented data summary and shared sample/input-output contract |
| 2. Prepare data | Implement repeatable loading and preprocessing, chronological train/validation/test splits, four client partitions, and simple synthetic augmentation method | Stage 1 | Member 1; Member 2 agrees augmentation interface | Rerunnable preparation and split checks |
| 3. Central prediction baseline | Implement a small, explainable model and train/evaluate it for the agreed 12-step target; save actual predictions and MAE/RMSE/time | Stages 1–2 | Member 2 | Baseline run artifacts and recorded metrics |
| 4. Federated training | Reuse the agreed model and data contract; train clients locally and aggregate updates; evaluate on the same test data | Stage 3 baseline/interface and Stage 2 client partitions | Member 3, with Member 2 | Completed four-client run, model/results, and actual metrics/time |
| 5. Augmentation comparison | Run centralized with augmentation; run federated with augmentation if schedule permits; record method and metrics | Stages 2–4 as applicable | Member 2; Member 3 for federated variant | Comparable experiment records; otherwise document as incomplete |
| 6. API and dashboard | Expose predictions/results; build dashboard against mock response contract, then connect real API outputs | Model output contract; API endpoints for live integration | Member 3 (API), Member 4 (UI) | Working local demo displaying actual results |
| 7. Integration and assessment | Run focused and end-to-end checks; record evidence; discuss performance, privacy/security, limitations; complete APA report, slides, recording, and ZIP checklist | Stages 1–6; actual results available | All; Member 4 coordinates | Evidence log, report, presentation, runnable demo, submission checklist |

## What must be completed first

1. Validate the dataset and agree one shared input/output contract, forecast target, chronological split policy, and client partition scheme.
2. Prepare the data and verify the 12-step forecast labels before model and federated work depend on them.
3. Produce a centralized baseline with real outputs and metrics. It anchors comparisons and confirms the model interface needed for federation.

## Parallel work

- On Day 1, Member 1 validates the dataset while Member 2 selects the simplest suitable baseline, Member 3 drafts the API contract/skeleton, and Member 4 sketches the dashboard using explicitly labelled mock data.
- Once the data contract is agreed, Members 2 and 3 can work in parallel on the centralized model and federated/API scaffolding; Member 3 integrates the real model when its interface is stable.
- Member 4 can build dashboard layout and API-bound views with mock responses while training runs proceed. Connect real API responses after the endpoint contract is available.
- Report outline and reference collection can begin early, but results and analysis must use observed experiment evidence.

## Practical one-week schedule

| Day | Focus |
|---|---|
| 1 | Agree scope and data contract; verify HDF5 load; settle model target and split approach; draft API contract and dashboard layout. |
| 2 | Finish data preparation and client partitions; train first centralized baseline; start mock-data dashboard and federated skeleton. |
| 3 | Record baseline metrics; run centralized augmentation comparison; connect federated training to the shared model/data contract. |
| 4 | Complete federated run and metrics; expose results through API; connect dashboard to real responses; resolve interface mismatches. |
| 5 | Run comparable experiments, prioritize reruns for failures, perform end-to-end checks, capture actual outputs, draft results/report. |
| 6 | Finish analysis, security/privacy discussion, limitations, APA references, slides and recording; rehearse demo. |
| 7 | Final integration and documented demo run; verify report/presentation/submission checklist and package required files. |

The plan is intentionally prioritized: if time becomes tight, complete data preparation, centralized baseline, federated run, and usable API/dashboard first. Treat federated-with-augmentation as optional; do not replace missing experiment results with estimates.
