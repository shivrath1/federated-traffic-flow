# Synthetic Augmentation Experiment — Draft 1

## Summary

This experiment tests a simple client-local noise augmentation with the existing four-client FedAvg forecaster. For each client, one synthetic input copy is created per original training window by adding independent Gaussian noise to the input history. The future target is copied unchanged. The four synthetic client files are stored separately from the original Stage 1 files and are concatenated with originals only inside that client's local training call.

This is not FedTPS and does not implement the research paper's federated diffusion generator. The selected paper trains a federated diffusion model to generate global synthetic traffic samples, which clients then use in a second federated prediction-model stage. This project instead applies a small, local perturbation method to existing training inputs only.

## Augmentation method and parameters

- Estimate each client's per-sensor standard deviation from that client's `X_train` only.
- For every original window, generate one input copy with independent zero-mean Gaussian noise. Per-sensor noise standard deviation is 5% of that sensor's training-input standard deviation.
- Clip jittered input values at a minimum of 0.0. The inspected training inputs are non-negative.
- Reuse the same original `y_train` target for its synthetic input copy. Targets are not jittered.
- Use base seed 42 and per-client seeds 42, 43, 44, and 45 in client ID order.
- Do not read, fit noise statistics from, or augment validation or test arrays. The synthetic-data manifest records both `validation_used: false` and `test_used: false`.

The rationale is to add mild measurement/input variation while keeping the prediction target unchanged. It is simple and reproducible but does not generate novel traffic regimes or cross-client patterns.

Observed noise-standard-deviation means across sensors were 0.9007, 0.9054, 0.9565, and 0.9165 dataset speed units for clients 1–4. The per-sensor noise standard deviations ranged from 0.4457 to 1.4196 across all four clients.

## Training example counts

Each client has 23,967 original multichannel training windows and receives 23,967 separate synthetic windows. Clients 1–3 each have 52 sensors; client 4 has 51.

| Count | Original | Synthetic | Combined local fit |
|---|---:|---:|---:|
| Multichannel client windows across four clients | 95,868 | 95,868 | 191,736 |
| Sensor-specific window examples across four clients | 4,961,169 | 4,961,169 | 9,922,338 |

The synthetic examples are separate artifacts under `models/federated_augmented/synthetic_training_data/`, with generation settings, per-client seeds, counts, and noise-scale summaries in `synthetic_manifest.json`.

## Federated configuration

The model architecture and evaluation procedure match the unaugmented FedAvg run. Training used 30 communication rounds, one local epoch per client per round, batch size 256 windows, learning rate 0.001, Adam with state reset for each client's round, and validation-MSE selection with patience 5. The best checkpoint was round 30. The test split was opened only after that selection.

Synthetic generation took 17.6795 seconds. Augmented FedAvg coordinator round-loop time was 123.6770 seconds, and summed local-fit time was 36.6516 seconds. The round-loop timer includes repeated synthetic/original client file loading, local training, aggregation, and validation; it is not directly comparable to the centralized training timer. The parameter payload was unchanged from FedAvg without augmentation: 74,880 bytes down and 74,880 bytes up over 30 rounds, excluding protocol overhead.

## Three-model results

All models were evaluated on the same 5,131-window METR-LA test split, aggregating equally over 12 forecast horizons and 207 sensors (12,745,404 scalar predictions). MAE is the mean absolute error and RMSE is the square root of the mean squared error, both in dataset speed units.

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Centralized baseline | 5.698667 | 11.969190 |
| FedAvg without augmentation | 5.522869 | 11.973663 |
| FedAvg with noise augmentation | 5.665383 | 11.883041 |

Compared with FedAvg without augmentation, noise augmentation raised MAE by 0.142514 (2.5804%) and lowered RMSE by 0.090622 (0.7568%). It therefore **did not consistently improve** the result: one metric worsened and the other improved slightly. Compared with the centralized baseline, augmented FedAvg had slightly lower MAE and RMSE, but this does not isolate an augmentation benefit because the unaugmented FedAvg result also differs from centralized training. The results are from one seed and one run; no statistical significance or generalization claim is made.

Per-horizon MAE and RMSE are included in `metrics.json` for all three models and shown in `three_model_test_comparison.png`. Errors generally increase at longer forecast leads.

## Limitations and privacy

The augmentation is input jitter, not a generative model, and it may not reflect realistic temporal or sensor noise. The fixed 5% perturbation was tested once rather than tuned across a validation-only parameter sweep. Clipping at zero is consistent with the observed non-negative values but may create a boundary effect near zero.

The experiment remains a one-machine simulation. Original client files and synthetic copies stay separated by client, and the coordinator aggregates model parameters, but model updates can leak information. No formal differential privacy or secure aggregation is implemented. This work does not reproduce FedTPS or establish a privacy guarantee.

## Stage 5 centralized augmentation and four-model consolidation

A separate centralized run was added using the same augmentation function and noise rule. It created one jittered input copy for each of the 23,967 central training windows (23,967 synthetic windows; 4,961,169 sensor-window examples each for original and synthetic). Scale fitting used the central `X_train` tensor only. The value for each sensor matches the client rule because the four clients partition sensors and use the same chronological training windows. The exact random perturbations differ: central uses seed 42 in one all-sensor array, while Stage 4 generated separate arrays with seeds 42, 43, 44, and 45.

The centralized trainer used the same model and horizon, 30-epoch maximum, batch size 256, learning rate 0.001, and persistent Adam state as the unaugmented centralized experiment. It selected epoch 30 by validation MSE before opening test data. The federated augmented run instead trains each client on its own original and synthetic files for one epoch each round over 30 rounds; Adam state resets for each local client invocation, and the coordinator selects by global validation MSE. These are necessary differences between centralized batch training and FedAvg; test metrics and forecasts remain aligned on the same split and dimensions.

On the shared test set, centralized augmentation scored MAE **5.790541** and RMSE **11.871505**, versus the centralized baseline's **5.698667** MAE and **11.969190** RMSE. Federated augmentation scored MAE **5.665383** and RMSE **11.883041**, versus unaugmented FedAvg's **5.522869** MAE and **11.973663** RMSE. In both comparisons MAE rose and RMSE fell, so neither supports a claim of consistent improvement. The full four-model table and interpretation are in [final-experiment-comparison.md](/Users/shivrath/Documents/MSE806-Federated-Traffic/reports/final-experiment-comparison.md), with consolidated plots and per-horizon metrics under `models/experiment_comparison/`.
