# Four-Model Experiment Comparison

## Scope and evaluation

All four configurations use the prepared METR-LA data, the shared per-sensor linear model, a 12-reading (one-hour) input, a 12-reading (one-hour) forecast, and the same chronological test split. Each test array has shape `X/y = (5131, 12, 207)`. The aggregate metrics weight every scalar prediction equally across windows, forecast leads, and sensors: 5,131 × 12 × 207 = 12,745,404 values. MAE is mean absolute error; RMSE is the square root of mean squared error. Both are in the dataset's speed units.

Validation MSE selects the centralized epoch or federated communication round. Test arrays are opened only after selection. No test observations are used for training or augmentation. Each result is one run with a fixed seed; there is no repeated-seed uncertainty analysis.

## Measured results

| Configuration | Best checkpoint | Test MAE | Test RMSE | Measured training time |
|---|---:|---:|---:|---|
| Centralized baseline | Epoch 29 / 30 | 5.698667 | 11.969190 | 28.5255 s train/validation |
| Centralized with noise augmentation | Epoch 30 / 30 | 5.790541 | 11.871505 | 49.0059 s train/validation; augmentation generation 2.9680 s |
| FedAvg without augmentation | Round 30 / 30 | 5.522869 | 11.973663 | 42.8984 s coordinator round loop; 16.9813 s summed local-fit time |
| FedAvg with noise augmentation | Round 30 / 30 | 5.665383 | 11.883041 | 123.6770 s coordinator round loop; 36.6516 s summed local-fit time; 17.6795 s synthetic generation |

The aggregate values and all twelve per-horizon MAE/RMSE values are stored in [`models/experiment_comparison/metrics.json`](/Users/shivrath/Documents/MSE806-Federated-Traffic/models/experiment_comparison/metrics.json). Per-horizon results are also graphed in [`four_model_per_horizon_comparison.png`](/Users/shivrath/Documents/MSE806-Federated-Traffic/models/experiment_comparison/four_model_per_horizon_comparison.png). Overall metrics are plotted in [`four_model_overall_comparison.png`](/Users/shivrath/Documents/MSE806-Federated-Traffic/models/experiment_comparison/four_model_overall_comparison.png).

![Four model test metrics](/Users/shivrath/Documents/MSE806-Federated-Traffic/models/experiment_comparison/four_model_overall_comparison.png)

## Interpretation

Centralized noise augmentation changed MAE from 5.698667 to 5.790541 (+1.612%) and RMSE from 11.969190 to 11.871505 (-0.816%). FedAvg noise augmentation changed MAE from 5.522869 to 5.665383 (+2.580%) and RMSE from 11.973663 to 11.883041 (-0.757%). In both settings the MAE became worse while RMSE became better, so augmentation did not consistently help. These small single-run differences do not establish a general effect.

## Augmentation and training differences

Both augmented configurations use the same simple method: one synthetic input copy per original training window, independent zero-mean Gaussian jitter at 5% of the corresponding sensor's training-input standard deviation, unchanged future target, and clipping at zero. Both use seed 42 as the base setting. Central noise scales are estimated from the full central training tensor; because sensor groups are disjoint and each uses the same time windows, each sensor's estimated scale matches its Stage 4 client scale. Perturbation draws differ: the central run uses one seed-42 all-sensor random stream, while FedAvg uses client-specific seeds 42–45 and writes separate synthetic client files.

Centralized training processes original and synthetic arrays as two dataset parts, samples combined minibatches, and retains Adam state across batches and epochs. FedAvg keeps client data local to its simulated client and performs one local epoch for each client in each of 30 rounds; the local Adam state resets for each client invocation and the coordinator applies weighted FedAvg. The same model, forecast horizon, test split, and error definitions are used. The runtime measures are not directly comparable: the federated coordinator time also includes repeated compressed-file loading, aggregation, and validation.

This is a single-machine four-client simulation. The noise method is input jitter, not a diffusion generator; the project does not reproduce FedTPS. FedAvg does not guarantee privacy, and this work has no formal differential privacy or secure aggregation.

## Artifacts and reproduction

- Centralized augmented model and run evidence: `models/centralized_augmented/`.
- Four-model comparison outputs: `models/experiment_comparison/`.
- Run the added experiment with `python3 -m src.models.centralized_augmented`.
- Rebuild comparison outputs with `python3 -m src.models.consolidate_experiments`.
- Tests cover separate-part training shape compatibility and enforce workflow order: training/validation load, training-only augmentation, validation checkpoint selection, then test loading.
