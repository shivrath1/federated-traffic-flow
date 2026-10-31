# Member 2 Centralized Model Report — Draft 1

**Status:** Centralized baseline implemented and evaluated. The Stage 4 train-only noise augmentation was also implemented and evaluated within the federated experiment; see the detailed report linked below.

## Model design

The baseline is a shared per-sensor linear autoregressive forecaster. For each sensor independently, it maps the previous 12 speed readings to the next 12 readings using one learned matrix and bias:

`prediction = input_history @ W + b`

`W` has shape `(12, 12)` and `b` has shape `(12,)`. The same parameters are applied to every sensor, so the model input/output shapes are `(samples, 12, sensors) -> (samples, 12, sensors)`. It has no sensor-specific weights, fixed channel count, or cross-sensor spatial operations.

This design is compatible with the planned four sensor-group clients: a client with 51 sensors and one with 52 sensors use the same parameter shapes and can receive/return the same model state for later FedAvg. Client updates can be weighted by the number of local sensor-window examples. The trade-off is that this simple model does not learn spatial relationships between sensors and assumes a shared temporal mapping across sensors. It is a practical baseline, not a reproduction of FedTPS or the selected paper's diffusion/attention architecture.

The model is initialized with persistence-like weights (each forecast step initially copies the latest input reading) and is refined by Adam updates. It is implemented in NumPy because PyTorch is not installed in the project environment; no dependency was installed.

## Prepared data and split use

The Stage 1 artifacts under `data/processed/metr_la` were used. Train arrays were confirmed as `X/y = (23967, 12, 207)`, validation as `(5129, 12, 207)`, and test as `(5131, 12, 207)`. Arrays are `float32`. The exact test shape was also recorded in Stage 1 metadata. Training code opens only train and validation files first. It trains on train windows, selects weights using validation MSE, then opens the test file once selection is complete. Test data was not used for training, epoch selection, or tuning.

## Training configuration

| Setting | Value |
|---|---:|
| Input lookback | 12 five-minute readings (one hour) |
| Forecast horizon | 12 future five-minute readings (one hour) |
| Optimizer | Adam, implemented in NumPy (β1 0.9, β2 0.999) |
| Learning rate | 0.001 |
| Batch size | 256 time windows (all sensors in each window are model examples) |
| Maximum epochs | 30 |
| Early stopping | Validation MSE, patience 4, minimum improvement 0.00001 |
| Random seed | 42 (for training-window order) |
| Checkpoint selected | Epoch 29 of 30 based on validation MSE |
| Training time | 28.5255 seconds for training/validation epochs; excludes data loading, final test evaluation, and plot writing |
| Normalization | None |

Training windows are shuffled only after the chronological split has already been created. Validation is evaluated after each epoch. The test split was evaluated after selecting the epoch 29 checkpoint.

## Test evaluation

MAE and RMSE are computed over every scalar in the test prediction tensor: all 5,131 windows × 12 forecast leads × 207 sensors = 12,745,404 values. Each sample/lead/sensor value has equal weight. MAE is the mean absolute prediction error; RMSE is the square root of the mean squared prediction error. Values are in the dataset's speed units; no unit conversion was applied.

| Metric | Test result |
|---|---:|
| MAE | 5.698667 |
| RMSE | 11.969190 |

Errors increase with lead time: MAE is 2.968485 at the first five-minute lead and 7.782167 at the 60-minute lead; corresponding RMSE values are 6.399858 and 15.190974. Per-lead metrics are saved in `models/centralized_baseline/metrics.json`. RMSE weights larger errors more strongly than MAE; these aggregate results do not identify the cause of the larger squared errors.

## Saved artifacts and rerunning

Run the experiment with:

```bash
python3 -m src.models.centralized_baseline
```

Artifacts are saved in `models/centralized_baseline/`:

- `shared_linear_forecaster.npz`: learned weights and bias.
- `config.json`: model dimensions, optimizer/training settings, checkpoint epoch, and client compatibility note.
- `metrics.json`: test MAE/RMSE, per-horizon metrics, training duration, and test array shape.
- `training_history.csv`: train MSE and validation MSE/MAE per epoch.
- `learning_curve.png`: training and validation MSE by epoch.
- `test_error_by_horizon.png`: test MAE/RMSE by five-minute lead time.

Load trained weights for later predictions with:

```python
from src.models.centralized_baseline import load_saved_model

model, config = load_saved_model("models/centralized_baseline")
forecast = model.predict(X)  # X shape: (samples, 12, sensors)
```

## Tests and limitations

The final full suite `python3 -m pytest -q` completed with **7 passed**. Model tests check output shapes for 51 and 52 sensors, persistence-like initialization behavior, training/checkpoint selection on a small fixture, and saving/loading model weights. `python3 -m compileall -q src/models tests` completed successfully. Training and plots were also inspected after the real-data run.

This baseline has no explicit spatial modeling, normalization, or learned per-sensor offsets. The data values' units were not converted or independently verified here, so metrics are reported in dataset speed units. Stage 2 did not run a naive benchmark comparison, federated learning, or synthetic augmentation. It should be interpreted as a simple centralized reference point.

## Stage 4 augmentation follow-up

The reusable `src/data/noise_augmentation.py` module creates one input-jittered copy per local training window using a fixed seed and 5% of each sensor's standard deviation, estimated from that client's training inputs only. It preserves the paired target, clips jittered inputs at zero, and writes generated data separately from original client training files. Stage 4 used this augmentation in four-client FedAvg, not in a separate centralized augmented run. On the shared test set, augmented FedAvg had MAE 5.665383 and RMSE 11.883041 versus unaugmented FedAvg MAE 5.522869 and RMSE 11.973663: MAE worsened by 2.5804%, while RMSE improved by 0.7568%. The result is mixed and does not support a general claim that augmentation improves prediction. The detailed method, example counts, comparison, and limitations are in [the synthetic augmentation report](/Users/shivrath/Documents/MSE806-Federated-Traffic/reports/synthetic-augmentation-report.md).

## Stage 5 centralized augmentation follow-up

The centralized augmented run reuses the same augmenter: a single extra input window for every original central training window, with independent Gaussian jitter at 5% of the corresponding sensor's training-input standard deviation, targets unchanged, and values clipped at zero. The seed is 42. Noise scales are calculated from the full central training input tensor; because each sensor's complete chronological training series belongs to exactly one client and each client uses the same training windows, this produces the same per-sensor scale as the client-local rule. Random draws differ: centralized uses one global seed-42 stream over all channels, while Stage 4 uses client seeds 42–45.

Training uses the same shared 12×12 weights plus 12-element bias, 12-step input and 12-step forecast, batch size 256 windows, learning rate 0.001, and up to 30 epochs as the centralized baseline. Adam state is retained across centralized batches/epochs. The best validation-MSE checkpoint was epoch 30; test was loaded afterward. The 23,967 original multichannel train windows and 23,967 synthetic windows were kept as separate arrays and combined per minibatch. That represents 4,961,169 original and the same number of synthetic sensor-window examples.

Test MAE was 5.790541 and RMSE 11.871505 over the same 12,745,404 scalar test predictions. Relative to the centralized baseline (5.698667 MAE, 11.969190 RMSE), augmentation raised MAE by 1.612% and lowered RMSE by 0.816%. It therefore did not improve both metrics. This single seeded run is descriptive, not a significance test. See [the consolidated comparison](/Users/shivrath/Documents/MSE806-Federated-Traffic/reports/final-experiment-comparison.md) for all four experiments and per-horizon results.
