# Member 3 Federated Learning Report — Draft 1

**Status:** Four-client FedAvg with and without train-only noise augmentation implemented and evaluated. The local FastAPI prediction/results API and Streamlit dashboard were added in Stage 6; see [the API/dashboard report](/Users/shivrath/Documents/MSE806-Federated-Traffic/reports/api-dashboard-report.md). This report focuses on the federated experiments.

## Architecture and client setup

The experiment reuses Member 2's `SharedLinearForecaster`: each sensor's 12-reading history maps to 12 future readings using a shared 12×12 weight matrix and a 12-element bias. The model parameters do not depend on a sensor count, so clients with 51 and 52 sensors can train and return identically shaped parameters. Each client receives the current global state and trains only against its own Stage 1 training file.

The prepared metadata confirms client sizes in ID order are `client_1: 52`, `client_2: 52`, `client_3: 52`, and `client_4: 51` sensors. This is the same set of counts as 51, 52, 52, 52, with the 51-sensor client last. These are sensor groups simulated sequentially in one process on one machine, not four organizations or devices.

## FedAvg workflow

At each communication round:

1. The coordinator broadcasts the current global 12×12 weights and 12-element bias to each simulated client.
2. Each client loads only its own `client_N_train.npz`, starts from the broadcast state, and performs one local epoch. No validation or test arrays enter client training.
3. Clients return their model parameters. The coordinator averages each parameter tensor using the local training sensor-window example count as the client weight:

   `global_parameter = sum(client_examples_i * local_parameter_i) / sum(client_examples_i)`

4. The coordinator evaluates the resulting global model against the shared validation split. The best validation-MSE checkpoint is retained; round selection and patience do not use test data.
5. After round selection ends, the selected checkpoint is evaluated on the held-out test split.

The training data itself is not combined or included in the aggregation. Only the model's `float32` weight and bias arrays are exchanged in this simulation.

## Configuration

| Setting | Value |
|---|---:|
| Clients | 4 sensor-group clients, one machine |
| Sensor counts | 52, 52, 52, 51 (client IDs 1–4) |
| Local epochs | 1 per communication round |
| Communication rounds | 30 run; best validation checkpoint at round 30 |
| Local batch size | 256 time windows per client |
| Optimizer | Same NumPy Adam update as centralized baseline; optimizer state resets per client per round |
| Learning rate | 0.001 |
| Round selection | Global validation MSE; patience 5, minimum improvement 0.00001 |
| Seed | 42 base seed; deterministic per-client/per-round shuffle offsets |
| Model parameters | 156 `float32` values (144 weights + 12 bias), 624 bytes per client payload |

The centralized model used 30 global epochs with Adam state retained between batches; the federated run used 30 communication rounds with one local epoch per client and reset Adam state for each local client invocation. This gives 30 local passes over each client's training windows while allowing a global aggregation between passes.

## Test results and centralized comparison

The central and federated models use the same model architecture, prepared METR-LA splits, 12-step input window, 12-step forecast horizon, and held-out test arrays. Both metrics aggregate error equally over each sample, lead time, and sensor: 5,131 × 12 × 207 = 12,745,404 scalar predictions. MAE is the mean absolute error; RMSE is the square root of mean squared error. Units are the dataset's speed units.

| Model | Test MAE | Test RMSE | Training/runtime measure |
|---|---:|---:|---|
| Centralized, best validation checkpoint epoch 29 | 5.698667 | 11.969190 | 28.5255 seconds for train and validation epochs |
| FedAvg, best validation checkpoint round 30 | 5.522869 | 11.973663 | 42.8984 seconds coordinator round-loop wall time; 16.9813 seconds summed local-fit time |

Compared with the centralized run, FedAvg MAE is 0.175798 lower (3.0849% lower), while RMSE is 0.004473 higher (0.0374% higher). The metrics therefore do not show a consistent improvement in both error measures. This is one run with one fixed seed, not a repeated-seed significance comparison. Per-horizon MAE/RMSE for both models is saved in `metrics.json` and plotted in `test_error_comparison_by_horizon.png`.

Coordinator round-loop time includes repeated client-file loading, local fitting, model aggregation, and validation per round, while the centralized training timer covers its training/validation epochs and excludes initial data loading. The two wall-time figures are not perfectly like-for-like. Summed local-fit time is provided separately. No real network transfer was performed.

## Communication accounting

The model has 156 `float32` parameters, or 624 bytes per parameter state. For each round, four clients each receive one state (2,496 downlink bytes) and return one updated state (2,496 uplink bytes). Across 30 rounds this is **74,880 bytes down + 74,880 bytes up = 149,760 total model-payload bytes** (about 146.25 KiB). This is a calculated tensor-payload estimate; it excludes transport protocol, message framing, client identifiers, and serialization overhead. The simulation does not claim measured network bandwidth or latency.

## Privacy and limitations

This simulation keeps raw arrays in client-specific files and the coordinator aggregates model parameters rather than concatenating client training arrays. However, all clients run on one machine under one process and this is not a security boundary. Model parameters/updates can reveal information about training data. No differential privacy, secure aggregation, encryption protocol, or formal privacy analysis was implemented. Federated learning alone does not guarantee privacy.

The shared temporal model does not represent spatial relationships between sensors and assumes common temporal weights. The result is a small practical FL exercise, not a reproduction of FedTPS. The one-run metric differences should not be generalized beyond this experiment.

## Saved artifacts and rerunning

Run with:

```bash
python3 -m src.federated.fedavg
```

Artifacts are in `models/federated_baseline/`:

- `fedavg_global_model.npz`: selected global model weights and bias.
- `config.json`: client counts, aggregation weights, training settings, selected round, communication payload basis, and privacy note.
- `metrics.json`: test metrics, per-horizon metrics, centralized comparison, timing, communication counts, and test shape.
- `round_history.csv`: client-train and global-validation loss by round.
- `round_learning_curve.png`: local training and validation loss by communication round.
- `test_error_comparison_by_horizon.png`: centralized and FedAvg test errors by lead time.

The global model can be reloaded with `load_federated_model("models/federated_baseline")` from `src.federated.fedavg`.

## Tests and issues

The full suite completed with **11 passed**. FedAvg tests cover weighted averaging arithmetic, invalid weights and shape mismatches, local training compatibility for 51/52 sensors, and global-model reload. The existing centralized-model and data-pipeline tests also pass. `python3 -m compileall -q src tests` completed successfully. Matplotlib printed non-fatal warnings about unavailable default cache/fontconfig directories; plots were still saved and visually inspected.

No code or experiment blocker remains. Review the small MAE/RMSE trade-off, the single-seed result, and the fact that the model has no spatial component before interpreting or extending the experiment.

## Stage 4 augmented FedAvg follow-up

Stage 4 used the same four-client coordinator, shared per-sensor linear model, validation-based round selection, and test split with one client-local input-jittered copy per original training window. Generated arrays were stored separately under `models/federated_augmented/synthetic_training_data/`; each client concatenated those copies with its originals only for its local fit. No validation/test data was used to generate noise or examples.

The actual per-sensor noise standard deviation was 5% of the corresponding client's training-input standard deviation. Mean noise standard deviations were 0.9007, 0.9054, 0.9565, and 0.9165 speed units for clients 1–4, respectively; across individual sensors, values ranged from 0.4457 to 1.4196.

The selected augmented checkpoint was round 30 of 30. Test MAE was 5.665383 and RMSE was 11.883041, compared with the unaugmented FedAvg's 5.522869 MAE and 11.973663 RMSE. Thus augmentation worsened MAE by 2.5804% and improved RMSE by 0.7568%; the outcomes are mixed. The run used one seed, so no significance claim is made. Augmentation generation took 17.6795 seconds; coordinator round-loop time was 123.6770 seconds, including repeated compressed client-array reads, local fits, aggregation, and validation. Summed local-fit time was 36.6516 seconds. The model parameter communication payload remained 149,760 bytes for 30 rounds. See [the Stage 4 augmentation report](/Users/shivrath/Documents/MSE806-Federated-Traffic/reports/synthetic-augmentation-report.md) for the full configuration, example counts, three-model metrics, and caveats.
