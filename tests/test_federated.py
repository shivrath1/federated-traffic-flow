import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from src.federated.fedavg import fedavg, load_federated_model
from src.models.centralized_baseline import SharedLinearForecaster, TrainingConfig, train_local_model


class FedAvgTests(unittest.TestCase):
    def test_weighted_aggregation_matches_expected_average(self):
        states = [
            {"weights": np.array([[1.0, 3.0]], dtype=np.float32), "bias": np.array([2.0], dtype=np.float32)},
            {"weights": np.array([[5.0, 7.0]], dtype=np.float32), "bias": np.array([6.0], dtype=np.float32)},
        ]
        aggregate = fedavg(states, [1, 3])
        np.testing.assert_allclose(aggregate["weights"], [[4.0, 6.0]])
        np.testing.assert_allclose(aggregate["bias"], [5.0])
        self.assertEqual(aggregate["weights"].dtype, np.float32)

    def test_aggregation_rejects_invalid_client_states(self):
        state = {"weights": np.zeros((2, 2), dtype=np.float32), "bias": np.zeros(2, dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "positive"):
            fedavg([state], [0])
        different_shape = {"weights": np.zeros((3, 2), dtype=np.float32), "bias": np.zeros(2, dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            fedavg([state, different_shape], [1, 1])

    def test_local_training_works_for_both_client_sensor_counts(self):
        config = TrainingConfig(
            lookback_steps=4,
            horizon_steps=3,
            max_epochs=1,
            batch_windows=2,
            learning_rate=0.001,
            seed=4,
        )
        initial = SharedLinearForecaster(4, 3).state()
        for sensor_count in (51, 52):
            rng = np.random.default_rng(sensor_count)
            X = rng.normal(size=(4, 4, sensor_count)).astype(np.float32)
            y = np.repeat(X[:, -1:, :], repeats=3, axis=1)
            local_model, losses, _ = train_local_model(
                X, y, initial, config, local_epochs=1, seed=7
            )
            self.assertEqual(local_model.weights.shape, (4, 3))
            self.assertEqual(local_model.bias.shape, (3,))
            self.assertTrue(np.isfinite(losses[0]))
            self.assertEqual(local_model.predict(X).shape, y.shape)

    def test_selected_global_model_can_be_loaded(self):
        model = SharedLinearForecaster(4, 3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(root / "fedavg_global_model.npz", **model.state())
            (root / "config.json").write_text(json.dumps({"lookback_steps": 4, "horizon_steps": 3}))
            loaded, _ = load_federated_model(root)
            self.assertEqual(loaded.predict(np.ones((2, 4, 51), dtype=np.float32)).shape, (2, 3, 51))


if __name__ == "__main__":
    unittest.main()
