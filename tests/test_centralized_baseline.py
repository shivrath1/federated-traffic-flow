import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from src.models.centralized_baseline import (
    SharedLinearForecaster,
    TrainingConfig,
    load_saved_model,
    train_model,
)


class SharedLinearForecasterTests(unittest.TestCase):
    def test_prediction_shapes_support_different_sensor_counts(self):
        model = SharedLinearForecaster(lookback_steps=4, horizon_steps=3)
        for sensor_count in (51, 52):
            X = np.ones((5, 4, sensor_count), dtype=np.float32)
            predictions = model.predict(X)
            self.assertEqual(predictions.shape, (5, 3, sensor_count))
            self.assertEqual(predictions.dtype, np.float32)

    def test_persistence_initialization_has_basic_prediction_behavior(self):
        model = SharedLinearForecaster(lookback_steps=4, horizon_steps=3)
        X = np.zeros((2, 4, 2), dtype=np.float32)
        X[:, -1, :] = np.array([[7.0, 9.0], [3.0, 5.0]], dtype=np.float32)
        predictions = model.predict(X)
        expected = np.repeat(X[:, -1:, :], repeats=3, axis=1)
        np.testing.assert_allclose(predictions, expected)

    def test_training_selects_epoch_from_validation(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(32, 4, 5)).astype(np.float32)
        y = np.repeat(X[:, -1:, :], repeats=3, axis=1)
        config = TrainingConfig(
            lookback_steps=4,
            horizon_steps=3,
            max_epochs=3,
            batch_windows=8,
            learning_rate=0.01,
            patience=3,
            seed=9,
        )
        model, history, best_epoch, elapsed = train_model(X, y, X[:8], y[:8], config)
        self.assertEqual(model.predict(X[:2]).shape, (2, 3, 5))
        self.assertGreaterEqual(best_epoch, 1)
        self.assertEqual(len(history), 3)
        self.assertGreaterEqual(elapsed, 0)

    def test_saved_weights_load_and_predict(self):
        model = SharedLinearForecaster(lookback_steps=4, horizon_steps=3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(root / "shared_linear_forecaster.npz", **model.state())
            (root / "config.json").write_text(json.dumps({"lookback_steps": 4, "horizon_steps": 3}))
            loaded, _ = load_saved_model(root)
            X = np.ones((2, 4, 51), dtype=np.float32)
            self.assertEqual(loaded.predict(X).shape, (2, 3, 51))
            np.testing.assert_allclose(loaded.predict(X), model.predict(X))


if __name__ == "__main__":
    unittest.main()
