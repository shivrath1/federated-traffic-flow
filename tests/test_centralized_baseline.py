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

    def test_model_state_rejects_non_finite_weights_and_biases(self):
        model = SharedLinearForecaster(lookback_steps=4, horizon_steps=3)
        original = model.state()
        for parameter, invalid in (("weights", np.nan), ("weights", np.inf), ("bias", np.nan), ("bias", -np.inf)):
            with self.subTest(parameter=parameter, value=invalid):
                state = {key: value.copy() for key, value in original.items()}
                state[parameter].flat[0] = invalid
                with self.assertRaisesRegex(ValueError, "finite"):
                    model.load_state(state)
                np.testing.assert_array_equal(model.weights, original["weights"])
                np.testing.assert_array_equal(model.bias, original["bias"])

    def test_prediction_rejects_non_finite_input_or_output(self):
        model = SharedLinearForecaster(lookback_steps=4, horizon_steps=3)
        invalid_input = np.ones((1, 4, 1), dtype=np.float32)
        invalid_input[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            model.predict(invalid_input)

        model.weights.fill(np.finfo(np.float32).max)
        with self.assertRaisesRegex(ValueError, "prediction contains non-finite"):
            model.predict(np.full((1, 4, 1), np.finfo(np.float32).max, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
