import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import src.models.centralized_augmented as augmented_module
from src.data.noise_augmentation import NoiseAugmentationConfig, augment_training_windows
from src.models.centralized_baseline import TrainingConfig, train_model_with_parts


class CentralizedAugmentedTests(unittest.TestCase):
    def test_two_part_training_has_compatible_forecast_shape(self):
        rng = np.random.default_rng(12)
        X = rng.uniform(0, 10, (24, 4, 3)).astype(np.float32)
        y = np.repeat(X[:, -1:, :], 3, axis=1)
        synthetic_X, synthetic_y, _ = augment_training_windows(
            X, y, NoiseAugmentationConfig(seed=7)
        )
        config = TrainingConfig(
            lookback_steps=4, horizon_steps=3, max_epochs=2, batch_windows=8,
            patience=2, seed=7,
        )
        model, history, best_epoch, _ = train_model_with_parts(
            [(X, y), (synthetic_X, synthetic_y)], X[:8], y[:8], config
        )
        self.assertEqual(model.predict(X[:5]).shape, (5, 3, 3))
        self.assertTrue(history)
        self.assertGreaterEqual(best_epoch, 1)

    def test_workflow_generates_only_after_train_validation_and_test_after_selection(self):
        rng = np.random.default_rng(13)
        train_X = rng.uniform(0, 10, (24, 4, 3)).astype(np.float32)
        train_y = np.repeat(train_X[:, -1:, :], 3, axis=1)
        validation_X, validation_y = train_X[:8].copy(), train_y[:8].copy()
        test_X, test_y = train_X[8:16].copy(), train_y[8:16].copy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "out"
            data.mkdir()
            (data / "metadata.json").write_text(json.dumps({
                "lookback_steps": 4, "forecast_horizon_steps": 3,
            }))
            for name, X, y in (("train", train_X, train_y),
                               ("validation", validation_X, validation_y),
                               ("test", test_X, test_y)):
                np.savez_compressed(data / f"{name}.npz", X=X, y=y)

            real_load = augmented_module._load_split
            real_augment = augmented_module.augment_training_windows
            real_train = augmented_module.train_model_with_parts
            events = []

            def load_spy(path, name):
                events.append(f"load:{name}")
                return real_load(path, name)

            def augment_spy(X, y, config, *, seed):
                self.assertEqual(events, ["load:train", "load:validation"])
                np.testing.assert_array_equal(X, train_X)
                np.testing.assert_array_equal(y, train_y)
                events.append("augment:training-only")
                return real_augment(X, y, config, seed=seed)

            def train_spy(*args, **kwargs):
                result = real_train(*args, **kwargs)
                events.append("selection:complete")
                return result

            with patch.object(augmented_module, "_load_split", side_effect=load_spy), \
                 patch.object(augmented_module, "augment_training_windows", side_effect=augment_spy), \
                 patch.object(augmented_module, "train_model_with_parts", side_effect=train_spy):
                metrics = augmented_module.run_experiment(
                    data, output,
                    TrainingConfig(lookback_steps=4, horizon_steps=3, max_epochs=1,
                                   batch_windows=8, patience=1, seed=21),
                    NoiseAugmentationConfig(seed=21),
                )
            self.assertEqual(events, [
                "load:train", "load:validation", "augment:training-only",
                "selection:complete", "load:test",
            ])
            manifest = json.loads((output / "augmentation_manifest.json").read_text())
            self.assertFalse(manifest["validation_used"])
            self.assertFalse(manifest["test_used"])
            self.assertTrue(metrics["test_opened_after_model_selection"])
            self.assertEqual(metrics["test"]["evaluated_values"], 8 * 3 * 3)


if __name__ == "__main__":
    unittest.main()
