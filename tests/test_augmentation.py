import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.noise_augmentation import (
    NoiseAugmentationConfig,
    augment_training_windows,
    generate_synthetic_client_files,
)
from src.models.centralized_baseline import SharedLinearForecaster


class NoiseAugmentationTests(unittest.TestCase):
    def make_data(self, sensors=3):
        rng = np.random.default_rng(12)
        X = rng.normal(loc=25.0, scale=4.0, size=(8, 4, sensors)).astype(np.float32)
        X = np.maximum(X, 0)
        y = np.repeat(X[:, -1:, :], repeats=3, axis=1)
        return X, y

    def test_noise_is_seeded_separate_and_targets_are_retained(self):
        X, y = self.make_data()
        X_before, y_before = X.copy(), y.copy()
        config = NoiseAugmentationConfig(noise_std_fraction=0.05, synthetic_copies_per_window=1, seed=17)
        synthetic_X, synthetic_y, details = augment_training_windows(X, y, config)
        repeat_X, repeat_y, _ = augment_training_windows(X, y, config)

        self.assertEqual(synthetic_X.shape, X.shape)
        self.assertEqual(synthetic_y.shape, y.shape)
        self.assertEqual(synthetic_X.dtype, np.float32)
        np.testing.assert_array_equal(synthetic_X, repeat_X)
        np.testing.assert_array_equal(synthetic_y, repeat_y)
        np.testing.assert_array_equal(synthetic_y, y)
        np.testing.assert_array_equal(X, X_before)
        np.testing.assert_array_equal(y, y_before)
        self.assertGreater(float(np.abs(synthetic_X - X).max()), 0)
        self.assertGreaterEqual(float(synthetic_X.min()), 0.0)
        self.assertEqual(details["fit_source"], "this client's X_train only")

    def test_generated_shapes_remain_model_compatible_for_51_and_52_sensors(self):
        config = NoiseAugmentationConfig(seed=3)
        model = SharedLinearForecaster(lookback_steps=4, horizon_steps=3)
        for sensors in (51, 52):
            X, y = self.make_data(sensors)
            X_synthetic, y_synthetic, _ = augment_training_windows(X, y, config)
            self.assertEqual(X_synthetic.shape, X.shape)
            self.assertEqual(y_synthetic.shape, y.shape)
            self.assertEqual(model.predict(X_synthetic).shape, y_synthetic.shape)

    def test_client_generation_reads_training_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "prepared"
            clients_dir = root / "clients"
            clients_dir.mkdir(parents=True)
            client_metadata = []
            for index in range(1, 5):
                X, y = self.make_data(2 if index < 4 else 1)
                train_path = clients_dir / f"client_{index}_train.npz"
                np.savez_compressed(train_path, X=X, y=y)
                client_metadata.append({"client_id": f"client_{index}", "train_file": f"clients/{train_path.name}"})
            (root / "metadata.json").write_text(json.dumps({"client_partition": {"clients": client_metadata}}))
            # If the generator ever tries to open either split, it will fail.
            (root / "validation.npz").write_text("must not be opened")
            (root / "test.npz").write_text("must not be opened")

            output = Path(tmp) / "synthetic"
            manifest = generate_synthetic_client_files(
                root,
                output,
                NoiseAugmentationConfig(seed=21),
            )
            self.assertFalse(manifest["validation_used"])
            self.assertFalse(manifest["test_used"])
            self.assertEqual(len(manifest["clients"]), 4)
            for client in manifest["clients"]:
                self.assertIn("_train.npz", client["source_file"])
                self.assertTrue((output / client["synthetic_file"]).is_file())


if __name__ == "__main__":
    unittest.main()
