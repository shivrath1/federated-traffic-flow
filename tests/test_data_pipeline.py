import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.metr_la import (
    load_client_training_data,
    load_prepared_data,
    prepare_dataset,
    validate_frame,
)


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2025-01-01", periods=120, freq="5min")
        values = np.arange(120 * 7, dtype=np.float64).reshape(120, 7)
        self.frame = pd.DataFrame(values, index=self.index, columns=[f"sensor_{i}" for i in range(7)])

    def test_validates_regular_frame(self):
        validate_frame(self.frame)

    def test_rejects_missing_values_duplicate_and_irregular_time(self):
        missing = self.frame.copy()
        missing.iloc[3, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            validate_frame(missing)

        duplicate = self.frame.copy()
        duplicate.index = duplicate.index.where(np.arange(len(duplicate)) != 5, duplicate.index[4])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_frame(duplicate)

        irregular = self.frame.drop(self.frame.index[5])
        with self.assertRaisesRegex(ValueError, "five-minute"):
            validate_frame(irregular)

    def test_prepare_splits_windows_and_sensor_clients(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "tiny.h5"
            self.frame.to_hdf(source, key="df")
            output = tmp_path / "processed"
            metadata = prepare_dataset(
                source,
                output,
                lookback=4,
                horizon=3,
                train_fraction=0.60,
                validation_fraction=0.20,
            )

            data = load_prepared_data(output)
            train, validation, test = (data["splits"][key] for key in ("train", "validation", "test"))
            self.assertEqual(train["X"].shape[1:], (4, 7))
            self.assertEqual(train["y"].shape[1:], (3, 7))
            self.assertEqual(train["X"].dtype, np.float32)
            np.testing.assert_array_equal(train["X"][0], self.frame.iloc[0:4].to_numpy(dtype=np.float32))
            np.testing.assert_array_equal(train["y"][0], self.frame.iloc[4:7].to_numpy(dtype=np.float32))
            self.assertEqual(metadata["forecast_horizon_steps"], 3)
            self.assertEqual(metadata["splits"]["train"]["row_end_exclusive"], 72)
            self.assertEqual(metadata["splits"]["validation"]["row_start_inclusive"], 72)
            self.assertEqual(metadata["splits"]["test"]["row_start_inclusive"], 96)
            self.assertEqual(train["y"].shape[0], 66)
            self.assertEqual(validation["y"].shape[0], 22)
            self.assertEqual(test["y"].shape[0], 22)

            clients = metadata["client_partition"]["clients"]
            self.assertEqual([client["sensor_count"] for client in clients], [2, 2, 2, 1])
            all_sensors = [sensor for client in clients for sensor in client["sensor_ids"]]
            self.assertEqual(len(all_sensors), len(set(all_sensors)))
            self.assertEqual(set(all_sensors), set(self.frame.columns))
            client_1 = load_client_training_data("client_1", output)
            self.assertEqual(client_1["X"].shape, (66, 4, 2))
            self.assertEqual(client_1["y"].shape, (66, 3, 2))

            # Training targets end before the validation boundary; later splits
            # are separate files and cannot enter the generated client training set.
            train_last_target_row = metadata["splits"]["train"]["target_start_row_last_sample"] + 3 - 1
            self.assertLess(train_last_target_row, metadata["splits"]["validation"]["row_start_inclusive"])
            self.assertTrue((output / "test.npz").is_file())
            self.assertFalse((output / "clients" / "client_1_test.npz").exists())


if __name__ == "__main__":
    unittest.main()
