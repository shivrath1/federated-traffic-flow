import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.app import MODEL_SPECS, ROOT, TrafficService, create_app


class TrafficApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())
        cls.sensor_id = "773869"
        cls.client.get("/api/health")
        cls.service = cls.client.app.state.traffic

    def _artifact_service(self, model_root: Path) -> TrafficService:
        service = TrafficService.__new__(TrafficService)
        service.model_root = model_root
        service.test_start = self.service.test_start
        service.test_end = self.service.test_end
        service.sensor_ids = self.service.sensor_ids
        service._models = {}
        return service

    def _copy_model(self, model_id: str, destination: Path) -> Path:
        source = ROOT / "models" / MODEL_SPECS[model_id]["directory"]
        model_dir = destination / MODEL_SPECS[model_id]["directory"]
        shutil.copytree(source, model_dir)
        return model_dir

    def test_health_sensors_and_history(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["dataset"], "METR-LA")
        sensors = self.client.get("/api/sensors").json()
        self.assertEqual(sensors["count"], 207)
        self.assertIn(self.sensor_id, sensors["sensor_ids"])
        history = self.client.get(f"/api/sensors/{self.sensor_id}/history?limit=36")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()["forecast_origin"], "2012-06-13T08:00:00")
        readings = history.json()["readings"]
        self.assertEqual(len(readings), 36)
        self.assertEqual(readings[-1]["timestamp"], "2012-06-13T07:55:00")

    def test_invalid_sensor_and_history_limit_are_rejected(self):
        unknown = self.client.get("/api/sensors/not-a-sensor/history")
        self.assertEqual(unknown.status_code, 404)
        self.assertIn("Sensor ID", unknown.json()["detail"])
        invalid_limit = self.client.get(f"/api/sensors/{self.sensor_id}/history?limit=500")
        self.assertEqual(invalid_limit.status_code, 422)

    def test_every_saved_model_returns_twelve_future_steps(self):
        model_list = self.client.get("/api/models").json()["models"]
        self.assertEqual(len(model_list), 4)
        self.assertTrue(all(item["available"] for item in model_list))
        for model_id in MODEL_SPECS:
            response = self.client.post("/api/predict", json={"sensor_id": self.sensor_id, "model_id": model_id})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["model_id"], model_id)
            self.assertEqual(result["forecast_origin"], "2012-06-13T08:00:00")
            self.assertEqual(len(result["forecasts"]), 12)
            self.assertEqual(result["forecasts"][0]["minutes_ahead"], 5)
            self.assertEqual(result["forecasts"][-1]["minutes_ahead"], 60)
            self.assertEqual(result["forecasts"][0]["timestamp"], "2012-06-13T08:00:00")
            self.assertEqual(result["forecasts"][-1]["timestamp"], "2012-06-13T08:55:00")
            self.assertEqual(len(result["actuals"]), 12)
            self.assertEqual(result["actuals"][0]["timestamp"], result["forecasts"][0]["timestamp"])
            self.assertEqual(
                [item["timestamp"] for item in result["actuals"]],
                [item["timestamp"] for item in result["persistence_baseline"]["forecasts"]],
            )
            self.assertLess(result["input_readings"][-1]["timestamp"], result["forecast_origin"])
            self.assertIn("not a live traffic service", result["context"])
            self.assertIn("metrics", result["persistence_baseline"])

    def test_selected_window_separates_past_inputs_from_future_targets(self):
        origin = "2012-06-13T08:00:00"
        response = self.client.post("/api/predict", json={
            "sensor_id": self.sensor_id,
            "model_id": "centralized_baseline",
            "forecast_origin": origin,
        })
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        position = int(self.service.frame.index.get_loc(pd.Timestamp(origin)))
        expected_input = self.service.frame[self.sensor_id].iloc[position - 12:position].to_numpy(dtype=np.float32)
        expected_target = self.service.frame[self.sensor_id].iloc[position:position + 12].to_numpy(dtype=np.float32)
        returned_input = np.array([row["value"] for row in result["input_readings"]], dtype=np.float32)
        returned_actual = np.array([row["value"] for row in result["actuals"]], dtype=np.float32)
        returned_prediction = np.array([row["value"] for row in result["forecasts"]], dtype=np.float32)

        np.testing.assert_array_equal(returned_input, expected_input)
        np.testing.assert_array_equal(returned_actual, expected_target)
        np.testing.assert_array_equal(
            returned_prediction,
            self.service.get_model("centralized_baseline").model.predict(expected_input.reshape(1, 12, 1))[0, :, 0],
        )
        self.assertEqual(len(result["input_readings"]), 12)
        self.assertTrue(all(pd.Timestamp(row["timestamp"]) < pd.Timestamp(origin) for row in result["input_readings"]))
        self.assertEqual([row["timestamp"] for row in result["forecasts"]], [row["timestamp"] for row in result["actuals"]])
        self.assertEqual([row["value"] for row in result["persistence_baseline"]["forecasts"]], [float(expected_input[-1])] * 12)

        actual_error = returned_prediction.astype(np.float64) - returned_actual.astype(np.float64)
        baseline_error = np.full(12, expected_input[-1], dtype=np.float64) - returned_actual.astype(np.float64)
        self.assertAlmostEqual(result["selected_window_metrics"]["mae"], np.abs(actual_error).mean(), places=7)
        self.assertAlmostEqual(result["selected_window_metrics"]["rmse"], np.sqrt(np.square(actual_error).mean()), places=7)
        self.assertAlmostEqual(result["persistence_baseline"]["metrics"]["mae"], np.abs(baseline_error).mean(), places=7)
        self.assertAlmostEqual(result["persistence_baseline"]["metrics"]["rmse"], np.sqrt(np.square(baseline_error).mean()), places=7)

    def test_forecast_window_options_and_timestamp_validation(self):
        options = self.client.get("/api/forecast-options").json()
        self.assertEqual(options["default_forecast_origin"], "2012-06-13T08:00:00")
        self.assertEqual(options["forecast_horizon_minutes"], 60)
        self.assertEqual([item["id"] for item in options["presets"]], [
            "morning_peak", "midday", "evening_peak", "late_night",
        ])
        invalid = self.client.post("/api/predict", json={
            "sensor_id": self.sensor_id,
            "model_id": "centralized_baseline",
            "forecast_origin": "2012-06-13T08:02:00",
        })
        self.assertEqual(invalid.status_code, 422)
        outside_test = self.client.post("/api/predict", json={
            "sensor_id": self.sensor_id,
            "model_id": "centralized_baseline",
            "forecast_origin": "2012-06-01T08:00:00",
        })
        self.assertEqual(outside_test.status_code, 422)

    def test_held_out_evaluation_includes_models_persistence_horizon_hour_sensor(self):
        result = self.client.get("/api/evaluation")
        self.assertEqual(result.status_code, 200, result.text)
        report = result.json()
        self.assertEqual(report["test_shape"]["X"], [5131, 12, 207])
        self.assertEqual(len(report["models"]), 4)
        self.assertEqual(len(report["persistence_baseline"]["mae_by_horizon"]), 12)
        self.assertEqual(len(report["persistence_baseline"]["by_hour"]), 24)
        self.assertEqual(len(report["persistence_baseline"]["by_sensor"]), 207)
        X, y = self.service._load_test_arrays()
        persistence_prediction = np.repeat(X[:, -1:, :], repeats=12, axis=1)
        persistence_error = persistence_prediction.astype(np.float64) - y.astype(np.float64)
        self.assertAlmostEqual(report["persistence_baseline"]["overall"]["mae"], np.abs(persistence_error).mean(), places=12)
        self.assertAlmostEqual(report["persistence_baseline"]["overall"]["rmse"], np.sqrt(np.square(persistence_error).mean()), places=12)
        for offset in (0, len(X) // 2, len(X) - 1):
            target_row = self.service.test_start + offset
            expected_target = self.service.frame.iloc[target_row:target_row + 12].to_numpy(dtype=np.float32)
            np.testing.assert_array_equal(y[offset], expected_target)
        for model in report["models"]:
            self.assertTrue(model["recomputed_overall_matches_saved"], model["model_id"])
            self.assertEqual(len(model["mae_by_horizon"]), 12)
            self.assertEqual(len(model["by_hour"]), 24)
            self.assertEqual(len(model["by_sensor"]), 207)
            prediction = self.service.get_model(model["model_id"]).model.predict(X)
            error = prediction.astype(np.float64) - y.astype(np.float64)
            self.assertAlmostEqual(model["recomputed_overall"]["mae"], np.abs(error).mean(), places=12)
            self.assertAlmostEqual(model["recomputed_overall"]["rmse"], np.sqrt(np.square(error).mean()), places=12)

    def test_invalid_model_and_missing_artifact_have_safe_errors(self):
        invalid = self.client.post("/api/predict", json={"sensor_id": self.sensor_id, "model_id": "other"})
        self.assertEqual(invalid.status_code, 422)
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(ROOT / "data/raw/metr-la.h5", Path(tmp)))
            unavailable = client.post(
                "/api/predict", json={"sensor_id": self.sensor_id, "model_id": "centralized_baseline"}
            )
            self.assertEqual(unavailable.status_code, 503)
            self.assertNotIn(str(ROOT), unavailable.text)

    def test_non_finite_model_weights_and_bias_are_rejected_as_unavailable(self):
        for parameter, invalid_value in (("weights", np.nan), ("weights", np.inf), ("bias", np.nan), ("bias", -np.inf)):
            with self.subTest(parameter=parameter, value=invalid_value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model_dir = self._copy_model("centralized_baseline", root)
                weights_path = model_dir / "shared_linear_forecaster.npz"
                with np.load(weights_path) as saved:
                    state = {key: saved[key].copy() for key in saved.files}
                state[parameter].flat[0] = invalid_value
                np.savez_compressed(weights_path, **state)

                original_root = self.service.model_root
                original_models = self.service._models
                try:
                    self.service.model_root = root
                    self.service._models = {}
                    response = self.client.post("/api/predict", json={
                        "sensor_id": self.sensor_id,
                        "model_id": "centralized_baseline",
                    })
                    self.assertEqual(response.status_code, 503)
                    self.assertNotIn(tmp, response.text)
                finally:
                    self.service.model_root = original_root
                    self.service._models = original_models

    def test_invalid_model_artifact_dimensions_config_and_metric_schema_are_rejected(self):
        cases = ("weight_shape", "bias_shape", "missing_config", "config_dimensions", "missing_metric", "non_finite_metric", "wrong_metric_count")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model_dir = self._copy_model("centralized_baseline", root)
                config_path = model_dir / "config.json"
                metrics_path = model_dir / "metrics.json"
                weights_path = model_dir / "shared_linear_forecaster.npz"
                if case in {"weight_shape", "bias_shape"}:
                    with np.load(weights_path) as saved:
                        state = {key: saved[key].copy() for key in saved.files}
                    if case == "weight_shape":
                        state["weights"] = state["weights"][:, :-1]
                    else:
                        state["bias"] = state["bias"][:-1]
                    np.savez_compressed(weights_path, **state)
                elif case == "missing_config":
                    config = json.loads(config_path.read_text())
                    del config["selection_split"]
                    config_path.write_text(json.dumps(config))
                elif case == "config_dimensions":
                    config = json.loads(config_path.read_text())
                    config["lookback_steps"] = 11
                    config_path.write_text(json.dumps(config))
                else:
                    metrics = json.loads(metrics_path.read_text())
                    if case == "missing_metric":
                        del metrics["test"]["rmse_by_horizon"]
                    elif case == "non_finite_metric":
                        metrics["test"]["mae"] = float("nan")
                    else:
                        metrics["test"]["evaluated_values"] += 1
                    metrics_path.write_text(json.dumps(metrics))

                service = self._artifact_service(root)
                with self.assertRaises(HTTPException) as raised:
                    service.get_model("centralized_baseline")
                self.assertEqual(raised.exception.status_code, 503)

    def test_valid_but_mismatched_saved_metrics_fail_evaluation_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._copy_model("centralized_baseline", root)
            metrics_path = model_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["test"]["mae"] += 0.25
            metrics_path.write_text(json.dumps(metrics))

            original_root = self.service.model_root
            original_models = self.service._models
            original_evaluation = self.service._evaluation
            try:
                self.service.model_root = root
                self.service._models = {}
                self.service._evaluation = None
                with self.assertRaises(HTTPException) as raised:
                    self.service.evaluation()
                self.assertEqual(raised.exception.status_code, 503)
                self.assertIn("do not match", raised.exception.detail)
                self.assertNotIn(tmp, raised.exception.detail)
            finally:
                self.service.model_root = original_root
                self.service._models = original_models
                self.service._evaluation = original_evaluation

    def test_malformed_metadata_is_a_controlled_initialization_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp) / "processed"
            processed.mkdir()
            (processed / "metadata.json").write_text(json.dumps({"splits": {"test": {}}}))
            client = TestClient(create_app(ROOT / "data/raw/metr-la.h5", ROOT / "models", processed))
            response = client.get("/api/health")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "METR-LA data could not be initialized"})
            self.assertNotIn(tmp, response.text)

    def test_missing_or_corrupted_test_arrays_return_safe_errors(self):
        service = TrafficService.__new__(TrafficService)
        with tempfile.TemporaryDirectory() as tmp:
            service.processed_root = Path(tmp)
            service.test_start = 0
            service.test_end = 15
            service.sensor_ids = ["a", "b"]
            expected_X = np.ones((4, 12, 2), dtype=np.float32)
            expected_y = np.ones((4, 12, 2), dtype=np.float32)

            np.savez_compressed(service.processed_root / "test.npz", X=expected_X)
            with self.assertRaises(HTTPException) as missing:
                service._load_test_arrays()
            self.assertEqual(missing.exception.status_code, 503)

            np.savez_compressed(service.processed_root / "test.npz", X=expected_X[:, :, :-1], y=expected_y)
            with self.assertRaises(HTTPException) as bad_shape:
                service._load_test_arrays()
            self.assertEqual(bad_shape.exception.status_code, 503)

            corrupted = expected_X.copy()
            corrupted[0, 0, 0] = np.nan
            np.savez_compressed(service.processed_root / "test.npz", X=corrupted, y=expected_y)
            with self.assertRaises(HTTPException) as bad_values:
                service._load_test_arrays()
            self.assertEqual(bad_values.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
