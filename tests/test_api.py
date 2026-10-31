import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import MODEL_SPECS, ROOT, create_app


class TrafficApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())
        cls.sensor_id = "773869"

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
            self.assertLess(result["input_readings"][-1]["timestamp"], result["forecast_origin"])
            self.assertIn("not a live traffic service", result["context"])
            self.assertIn("metrics", result["persistence_baseline"])

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
        for model in report["models"]:
            self.assertTrue(model["recomputed_overall_matches_saved"], model["model_id"])
            self.assertEqual(len(model["mae_by_horizon"]), 12)
            self.assertEqual(len(model["by_hour"]), 24)
            self.assertEqual(len(model["by_sensor"]), 207)

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


if __name__ == "__main__":
    unittest.main()
