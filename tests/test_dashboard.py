import unittest
from datetime import date, time
from pathlib import Path
from unittest.mock import patch

import streamlit as st
import altair as alt
from streamlit.testing.v1 import AppTest


class DashboardFailureTests(unittest.TestCase):
    def test_initialization_error_is_readable_without_exposing_local_path(self):
        internal_path = "/private/tmp/research-artifact/location"

        class BrokenTrafficService:
            def __init__(self, *_args, **_kwargs):
                raise OSError(f"missing file: {internal_path}")

        st.cache_resource.clear()
        with patch("src.api.app.TrafficService", BrokenTrafficService):
            dashboard = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
            app = AppTest.from_file(str(dashboard), default_timeout=60).run()

        self.assertEqual(len(app.exception), 0)
        error_text = " ".join(element.value for element in app.error)
        self.assertIn("Required project files may be missing or invalid", error_text)
        self.assertNotIn(internal_path, error_text)

    def test_missing_model_artifacts_show_a_clear_dashboard_state(self):
        class ServiceWithoutModels:
            sensor_ids = ["773869"]

            def __init__(self, *_args, **_kwargs):
                pass

            def model_list(self):
                return {"models": [{"name": "Centralized baseline", "available": False}]}

            def forecast_options(self):
                return {}

            def evaluation(self):
                return {}

        st.cache_resource.clear()
        with patch("src.api.app.TrafficService", ServiceWithoutModels):
            dashboard = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
            app = AppTest.from_file(str(dashboard), default_timeout=60).run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("No saved model artifacts are available", " ".join(e.value for e in app.error))

    def test_actual_dashboard_renders_and_controls_forecast_window(self):
        st.cache_resource.clear()
        rendered_charts = []
        original_altair_chart = st.altair_chart

        def record_chart(chart, *args, **kwargs):
            rendered_charts.append(chart)
            return original_altair_chart(chart, *args, **kwargs)

        dashboard = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
        with patch("streamlit.altair_chart", side_effect=record_chart):
            app = AppTest.from_file(str(dashboard), default_timeout=120).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.title[0].value, "Federated Traffic Flow Prediction")
            self.assertEqual(app.selectbox[0].label, "Traffic sensor")
            self.assertEqual(app.selectbox[1].label, "Forecast model")
            self.assertEqual(len(app.metric), 5)
            self.assertTrue(any(metric.label == "Selected-window MAE" for metric in app.metric))
            self.assertTrue(any(metric.label == "Persistence RMSE" for metric in app.metric))

            selected_sensor = app.selectbox[0].options[1]
            selected_model = "Centralized baseline"
            app.selectbox[0].set_value(selected_sensor)
            app.selectbox[1].set_value(selected_model)
            app.date_input[0].set_value(date(2012, 6, 14))
            app.time_input[0].set_value(time(9, 0))
            app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, selected_sensor)
        self.assertEqual(app.selectbox[1].value, selected_model)
        self.assertEqual(app.date_input[0].value, date(2012, 6, 14))
        self.assertEqual(app.time_input[0].value, time(9, 0))
        captions = " ".join(element.value for element in app.caption)
        self.assertIn("2012-06-14T09:00:00", captions)
        self.assertTrue(any(metric.label == "Selected-window MAE" for metric in app.metric))
        self.assertGreaterEqual(len(rendered_charts), 4)
        self.assertTrue(all(isinstance(chart, (alt.Chart, alt.LayerChart, alt.HConcatChart, alt.VConcatChart)) for chart in rendered_charts))


if __name__ == "__main__":
    unittest.main()
