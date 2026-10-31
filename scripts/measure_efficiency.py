"""Measure local dashboard/service timings without modifying experiment artifacts."""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import MODEL_SPECS, TrafficService


def timed(callable_):
    start = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - start


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and most other supported Unix systems report KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    data_path = ROOT / "data/raw/metr-la.h5"
    model_root = ROOT / "models"
    processed_root = ROOT / "data/processed/metr_la"

    service, service_seconds = timed(lambda: TrafficService(data_path, model_root, processed_root))
    loaded_models, artifact_seconds = timed(lambda: [service.get_model(model_id) for model_id in MODEL_SPECS])
    peak_rss_after_models = peak_rss_bytes()

    sensor_id = service.sensor_ids[0]
    origin = service.default_origin.to_pydatetime()
    model_id = "fedavg"
    _, first_forecast_seconds = timed(lambda: service.predict(sensor_id, model_id, origin))
    forecast_samples = []
    for _ in range(10):
        _, duration = timed(lambda: service.predict(sensor_id, model_id, origin))
        forecast_samples.append(duration)

    evaluation, evaluation_seconds = timed(service.evaluation)
    peak_rss_after_evaluation = peak_rss_bytes()
    _, cached_evaluation_seconds = timed(service.evaluation)

    # AppTest starts a fresh Streamlit script context in this process. This is a
    # cold dashboard-render proxy, not a production browser/server latency test.
    st.cache_resource.clear()
    app_path = ROOT / "app/dashboard.py"
    app, dashboard_seconds = timed(lambda: AppTest.from_file(str(app_path), default_timeout=120).run())
    if app.exception:
        raise RuntimeError(f"Dashboard AppTest failed: {app.exception[0].message}")

    # Include a separate frame size only as a useful reproducibility context;
    # it is not a process memory measurement.
    frame_bytes = int(service.frame.memory_usage(index=True, deep=True).sum())

    result = {
        "measurement_date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "streamlit": st.__version__,
            "cpu_count_logical": os.cpu_count(),
        },
        "method": {
            "clock": "time.perf_counter (wall-clock elapsed)",
            "dashboard_cold_start": "one Streamlit AppTest render after clearing Streamlit resource cache; excludes OS/process startup and is not deployment latency",
            "service_and_artifact_load": "one new TrafficService followed by first load of all four saved model artifacts",
            "selected_forecast": "one first prediction, then ten repeated warm predictions for the service default held-out window and sensor 773869-equivalent first sensor",
            "evaluation": "first full test evaluation, then one cached evaluation call",
            "memory": "resource.getrusage maximum resident set size after each stage, normalized to bytes; process-wide high-water mark",
        },
        "sample_count": {
            "dashboard_cold_start": 1,
            "service_initialization": 1,
            "all_model_artifacts_load": 1,
            "selected_forecast_first": 1,
            "selected_forecast_warm": len(forecast_samples),
            "full_evaluation_first": 1,
            "full_evaluation_cached": 1,
            "rss_snapshots": 2,
        },
        "timings_seconds": {
            "dashboard_cold_apptest_render": dashboard_seconds,
            "service_initialization_data_and_metadata": service_seconds,
            "all_four_model_artifacts_load": artifact_seconds,
            "selected_forecast_first": first_forecast_seconds,
            "selected_forecast_warm_median": statistics.median(forecast_samples),
            "selected_forecast_warm_min": min(forecast_samples),
            "selected_forecast_warm_max": max(forecast_samples),
            "full_test_evaluation_initialization": evaluation_seconds,
            "full_test_evaluation_cached": cached_evaluation_seconds,
        },
        "memory_bytes": {
            "process_peak_rss_after_models": peak_rss_after_models,
            "process_peak_rss_after_evaluation": peak_rss_after_evaluation,
            "dataframe_deep_size": frame_bytes,
            "note": "RSS includes interpreter and libraries; this is a process-wide high-water mark, not attributable incremental memory.",
        },
        "evaluated_test_shape": evaluation["test_shape"],
        "selected_forecast": {
            "sensor_id": sensor_id,
            "model_id": model_id,
            "forecast_origin": origin.isoformat(),
        },
        "dashboard_rendered_without_exception": not bool(app.exception),
    }
    output = ROOT / "reports/efficiency-measurements.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
