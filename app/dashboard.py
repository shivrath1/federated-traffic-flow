"""Research dashboard for held-out METR-LA traffic forecasts."""

from __future__ import annotations

import sys
from datetime import date, datetime, time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from fastapi import HTTPException

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.api.app import TrafficService

MODEL_COLORS = {
    "Centralized baseline": "#4f6bed",
    "Centralized with noise augmentation": "#9b72cf",
    "FedAvg without augmentation": "#078b83",
    "FedAvg with noise augmentation": "#d18a28",
    "Persistence baseline": "#7a8496",
}

st.set_page_config(
    page_title="Federated Traffic Flow Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      .stApp { background: #f4f6fa; color: #19263c; }
      html, body, [class*="css"] { font-family: 'Avenir Next', 'Segoe UI', system-ui, sans-serif; }
      h1, h2, h3 { font-family: 'Avenir Next', 'Segoe UI', system-ui, sans-serif; color: #172742; letter-spacing: -0.025em; }
      .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
      .hero { padding: 1.5rem 1.8rem; border-radius: 18px; color: white;
              background: linear-gradient(115deg,#132744 0%,#1d4260 62%,#087e7b 100%); margin-bottom: 1rem; }
      .hero .sub { color: #d5e8ed; font-size: 1.02rem; margin-bottom: .8rem; }
      .hero .desc { color: #f2f7f8; max-width: 1100px; font-size: .96rem; line-height: 1.55; }
      .section-kicker { color: #087e7b; text-transform: uppercase; letter-spacing: .1em;
                         font-weight: 700; font-size: .75rem; margin-bottom: -.55rem; }
      .soft-panel { padding: 1rem 1.15rem; background: white; border: 1px solid #e3e8ef;
                    border-radius: 14px; min-height: 95px; }
      .soft-panel b { color: #172742; }
      .muted { color: #566579; }
      [data-testid="stMetric"] { background: white; border: 1px solid #e4e9f0; border-radius: 12px; padding: .75rem 1rem; }
      div[data-testid="stDataFrame"] { border: 1px solid #e4e9f0; border-radius: 12px; overflow: hidden; }
      div.stButton > button { border-radius: 9px; border: 1px solid #d8e0e9; color: #233a56; }
      div.stButton > button:hover { border-color: #078b83; color: #087e7b; }
      hr { border-color: #e1e6ed; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Federated Traffic Flow Prediction")


@st.cache_resource(show_spinner=False)
def load_research_service() -> TrafficService:
    """Load tracked METR-LA data and saved artifacts once per Streamlit process."""
    return TrafficService(
        REPOSITORY_ROOT / "data/raw/metr-la.h5",
        REPOSITORY_ROOT / "models",
        REPOSITORY_ROOT / "data/processed/metr_la",
    )


def section_heading(kicker: str, heading: str, explanation: str | None = None) -> None:
    st.markdown(f"<div class='section-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"## {heading}")
    if explanation:
        st.markdown(f"<div class='muted'>{explanation}</div>", unsafe_allow_html=True)


def chart_line(data: pd.DataFrame, *, x: str, y: str, color: str, tooltip: list[str], height: int = 300) -> alt.Chart:
    return alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        x=alt.X(x, title=None, axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y(y, title=None, scale=alt.Scale(zero=False)),
        color=alt.Color(color, scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))),
        tooltip=tooltip,
    ).properties(height=height).interactive()


try:
    with st.spinner("Loading METR-LA data, saved models, and held-out evaluation results…"):
        research = load_research_service()
        sensor_data = {"sensor_ids": research.sensor_ids}
        model_data = research.model_list()
        options = research.forecast_options()
        evaluation = research.evaluation()
except (HTTPException, RuntimeError, OSError, ValueError) as exc:
    detail = getattr(exc, "detail", str(exc))
    st.error(f"Could not load the research data or saved model artifacts: {detail}")
    st.stop()

available_models = [item for item in model_data["models"] if item["available"]]
if not available_models:
    st.error("No saved model artifacts are available in this repository checkout.")
    st.stop()

default_origin = datetime.fromisoformat(options["default_forecast_origin"])
if "forecast_time" not in st.session_state:
    st.session_state["forecast_time"] = default_origin.time()

st.markdown(
    """
    <div class="hero">
      <div class="sub"><i>Evaluating privacy-conscious collaborative traffic forecasting across distributed sensor groups.</i></div>
      <div class="desc">Traffic forecasting can help researchers understand changing road conditions and support transportation planning. Centralized learning gathers traffic records in one place. Federated learning instead lets simulated client groups train locally and share model updates, rather than pooling their raw training arrays. This project evaluates that approach on historical METR-LA by comparing centralized models, FedAvg, and noise-augmented variants.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    "Research prototype using historical METR-LA observations from 2012. Forecasts are evaluated on held-out historical windows; this is not a live traffic service. The clients are simulated on one machine, and the experiment does not provide formal privacy guarantees.",
    icon="ℹ️",
)

overview = st.columns(3)
for column, title, text in zip(
    overview,
    ("Research question", "What is federated here?", "What is being tested?"),
    (
        "Can collaborative training produce useful one-hour traffic-speed forecasts?",
        "Four sensor groups train locally in a single-machine simulation; the coordinator averages model parameters with FedAvg.",
        "Four trained configurations share the same held-out test split. Noise augmentation is evaluated as an experiment, not assumed to help.",
    ),
):
    with column:
        st.markdown(f"<div class='soft-panel'><b>{title}</b><br><span class='muted'>{text}</span></div>", unsafe_allow_html=True)

st.write("")
section_heading("01 · Forecast explorer", "Choose a held-out traffic window", "Select a sensor, saved model, and forecast start time. The date picker is restricted to the test period so every displayed forecast can be checked against actual held-out readings.")
choice_left, choice_right = st.columns([1, 1])
with choice_left:
    sensor_id = st.selectbox("Traffic sensor", sensor_data["sensor_ids"], index=0, help="Each model uses the same shared temporal weights for the selected sensor.")
with choice_right:
    models_by_name = {item["name"]: item for item in available_models}
    model_names = list(models_by_name)
    default_model_index = next((i for i, item in enumerate(model_names) if models_by_name[item]["model_id"] == "fedavg"), 0)
    selected_model_name = st.selectbox("Forecast model", model_names, index=default_model_index)
selected_model = models_by_name[selected_model_name]
model_id = selected_model["model_id"]

st.markdown("**Quick time presets** · presets set the forecast start on the date selected below.")
preset_cols = st.columns(4)
for col, preset in zip(preset_cols, options["presets"]):
    with col:
        if st.button(preset["label"], key=f"preset_{preset['id']}", use_container_width=True):
            st.session_state["forecast_time"] = time.fromisoformat(preset["time"])
            st.rerun()

start_dt = datetime.fromisoformat(options["held_out_test_origin_start"])
end_dt = datetime.fromisoformat(options["held_out_test_origin_end"])
date_col, time_col, horizon_col = st.columns([1.2, 1, 1.2])
with date_col:
    selected_date: date = st.date_input(
        "Historical test date",
        value=default_origin.date(),
        min_value=start_dt.date(),
        max_value=end_dt.date(),
        help="Only dates in the chronological held-out test period are available.",
    )
with time_col:
    selected_time: time = st.time_input("Forecast start time", key="forecast_time", step=300)
with horizon_col:
    st.metric("Forecast horizon", f"{options['forecast_horizon_minutes']} minutes", "12 readings · 5 minutes apart")

if selected_time.minute % options["sampling_interval_minutes"] != 0 or selected_time.second != 0:
    st.error("Choose a time aligned to METR-LA's five-minute sampling interval (for example, 08:00 or 08:05).")
    st.stop()
forecast_origin = datetime.combine(selected_date, selected_time)
forecast_origin_iso = forecast_origin.isoformat()
st.caption(
    f"Forecast origin: **{forecast_origin_iso}** (first predicted reading) · "
    f"last input reading: {forecast_origin - pd.Timedelta(minutes=5)} · "
    f"test origins available: {start_dt.isoformat()} to {end_dt.isoformat()}"
)

try:
    with st.spinner(f"Evaluating sensor {sensor_id} with {selected_model['name']}…"):
        forecast_result = research.predict(sensor_id, model_id, forecast_origin)
except (HTTPException, RuntimeError, OSError, ValueError) as exc:
    detail = getattr(exc, "detail", str(exc))
    st.error(f"This forecast window could not be evaluated: {detail}")
    st.stop()

st.markdown(f"### Sensor {sensor_id} · {forecast_result['model_name']}")
st.caption(
    f"Held-out origin {forecast_result['forecast_origin']} · actual target interval "
    f"{forecast_result['actuals'][0]['timestamp']}–{forecast_result['actuals'][-1]['timestamp']}"
)

history_rows = [
    {"timestamp": x["timestamp"], "series": "Earlier history", "value": x["value"], "actual": None, "predicted": None}
    for x in forecast_result["history_readings"][:-12]
]
input_rows = [
    {"timestamp": x["timestamp"], "series": "Model input · previous 60 min", "value": x["value"], "actual": None, "predicted": None}
    for x in forecast_result["input_readings"]
]
actual_by_time = {item["timestamp"]: item["value"] for item in forecast_result["actuals"]}
prediction_by_time = {item["timestamp"]: item["value"] for item in forecast_result["forecasts"]}
actual_rows = [
    {"timestamp": x["timestamp"], "series": "Held-out actual", "value": x["value"],
     "actual": x["value"], "predicted": prediction_by_time[x["timestamp"]]}
    for x in forecast_result["actuals"]
]
prediction_rows = [
    {"timestamp": x["timestamp"], "series": "Model forecast", "value": x["value"],
     "actual": actual_by_time[x["timestamp"]], "predicted": x["value"]}
    for x in forecast_result["forecasts"]
]
window_chart_data = pd.DataFrame(history_rows + input_rows + actual_rows + prediction_rows)
window_chart_data["timestamp"] = pd.to_datetime(window_chart_data["timestamp"])
window_series = ["Earlier history", "Model input · previous 60 min", "Held-out actual", "Model forecast"]
window_colors = ["#8c9aab", "#087e7b", "#d18a28", "#c04d67"]
window_line = alt.Chart(window_chart_data).mark_line(point=alt.OverlayMarkDef(size=42, filled=True)).encode(
    x=alt.X("timestamp:T", title="Historical time → forecast time", axis=alt.Axis(format="%b %d, %H:%M", labelAngle=-25, grid=False)),
    y=alt.Y("value:Q", title="Traffic speed (dataset units)", scale=alt.Scale(zero=False)),
    color=alt.Color("series:N", title="Series", scale=alt.Scale(domain=window_series, range=window_colors)),
    tooltip=[
        alt.Tooltip("timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
        alt.Tooltip("series:N", title="Series"),
        alt.Tooltip("value:Q", title="This series", format=".2f"),
        alt.Tooltip("actual:Q", title="Held-out actual", format=".2f"),
        alt.Tooltip("predicted:Q", title="Model prediction", format=".2f"),
    ],
)
origin_rule = alt.Chart(pd.DataFrame({"timestamp": [pd.Timestamp(forecast_result["forecast_origin"])]})).mark_rule(color="#172742", strokeDash=[5, 4], strokeWidth=2).encode(
    x=alt.X("timestamp:T", title="Forecast origin"),
    tooltip=[alt.Tooltip("timestamp:T", title="Forecast begins", format="%Y-%m-%d %H:%M")],
)
st.altair_chart((window_line + origin_rule).properties(height=390, title=f"{forecast_result['model_name']} · 60-minute held-out forecast").interactive(), use_container_width=True)
st.caption("The dashed line marks the first predicted time. The teal line is the exact 12-reading input window; earlier readings provide visual context. Actual future values are shown only to evaluate the forecast, never as model input.")

window_metrics = forecast_result["selected_window_metrics"]
persistence_metrics = forecast_result["persistence_baseline"]["metrics"]
mae_gain = 100 * (persistence_metrics["mae"] - window_metrics["mae"]) / persistence_metrics["mae"] if persistence_metrics["mae"] else 0.0
rmse_gain = 100 * (persistence_metrics["rmse"] - window_metrics["rmse"]) / persistence_metrics["rmse"] if persistence_metrics["rmse"] else 0.0
st.markdown("#### How accurate was this selected forecast?")
st.caption("Lower error is better. These metrics use this one sensor and these same 12 held-out targets; they are not the overall test-set scores.")
metrics_cols = st.columns(4)
metrics_cols[0].metric("Selected-window MAE", f"{window_metrics['mae']:.2f}", f"{mae_gain:+.1f}% vs persistence")
metrics_cols[1].metric("Selected-window RMSE", f"{window_metrics['rmse']:.2f}", f"{rmse_gain:+.1f}% vs persistence")
metrics_cols[2].metric("Persistence MAE", f"{persistence_metrics['mae']:.2f}", "repeat last input")
metrics_cols[3].metric("Persistence RMSE", f"{persistence_metrics['rmse']:.2f}", "same 12 targets")

step_frame = pd.DataFrame({
    "Forecast step": [f"+{i * 5} min" for i in range(1, 13)],
    "Forecast time": [x["timestamp"] for x in forecast_result["forecasts"]],
    "Held-out actual": [x["value"] for x in forecast_result["actuals"]],
    "Model prediction": [x["value"] for x in forecast_result["forecasts"]],
    "Persistence prediction": [x["value"] for x in forecast_result["persistence_baseline"]["forecasts"]],
    "Absolute model error": window_metrics["mae_by_step"],
})
with st.expander("Inspect each five-minute forecast step", expanded=False):
    st.dataframe(step_frame, use_container_width=True, hide_index=True, column_config={
        "Held-out actual": st.column_config.NumberColumn(format="%.2f"),
        "Model prediction": st.column_config.NumberColumn(format="%.2f"),
        "Persistence prediction": st.column_config.NumberColumn(format="%.2f"),
        "Absolute model error": st.column_config.NumberColumn(format="%.2f"),
    })

st.divider()
section_heading("02 · Held-out results", "How do the trained models compare?", "These are the saved full test-set scores over the same 5,131 windows, 12 forecast steps, and 207 sensors. Time-of-day, sensor-specific, and persistence diagnostics below are newly computed from that same test split.")
overall_rows = [{"Model": item["name"], "MAE": item["overall"]["mae"], "RMSE": item["overall"]["rmse"], "Source": "Saved model result"} for item in evaluation["models"]]
persistence_eval = evaluation["persistence_baseline"]
overall_rows.append({"Model": "Persistence baseline", "MAE": persistence_eval["overall"]["mae"], "RMSE": persistence_eval["overall"]["rmse"], "Source": "New calculation on saved test split"})
overall_frame = pd.DataFrame(overall_rows)
all_metric_cols = st.columns(2)
for column, metric, title in zip(all_metric_cols, ("MAE", "RMSE"), ("Average absolute error", "Root mean squared error")):
    plot = alt.Chart(overall_frame).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("Model:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-24, labelLimit=190)),
        y=alt.Y(f"{metric}:Q", title=f"{metric} · dataset speed units", scale=alt.Scale(zero=True)),
        color=alt.Color("Model:N", scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())), legend=None),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip(f"{metric}:Q", format=".4f"), alt.Tooltip("Source:N")],
    ).properties(height=300, title=title)
    with column:
        st.altair_chart(plot, use_container_width=True)
st.dataframe(overall_frame.style.format({"MAE": "{:.6f}", "RMSE": "{:.6f}"}), hide_index=True, use_container_width=True)
st.caption("MAE averages absolute errors. RMSE takes the square root of average squared errors and puts more weight on large misses. Both aggregate equally over each window × forecast step × sensor value; lower is better.")

mae_by_horizon_rows, rmse_by_horizon_rows = [], []
for item in evaluation["models"]:
    for step, value in enumerate(item["mae_by_horizon"], start=1):
        mae_by_horizon_rows.append({"Forecast lead": step * 5, "Model": item["name"], "Error": value})
    for step, value in enumerate(item["rmse_by_horizon"], start=1):
        rmse_by_horizon_rows.append({"Forecast lead": step * 5, "Model": item["name"], "Error": value})
for step, (mae, rmse) in enumerate(zip(persistence_eval["mae_by_horizon"], persistence_eval["rmse_by_horizon"]), start=1):
    mae_by_horizon_rows.append({"Forecast lead": step * 5, "Model": "Persistence baseline", "Error": mae})
    rmse_by_horizon_rows.append({"Forecast lead": step * 5, "Model": "Persistence baseline", "Error": rmse})
hz_left, hz_right = st.columns(2)
for column, rows, label in ((hz_left, mae_by_horizon_rows, "MAE by forecast lead"), (hz_right, rmse_by_horizon_rows, "RMSE by forecast lead")):
    with column:
        st.altair_chart(chart_line(pd.DataFrame(rows), x="Forecast lead:Q", y="Error:Q", color="Model:N", tooltip=["Forecast lead:Q", "Model:N", "Error:Q"]).properties(title=label), use_container_width=True)

tod_col, sensor_col = st.columns([1, 1])
with tod_col:
    st.markdown("#### Error by time of day")
    tod_metric = st.radio("Time-of-day metric", ["MAE", "RMSE"], horizontal=True, key="tod_metric")
    tod_rows = []
    for item in evaluation["models"]:
        for point in item["by_hour"]:
            if point[tod_metric.lower()] is not None:
                tod_rows.append({"Forecast origin hour": point["hour"], "Model": item["name"], "Error": point[tod_metric.lower()]})
    for point in persistence_eval["by_hour"]:
        if point[tod_metric.lower()] is not None:
            tod_rows.append({"Forecast origin hour": point["hour"], "Model": "Persistence baseline", "Error": point[tod_metric.lower()]})
    st.altair_chart(chart_line(pd.DataFrame(tod_rows), x="Forecast origin hour:Q", y="Error:Q", color="Model:N", tooltip=["Forecast origin hour:Q", "Model:N", "Error:Q"]).properties(title=f"Held-out {tod_metric} by forecast start hour"), use_container_width=True)
    st.caption("Each test window is grouped by the hour of its first target reading. Windows overlap as in the original rolling-window test evaluation.")

with sensor_col:
    st.markdown(f"#### Held-out results for sensor {sensor_id}")
    sensor_rows = []
    for item in evaluation["models"]:
        sensor_stats = next(record for record in item["by_sensor"] if record["sensor_id"] == sensor_id)
        sensor_rows.append({"Model": item["name"], "MAE": sensor_stats["mae"], "RMSE": sensor_stats["rmse"]})
    baseline_sensor = next(record for record in persistence_eval["by_sensor"] if record["sensor_id"] == sensor_id)
    sensor_rows.append({"Model": "Persistence baseline", "MAE": baseline_sensor["mae"], "RMSE": baseline_sensor["rmse"]})
    st.dataframe(pd.DataFrame(sensor_rows).style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}"}), hide_index=True, use_container_width=True)
    st.caption("Per-sensor MAE/RMSE are newly computed across every held-out test window and forecast lead for this sensor.")

section_heading("03 · What the results say", "A trade-off, not a universal augmentation win")
st.markdown(
    ""
    "Across the full test set, FedAvg has the lowest MAE (5.522869), while the centralized baseline has the lowest RMSE among unaugmented models (11.969190 vs 11.973663). Noise augmentation **increased MAE but reduced RMSE** in both training settings. Centralized augmentation moved MAE from 5.698667 to 5.790541 and RMSE from 11.969190 to 11.871505. FedAvg augmentation moved MAE from 5.522869 to 5.665383 and RMSE from 11.973663 to 11.883041. Each run used one seed, so these results are descriptive; they do not establish a reliable general improvement."
)
st.markdown(
    "The persistence baseline is an additional reference that repeats the last observed input for the next 12 steps. The selected-window scores above use the same actual test targets for both the chosen model and persistence, making it easier to judge whether the learned forecast adds value for this one sensor and time."
)

st.divider()
section_heading("04 · Methods and limitations", "Technical details", "Expand a section for implementation and evaluation details.")
with st.expander("Dataset and chronological evaluation"):
    st.markdown(
        """
        - **Dataset:** METR-LA, 207 sensors and 34,272 timestamps, from 2012-03-01 00:00 to 2012-06-27 23:55.
        - **Sampling:** one reading every five minutes. Values are shown in dataset speed units; no unit conversion is applied here.
        - **Splits:** chronological 70% train, 15% validation, 15% test. The held-out test split begins at 2012-06-10 03:30.
        - **Forecasting:** 12 readings (one hour) as input and the following 12 readings (one hour) as target.
        - **Evaluation:** saved overall metrics aggregate every test window × forecast lead × sensor (5,131 × 12 × 207 = 12,745,404 scalar predictions). Checkpoint selection used validation, not test data.
        - **Selected-window validation:** dates are restricted to test origins with all 12 future targets present. The chart's actual line comes from those held-out targets; model inputs stop one reading before the forecast origin.
        """
    )
with st.expander("Model architecture and training"):
    st.markdown(
        """
        All models share one per-sensor linear temporal map: a 12 × 12 weight matrix and 12-element bias. It applies the same learned transformation to each sensor independently. The model has no spatial graph or cross-sensor interactions. Centralized training uses NumPy Adam, up to 30 epochs, 256-window batches, learning rate 0.001, and validation-MSE checkpoint selection. The centralized augmented run uses the same training schedule and one synthetic copy per training window.
        """
    )
    for model in model_data["models"]:
        if model["available"]:
            with st.container():
                st.markdown(f"**{model['name']}**")
                st.json(model.get("configuration", {}))
with st.expander("Federated simulation and communication assumptions"):
    st.markdown(
        """
        - Four clients are simulated sensor groups on one machine, with 52, 52, 52, and 51 sensors. They do not represent four real organizations or devices.
        - Each client trains locally for one epoch per communication round. The coordinator aggregates weights and bias using FedAvg, weighted by local sensor-window example count.
        - FedAvg runs 30 rounds; client Adam state resets for each local invocation. Only model parameters are aggregated in the simulation; no real network transport is used.
        - The 156-parameter model state is 624 float32 bytes. Four clients exchanging one model state in each direction for 30 rounds corresponds to 149,760 model-payload bytes, excluding protocol overhead.
        - Federated learning alone does not guarantee privacy. The project has no differential privacy, secure aggregation, encryption protocol, or formal privacy analysis.
        """
    )
with st.expander("Noise augmentation and interpretation"):
    st.markdown(
        """
        The simplified augmentation adds one independent Gaussian-jittered copy of each training input. Noise scale is 5% of each sensor's training-input standard deviation; targets remain unchanged and perturbed inputs are clipped at zero. Noise statistics come from training inputs only. The centralized run uses seed 42; client-local FedAvg augmentation uses seeds 42–45. This is not the paper's federated diffusion-based generator and does not reproduce FedTPS. One-seed metrics show mixed augmentation effects, so no consistent improvement is claimed.
        """
    )
with st.expander("Evaluation metric sources"):
    st.markdown(
        """
        - **Saved overall test MAE/RMSE and error by horizon:** read from each existing model's saved `metrics.json`; these published model results are not recomputed or overwritten by the dashboard.
        - **Selected-window errors:** newly calculated for one sensor and the selected 12 held-out targets.
        - **Persistence baseline:** newly calculated from the same test inputs/targets; for every window it repeats that window's final input reading.
        - **Time-of-day and per-sensor diagnostics:** newly calculated from the existing test arrays using the same rolling test windows. They are diagnostic views only and are not used to retrain, tune, or select checkpoints.
        """
    )

st.caption("Federated Traffic Flow Prediction · METR-LA historical research prototype · No live traffic feed · No formal privacy guarantee")
