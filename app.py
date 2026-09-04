from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import load_config
from data.synthetic_generator import DEMO_SCENARIOS, MODELS
from inference.pipeline import AegisPipeline
from ui.components import CSS, SCENARIO_LABELS, VARIABLE_META, chart, download_summary, fmt, pct, selected_cell_frame, status_badge
from verification.benchmarks import benchmark, benchmark_by_lead
from visualization.charts import line_comparison
from visualization.maps import heatmap

st.set_page_config(page_title="AEGIS-WX | SIH26081", page_icon="🌦️", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

NAVIGATION = ["Overview", "NWP Inputs", "Model Disagreement", "Weather Regime", "AI Weighting Engine", "Final Forecast", "Uncertainty & Alerts", "Explainability", "Verification", "System Information"]
PRESENTATION_NAVIGATION = ["Overview", "NWP Inputs", "Model Disagreement", "AI Weighting Engine", "Final Forecast", "Uncertainty & Alerts", "Verification"]


def initialise_state() -> None:
    for key, value in {"scenario": "NORMAL", "scenario_seed": 26081, "selected_lead": 24, "fallback_threshold": .75, "presentation_mode": False}.items():
        st.session_state.setdefault(key, value)


@st.cache_data(show_spinner=False, max_entries=32)
def run_demo(seed: int, scenario: str, data_mode: str, fallback_threshold: float) -> dict:
    config = load_config()
    config["thresholds"]["uncertainty"] = fallback_threshold
    return AegisPipeline(config).run(seed=seed, mode=data_mode, scenario=scenario)


def map_scale(fields: list[np.ndarray], enabled: bool) -> tuple[float | None, float | None]:
    if not enabled:
        return None, None
    values = np.concatenate([field[np.isfinite(field)].ravel() for field in fields if np.isfinite(field).any()])
    return (float(np.percentile(values, 2)), float(np.percentile(values, 98))) if len(values) else (None, None)


def forecast_csv(s: object, field: np.ndarray, lead_index: int, label: str) -> bytes:
    lat, lon = np.meshgrid(s.latitude, s.longitude, indexing="ij")
    return pd.DataFrame({"latitude": lat.ravel(), "longitude": lon.ravel(), label: field[lead_index].ravel()}).to_csv(index=False).encode()


def weights_csv(s: object, weights: np.ndarray, lead_index: int) -> bytes:
    lat, lon = np.meshgrid(s.latitude, s.longitude, indexing="ij")
    frame = pd.DataFrame({"latitude": lat.ravel(), "longitude": lon.ravel()})
    for index, model in enumerate(MODELS):
        frame[f"{model}_weight"] = weights[lead_index, index].ravel()
    return frame.to_csv(index=False).encode()


def render_hero(result: dict, lead: int, output: dict) -> None:
    st.markdown("""<div class='hero'><div class='eyebrow'>SIH26081 · Ministry of Earth Sciences</div><h1>AEGIS-WX</h1><p>Hybrid AI–NWP Multi-Model Forecast Blending System</p><p>Adaptive, uncertainty-aware multi-model weather forecasting.</p><p class='hero-explainer'>AEGIS-WX combines multiple numerical weather prediction models and dynamically adjusts their influence based on weather regime, forecast lead time, disagreement and confidence.</p></div>""", unsafe_allow_html=True)
    steps = [
        ("NWP Inputs", "Forecasts from multiple weather models are used as input."),
        ("Feature Analysis", "The system extracts local spread, bias and lead-time behaviour."),
        ("Model Disagreement", "Measures how strongly the forecast models disagree."),
        ("Weather Regime", "Identifies patterns such as normal, monsoon, cyclone or heatwave."),
        ("Adaptive Weighting", "Assigns more importance to models expected to perform better now."),
        ("Final Forecast", "Combines weighted model forecasts into one AEGIS-WX prediction."),
        ("Verification", "Compares forecasts against synthetic hindcast truth in demo mode."),
    ]
    st.markdown("<div class='flow'>" + "".join(f"<div class='flow-step' title='{tip}'><b>{i+1:02d}</b><span>{name}</span></div>" for i, (name, tip) in enumerate(steps)) + "</div>", unsafe_allow_html=True)
    st.caption("DEMO MODE — This prototype uses synthetic NWP-style forecasts and synthetic hindcast truth to demonstrate the AEGIS-WX architecture. Results are not operational weather forecasts.")
    columns = st.columns(6)
    metrics = [("Scenario", SCENARIO_LABELS[result["scenario"].scenario], None), ("Detected Regime", result["detected_regime"], None), ("Forecast Lead", f"{lead}h", None), ("Blend Status", "ACTIVE" if output["fallback_status"] == "ADAPTIVE BLEND ACTIVE" else "FALLBACK", "Adaptive model weights are currently being used."), ("Domain Mean Confidence", pct(float(output["confidence"].mean())), "Average confidence across the complete forecast grid."), ("Runtime", fmt(result["runtime_seconds"], 2, " s"), None)]
    for column, (label, value, help_text) in zip(columns, metrics):
        column.metric(label, value, help=help_text)


def render_final(result: dict, lead_index: int, lat_i: int, lon_i: int, compact: bool = False) -> None:
    s, rain = result["scenario"], result["outputs"]["precipitation"]
    status_badge(rain["fallback_status"], rain["fallback_reasons"], rain["fallback_fraction"])
    st.markdown("<div class='section-kicker'>Final AEGIS-WX Output</div>", unsafe_allow_html=True)
    st.subheader("Bias-aware blended weather forecast with confidence information")
    with st.container(horizontal=True):
        for label in ("Rainfall", "Temperature", "Wind", "Confidence"):
            st.badge(label, color="blue")
    items = [("AEGIS-WX Rainfall", rain["final"][lead_index, lat_i, lon_i], "mm"), *[(f"{model} Rainfall", rain["members"][lead_index, i, lat_i, lon_i], "mm") for i, model in enumerate(MODELS)], ("Selected Location Confidence", rain["confidence"][lead_index, lat_i, lon_i], "%"), ("Uncertainty (0–1)", rain["uncertainty"][lead_index, lat_i, lon_i], ""), ("Detected Regime", result["detected_regime"], "")]
    for index, (column, (label, value, unit)) in enumerate(zip(st.columns(7), items)):
        shown = pct(float(value)) if label == "Selected Location Confidence" else fmt(float(value), 2, f" {unit}" if unit else "") if isinstance(value, (float, np.floating)) else str(value)
        card_class = " primary-forecast" if index == 0 else ""
        column.markdown(f"<div class='forecast-card{card_class}'><small>{label}</small><strong>{shown}</strong></div>", unsafe_allow_html=True)
    maps = [("precipitation", "AEGIS-WX Rainfall Forecast"), ("temperature_2m", "AEGIS-WX Temperature Forecast"), ("wind_speed_10m", "AEGIS-WX Wind Forecast"), ("confidence", "Forecast Confidence")]
    for column, (variable, title) in zip(st.columns(2 if compact else 4), maps):
        if variable == "confidence":
            chart(column, heatmap(rain["confidence"][lead_index] * 100, s.latitude, s.longitude, title, "Greens", "%", 0, 100, 300))
        else:
            _, unit, colorscale = VARIABLE_META[variable]
            chart(column, heatmap(result["outputs"][variable]["final"][lead_index], s.latitude, s.longitude, title, colorscale, unit, height=300))


def render_inputs(result: dict, variable: str, lead_index: int, same_scale: bool) -> None:
    s = result["scenario"]
    label, unit, colorscale = VARIABLE_META[variable]
    st.header("NWP Inputs")
    st.caption("What you are seeing: three GFS-, ECMWF- and NCUM-style source forecasts before AEGIS-WX blends them. These are not live operational feeds.")
    vmin, vmax = map_scale([s.forecasts[model][variable][lead_index] for model in MODELS], same_scale)
    for column, model in zip(st.columns(3), MODELS):
        field = s.forecasts[model][variable][lead_index]
        if np.isfinite(field).any():
            chart(column, heatmap(field, s.latitude, s.longitude, f"{model} {label}", colorscale, unit, vmin, vmax))
        else:
            column.warning(f"{model} input is intentionally unavailable in this scenario. AEGIS-WX has switched to safe two-model blending.")
    st.caption("Common colour scale is " + ("enabled for direct comparison." if same_scale else "disabled; enable it in the sidebar to compare sources directly."))


def render_disagreement(result: dict, variable: str, lead_index: int) -> None:
    s, members = result["scenario"], result["outputs"][variable]["members"][lead_index]
    mean, std = np.nanmean(members, axis=0), np.nanstd(members, axis=0)
    spread = np.nanmax(members, axis=0) - np.nanmin(members, axis=0)
    score = np.clip(std / max(float(np.nanpercentile(std, 95)), .01), 0, 1)
    value = float(np.nanmean(score))
    explanation = "Low — models broadly agree." if value < .35 else "Moderate — some divergence exists." if value < .65 else "High — forecast sources strongly disagree."
    st.header("Model Disagreement")
    for column, (label, value) in zip(st.columns(4), [("Mean forecast", float(np.nanmean(mean))), ("Mean standard deviation", float(np.nanmean(std))), ("Maximum difference", float(np.nanmax(spread))), ("Normalized score", value)]):
        column.metric(label, pct(value) if label == "Normalized score" else fmt(value, 2))
    st.markdown(f"<div class='evidence'><b>Disagreement: {explanation}</b> The normalized score is {pct(value)}. Higher disagreement lowers confidence and can activate the safety fallback in affected cells.</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    chart(left, heatmap(std, s.latitude, s.longitude, "Forecast standard deviation", "Magma", height=360))
    chart(right, heatmap(score, s.latitude, s.longitude, "Normalized disagreement score", "Magma", "0–1", 0, 1, 360))


def render_regime(result: dict) -> None:
    probabilities = pd.DataFrame({"Regime": list(result["regime_probabilities"]), "Probability": list(result["regime_probabilities"].values())}).sort_values("Probability", ascending=False)
    selected = result["detected_regime"]
    st.header("Weather Regime")
    left, right = st.columns([1, 1.8])
    top_probability = float(probabilities.iloc[0]["Probability"])
    left.success(f"Detected regime: {selected}")
    left.metric("Regime confidence", pct(top_probability))
    left.write(f"The current pattern most closely matches **{selected.lower()}** conditions. Transparent rule-based detection is used; this prototype does not claim a trained ML classifier.")
    features = result["regime_features"]
    with left.expander("Advanced Details"):
        left.dataframe(pd.DataFrame({"Feature": [key.replace("_", " ").title() for key in features], "Value": [fmt(value, 2) for value in features.values()]}), hide_index=True, width="stretch")
    figure = px.bar(probabilities, x="Probability", y="Regime", orientation="h", text=probabilities["Probability"].map(lambda value: pct(value, 1)), color="Regime", color_discrete_sequence=["#41c7a1" if item == selected else "#397da8" for item in probabilities["Regime"]]).update_layout(showlegend=False, xaxis_tickformat=".0%", paper_bgcolor="#0b1c2d", plot_bgcolor="#0b1c2d", font={"color": "#eaf3fb"})
    chart(right, figure)


def render_weights(result: dict, variable: str, lead_index: int, lat_i: int, lon_i: int) -> None:
    s, output = result["scenario"], result["outputs"][variable]
    weights = output["weights"][lead_index, :, lat_i, lon_i]
    st.header("AI Weighting Engine")
    st.caption("What you are seeing: how much each source forecast contributes at the selected location, after regime, lead time, local disagreement and synthetic-hindcast bias information are considered.")
    st.info("Adaptive Weighting Prototype · transparent regime priors, local disagreement, lead time and current synthetic-hindcast bias correction drive the weights. This demonstrator does not claim a trained ML model.", icon="ℹ️")
    steps = ["NWP Forecasts", "Feature Engineering", "Model Disagreement", "Weather Regime", "Lead Time", "Local Bias / Historical Error", "Rule-based Weight Predictor", "Adaptive Weights", "AEGIS-WX Forecast"]
    st.markdown("<div class='flow'>" + "".join(f"<div class='flow-step'><b>{i+1:02d}</b><span>{name}</span></div>" for i, name in enumerate(steps)) + "</div>", unsafe_allow_html=True)
    frame = pd.DataFrame({"Model": MODELS, "Weight": weights})
    chart(st, px.bar(frame, x="Weight", y="Model", orientation="h", text=frame["Weight"].map(lambda value: pct(value, 1)), range_x=[0, 1], color="Model", color_discrete_sequence=["#6dd3ff", "#41c7a1", "#f7c65d"]).update_layout(showlegend=False, xaxis_tickformat=".0%", paper_bgcolor="#0b1c2d", plot_bgcolor="#0b1c2d", font={"color": "#eaf3fb"}))
    lead_model = MODELS[int(np.nanargmax(weights))]
    st.markdown(f"<div class='evidence'><b>{lead_model}</b> currently receives the highest local weight ({pct(float(np.nanmax(weights)), 1)}). Higher weights indicate greater trust for this grid cell, lead time and weather regime; the weights are softmax-normalized and sum to {pct(float(weights.sum()))}.</div>", unsafe_allow_html=True)
    for index, (column, model) in enumerate(zip(st.columns(3), MODELS)):
        chart(column, heatmap(output["weights"][lead_index, index], s.latitude, s.longitude, f"{model} spatial weight", "Blues", "%", 0, 1, 300))


def render_uncertainty(result: dict, lead_index: int) -> None:
    s, rain, wind, temp = result["scenario"], result["outputs"]["precipitation"], result["outputs"]["wind_speed_10m"], result["outputs"]["temperature_2m"]
    st.header("Uncertainty & Alerts")
    left, right = st.columns(2)
    chart(left, heatmap(rain["uncertainty"][lead_index], s.latitude, s.longitude, "Uncertainty", "Reds", "0–1", 0, 1))
    chart(right, heatmap(rain["confidence"][lead_index] * 100, s.latitude, s.longitude, "Confidence", "Greens", "%", 0, 100))
    confidence = float(rain["confidence"][lead_index].mean())
    category = "High confidence" if confidence >= .7 else "Moderate confidence" if confidence >= .45 else "Low confidence"
    st.markdown(f"<div class='evidence'><b>{category}.</b> Mean confidence is {pct(confidence)}. Confidence expresses model agreement and estimated error in this synthetic case; it is not a guarantee. The configured per-cell fallback threshold is {pct(st.session_state.fallback_threshold)} uncertainty.</div>", unsafe_allow_html=True)
    alerts = [("Heavy rain", rain["final"], 20., "mm"), ("High wind", wind["final"], 17., "m/s"), ("Heatwave", temp["final"], 40., "°C"), ("High uncertainty", rain["uncertainty"], st.session_state.fallback_threshold, "")]
    for column, (name, values, threshold, unit) in zip(st.columns(4), alerts):
        field, count = values[lead_index], int(np.sum(values[lead_index] >= threshold))
        no_alert = f"No active {name.lower()} alert in the selected forecast lead"
        message = f"{count} grid cells exceed the {name.lower()} threshold · max {fmt(float(np.nanmax(field)), 2, ' '+unit if unit else '')}" if count else no_alert
        (column.warning if count else column.success)(f"{name}\n\n{message}\n\nThreshold {fmt(threshold, 2, ' '+unit if unit else '')}")


def render_explain(result: dict, variable: str, lead_index: int, lat_i: int, lon_i: int) -> None:
    s, output = result["scenario"], result["outputs"][variable]
    _, unit, _ = VARIABLE_META[variable]
    members, weights = output["members"][lead_index, :, lat_i, lon_i], output["weights"][lead_index, :, lat_i, lon_i]
    top = int(np.nanargmax(weights)); consensus = float(np.nanmean(members)); deviation = abs(float(members[top])-consensus)
    st.header("Why did AEGIS-WX choose these weights?")
    st.caption(f"Selected location: {s.latitude[lat_i]:.1f}°N, {s.longitude[lon_i]:.1f}°E · {int(s.leads[lead_index])}h")
    left, right = st.columns([1.5, 1])
    table = selected_cell_frame(MODELS, members, weights, unit)
    left.dataframe(table.style.format({f"Forecast ({unit})": "{:.2f}", "Weight": "{:.1%}", f"Deviation ({unit})": "{:.2f}"}).highlight_max(subset=["Weight"], color="#164c41"), hide_index=True, width="stretch")
    right.metric("AEGIS-WX final", fmt(float(output["final"][lead_index, lat_i, lon_i]), 2, f" {unit}"))
    right.metric("Local disagreement", fmt(float(np.nanstd(members)), 2, f" {unit}"))
    right.metric("Confidence", pct(float(output["confidence"][lead_index, lat_i, lon_i])))
    st.markdown(f"<div class='evidence'><b>{MODELS[top]}</b> receives the highest weight ({pct(float(weights[top]), 1)}) because it is closest to the local consensus ({fmt(deviation, 2, ' '+unit)} difference), after the <b>{result['detected_regime']}</b> regime prior and {int(s.leads[lead_index])}h lead-time adjustment. The final value uses measured synthetic-hindcast bias correction.</div>", unsafe_allow_html=True)


def render_verification(result: dict, variable: str, lat_i: int, lon_i: int) -> None:
    s, output, table = result["scenario"], result["outputs"][variable], benchmark(result, variable)
    st.header("Demo Validation Against Baselines")
    st.caption("DEMO MODE — Comparison using synthetic hindcast truth. Scores are recomputed for the generated case and are not operational performance claims.")
    best = table.loc[table["RMSE"].idxmin()]
    best_rmse = table.loc[table["RMSE"].idxmin(), "Model"]
    best_mae = table.loc[table["MAE"].idxmin(), "Model"]
    best_correlation = table.loc[table["Correlation"].idxmax(), "Model"]
    comparison_metrics = ["RMSE", "MAE", "Correlation", "CSI", "ROC-AUC"]
    aegis_wins = sum(
        table.loc[table[metric].idxmin() if metric in {"RMSE", "MAE"} else table[metric].idxmax(), "Model"] == "AEGIS-WX"
        for metric in comparison_metrics
    )
    metrics = [("Best RMSE", str(best_rmse), "Lower is better."), ("Best MAE", str(best_mae), "Lower is better."), ("Best correlation", str(best_correlation), "Closer to 1 is better."), ("AEGIS-WX wins", f"{aegis_wins} of {len(comparison_metrics)}", "Count across the displayed metrics.")]
    for column, (label, value, help_text) in zip(st.columns(4), metrics):
        column.metric(label, value, help=help_text)
    st.success(f"Best model for this generated case: {best['Model']} (lowest RMSE). Lower RMSE and MAE are better; correlation closer to 1 is better.")
    st.dataframe(table.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "Bias": "{:+.3f}", "Correlation": "{:.3f}", "CSI": "{:.3f}", "ROC-AUC": "{:.3f}"}).highlight_min(subset=["RMSE", "MAE"], color="#164c41").highlight_max(subset=["Correlation", "CSI", "ROC-AUC"], color="#164c41"), hide_index=True, width="stretch")
    chart(st, line_comparison(benchmark_by_lead(result, variable), title="Measured RMSE by forecast lead"))
    series = {model: s.forecasts[model][variable][:, lat_i, lon_i] for model in MODELS} | {"AEGIS-WX": output["final"][:, lat_i, lon_i], "Synthetic truth": s.truth[variable][:, lat_i, lon_i]}
    st.line_chart(pd.DataFrame(series, index=s.leads))
    st.download_button("Download validation metrics as CSV", table.to_csv(index=False).encode(), "aegis_validation_metrics.csv", "text/csv", width="stretch")


def render_objective() -> None:
    with st.container(border=True):
        st.subheader("SIH26081 Objective")
        st.write("AEGIS-WX demonstrates how multiple numerical weather prediction models can be blended transparently, with local confidence and visible safety fallbacks for uncertain conditions.")


def render_why_aegis() -> None:
    st.subheader("Why AEGIS-WX?")
    columns = st.columns(4)
    points = [
        ("Multiple sources", "Uses three complementary NWP-style inputs."),
        ("Adaptive blending", "Changes local weights by conditions and lead time."),
        ("Confidence first", "Shows disagreement, uncertainty and fallback use."),
        ("Explainable", "Lets a reviewer inspect the decision at any grid point."),
    ]
    for column, (title, detail) in zip(columns, points):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(detail)


def render_system() -> None:
    st.header("AEGIS-WX, SIH26081 and prototype transparency")
    left, right = st.columns(2)
    left.markdown("**Data mode:** Synthetic demo\n\n**Spatial resolution:** 1.0° demo grid\n\n**Forecast sources:** GFS-style, ECMWF-style, NCUM-style\n\n**Truth data:** Synthetic truth\n\n**Operational status:** Research/demo prototype")
    right.warning("This prototype demonstrates the architecture and blending methodology. Operational deployment requires real-time NWP feeds, observational ground truth, retraining and meteorological validation.")
    st.subheader("SIH26081 objective")
    st.write("Demonstrate an interpretable hybrid AI–NWP approach that combines multiple weather-model forecasts, communicates uncertainty clearly, and keeps safety fallbacks visible when confidence is limited.")
    st.subheader("Operational Integration Roadmap")
    st.markdown("**Current demo** — Synthetic NWP → synthetic observations → transparent adaptive blending\n\n↓\n\n**Real deployment** — GFS GRIB2 → ECMWF data → NCUM/IMD feed → IMD station observations → satellite/radar inputs → quality control → regridding → bias correction → model training → operational validation")
    with st.expander("Common questions"):
        st.markdown("**Is this a live forecast?** No. It is a synthetic, presentation-ready demonstrator.\n\n**What does confidence mean?** It summarizes local model agreement and estimated synthetic-hindcast error, not a certainty guarantee.\n\n**When does fallback activate?** For missing inputs, severe disagreement, high uncertainty or invalid forecast values.")


initialise_state()
with st.sidebar:
    st.header("Demo Controls")
    st.toggle("Presentation Mode", key="presentation_mode")
    st.caption("DEMO SETTINGS")
    data_mode = "DEMO" if st.session_state.presentation_mode else st.selectbox("Data mode", ["DEMO", "REAL DATA"], help="REAL DATA clearly falls back until source files are supplied.")
    st.selectbox("Scenario", list(DEMO_SCENARIOS), key="scenario", format_func=lambda item: SCENARIO_LABELS[item])
    buttons = st.columns(2)
    if buttons[0].button("Random Scenario", width="stretch"):
        rng = np.random.default_rng(int(st.session_state.scenario_seed)); st.session_state.scenario = str(rng.choice(DEMO_SCENARIOS)); st.session_state.scenario_seed = int(rng.integers(1, 999_999))
    if buttons[1].button("Reset Demo", width="stretch"):
        st.session_state.scenario, st.session_state.scenario_seed, st.session_state.selected_lead, st.session_state.fallback_threshold = "NORMAL", 26081, 24, .75
    same_scale = True
    if not st.session_state.presentation_mode:
        with st.expander("Advanced Demo Settings"):
            st.number_input("Deterministic seed", min_value=1, step=1, key="scenario_seed", help="Using the same seed recreates the same synthetic case.")
        st.caption("SAFETY SETTINGS")
        st.slider("Fallback uncertainty threshold", .50, .95, key="fallback_threshold", step=.05, help="Per-cell safety safeguard with a visible reason.")
        same_scale = st.checkbox("Use same colour scale for all models", value=True, help="Use identical map bounds to compare source magnitudes directly.")
        with st.expander("How to use this demo"):
            st.markdown("1. Choose a scenario.\n2. Compare source forecasts and disagreement.\n3. Inspect the regime and adaptive weights.\n4. Review the final forecast, confidence and synthetic validation.\n\nTip: select a location below to see a local explanation.")

with st.spinner("Generating synthetic NWP inputs and adaptive weights…"):
    try:
        result = run_demo(int(st.session_state.scenario_seed), st.session_state.scenario, data_mode, float(st.session_state.fallback_threshold))
    except Exception:
        st.error("AEGIS-WX could not prepare this demo case. Please reset the demo and try again."); st.stop()

s = result["scenario"]
lead_values = [int(item) for item in s.leads]
if st.session_state.selected_lead not in lead_values: st.session_state.selected_lead = lead_values[0]
latitude_options, longitude_options = [float(value) for value in s.latitude], [float(value) for value in s.longitude]
if st.session_state.get("latitude_value") not in latitude_options: st.session_state.latitude_value = latitude_options[len(latitude_options) // 2]
if st.session_state.get("longitude_value") not in longitude_options: st.session_state.longitude_value = longitude_options[len(longitude_options) // 2]
with st.sidebar:
    st.caption("FORECAST SETTINGS")
    st.select_slider("Forecast lead", options=lead_values, key="selected_lead", format_func=lambda value: f"{value}h")
    st.selectbox("Map variable", list(VARIABLE_META), format_func=lambda item: f"{VARIABLE_META[item][0]} ({VARIABLE_META[item][1]})", key="map_variable")
    st.caption("LOCATION INSPECTOR")
    st.select_slider("Latitude", options=latitude_options, key="latitude_value", format_func=lambda value: f"{value:.1f}°N")
    st.select_slider("Longitude", options=longitude_options, key="longitude_value", format_func=lambda value: f"{value:.1f}°E")
    st.info(f"Inspecting {st.session_state.latitude_value:.1f}°N, {st.session_state.longitude_value:.1f}°E", icon="📍")
    if not st.session_state.presentation_mode:
        with st.expander("Advanced Details"):
            st.caption(f"Grid indices: latitude {latitude_options.index(st.session_state.latitude_value)}, longitude {longitude_options.index(st.session_state.longitude_value)}")
    st.caption("NAVIGATION")
    page = st.selectbox("View", PRESENTATION_NAVIGATION if st.session_state.presentation_mode else NAVIGATION)

lead_index = lead_values.index(st.session_state.selected_lead)
variable = st.session_state.map_variable
lat_i, lon_i = latitude_options.index(st.session_state.latitude_value), longitude_options.index(st.session_state.longitude_value)
output = result["outputs"][variable]

if page == "Overview":
    render_hero(result, lead_values[lead_index], output)
    if st.session_state.presentation_mode:
        with st.expander("30-second explanation"):
            st.write("AEGIS-WX compares three weather-model forecasts, detects the current weather pattern, gives more weight to the most reliable local sources, and shows confidence alongside the final blended forecast. If inputs are missing or uncertainty is too high, the safety fallback is stated plainly.")
    render_objective(); render_final(result, lead_index, lat_i, lon_i, st.session_state.presentation_mode); render_why_aegis()
    if not st.session_state.presentation_mode:
        st.subheader("Export Results")
        st.caption("Download the currently selected lead, variable and generated-case evidence.")
        buttons = st.columns(4)
        buttons[0].download_button("Download Forecast CSV", forecast_csv(s, output["final"], lead_index, "aegis_wx_forecast"), "aegis_selected_forecast.csv", "text/csv", width="stretch")
        buttons[1].download_button("Download Model Weights CSV", weights_csv(s, output["weights"], lead_index), "aegis_adaptive_weights.csv", "text/csv", width="stretch")
        buttons[2].download_button("Download Validation Metrics CSV", benchmark(result, variable).to_csv(index=False).encode(), "aegis_validation_metrics.csv", "text/csv", width="stretch")
        buttons[3].download_button("Download Demo Summary JSON", download_summary(result, variable, lead_index), "aegis_demo_summary.json", "application/json", width="stretch")
elif page == "NWP Inputs": render_inputs(result, variable, lead_index, same_scale)
elif page == "Model Disagreement": render_disagreement(result, variable, lead_index)
elif page == "Weather Regime": render_regime(result)
elif page == "AI Weighting Engine": render_weights(result, variable, lead_index, lat_i, lon_i)
elif page == "Final Forecast": render_final(result, lead_index, lat_i, lon_i)
elif page == "Uncertainty & Alerts": render_uncertainty(result, lead_index)
elif page == "Explainability": render_explain(result, variable, lead_index, lat_i, lon_i)
elif page == "Verification": render_verification(result, variable, lat_i, lon_i)
else: render_system()
