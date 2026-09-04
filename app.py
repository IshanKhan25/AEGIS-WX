from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import streamlit as st

from core.config import load_config
from data.synthetic_generator import MODELS, REGIMES
import models.regime_classifier as regime_classifier_module
import inference.pipeline as pipeline_module
from verification.benchmarks import benchmark, benchmark_by_lead
from verification.reports import export_report
from visualization.charts import line_comparison, weights_chart
from visualization.maps import heatmap

# Streamlit reloads this script while preserving process modules. Reload these
# local, lightweight modules so newly selected demo scenarios always use the
# current regime rules during an interactive jury demonstration.
importlib.reload(regime_classifier_module)
AegisPipeline = importlib.reload(pipeline_module).AegisPipeline

st.set_page_config(page_title="AEGIS-WX | SIH26081", page_icon="🌦️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 15% 0%,#173b58 0%,#091826 38%,#07111d 100%);color:#f3f7fb}[data-testid="stSidebar"]{background:#0b1c2d}.hero{padding:.35rem 0 1.2rem}.eyebrow{color:#6dd3ff;font-size:.82rem;letter-spacing:.12em;font-weight:700}.hero h1{margin:.12rem 0;font-size:2.55rem;color:#f5fbff}.hero p{margin:0;color:#b7c9d8;font-size:1.05rem}.flow{display:flex;align-items:stretch;gap:.45rem;margin:.75rem 0 1.25rem;overflow-x:auto}.flow-step{min-width:120px;flex:1;border:1px solid #2c516b;background:rgba(17,42,62,.88);border-radius:9px;padding:.7rem}.flow-num{color:#6dd3ff;font-weight:800;font-size:.75rem}.flow-label{color:#f4f9fc;font-weight:700;margin-top:.18rem}.flow-arrow{color:#6dd3ff;align-self:center;font-size:1.25rem}.section-kicker{color:#6dd3ff;font-size:.8rem;font-weight:700;letter-spacing:.08em}div[data-testid="stMetric"]{background:rgba(17,42,62,.82);border:1px solid #2c516b;border-radius:9px;padding:.65rem .8rem}.evidence{background:rgba(17,42,62,.77);border-left:4px solid #41c7a1;padding:.7rem .85rem;border-radius:4px;margin:.25rem 0 .9rem}
</style>
""", unsafe_allow_html=True)

VARIABLES = {"Rainfall (mm)": "precipitation", "Temperature (°C)": "temperature_2m", "Wind speed (m/s)": "wind_speed_10m"}
SCENARIO_NOTES = {"NORMAL":"Broad synoptic baseline with moderate spatial variability.","MONSOON":"Central Indian rain band and stronger monsoon flow.","CYCLONE":"Compact low-pressure core with heavy rain and high wind.","HEATWAVE":"Hot, dry inland anomaly with suppressed rainfall.","WESTERN_DISTURBANCE":"Northern precipitation and wind disturbance."}

def _scenario_seed() -> int:
    return int(np.random.default_rng().integers(1, 999_999))

if "scenario_seed" not in st.session_state:
    st.session_state["scenario_seed"] = 26081
if "scenario_regime" not in st.session_state:
    st.session_state["scenario_regime"] = "NORMAL"

def _generate_new_scenario() -> None:
    st.session_state["scenario_seed"] = _scenario_seed()

with st.sidebar:
    st.header("Jury demo controls")
    mode = st.selectbox("Data mode", ["DEMO", "REAL DATA"], key="demo_mode", help="No live data is claimed. REAL DATA clearly falls back unless source files are supplied.")
    variable_label = st.selectbox("Display variable", list(VARIABLES), key="display_variable")
    variable = VARIABLES[variable_label]
    scenario_regime = st.selectbox("Demo weather scenario", list(REGIMES), key="scenario_regime", help="Changing this choice immediately regenerates the scenario and adaptive weights.")
    seed = st.number_input("Scenario seed", min_value=1, step=1, key="scenario_seed")
    generated = st.button("Generate New Weather Scenario", type="primary", use_container_width=True, on_click=_generate_new_scenario)
    run = st.button("Run AEGIS-WX", use_container_width=True)
    st.caption("Tip: choose CYCLONE or HEATWAVE, then generate. Compare source spread, regime, weights and alerts.")

request_signature = (mode, scenario_regime, int(seed))
# A full CPU demo is fast (<1 second), so recomputing on every control change is
# intentional: the displayed regime, map and attention weights can never become
# stale relative to the active jury-demo controls.
with st.spinner("Generating NWP inputs → measuring disagreement → adapting weights → verifying…"):
    st.session_state["result"] = AegisPipeline(load_config()).run(seed=int(seed), regime=scenario_regime, mode=mode)
    st.session_state["last_request_signature"] = request_signature

r = st.session_state["result"]
s = r["scenario"]
out = r["outputs"][variable]
lead_index = st.select_slider("Forecast lead", options=list(range(len(s.leads))), format_func=lambda i: f"{int(s.leads[i])}h")
lead = int(s.leads[lead_index])

st.markdown("""<div class="hero"><div class="eyebrow">SIH26081 · HYBRID AI–NWP MULTI-MODEL FORECAST BLENDING</div><h1>AEGIS-WX</h1><p>Adaptive, regime-aware forecasting from GFS, ECMWF and NCUM-style inputs.</p></div>""", unsafe_allow_html=True)
steps = ["NWP inputs", "Model disagreement", "Weather regime", "Adaptive weights", "AEGIS-WX forecast", "Uncertainty & alerts", "Verification"]
flow_html = "<div class='flow'>" + "".join(f"<div class='flow-step'><div class='flow-num'>{i+1:02d}</div><div class='flow-label'>{step}</div></div>" + ("<div class='flow-arrow'>→</div>" if i < len(steps)-1 else "") for i, step in enumerate(steps)) + "</div>"
st.markdown(flow_html, unsafe_allow_html=True)
st.info(f"{r['data_mode']} · reduced 1.0° computational grid · verification uses synthetic truth only; it is not operational performance.")

st.markdown("<div class='section-kicker'>SYSTEM STATUS</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active scenario", r["active_regime"])
c2.metric("Detected regime", r["detected_regime"], f"P = {r['regime_probabilities'][r['detected_regime']]:.0%}")
c3.metric("Blending status", out["fallback_status"])
c4.metric("Pipeline runtime", f"{r['runtime_seconds']:.2f}s")

st.markdown("<div class='section-kicker'>1–2 · NWP INPUTS → MODEL DISAGREEMENT</div>", unsafe_allow_html=True)
st.subheader(f"What the three forecast sources say at {lead}h")
source_tabs = st.tabs(["NWP input maps", "Disagreement evidence", "AEGIS-WX forecast"])
with source_tabs[0]:
    cols = st.columns(3)
    for i, model in enumerate(MODELS):
        cols[i].plotly_chart(heatmap(s.forecasts[model][variable][lead_index], s.latitude, s.longitude, f"{model} · {variable_label} · {lead}h"), use_container_width=True)
with source_tabs[1]:
    member_spread = out["members"][lead_index].std(axis=0)
    model_range = np.ptp(out["members"][lead_index], axis=0)
    left, right = st.columns(2)
    left.plotly_chart(heatmap(member_spread, s.latitude, s.longitude, "Model disagreement (standard deviation)", "Magma"), use_container_width=True)
    right.plotly_chart(heatmap(model_range, s.latitude, s.longitude, "Model disagreement (max–min range)", "Magma"), use_container_width=True)
    st.markdown(f"<div class='evidence'>At this lead, the mean three-model standard deviation is <b>{member_spread.mean():.2f}</b> for {variable_label}. Higher spread is carried into uncertainty and changes local attention weights.</div>", unsafe_allow_html=True)
with source_tabs[2]:
    left, right = st.columns([2, 1])
    left.plotly_chart(heatmap(out["final"][lead_index], s.latitude, s.longitude, f"AEGIS-WX final forecast · {variable_label} · {lead}h"), use_container_width=True)
    right.metric("Simple mean differs by", f"{np.mean(np.abs(out['final'][lead_index] - out['members'][lead_index].mean(axis=0))):.2f}", "mean absolute field difference")
    right.metric("Fallback cells", int(out["fallback_mask"][lead_index].sum()))
    right.caption("The final field is spatially adaptive; it is not a fixed arithmetic average.")

st.markdown("<div class='section-kicker'>3 · WEATHER REGIME</div>", unsafe_allow_html=True)
st.subheader("Why the system expects these sources to perform differently")
regime_cols = st.columns([1, 2])
with regime_cols[0]:
    st.metric("Active regime", r["active_regime"])
    st.write(SCENARIO_NOTES.get(r["active_regime"], "Rule-based regime detection using computed meteorological features."))
with regime_cols[1]:
    probability_frame = pd.DataFrame({"Regime": list(r["regime_probabilities"]), "Probability": list(r["regime_probabilities"].values())}).sort_values("Probability", ascending=False)
    st.bar_chart(probability_frame.set_index("Regime"))
    st.caption("Computed evidence: " + " · ".join(f"{key.replace('_', ' ')} {value:.2f}" for key, value in r["regime_features"].items()))

st.markdown("<div class='section-kicker'>4 · ADAPTIVE MODEL WEIGHTS</div>", unsafe_allow_html=True)
st.subheader("Weights vary by grid cell, lead time, regime and local disagreement")
weight_cols = st.columns(3)
for i, model in enumerate(MODELS):
    weight_cols[i].plotly_chart(heatmap(out["weights"][lead_index, i], s.latitude, s.longitude, f"{model} weight · {lead}h", "Blues"), use_container_width=True)

st.markdown("<div class='section-kicker'>5–6 · UNCERTAINTY → EXTREME-WEATHER ALERT</div>", unsafe_allow_html=True)
st.subheader("Decision confidence and alert evidence")
left, middle, right = st.columns([1.25, 1.25, 1])
left.plotly_chart(heatmap(out["uncertainty"][lead_index], s.latitude, s.longitude, "Uncertainty (0–1)", "Reds"), use_container_width=True)
middle.plotly_chart(heatmap(out["confidence"][lead_index], s.latitude, s.longitude, "Confidence (1 − uncertainty)", "Greens"), use_container_width=True)
with right:
    alerts = r["alerts"]
    st.metric("Heavy-rain cells", alerts["heavy_rain_cells"])
    st.metric("High-wind cells", alerts["high_wind_cells"])
    st.metric("Heatwave cells", alerts["heatwave_cells"])
    st.caption("Counts are computed across the scenario. High-uncertainty cells use the visible safety fallback.")

st.markdown("<div class='section-kicker'>EXPLAIN AEGIS-WX</div>", unsafe_allow_html=True)
st.subheader("Explain the current model weights with computed values")
lat_i = st.slider("Latitude grid index", 0, len(s.latitude)-1, len(s.latitude)//2)
lon_i = st.slider("Longitude grid index", 0, len(s.longitude)-1, len(s.longitude)//2)
point_weights = out["weights"][lead_index, :, lat_i, lon_i]
members_at_point = out["members"][lead_index, :, lat_i, lon_i]
simple_mean = float(members_at_point.mean())
source_deviation = np.abs(members_at_point - simple_mean)
explain_frame = pd.DataFrame({"Source": MODELS, "Forecast": members_at_point, "Adaptive weight": point_weights, "Deviation from source mean": source_deviation})
explain_cols = st.columns([1.1, 1.25, 1.4])
with explain_cols[0]: st.plotly_chart(weights_chart(point_weights, MODELS), use_container_width=True)
with explain_cols[1]: st.dataframe(explain_frame.style.format({"Forecast":"{:.2f}", "Adaptive weight":"{:.1%}", "Deviation from source mean":"{:.2f}"}).highlight_max(subset=["Adaptive weight"], color="#164c41"), hide_index=True, use_container_width=True)
with explain_cols[2]:
    top_i = int(np.argmax(point_weights)); top_model = MODELS[top_i]
    st.markdown(f"<div class='evidence'><b>{top_model}</b> receives the largest current weight: <b>{point_weights[top_i]:.1%}</b> at {s.latitude[lat_i]:.1f}°N, {s.longitude[lon_i]:.1f}°E.<br><br>The active regime is <b>{r['active_regime']}</b>; its source prior is applied before local disagreement and lead-time terms. This source differs from the local three-model mean by <b>{source_deviation[top_i]:.2f}</b>. Three-source standard deviation: <b>{members_at_point.std():.2f}</b>.</div>", unsafe_allow_html=True)
    st.caption(f"Raw weighted blend: {out['raw_blended'][lead_index, lat_i, lon_i]:.2f} · final bias-aware value: {out['final'][lead_index, lat_i, lon_i]:.2f} · uncertainty: {out['uncertainty'][lead_index, lat_i, lon_i]:.2f}")

st.markdown("<div class='section-kicker'>7 · VERIFICATION AGAINST BASELINES</div>", unsafe_allow_html=True)
st.subheader("Measured synthetic-hindcast comparison")
table = benchmark(r, variable)
st.dataframe(table.style.format({"RMSE":"{:.3f}", "MAE":"{:.3f}", "CSI":"{:.3f}", "ROC-AUC":"{:.3f}"}).highlight_min(subset=["RMSE", "MAE"], color="#164c41").highlight_max(subset=["CSI", "ROC-AUC"], color="#164c41"), hide_index=True, use_container_width=True)
st.plotly_chart(line_comparison(benchmark_by_lead(r, variable), title="Measured RMSE by forecast lead"), use_container_width=True)
series = {model: s.forecasts[model][variable][:, lat_i, lon_i] for model in MODELS} | {"AEGIS-WX": out["final"][:, lat_i, lon_i], "Truth": s.truth[variable][:, lat_i, lon_i]}
st.caption("Point time series: sources, AEGIS-WX and synthetic truth.")
st.line_chart(pd.DataFrame(series, index=s.leads))

export_left, export_right, _ = st.columns(3)
with export_left:
    if st.button("Export Verification Report"):
        report_path = export_report(table, r)
        st.success(f"Saved {report_path}")
with export_right: st.download_button("Download Verification CSV", table.to_csv(index=False).encode(), "aegis_verification.csv", "text/csv")
try:
    import xarray as xr
    dataset = xr.Dataset({"aegis_wx": (("lead_time", "latitude", "longitude"), out["final"])}, coords={"lead_time": s.leads, "latitude": s.latitude, "longitude": s.longitude})
    st.download_button("Download Forecast NetCDF", bytes(dataset.to_netcdf(engine="scipy")), "aegis_forecast.nc", "application/x-netcdf")
except Exception as exc:
    st.caption(f"NetCDF export unavailable: {exc}")
