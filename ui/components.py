from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


VARIABLE_META = {
    "precipitation": ("Rainfall", "mm", "Blues"),
    "temperature_2m": ("Temperature", "°C", "RdYlBu_r"),
    "wind_speed_10m": ("Wind speed", "m/s", "Turbo"),
}

SCENARIO_LABELS = {
    "NORMAL": "Normal", "MONSOON": "Monsoon", "HEAVY_RAIN": "Heavy Rain",
    "CYCLONE": "Cyclone", "HEATWAVE": "Heatwave", "HIGH_WIND": "High Wind",
    "EXTREME_DISAGREEMENT": "Extreme Disagreement", "MISSING_MODEL_INPUT": "Missing Model Input",
}


def fmt(value: float | int | None, digits: int = 2, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "Unavailable"
    return f"{value:.{digits}f}{suffix}"


def pct(value: float | None, digits: int = 0) -> str:
    return fmt(None if value is None else value * 100, digits, "%")


def chart(container: Any, figure: Any, **kwargs: Any) -> None:
    container.plotly_chart(figure, width="stretch", config={"displayModeBar": False, "responsive": True}, **kwargs)


def status_badge(status: str, reasons: list[str], fallback_fraction: float) -> None:
    if status == "ADAPTIVE BLEND ACTIVE":
        st.success(f"Adaptive blend active — dynamic model weights are being used across {pct(1-fallback_fraction)} of the forecast grid.")
        if reasons:
            st.caption("Remaining cells use safety fallback where uncertainty or disagreement is high.")
        else:
            st.caption("Safety rules enabled: high uncertainty and extreme disagreement checks.")
    else:
        reason_text = {
            "High uncertainty": "uncertainty exceeded the configured threshold.",
            "Extreme disagreement": "model disagreement exceeded the safe operating threshold.",
            "Missing model input": "one or more forecast sources are unavailable.",
            "Weight computation failed": "the model-weight computation did not pass a safety check.",
        }
        st.error("Safety fallback activated")
        st.write("Reason: " + (reason_text.get(reasons[0], "a safety check did not pass.") if reasons else "a safety check did not pass."))


def download_summary(result: dict[str, Any], variable: str, lead_index: int) -> bytes:
    scenario = result["scenario"]
    output = result["outputs"][variable]
    payload = {
        "data_mode": result["data_mode"], "scenario": scenario.scenario, "detected_regime": result["detected_regime"],
        "lead_time_hours": int(scenario.leads[lead_index]), "blend_status": output["fallback_status"],
        "mean_confidence": float(output["confidence"][lead_index].mean()), "mean_uncertainty": float(output["uncertainty"][lead_index].mean()),
        "alerts": result["alerts"],
    }
    return json.dumps(payload, indent=2).encode()


def selected_cell_frame(models: tuple[str, ...], members: np.ndarray, weights: np.ndarray, unit: str) -> pd.DataFrame:
    consensus = float(np.nanmean(members))
    return pd.DataFrame({
        "Source": models,
        f"Forecast ({unit})": members,
        "Weight": weights,
        f"Deviation ({unit})": np.abs(members-consensus),
    })


CSS = """
<style>
.stApp { background: radial-gradient(circle at 10% 0%, #173b58 0%, #091826 42%, #07111d 100%); color: #eef6fb; overflow-x: hidden; }
[data-testid="stSidebar"] { background: #091a2a; }
.hero { padding: .7rem 0 1.1rem; }
.eyebrow, .section-kicker { color: #6dd3ff; font-size: .77rem; letter-spacing: .12em; font-weight: 800; text-transform: uppercase; }
.hero h1 { margin: .08rem 0; font-size: clamp(2.1rem, 5vw, 3.25rem); color: #f6fbff; }
.hero p { margin: 0; color: #c0d4e3; font-size: 1.05rem; }.hero .hero-explainer { margin-top: .6rem; max-width: 72rem; color: #d4e3ed; font-size: .96rem; }
.flow { display: flex; flex-wrap: wrap; gap: .45rem; margin: .8rem 0 1.15rem; }
.flow-step { flex: 1 1 110px; border: 1px solid #2c516b; background: rgba(17,42,62,.88); border-radius: 10px; padding: .7rem; box-sizing: border-box; min-width: 0; }
.flow-step b { display: block; color: #6dd3ff; font-size: .74rem; }.flow-step span { display: block; margin-top: .18rem; font-weight: 700; overflow-wrap: anywhere; }
.forecast-card { background: rgba(17,42,62,.86); border: 1px solid #2c516b; border-radius: 10px; padding: .85rem; min-height: 92px; box-sizing: border-box; margin-bottom: .5rem; }.forecast-card.primary-forecast { border-color: #58bedf; box-shadow: 0 0 0 1px rgba(109,211,255,.15), 0 8px 22px rgba(0,0,0,.16); }
.forecast-card small { color: #a9c1d1; display: block; }.forecast-card strong { font-size: 1.28rem; color: #f4fbff; display: block; margin-top: .22rem; }
.evidence { background: rgba(17,42,62,.77); border-left: 4px solid #41c7a1; padding: .75rem .9rem; border-radius: 4px; margin: .35rem 0 .9rem; overflow-wrap: anywhere; }
div[data-testid="stMetric"] { background: rgba(17,42,62,.82); border: 1px solid #2c516b; border-radius: 10px; padding: .7rem .8rem; min-width: 0; }
@media (max-width: 700px) { .block-container { padding: 1rem .8rem 2rem; } .hero { padding-top: .2rem; } .flow-step { flex-basis: 100px; } [data-testid="stHorizontalBlock"] { gap: .55rem; flex-wrap: wrap; } [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 100%; min-width: 0; } }
</style>
"""
