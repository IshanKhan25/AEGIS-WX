from __future__ import annotations
from pathlib import Path
from .nwp_adapters import DemoAdapter, GFSAdapter, ECMWFAdapter, NCUMAdapter

def load_scenario(config: dict, mode: str = "DEMO", **kwargs):
    """REAL DATA never masquerades as live data; unavailable input explicitly falls back to DEMO."""
    real_requested = mode.upper().startswith("REAL")
    if real_requested and kwargs.get("paths"):
        try:
            paths = kwargs["paths"]
            return {name: cls().load_forecast(paths[name]) for name, cls in {"GFS":GFSAdapter,"ECMWF":ECMWFAdapter,"NCUM":NCUMAdapter}.items()}, "REAL DATA"
        except (OSError, RuntimeError, KeyError) as exc:
            return DemoAdapter(config).load_forecast(seed=kwargs.get("seed")), f"DEMO MODE — real input unavailable ({exc})"
    label = "DEMO MODE — real inputs not supplied; Synthetic Hindcast Data" if real_requested else "DEMO MODE — Synthetic Hindcast Data"
    return DemoAdapter(config).load_forecast(seed=kwargs.get("seed"), regime=kwargs.get("regime"), scenario=kwargs.get("scenario")), label
