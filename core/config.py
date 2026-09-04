from __future__ import annotations
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "domain": {"min_lat": 5., "max_lat": 40., "min_lon": 60., "max_lon": 100.},
    "grid": {"resolution": 1.},
    "forecast": {"lead_times": [0, 24, 48, 72]},
    "thresholds": {"uncertainty": .62, "heavy_rain": 20., "severe_wind": 17., "heatwave": 40.},
    "demo": {"random_seed": 26081}, "runtime": {"device": "auto", "batch_size": 2},
}

def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML when available; defaults keep the demo usable in bare Python."""
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    try:
        import yaml
        source = yaml.safe_load(Path(path).read_text()) or {}
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict): cfg[key].update(value)
            else: cfg[key] = value
    except (ImportError, FileNotFoundError):
        pass
    return cfg
