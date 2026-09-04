from __future__ import annotations
import numpy as np
from .synthetic_generator import WeatherScenario, MODELS, VARIABLES

class DataValidationError(ValueError): pass

def validate_scenario(s: WeatherScenario) -> list[str]:
    """Strictly validates the internal (lead, latitude, longitude) forecast schema."""
    errors: list[str] = []
    if not (np.all(np.diff(s.latitude) > 0) and np.all(np.diff(s.longitude) > 0)): errors.append("Coordinates must be strictly ascending")
    if len(np.unique(s.latitude)) != len(s.latitude) or len(np.unique(s.longitude)) != len(s.longitude): errors.append("Duplicate coordinates")
    shape = (len(s.leads), len(s.latitude), len(s.longitude))
    for source, values in [("truth", s.truth), *s.forecasts.items()]:
        fields = values if source == "truth" else values
        if source != "truth" and source not in MODELS: errors.append(f"Unknown model {source}")
        for v in VARIABLES:
            if v not in fields: errors.append(f"{source}: missing {v}"); continue
            a = fields[v]
            if a.shape != shape: errors.append(f"{source}/{v}: expected {shape}, got {a.shape}")
            if not np.isfinite(a).all(): errors.append(f"{source}/{v}: NaN or infinity")
    if np.any(s.truth["precipitation"] < 0) or np.any(s.truth["wind_speed_10m"] < 0): errors.append("Negative physical value")
    return errors

def assert_valid(s: WeatherScenario) -> None:
    errors = validate_scenario(s)
    if errors: raise DataValidationError("; ".join(errors))
