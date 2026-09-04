from __future__ import annotations
import numpy as np
from data.synthetic_generator import REGIMES

class RuleBasedRegimeClassifier:
    """Transparent prototype classifier; probabilities derive from observed aggregate features."""
    def predict(self, forecasts: dict[str, dict[str, np.ndarray]]) -> tuple[str, dict[str, float], dict[str, float]]:
        rain = np.nanmean([x["precipitation"] for x in forecasts.values()], axis=0)
        temp = np.nanmean([x["temperature_2m"] for x in forecasts.values()], axis=0)
        wind = np.nanmean([x["wind_speed_10m"] for x in forecasts.values()], axis=0)
        p = np.nanmean([x["pressure"] for x in forecasts.values()], axis=0)
        # The demo domain runs south-to-north.  These are transparent spatial
        # signatures, rather than opaque labels tied to a scenario seed.
        central_rain = float(np.mean(rain[:, 10:24]))
        northern_rain = float(np.mean(rain[:, -10:]))
        max_temperature = float(np.max(temp))
        max_wind = float(np.max(wind))
        min_pressure = float(np.min(p))
        mean_rain = float(np.mean(rain))
        scores = {
            "NORMAL": 1.0,
            "MONSOON": max(0.0, (central_rain - 4.0) * 1.2),
            "CYCLONE": max(0.0, (1000.0 - min_pressure) / 3.0) + max(0.0, (max_wind - 15.0) / 3.0),
            "HEATWAVE": max(0.0, (max_temperature - 38.0) * 2.0) + max(0.0, (2.0 - mean_rain) * .5),
            # A normal northern-orographic rainfall background is expected;
            # disturbance scoring activates only above that baseline.
            "WESTERN_DISTURBANCE": max(0.0, (northern_rain - 8.2) * 1.4) + max(0.0, (max_wind - 9.0) / 4.0),
        }
        vec = np.array([scores[r] for r in REGIMES]); prob = np.exp(vec-vec.max()); prob /= prob.sum()
        probabilities = dict(zip(REGIMES, map(float,prob)))
        regime = max(probabilities, key=probabilities.get)
        explanation = {"mean_rainfall_mm": mean_rain, "central_rainfall_mm": central_rain, "northern_rainfall_mm": northern_rain, "max_temperature_c": max_temperature, "max_wind_ms": max_wind, "min_pressure_hpa": min_pressure}
        return regime, probabilities, explanation

class MLRegimeClassifier(RuleBasedRegimeClassifier):
    """Drop-in placeholder for a persisted sklearn/PyTorch classifier; rules remain reliable offline."""
