from __future__ import annotations
import numpy as np
from dataclasses import dataclass

MODELS = ("GFS", "ECMWF", "NCUM")
REGIMES = ("NORMAL", "MONSOON", "CYCLONE", "HEATWAVE", "WESTERN_DISTURBANCE")
VARIABLES = ("precipitation", "temperature_2m", "wind_speed_10m", "wind_direction_10m", "pressure")

@dataclass
class WeatherScenario:
    latitude: np.ndarray
    longitude: np.ndarray
    leads: np.ndarray
    truth: dict[str, np.ndarray]          # (lead, lat, lon)
    forecasts: dict[str, dict[str, np.ndarray]] # model -> variable -> (lead, lat, lon)
    regime: str
    seed: int

def _smooth(x: np.ndarray, passes: int = 2) -> np.ndarray:
    for _ in range(passes):
        x = (x + np.roll(x,1,0) + np.roll(x,-1,0) + np.roll(x,1,1) + np.roll(x,-1,1)) / 5
    return x

class SyntheticNWPGenerator:
    """Produces correlated Indian-domain hindcasts with regime/source-dependent errors."""
    def __init__(self, config: dict): self.config = config
    def generate(self, seed: int | None = None, regime: str | None = None) -> WeatherScenario:
        d, g, f = self.config["domain"], self.config["grid"], self.config["forecast"]
        seed = self.config["demo"]["random_seed"] if seed is None else seed
        rng = np.random.default_rng(seed)
        lat = np.arange(d["min_lat"], d["max_lat"] + .01, g["resolution"])
        lon = np.arange(d["min_lon"], d["max_lon"] + .01, g["resolution"])
        yy, xx = np.meshgrid(lat, lon, indexing="ij"); leads = np.asarray(f["lead_times"], dtype=float)
        regime = regime if regime in REGIMES else REGIMES[seed % len(REGIMES)]
        fields = {v: [] for v in VARIABLES}
        # physically motivated land/monsoon/cyclonic patterns; values are synthetic hindcasts.
        coast = np.exp(-((xx-76)/5)**2) + .7*np.exp(-((xx-88)/6)**2)
        himalaya = np.exp(-((yy-32)/4)**2)
        for lead in leads:
            phase = lead / 24
            rain = 1.5 + 2*coast + 3*himalaya
            temp = 31 - .38*(yy-18) + 1.3*np.sin((xx-64)/5 + phase)
            wind = 4 + 2*coast + .7*np.cos(yy/4 - phase)
            pressure = 1010 - 2*np.sin((xx+yy)/7 + phase)
            if regime == "MONSOON":
                band = np.exp(-((yy - (19 + 1.5*np.sin(phase)))/5)**2)
                rain += 26*band*(.35+.65*coast); wind += 4*band; temp -= 2.2*band; pressure -= 3*band
            elif regime == "CYCLONE":
                cx, cy = 87 + .5*phase, 17 + .2*phase; r = np.hypot((xx-cx)/2.2, (yy-cy)/2.2)
                core = np.exp(-r*r); rain += 55*core; wind += 23*core; pressure -= 24*core
            elif regime == "HEATWAVE":
                hot = np.exp(-((xx-77)/8)**2-((yy-25)/6)**2); temp += 11*hot; rain *= .15; wind -= hot
            elif regime == "WESTERN_DISTURBANCE":
                wd = np.exp(-((xx-72-1.1*phase)/7)**2-((yy-31)/4)**2); rain += 22*wd; wind += 7*wd; temp -= 3*wd; pressure -= 7*wd
            else: rain += 3*np.maximum(0, np.sin((xx+yy)/6+phase))
            texture = _smooth(rng.normal(size=yy.shape), 3)
            rain = np.maximum(0, rain + 2.2*texture); temp += .7*texture; wind = np.maximum(0, wind + .9*texture)
            direction = (210 + 35*np.sin(xx/7) + 25*np.cos(yy/6) + phase*8) % 360
            for name, value in zip(VARIABLES, (rain,temp,wind,direction,pressure)): fields[name].append(value)
        truth = {k: np.asarray(v) for k,v in fields.items()}
        forecasts: dict[str, dict[str, np.ndarray]] = {}
        # lower values signify nominal RMSE: ECMWF better in monsoon/cyclone, NCUM in heat/WD.
        skill = {"NORMAL": (.85,.65,.76), "MONSOON": (1.15,.55,.78), "CYCLONE": (1.18,.62,.92), "HEATWAVE": (.86,.85,.55), "WESTERN_DISTURBANCE": (.96,.78,.52)}[regime]
        for m, factor in zip(MODELS, skill):
            model: dict[str, np.ndarray] = {}
            spatial_bias = {"GFS": .9*np.sin(xx/7)+.25*(yy-20), "ECMWF": -.7*np.cos(yy/6), "NCUM": .55*np.sin((xx+yy)/8)}[m]
            for var, actual in truth.items():
                err = _smooth(rng.normal(size=actual.shape), 1) * factor
                lead_growth = (1 + leads[:,None,None]/120)
                if var == "precipitation": model[var] = np.maximum(0, actual + 2.8*factor*spatial_bias + 5.2*err*lead_growth)
                elif var == "temperature_2m": model[var] = actual + .35*factor*spatial_bias + .95*err*lead_growth
                elif var == "wind_speed_10m": model[var] = np.maximum(0, actual + .28*factor*spatial_bias + 1.35*err*lead_growth)
                elif var == "wind_direction_10m": model[var] = (actual + 8*err) % 360
                else: model[var] = actual + .25*factor*spatial_bias + .5*err
            forecasts[m] = model
        return WeatherScenario(lat, lon, leads, truth, forecasts, regime, seed)
