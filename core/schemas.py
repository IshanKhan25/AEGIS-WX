from __future__ import annotations
from typing import Literal
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel: pass
    def Field(*args, **kwargs): return None

class ForecastRequest(BaseModel):
    lead_time: int = 24
    variable: Literal["precipitation", "temperature_2m", "wind_speed_10m"] = "precipitation"
    regime: str = "AUTO"
    seed: int = 26081

class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str = "DEMO"
