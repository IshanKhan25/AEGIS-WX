from __future__ import annotations
from fastapi import FastAPI
from core.schemas import ForecastRequest, HealthResponse
from inference.pipeline import AegisPipeline
from verification.benchmarks import benchmark

app=FastAPI(title="AEGIS-WX API",version="0.1.0")
def _run(request: ForecastRequest | None=None): return AegisPipeline().run(seed=request.seed if request else None,regime=request.regime if request else None)
@app.get("/health",response_model=HealthResponse)
def health(): return HealthResponse()
@app.get("/models")
def models(): return {"available":["GFS","ECMWF","NCUM"],"mode":"DEMO MODE — Synthetic Hindcast Data"}
@app.post("/forecast")
def forecast(request: ForecastRequest):
    r=_run(request); o=r["outputs"][request.variable]; return {"mode":r["data_mode"],"regime":r["active_regime"],"lead_times":r["scenario"].leads.tolist(),"forecast_shape":list(o["final"].shape),"fallback":o["fallback_status"]}
@app.post("/blend")
def blend(request: ForecastRequest):
    r=_run(request); o=r["outputs"][request.variable]; return {"weights_sum_range":[float(o["weights"].sum(1).min()),float(o["weights"].sum(1).max())],"fallback":o["fallback_status"]}
@app.post("/verify")
def verify(request: ForecastRequest): return benchmark(_run(request),request.variable).to_dict(orient="records")
@app.get("/regime")
def regime():
    r=_run(); return {"detected":r["detected_regime"],"probabilities":r["regime_probabilities"],"features":r["regime_features"]}
@app.get("/weights")
def weights():
    r=_run(); o=r["outputs"]["precipitation"]; return {"shape":list(o["weights"].shape),"sum_min":float(o["weights"].sum(1).min()),"sum_max":float(o["weights"].sum(1).max())}
@app.get("/uncertainty")
def uncertainty():
    r=_run(); o=r["outputs"]["precipitation"]; return {"mean_uncertainty":float(o["uncertainty"].mean()),"mean_confidence":float(o["confidence"].mean())}
