from __future__ import annotations

import numpy as np
import pandas as pd

from data.synthetic_generator import MODELS
from .metrics import bias, correlation, rmse, mae, csi, roc_auc


METRIC_DIRECTIONS = {
    "RMSE": "min",
    "MAE": "min",
    "Bias": "absolute_min",
    "Correlation": "max",
    "CSI": "max",
    "ROC-AUC": "max",
}


def finite_metric_values(table: pd.DataFrame, metric: str) -> pd.Series:
    """Coerce one metric column and retain only finite numeric values."""
    if metric not in table:
        return pd.Series(dtype=float)
    values = pd.to_numeric(table[metric], errors="coerce")
    return values.loc[np.isfinite(values)]


def get_best_model_for_metric(table: pd.DataFrame, metric: str) -> str | None:
    """Return the scientifically appropriate best model, or None when undefined."""
    if metric not in METRIC_DIRECTIONS or "Model" not in table:
        return None
    values = finite_metric_values(table, metric)
    if values.empty:
        return None
    direction = METRIC_DIRECTIONS[metric]
    index = values.idxmin() if direction == "min" else values.abs().idxmin() if direction == "absolute_min" else values.idxmax()
    model = table.loc[index, "Model"]
    return str(model) if pd.notna(model) else None


def sanitise_metric_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a display/export-safe copy with infinities represented as missing."""
    clean = table.copy()
    for metric in METRIC_DIRECTIONS:
        if metric in clean:
            values = pd.to_numeric(clean[metric], errors="coerce")
            clean[metric] = values.where(np.isfinite(values), np.nan)
    return clean


def metric_records(table: pd.DataFrame) -> list[dict[str, object]]:
    """Create JSON-safe records, representing undefined metrics as null."""
    clean = sanitise_metric_table(table).astype(object)
    return clean.where(pd.notna(clean), None).to_dict(orient="records")

def fixed_weighted_mean(members: np.ndarray) -> np.ndarray:
    """A transparent fixed-weight baseline; unavailable sources are renormalised."""
    base=np.asarray([.33,.34,.33])[None,:,None,None]
    available=np.isfinite(members)
    weights=base*available
    weights/=np.maximum(weights.sum(axis=1,keepdims=True),1e-12)
    return np.nansum(members*weights,axis=1)

def benchmark(result: dict, variable: str="precipitation") -> pd.DataFrame:
    s=result["scenario"]; out=result["outputs"][variable]; truth=s.truth[variable]; threshold=20. if variable=="precipitation" else 17.
    candidates={**{m:s.forecasts[m][variable] for m in MODELS},"Simple Mean":np.nanmean(out["members"],axis=1),"Fixed Weighted Mean":fixed_weighted_mean(out["members"]),"AEGIS-WX":out["final"]}
    rows=[]
    for name, pred in candidates.items():
        rows.append({"Model": name, "RMSE": rmse(pred, truth), "MAE": mae(pred, truth), "Bias": bias(pred, truth), "Correlation": correlation(pred, truth), "CSI": csi(pred, truth, threshold), "ROC-AUC": roc_auc(pred, truth, threshold)})
    return pd.DataFrame(rows)

def benchmark_by_lead(result: dict, variable: str="precipitation") -> pd.DataFrame:
    s=result["scenario"]; out=result["outputs"][variable]; rows=[]
    for i,lead in enumerate(s.leads):
        local={**result,"scenario":s}; table=benchmark(local,variable)
        # calculate per-lead instead of relabeling full-period score
        for model in table.Model:
            p=out["final"][i] if model=="AEGIS-WX" else np.nanmean(out["members"][i],axis=0) if model=="Simple Mean" else fixed_weighted_mean(out["members"])[i] if model=="Fixed Weighted Mean" else s.forecasts[model][variable][i]
            rows.append({"lead_time":int(lead),"Model":model,"RMSE":rmse(p,s.truth[variable][i]),"MAE":mae(p,s.truth[variable][i])})
    return pd.DataFrame(rows)
