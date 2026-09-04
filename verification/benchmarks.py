from __future__ import annotations
import numpy as np
import pandas as pd
from data.synthetic_generator import MODELS
from .metrics import rmse,mae,csi,roc_auc

def benchmark(result: dict, variable: str="precipitation") -> pd.DataFrame:
    s=result["scenario"]; out=result["outputs"][variable]; truth=s.truth[variable]; threshold=20. if variable=="precipitation" else 17.
    candidates={**{m:s.forecasts[m][variable] for m in MODELS},"Simple Mean":out["members"].mean(axis=1),"AEGIS-WX":out["final"]}
    rows=[]
    for name,pred in candidates.items(): rows.append({"Model":name,"RMSE":rmse(pred,truth),"MAE":mae(pred,truth),"CSI":csi(pred,truth,threshold),"ROC-AUC":roc_auc(pred,truth,threshold)})
    return pd.DataFrame(rows)

def benchmark_by_lead(result: dict, variable: str="precipitation") -> pd.DataFrame:
    s=result["scenario"]; out=result["outputs"][variable]; rows=[]
    for i,lead in enumerate(s.leads):
        local={**result,"scenario":s}; table=benchmark(local,variable)
        # calculate per-lead instead of relabeling full-period score
        for model in table.Model:
            p=out["final"][i] if model=="AEGIS-WX" else out["members"][i].mean(0) if model=="Simple Mean" else s.forecasts[model][variable][i]
            rows.append({"lead_time":int(lead),"Model":model,"RMSE":rmse(p,s.truth[variable][i]),"MAE":mae(p,s.truth[variable][i])})
    return pd.DataFrame(rows)
