from __future__ import annotations
import numpy as np
def _valid(pred, truth):
    p,t=np.asarray(pred),np.asarray(truth); mask=np.isfinite(p)&np.isfinite(t); return p[mask],t[mask]
def rmse(pred, truth):
    p,t=_valid(pred,truth); return float(np.sqrt(np.mean((p-t)**2))) if len(p) else float("nan")
def mae(pred, truth):
    p,t=_valid(pred,truth); return float(np.mean(np.abs(p-t))) if len(p) else float("nan")
def bias(pred, truth):
    p,t=_valid(pred,truth); return float(np.mean(p-t)) if len(p) else float("nan")
def correlation(pred, truth):
    p,t=_valid(pred,truth)
    if len(p) < 2 or np.std(p) == 0 or np.std(t) == 0: return float("nan")
    return float(np.corrcoef(p,t)[0,1])
def csi(pred, truth, threshold: float):
    p,t=_valid(pred,truth)
    if not len(p): return float("nan")
    p=p>=threshold; t=t>=threshold; hits=np.sum(p&t); denom=hits+np.sum(p&~t)+np.sum(~p&t); return float(hits/denom) if denom else 1.0
def roc_auc(pred, truth, threshold: float):
    scores,actual=_valid(pred,truth); y=(actual>=threshold).astype(int)
    if len(np.unique(y))<2: return float("nan")
    try:
        from sklearn.metrics import roc_auc_score; return float(roc_auc_score(y,scores))
    except ImportError:
        pos=scores[y==1]; neg=scores[y==0]; return float((pos[:,None]>neg).mean()+.5*(pos[:,None]==neg).mean())
def crps_ensemble(members, truth):
    m=np.asarray(members); y=np.asarray(truth)[:,None]; term1=np.mean(np.abs(m-y),axis=1); term2=.5*np.mean(np.abs(m[:,:,None]-m[:,None,:]),axis=(1,2)); return float(np.mean(term1-term2))
