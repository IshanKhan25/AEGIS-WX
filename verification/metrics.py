from __future__ import annotations
import numpy as np
def rmse(pred, truth): return float(np.sqrt(np.mean((np.asarray(pred)-np.asarray(truth))**2)))
def mae(pred, truth): return float(np.mean(np.abs(np.asarray(pred)-np.asarray(truth))))
def csi(pred, truth, threshold: float):
    p=np.asarray(pred)>=threshold; t=np.asarray(truth)>=threshold; hits=np.sum(p&t); denom=hits+np.sum(p&~t)+np.sum(~p&t); return float(hits/denom) if denom else 1.0
def roc_auc(pred, truth, threshold: float):
    y=(np.asarray(truth).ravel()>=threshold).astype(int); scores=np.asarray(pred).ravel()
    if len(np.unique(y))<2: return float("nan")
    try:
        from sklearn.metrics import roc_auc_score; return float(roc_auc_score(y,scores))
    except ImportError:
        pos=scores[y==1]; neg=scores[y==0]; return float((pos[:,None]>neg).mean()+.5*(pos[:,None]==neg).mean())
def crps_ensemble(members, truth):
    m=np.asarray(members); y=np.asarray(truth)[:,None]; term1=np.mean(np.abs(m-y),axis=1); term2=.5*np.mean(np.abs(m[:,:,None]-m[:,None,:]),axis=(1,2)); return float(np.mean(term1-term2))
