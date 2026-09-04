from __future__ import annotations
import numpy as np

class SafetyFallback:
    def __init__(self, threshold: float=.62): self.threshold=threshold
    def apply(self, ai_prediction: np.ndarray, members: np.ndarray, weights: np.ndarray, uncertainty: np.ndarray) -> tuple[np.ndarray,np.ndarray,str]:
        valid=np.isfinite(ai_prediction)&np.isfinite(weights).all(axis=1)&(np.abs(weights.sum(axis=1)-1)<1e-4)
        mask=(uncertainty>self.threshold)|~valid
        fallback=np.nanmean(members,axis=1); final=np.where(mask,fallback,ai_prediction)
        status="FALLBACK MODE" if np.any(mask) else "NORMAL AI BLENDING"
        return final,mask,status
