from __future__ import annotations
import numpy as np

class SafetyFallback:
    def __init__(self, threshold: float=.75):
        self.threshold=threshold
        self.last_reasons: list[str] = []
        self.last_fraction: float = 0.0

    def apply(self, ai_prediction: np.ndarray, members: np.ndarray, weights: np.ndarray, uncertainty: np.ndarray) -> tuple[np.ndarray,np.ndarray,str]:
        available=np.isfinite(members).sum(axis=1)
        valid=np.isfinite(ai_prediction)&np.isfinite(weights).all(axis=1)&(np.abs(weights.sum(axis=1)-1)<1e-4)&(available>=2)
        high_uncertainty=uncertainty>self.threshold
        missing_input=available<3
        # Relative spread catches deliberately divergent sources without
        # incorrectly treating a uniformly high meteorological field as unsafe.
        model_mean=np.nanmean(members,axis=1)
        spread=np.nanmax(members,axis=1)-np.nanmin(members,axis=1)
        extreme_disagreement=spread/np.maximum(np.abs(model_mean),1.)>2.5
        mask=high_uncertainty|missing_input|extreme_disagreement|~valid
        fallback=np.nanmean(members,axis=1); final=np.where(mask,fallback,ai_prediction)
        self.last_fraction=float(mask.mean())
        reasons=[]
        if np.any(missing_input): reasons.append("Missing model input")
        if np.any(high_uncertainty): reasons.append("High uncertainty")
        if np.any(extreme_disagreement): reasons.append("Extreme disagreement")
        if np.any(~valid & ~missing_input): reasons.append("Weight computation failed")
        self.last_reasons=reasons
        # Local safeguards can operate while the system remains an adaptive
        # blend overall. A full fallback is reserved for an unavailable source
        # or a substantial fraction of unsafe cells.
        status="FALLBACK MODE" if np.any(missing_input) or self.last_fraction>=.20 else "ADAPTIVE BLEND ACTIVE"
        return final,mask,status
