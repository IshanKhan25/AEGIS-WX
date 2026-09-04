from __future__ import annotations
import numpy as np

class DisagreementUncertainty:
    def calculate(self, members: np.ndarray, weights: np.ndarray, historical_error: float = 0.0) -> tuple[np.ndarray,np.ndarray]:
        """Interpretable 0–1 uncertainty from weighted spread plus measured residual error."""
        safe_members=np.where(np.isfinite(members),members,0.)
        centre=(safe_members*weights).sum(axis=1,keepdims=True)
        spread=np.sqrt((weights*(safe_members-centre)**2).sum(axis=1))
        scale=np.percentile(spread,95)+max(historical_error,.1)
        u=np.clip((spread+historical_error)/(scale+historical_error),0,1); return u,1-u
    def exceedance_probability(self, members: np.ndarray, threshold: float, softness: float = 2.) -> np.ndarray:
        return np.mean(1/(1+np.exp(-(members-threshold)/softness)),axis=1)
