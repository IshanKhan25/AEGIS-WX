from __future__ import annotations
import numpy as np

class RollingBiasCorrector:
    """Bias estimate is calculated from the current synthetic hindcast/available history."""
    def fit(self, prediction: np.ndarray, truth: np.ndarray) -> "RollingBiasCorrector":
        self.bias = np.mean(prediction-truth,axis=0,keepdims=True); self.residual_std=float(np.std(prediction-truth)); return self
    def transform(self, prediction: np.ndarray, variable: str) -> np.ndarray:
        out=prediction-self.bias
        return np.maximum(0,out) if variable in {"precipitation","wind_speed_10m"} else out
