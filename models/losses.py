from __future__ import annotations
import numpy as np
def asymmetric_extreme_mse(prediction, target, threshold: float, underprediction_weight: float=3.0):
    error=np.asarray(prediction)-np.asarray(target); severe=np.asarray(target)>=threshold
    weight=np.where(severe & (error<0),underprediction_weight,1.0); return float(np.mean(weight*error**2))
