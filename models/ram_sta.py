from __future__ import annotations
import numpy as np
from data.synthetic_generator import MODELS

REGIME_PRIORS = {
 "NORMAL": [0.31,.38,.31], "MONSOON": [.20,.52,.28], "CYCLONE": [.20,.55,.25], "HEATWAVE": [.27,.29,.44], "WESTERN_DISTURBANCE": [.24,.32,.44]}

class RAMSTABlender:
    """Numerically stable regime-aware spatial attention with softmax-normalised per-cell weights."""
    def weights(self, forecasts: dict[str, dict[str,np.ndarray]], variable: str, regime: str, leads: np.ndarray) -> np.ndarray:
        stack = np.stack([forecasts[m][variable] for m in MODELS], axis=1) # lead model lat lon
        mean = stack.mean(axis=1, keepdims=True); disagreement = np.abs(stack-mean)
        scale = np.maximum(np.std(stack,axis=1,keepdims=True), .25)
        priors = np.log(np.asarray(REGIME_PRIORS.get(regime, REGIME_PRIORS["NORMAL"])))[None,:,None,None]
        # Higher local disagreement modestly reduces a source's attention; lead modifies priors spatially.
        yy, xx = np.indices(stack.shape[-2:]); texture = .15*np.sin(xx/6) + .10*np.cos(yy/5)
        source_texture = np.stack([texture, -texture*.6, texture*.25], axis=0)
        logits = priors - .62*disagreement/scale + source_texture[None] - np.asarray(leads)[:,None,None,None]/360 * np.array([.12,.05,.08])[None,:,None,None]
        logits -= logits.max(axis=1,keepdims=True); exp = np.exp(logits); return exp/exp.sum(axis=1,keepdims=True)
    def blend(self, forecasts: dict[str, dict[str,np.ndarray]], variable: str, weights: np.ndarray) -> np.ndarray:
        stack=np.stack([forecasts[m][variable] for m in MODELS],axis=1); return (stack*weights).sum(axis=1)
