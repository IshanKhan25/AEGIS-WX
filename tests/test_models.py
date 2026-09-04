import numpy as np
from inference.pipeline import AegisPipeline
def test_weight_normalization_and_blending():
    r=AegisPipeline().run(regime="CYCLONE"); o=r["outputs"]["precipitation"]
    assert np.max(np.abs(o["weights"].sum(axis=1)-1)) < 1e-6
    assert not np.allclose(o["raw_blended"],o["members"].mean(axis=1))
def test_uncertainty_and_fallback():
    r=AegisPipeline().run(regime="CYCLONE"); o=r["outputs"]["precipitation"]
    assert np.all((o["uncertainty"]>=0)&(o["uncertainty"]<=1))
    assert o["fallback_mask"].dtype == bool
