import numpy as np
from verification.metrics import rmse,mae,csi
def test_continuous_metrics():
    assert rmse([1,3],[1,1]) == np.sqrt(2)
    assert mae([1,3],[1,1]) == 1
def test_csi(): assert csi([25,0],[30,0],20)==1
