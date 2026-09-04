from core.config import load_config
from data.synthetic_generator import SyntheticNWPGenerator
from data.validators import validate_scenario
from data.regridding import Regridder
import numpy as np
def test_generator_schema_and_distinct_models():
    s=SyntheticNWPGenerator(load_config()).generate(regime="MONSOON")
    assert not validate_scenario(s)
    assert not np.allclose(s.forecasts["GFS"]["precipitation"],s.forecasts["ECMWF"]["precipitation"])
def test_regridder():
    r=Regridder([0,1],[0,1],[0,.5,1],[0,.5,1]); out=r.regrid(np.array([[0.,1.],[1.,2.]])); assert out.shape==(3,3) and np.isclose(out[1,1],1)
