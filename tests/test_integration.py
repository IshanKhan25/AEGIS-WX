import numpy as np
import pytest
from core.config import load_config
from data.loaders import load_scenario
from data.synthetic_generator import REGIMES
from inference.pipeline import AegisPipeline
from verification.benchmarks import benchmark, benchmark_by_lead
from verification.metrics import crps_ensemble, roc_auc
from verification.reports import export_report

@pytest.mark.parametrize("regime", REGIMES)
def test_every_demo_regime_and_variable(regime):
    result = AegisPipeline().run(seed=26081, regime=regime)
    scenario = result["scenario"]
    assert result["detected_regime"] == regime
    for variable in ("precipitation", "temperature_2m", "wind_speed_10m"):
        output = result["outputs"][variable]
        assert output["final"].shape == scenario.truth[variable].shape
        assert np.allclose(output["weights"].sum(axis=1), 1.0, atol=1e-8)
        assert np.isfinite(output["final"]).all()
        assert np.isfinite(output["uncertainty"]).all()
        assert len(benchmark(result, variable)) == 6
        assert len(benchmark_by_lead(result, variable)) == len(scenario.leads) * 6

def test_real_request_is_honest_demo_fallback():
    scenario, label = load_scenario(load_config(), mode="REAL DATA", seed=1)
    assert "DEMO MODE" in label and "real inputs not supplied" in label
    assert hasattr(scenario, "forecasts")

def test_xarray_forecast_export_dimensions():
    xr = pytest.importorskip("xarray")
    result = AegisPipeline().run(); scenario=result["scenario"]; field=result["outputs"]["precipitation"]["final"]
    ds=xr.Dataset({"aegis_wx":(("lead_time","latitude","longitude"),field)},coords={"lead_time":scenario.leads,"latitude":scenario.latitude,"longitude":scenario.longitude})
    payload=bytes(ds.to_netcdf(engine="scipy"))
    assert payload and ds["aegis_wx"].dims == ("lead_time","latitude","longitude")

def test_probabilistic_metrics_shape_and_degenerate_roc():
    members=np.array([[[1.,2.],[3.,4.]],[[2.,3.],[4.,5.]]])
    truth=np.array([[1.5,2.5],[3.5,4.5]])
    assert np.isfinite(crps_ensemble(members, truth))
    assert np.isnan(roc_auc([1,2], [0,0], 20))

def test_api_all_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.main import app
    client=TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/models").status_code == 200
    assert client.get("/regime").status_code == 200
    weights=client.get("/weights"); assert weights.status_code == 200 and weights.json()["sum_min"] > .999999
    assert client.get("/uncertainty").status_code == 200
    for endpoint in ("/forecast", "/blend", "/verify"):
        response=client.post(endpoint,json={"lead_time":24,"variable":"precipitation","regime":"MONSOON","seed":26081})
        assert response.status_code == 200

def test_plot_builders_and_report_export(tmp_path):
    pytest.importorskip("plotly")
    from visualization.maps import heatmap
    from visualization.charts import line_comparison, weights_chart
    result=AegisPipeline().run(); scenario=result["scenario"]; table=benchmark(result)
    assert heatmap(result["outputs"]["precipitation"]["final"][0],scenario.latitude,scenario.longitude,"Forecast").to_json()
    assert line_comparison(benchmark_by_lead(result)).to_json()
    assert weights_chart(np.array([.2,.5,.3]), ["GFS","ECMWF","NCUM"]).to_json()
    report=export_report(table,result,tmp_path / "report.json")
    assert report and (tmp_path / "report.json").exists()
