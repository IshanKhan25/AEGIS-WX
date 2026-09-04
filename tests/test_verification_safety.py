import numpy as np
import pandas as pd
import pytest

from data.synthetic_generator import DEMO_SCENARIOS
from inference.pipeline import AegisPipeline
from verification.benchmarks import (
    METRIC_DIRECTIONS,
    benchmark,
    benchmark_by_lead,
    get_best_model_for_metric,
    metric_records,
    sanitise_metric_table,
)
from verification.metrics import bias, correlation, csi, mae, rmse
from visualization.charts import line_comparison


def test_all_undefined_correlation_has_no_best_model():
    table = pd.DataFrame({"Model": ["GFS", "ECMWF", "NCUM"], "Correlation": [np.nan, np.inf, -np.inf]})
    assert get_best_model_for_metric(table, "Correlation") is None


def test_one_finite_correlation_wins_over_undefined_values():
    table = pd.DataFrame({"Model": ["GFS", "ECMWF", "NCUM"], "Correlation": [np.nan, .82, np.nan]})
    assert get_best_model_for_metric(table, "Correlation") == "ECMWF"


def test_zero_variance_correlation_is_undefined():
    assert np.isnan(correlation([2, 2, 2], [1, 2, 3]))
    assert np.isnan(correlation([1, 2, 3], [5, 5, 5]))


def test_csi_with_no_forecast_or_observed_event_is_undefined():
    assert np.isnan(csi([1, 2, 3], [2, 3, 4], threshold=20))


def test_all_undefined_csi_has_no_best_model():
    table = pd.DataFrame({"Model": ["GFS", "ECMWF", "NCUM"], "CSI": [np.nan, np.nan, np.nan]})
    assert get_best_model_for_metric(table, "CSI") is None


def test_all_undefined_roc_auc_has_no_best_model():
    table = pd.DataFrame({"Model": ["GFS", "ECMWF", "NCUM"], "ROC-AUC": [np.nan, np.nan, np.nan]})
    assert get_best_model_for_metric(table, "ROC-AUC") is None


def test_bias_ranking_uses_distance_from_zero():
    table = pd.DataFrame({"Model": ["GFS", "ECMWF", "NCUM"], "Bias": [.35, -.04, .10]})
    assert get_best_model_for_metric(table, "Bias") == "ECMWF"


def test_finite_metrics_and_metric_sanitising_are_safe():
    truth, prediction = np.array([1., 2., 3.]), np.array([1.2, 1.8, 3.1])
    assert np.isfinite(rmse(prediction, truth))
    assert np.isfinite(mae(prediction, truth))
    assert np.isfinite(bias(prediction, truth))
    assert np.isfinite(correlation(prediction, truth))
    clean = sanitise_metric_table(pd.DataFrame({"Model": ["GFS"], "RMSE": [np.inf], "Correlation": [np.nan]}))
    assert clean["RMSE"].isna().all()
    assert clean["Correlation"].isna().all()
    records = metric_records(clean)
    assert records[0]["RMSE"] is None
    assert records[0]["Correlation"] is None


@pytest.mark.parametrize("scenario_name", DEMO_SCENARIOS)
def test_every_demo_scenario_and_lead_time_has_safe_verification(scenario_name):
    result = AegisPipeline().run(seed=26081, scenario=scenario_name)
    for variable in ("precipitation", "temperature_2m", "wind_speed_10m"):
        table = sanitise_metric_table(benchmark(result, variable))
        assert len(table) == 6
        for metric in METRIC_DIRECTIONS:
            # Undefined scientific metrics are expected in some cases; ranking must never raise.
            assert get_best_model_for_metric(table, metric) in set(table["Model"]) | {None}

        by_lead = benchmark_by_lead(result, variable)
        assert set(by_lead["lead_time"]) == {0, 24, 48, 72}
        assert all(len(by_lead[by_lead["lead_time"] == lead]) == 6 for lead in (0, 24, 48, 72))
        assert line_comparison(by_lead).to_json()


def test_verification_chart_handles_an_entirely_undefined_metric():
    frame = pd.DataFrame({"lead_time": [0, 24], "Model": ["GFS", "GFS"], "RMSE": [np.nan, np.inf]})
    assert "metric undefined" in line_comparison(frame).to_json()


def test_verify_api_serialises_undefined_metrics_as_null():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.main import app

    response = TestClient(app).post("/verify", json={"lead_time": 24, "variable": "precipitation", "regime": "NORMAL", "seed": 26081})
    assert response.status_code == 200
    assert all(row["ROC-AUC"] is None for row in response.json())
