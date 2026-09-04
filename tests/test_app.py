from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_overview_and_safety_scenarios():
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert any("Adaptive blend active" in item.value for item in app.success)

    app.selectbox(key="scenario").select("EXTREME_DISAGREEMENT").run()
    assert not app.exception
    assert any("Safety fallback activated" in item.value for item in app.error)

    app.selectbox(key="scenario").select("MISSING_MODEL_INPUT").run()
    assert not app.exception
    assert any("Safety fallback activated" in item.value for item in app.error)

    app.toggle(key="presentation_mode").set_value(True).run()
    assert not app.exception


def test_verification_page_handles_missing_model_metrics():
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30).run()
    app.selectbox(key="scenario").select("MISSING_MODEL_INPUT").run()
    app.selectbox[-1].select("Verification").run()
    assert not app.exception
