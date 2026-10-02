"""Smoke test of the Streamlit app itself (runs in CI, where Streamlit is installed)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _demo_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["source"] = "Demo data (offline)"
    return at.run()


def test_app_runs_on_demo_data():
    at = _demo_app()
    assert not at.exception
    assert not at.error


def test_backtest_toggle_runs():
    at = _demo_app()
    at.toggle(key="run_bt").set_value(True).run()
    assert not at.exception
