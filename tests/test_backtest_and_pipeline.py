import numpy as np
import pandas as pd
import pytest

from portfolio_risk import analysis, backtest, data

from .helpers import sample_returns

R = sample_returns()


def test_buy_one_asset_reproduces_its_returns():
    strategies = {"all EQ1": lambda win: pd.Series({"EQ1": 1.0})}
    res = backtest.walk_forward(R, strategies, lookback=252, frequency="Quarterly")
    expected = R["EQ1"].loc[res.returns.index].values
    np.testing.assert_allclose(res.returns["all EQ1"].values, expected)


def test_no_look_ahead():
    """Strategies only ever see data up to the rebalance date."""
    seen_until = []

    def spy(win):
        seen_until.append(win.index[-1])
        assert len(win) == 126
        return pd.Series(1 / win.shape[1], index=win.columns)

    res = backtest.walk_forward(R, {"spy": spy}, lookback=126, frequency="Monthly")
    assert all(d < res.returns.index[0] or d in res.weights["spy"].index for d in seen_until)
    assert res.returns.index[0] > seen_until[0]


def test_costs_reduce_returns():
    fns = {"EW": lambda win: pd.Series(1 / win.shape[1], index=win.columns)}
    free = backtest.walk_forward(R, fns, 252, "Monthly", 0.0).returns["EW"]
    costly = backtest.walk_forward(R, fns, 252, "Monthly", 50.0).returns["EW"]
    assert (1 + costly).prod() < (1 + free).prod()


def test_failed_optimiser_falls_back():
    def broken(win):
        raise RuntimeError("boom")

    res = backtest.walk_forward(R, {"broken": broken}, 252, "Quarterly")
    assert res.fallbacks["broken"] > 0
    assert res.returns["broken"].notna().all()


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        backtest.walk_forward(R.iloc[:100], {"x": lambda w: w.iloc[0] * 0 + 0.5}, lookback=252)


def test_full_pipeline_runs_for_every_strategy():
    settings = analysis.Settings(views=(("EQ2", 0.12, 0.7),), bounds=(0.0, 0.5))
    res = analysis.run_analysis(R, settings)
    assert not res.errors
    assert set(res.weights) == set(analysis.STRATEGIES)
    for w in res.weights.values():
        assert np.isclose(w.sum(), 1.0)
    bt = backtest.walk_forward(R, analysis.strategy_functions(settings), 252, "Quarterly", 10)
    assert list(bt.returns.columns) == analysis.STRATEGIES


def test_parse_tickers():
    assert data.parse_tickers("aapl, msft;ENI.MI\n gld  AAPL") == ["AAPL", "MSFT", "ENI.MI", "GLD"]
