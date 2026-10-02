import numpy as np
import pandas as pd
from scipy.stats import norm

from portfolio_risk import risk

from .helpers import sample_returns

RNG = np.random.default_rng(1)
NORMAL = pd.Series(RNG.normal(0.0, 0.01, 200_000))


def test_historical_var_matches_normal_quantile():
    res = risk.historical_var(NORMAL, 0.99)
    assert np.isclose(res.var, 0.01 * norm.ppf(0.99), rtol=0.02)


def test_parametric_var_and_cvar_closed_form():
    res = risk.parametric_var(NORMAL, 0.95)
    sigma = NORMAL.std()
    assert np.isclose(res.var, -(NORMAL.mean() - 1.6449 * sigma), rtol=1e-3)
    assert np.isclose(res.cvar, sigma * norm.pdf(norm.ppf(0.95)) / 0.05 - NORMAL.mean(), rtol=1e-3)


def test_cvar_is_at_least_var_for_every_method():
    returns = sample_returns()
    w = pd.Series(1 / returns.shape[1], index=returns.columns)
    table = risk.var_table(returns, {"EW": w}, 0.95, horizon=5, n_sims=5000)
    for method in ("Historical", "Parametric", "Monte Carlo"):
        assert table.loc["EW", (method, "CVaR")] >= table.loc["EW", (method, "VaR")]


def test_gaussian_monte_carlo_agrees_with_parametric():
    returns = sample_returns()
    w = pd.Series(1 / returns.shape[1], index=returns.columns)
    port = risk.portfolio_returns(returns, w)
    mc = risk.monte_carlo_var(returns, w, 0.95, 1, 50_000, "gaussian")
    param = risk.parametric_var(port, 0.95, 1)
    assert np.isclose(mc.var, param.var, rtol=0.05)


def test_bootstrap_is_reproducible():
    returns = sample_returns()
    w = pd.Series(1 / returns.shape[1], index=returns.columns)
    a = risk.monte_carlo_var(returns, w, 0.95, 10, 2000, "bootstrap", seed=3)
    b = risk.monte_carlo_var(returns, w, 0.95, 10, 2000, "bootstrap", seed=3)
    assert a.var == b.var


def test_var_grows_with_horizon():
    r = sample_returns()["EQ1"]
    assert risk.historical_var(r, 0.95, 10).var > risk.historical_var(r, 0.95, 1).var


def test_var_backtest_breach_rate_is_reasonable():
    out = risk.var_backtest(NORMAL.iloc[:5000], 0.95)
    assert 0.03 < out["breach_rate"] < 0.07


def test_drawdown_and_metrics():
    r = pd.Series([0.10, -0.50, 0.20])
    dd = risk.drawdown_series(r)
    assert np.isclose(dd.min(), -0.5)
    m = risk.performance_metrics(sample_returns()["EQ1"])
    assert m["Max drawdown"] <= 0 and m["Volatility"] > 0
