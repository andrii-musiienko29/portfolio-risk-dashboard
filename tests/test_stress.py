import numpy as np
import pandas as pd

from portfolio_risk import stress

DATES = pd.bdate_range("2020-01-01", "2020-01-10")
PRICES = pd.DataFrame(
    {
        "A": np.linspace(100, 80, len(DATES)),   # -20%
        "B": np.linspace(50, 55, len(DATES)),    # +10%
        "NEW": [np.nan] * 4 + list(np.linspace(10, 12, len(DATES) - 4)),
    },
    index=DATES,
)


def test_portfolio_return_is_weighted_buy_and_hold():
    res = stress.historical_scenario(PRICES, pd.Series({"A": 0.5, "B": 0.5}),
                                     "2020-01-01", "2020-01-10")
    assert np.isclose(res.portfolio_return, 0.5 * -0.2 + 0.5 * 0.1)
    assert res.coverage == 1.0
    assert res.max_drawdown < 0


def test_missing_assets_are_reported_not_hidden():
    w = pd.Series({"A": 0.5, "B": 0.25, "NEW": 0.25})
    res = stress.historical_scenario(PRICES, w, "2020-01-01", "2020-01-10")
    assert res.missing == ["NEW"]
    assert np.isclose(res.coverage, 0.75)
    assert np.isclose(res.portfolio_return, (0.5 * -0.2 + 0.25 * 0.1) / 0.75)


def test_window_without_data():
    res = stress.historical_scenario(PRICES, pd.Series({"A": 1.0}), "1999-01-01", "1999-02-01")
    assert np.isnan(res.portfolio_return) and res.coverage == 0.0


def test_betas_and_beta_shock():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.01, 2000))
    r = pd.DataFrame({"X": 2 * m + rng.normal(0, 0.001, 2000), "Y": -0.5 * m})
    b = stress.betas(r, m)
    assert np.isclose(b["X"], 2.0, atol=0.02) and np.isclose(b["Y"], -0.5)
    impact = stress.beta_shock(pd.Series({"X": 0.5, "Y": 0.5}), b, -0.10)
    assert np.isclose(impact, 0.5 * b["X"] * -0.1 + 0.5 * b["Y"] * -0.1)
