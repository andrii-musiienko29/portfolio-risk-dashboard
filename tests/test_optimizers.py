import numpy as np
import pandas as pd
import pytest

from portfolio_risk import estimators, optimizers as opt

from .helpers import sample_returns

R = sample_returns()
MU = estimators.expected_returns(R)
COV = estimators.covariance(R)


def _assert_valid(w: pd.Series, bounds=(0.0, 1.0)):
    assert np.isclose(w.sum(), 1.0)
    assert (w >= bounds[0] - 1e-6).all() and (w <= bounds[1] + 1e-6).all()


def test_min_volatility_beats_every_other_portfolio():
    w = opt.min_volatility(COV)
    _assert_valid(w)
    vol = opt.portfolio_performance(w, MU, COV)[1]
    for other in (opt.equal_weight(MU.index), opt.max_sharpe(MU, COV), opt.hrp(COV)):
        assert vol <= opt.portfolio_performance(other, MU, COV)[1] + 1e-8


def test_max_sharpe_has_highest_sharpe():
    rf = 0.01
    w = opt.max_sharpe(MU, COV, rf)
    _assert_valid(w)
    best = opt.portfolio_performance(w, MU, COV, rf)[2]
    rng = np.random.default_rng(0)
    for _ in range(200):
        rand = pd.Series(rng.dirichlet(np.ones(len(MU))), index=MU.index)
        assert best >= opt.portfolio_performance(rand, MU, COV, rf)[2] - 1e-8


def test_bounds_are_respected():
    bounds = (0.05, 0.3)
    for w in (opt.max_sharpe(MU, COV, 0.0, bounds), opt.min_volatility(COV, bounds)):
        _assert_valid(w, bounds)


def test_infeasible_bounds_raise():
    with pytest.raises(ValueError):
        opt.min_volatility(COV, (0.0, 0.1))  # 6 assets x 10% < 100%


def test_max_sharpe_rejects_all_negative_returns():
    with pytest.raises(ValueError):
        opt.max_sharpe(MU * 0 - 0.05, COV, rf=0.0)


def test_risk_parity_equalises_risk_contributions():
    w = opt.risk_parity(COV)
    _assert_valid(w)
    rc = opt.risk_contributions(w, COV)
    np.testing.assert_allclose(rc.values, 1 / len(rc), atol=1e-6)


def test_risk_parity_custom_budgets():
    budgets = pd.Series([3, 1, 1, 1, 1, 1], index=COV.index, dtype=float)
    rc = opt.risk_contributions(opt.risk_parity(COV, budgets), COV)
    np.testing.assert_allclose(rc.values, (budgets / budgets.sum()).values, atol=1e-6)


def test_hrp_is_long_only_and_sums_to_one():
    w = opt.hrp(COV)
    _assert_valid(w)
    assert (w > 0).all()


def test_hrp_with_uncorrelated_assets_is_inverse_variance():
    var = np.array([0.04, 0.09, 0.01, 0.16])
    cov = pd.DataFrame(np.diag(var), index=list("ABCD"), columns=list("ABCD"))
    expected = (1 / var) / (1 / var).sum()
    np.testing.assert_allclose(opt.hrp(cov).values, expected, rtol=1e-8)


def test_efficient_frontier_is_monotonic():
    frontier = opt.efficient_frontier(MU, COV, (0.0, 1.0), 15)
    assert len(frontier) >= 10
    assert frontier["return"].is_monotonic_increasing
    assert (frontier["volatility"].diff().dropna() > -1e-6).all()
