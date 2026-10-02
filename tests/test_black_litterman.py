import numpy as np
import pandas as pd

from portfolio_risk import black_litterman as bl, estimators, optimizers as opt

from .helpers import sample_returns

COV = estimators.covariance(sample_returns())
ASSETS = list(COV.index)
PRIOR = pd.Series([0.3, 0.2, 0.1, 0.25, 0.1, 0.05], index=ASSETS)


def test_no_views_returns_the_prior():
    P, Q, c = bl.absolute_views(ASSETS, [])
    res = bl.black_litterman(COV, PRIOR, P, Q, c)
    np.testing.assert_allclose(res.posterior.values, res.prior.values)


def test_no_views_max_sharpe_recovers_market_portfolio():
    """Reverse optimisation round-trip: the tangency portfolio of the implied returns is
    the prior portfolio itself."""
    P, Q, c = bl.absolute_views(ASSETS, [])
    res = bl.black_litterman(COV, PRIOR, P, Q, c)
    w = opt.max_sharpe(res.posterior, res.posterior_cov, rf=0.0)
    np.testing.assert_allclose(w.values, PRIOR.values, atol=1e-4)


def test_confidence_pulls_posterior_towards_view():
    view = 0.25
    gaps = []
    for conf in (0.1, 0.5, 0.95):
        P, Q, c = bl.absolute_views(ASSETS, [("GOLD", view, conf)])
        res = bl.black_litterman(COV, PRIOR, P, Q, c)
        gaps.append(abs(res.posterior["GOLD"] - view))
    assert gaps[0] > gaps[1] > gaps[2]


def test_fifty_percent_confidence_lands_halfway():
    P, Q, c = bl.absolute_views(ASSETS, [("EQ1", 0.30, 0.5)])
    res = bl.black_litterman(COV, PRIOR, P, Q, c)
    midpoint = (res.prior["EQ1"] + 0.30) / 2
    assert np.isclose(res.posterior["EQ1"], midpoint)


def test_views_on_unknown_assets_are_ignored():
    P, Q, c = bl.absolute_views(ASSETS, [("NOT_THERE", 0.1, 0.5)])
    assert P.shape == (0, len(ASSETS))


def test_absolute_views_subtract_risk_free_rate():
    _, Q, _ = bl.absolute_views(ASSETS, [("EQ1", 0.08, 0.5)], rf=0.03)
    assert np.isclose(Q[0], 0.05)
