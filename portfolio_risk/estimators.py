"""Estimators for expected returns and covariance (annualised)."""

from __future__ import annotations

import pandas as pd

from .data import TRADING_DAYS


def expected_returns(
    returns: pd.DataFrame,
    method: str = "historical",
    span: int = 180,
    frequency: int = TRADING_DAYS,
) -> pd.Series:
    """Annualised expected returns.

    method="historical": arithmetic mean of daily returns.
    method="ema": exponentially weighted mean, giving recent data more weight.
    """
    if method == "historical":
        mu = returns.mean()
    elif method == "ema":
        mu = returns.ewm(span=span).mean().iloc[-1]
    else:
        raise ValueError(f"Unknown expected-return method: {method!r}")
    return mu * frequency


def covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    frequency: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Annualised covariance matrix.

    method="sample": plain sample covariance.
    method="ledoit_wolf": Ledoit-Wolf shrinkage towards a scaled identity. More stable,
    which matters a lot for optimisers that otherwise amplify estimation noise.
    """
    if method == "sample":
        cov = returns.cov().values
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        cov = LedoitWolf().fit(returns.values).covariance_
    else:
        raise ValueError(f"Unknown covariance method: {method!r}")
    return pd.DataFrame(cov * frequency, index=returns.columns, columns=returns.columns)


def correlation(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix."""
    std = pd.Series(cov.values.diagonal() ** 0.5, index=cov.index)
    return cov.div(std, axis=0).div(std, axis=1).clip(-1.0, 1.0)
