"""Black-Litterman model (He & Litterman 1999, with Idzorek-style confidences).

The model starts from the returns the market *implies* (reverse optimisation of a prior
portfolio) and blends in the investor's views, weighted by how confident they are. The
result is a set of expected returns that stay close to equilibrium unless a view says
otherwise, which gives far more stable portfolios than raw historical means.

All returns in this module are annualised *excess* returns (over the risk-free rate).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BlackLittermanResult:
    prior: pd.Series          # implied equilibrium excess returns (pi)
    posterior: pd.Series      # blended excess returns
    posterior_cov: pd.DataFrame
    market_weights: pd.Series


def implied_excess_returns(
    cov: pd.DataFrame, market_weights: pd.Series, risk_aversion: float = 2.5
) -> pd.Series:
    """Reverse optimisation: pi = delta * Sigma * w_mkt."""
    w = market_weights.reindex(cov.index).fillna(0.0)
    return risk_aversion * cov @ w


def market_risk_aversion(market_returns: pd.Series, rf: float = 0.0, frequency: int = 252) -> float:
    """delta = (E[R_m] - rf) / Var(R_m), estimated from a market proxy."""
    excess = market_returns.mean() * frequency - rf
    return float(excess / (market_returns.var() * frequency))


def absolute_views(
    assets: Sequence[str],
    views: Sequence[tuple[str, float, float]],
    rf: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn absolute views like ("GLD", 0.08, 0.6) - "gold returns 8% a year, 60% sure" -
    into the pick matrix P, the excess view returns Q and the confidences."""
    assets = list(assets)
    rows, q, conf = [], [], []
    for asset, expected, confidence in views:
        if asset not in assets:
            continue
        row = np.zeros(len(assets))
        row[assets.index(asset)] = 1.0
        rows.append(row)
        q.append(expected - rf)
        conf.append(confidence)
    if not rows:
        return np.zeros((0, len(assets))), np.zeros(0), np.zeros(0)
    return np.vstack(rows), np.array(q), np.array(conf)


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    P: np.ndarray,
    Q: np.ndarray,
    confidences: np.ndarray | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> BlackLittermanResult:
    """Posterior expected returns and covariance.

    View uncertainty Omega is diagonal with omega_k = (1 - c_k) / c_k * (P tau Sigma P')_kk,
    so a confidence of 50% weights a view equally with the prior, and confidences near
    100% pull the posterior almost all the way to the view.
    """
    sigma = cov.values
    pi = implied_excess_returns(cov, market_weights, risk_aversion)
    ts = tau * sigma

    if P.shape[0] == 0:
        posterior, post_cov = pi.values, sigma + ts
    else:
        c = np.full(P.shape[0], 0.5) if confidences is None else np.asarray(confidences, float)
        c = np.clip(c, 1e-3, 0.999)
        base = np.diag(P @ ts @ P.T)
        omega = np.diag((1.0 - c) / c * base)
        middle = np.linalg.inv(P @ ts @ P.T + omega)
        posterior = pi.values + ts @ P.T @ middle @ (Q - P @ pi.values)
        post_cov = sigma + ts - ts @ P.T @ middle @ P @ ts

    idx = cov.index
    return BlackLittermanResult(
        prior=pi,
        posterior=pd.Series(posterior, index=idx),
        posterior_cov=pd.DataFrame((post_cov + post_cov.T) / 2, index=idx, columns=idx),
        market_weights=market_weights.reindex(idx).fillna(0.0),
    )
