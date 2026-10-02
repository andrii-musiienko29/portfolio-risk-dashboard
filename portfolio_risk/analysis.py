"""End-to-end pipeline: settings in, every result the dashboard displays out.

Keeping this separate from the Streamlit app means all the maths is unit-testable and
the app file only deals with layout and charts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import black_litterman as bl
from . import estimators
from . import optimizers as opt
from .backtest import StrategyFn

STRATEGIES = [
    "Equal weight",
    "Max Sharpe (MVO)",
    "Min volatility (MVO)",
    "Black-Litterman",
    "Risk parity",
    "HRP",
]

View = tuple[str, float, float]  # (asset, expected annual return, confidence 0-1)


@dataclass(frozen=True)
class Settings:
    return_method: str = "historical"      # "historical" | "ema"
    cov_method: str = "ledoit_wolf"        # "ledoit_wolf" | "sample"
    rf: float = 0.02
    bounds: tuple[float, float] = (0.0, 0.4)
    prior: str = "equal"                   # "equal" | "inverse_vol" | "custom"
    tau: float = 0.05
    risk_aversion: float = 2.5
    views: tuple[View, ...] = ()


@dataclass
class AnalysisResult:
    mu: pd.Series
    cov: pd.DataFrame
    weights: dict[str, pd.Series]
    errors: dict[str, str]
    bl_result: bl.BlackLittermanResult
    bl_returns: pd.Series                  # posterior *total* returns
    notes: list[str] = field(default_factory=list)


def prior_weights(
    cov: pd.DataFrame, mode: str, custom: pd.Series | None = None
) -> pd.Series:
    """Black-Litterman prior ('market') portfolio."""
    if mode == "inverse_vol":
        iv = 1.0 / np.sqrt(np.diag(cov.values))
        return pd.Series(iv / iv.sum(), index=cov.index)
    if mode == "custom" and custom is not None:
        c = custom.reindex(cov.index)
        if c.notna().all() and c.sum() > 0:
            return c / c.sum()
    return opt.equal_weight(cov.index)


def _black_litterman(
    mu_index: Sequence[str], cov: pd.DataFrame, s: Settings, custom_prior: pd.Series | None
) -> tuple[bl.BlackLittermanResult, pd.Series]:
    prior = prior_weights(cov, s.prior, custom_prior)
    P, Q, conf = bl.absolute_views(list(mu_index), s.views, s.rf)
    res = bl.black_litterman(cov, prior, P, Q, conf, tau=s.tau, risk_aversion=s.risk_aversion)
    return res, res.posterior + s.rf


def run_analysis(
    returns: pd.DataFrame, s: Settings, custom_prior: pd.Series | None = None
) -> AnalysisResult:
    mu = estimators.expected_returns(returns, s.return_method)
    cov = estimators.covariance(returns, s.cov_method)
    bl_res, bl_mu = _black_litterman(mu.index, cov, s, custom_prior)

    builders = {
        "Equal weight": lambda: opt.equal_weight(mu.index),
        "Max Sharpe (MVO)": lambda: opt.max_sharpe(mu, cov, s.rf, s.bounds),
        "Min volatility (MVO)": lambda: opt.min_volatility(cov, s.bounds),
        "Black-Litterman": lambda: opt.max_sharpe(bl_mu, bl_res.posterior_cov, s.rf, s.bounds),
        "Risk parity": lambda: opt.risk_parity(cov),
        "HRP": lambda: opt.hrp(cov),
    }
    weights, errors = {}, {}
    for name in STRATEGIES:
        try:
            weights[name] = builders[name]()
        except Exception as exc:  # noqa: BLE001 - surface the message in the UI
            errors[name] = str(exc)

    notes = []
    caps_missing = custom_prior is None or custom_prior.reindex(cov.index).isna().any()
    if s.prior == "custom" and caps_missing:
        notes.append("Market caps were unavailable for some assets, so the Black-Litterman "
                     "prior fell back to equal weights.")
    return AnalysisResult(mu, cov, weights, errors, bl_res, bl_mu, notes)


def strategy_functions(
    s: Settings, custom_prior: pd.Series | None = None
) -> dict[str, StrategyFn]:
    """Strategies as functions of a returns window, for the walk-forward backtest."""

    def estimate(win):
        return (estimators.expected_returns(win, s.return_method),
                estimators.covariance(win, s.cov_method))

    def ms(win):
        mu, cov = estimate(win)
        return opt.max_sharpe(mu, cov, s.rf, s.bounds)

    def mv(win):
        return opt.min_volatility(estimators.covariance(win, s.cov_method), s.bounds)

    def blm(win):
        mu, cov = estimate(win)
        res, bl_mu = _black_litterman(mu.index, cov, s, custom_prior)
        return opt.max_sharpe(bl_mu, res.posterior_cov, s.rf, s.bounds)

    return {
        "Equal weight": lambda win: opt.equal_weight(win.columns),
        "Max Sharpe (MVO)": ms,
        "Min volatility (MVO)": mv,
        "Black-Litterman": blm,
        "Risk parity": lambda win: opt.risk_parity(estimators.covariance(win, s.cov_method)),
        "HRP": lambda win: opt.hrp(estimators.covariance(win, s.cov_method)),
    }
