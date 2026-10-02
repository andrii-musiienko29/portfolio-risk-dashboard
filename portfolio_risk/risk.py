"""Risk measures: Value-at-Risk, Conditional VaR (expected shortfall), drawdowns and
performance statistics.

VaR and CVaR are reported as positive numbers: a 95% 1-day VaR of 0.02 means "on 95% of
days the portfolio loses less than 2%"; the CVaR is the average loss on the other 5%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from .data import TRADING_DAYS


@dataclass
class VaRResult:
    var: float
    cvar: float
    sample: np.ndarray | None = None  # the return distribution the estimate came from


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily portfolio returns for constant weights (i.e. daily rebalancing)."""
    w = weights.reindex(returns.columns).fillna(0.0)
    return returns @ w


def horizon_returns(r: pd.Series, horizon: int) -> pd.Series:
    """Overlapping compounded h-day returns."""
    if horizon <= 1:
        return r
    return np.expm1(np.log1p(r).rolling(horizon).sum()).dropna()


def _from_sample(x: np.ndarray, alpha: float) -> tuple[float, float]:
    q = np.quantile(x, 1.0 - alpha)
    tail = x[x <= q]
    return float(-q), float(-tail.mean()) if tail.size else float(-q)


def historical_var(r: pd.Series, alpha: float = 0.95, horizon: int = 1) -> VaRResult:
    """Empirical quantile of past (horizon) returns. No distributional assumption, but
    only as good as the history it sees."""
    x = horizon_returns(r, horizon).values
    var, cvar = _from_sample(x, alpha)
    return VaRResult(var, cvar, x)


def parametric_var(r: pd.Series, alpha: float = 0.95, horizon: int = 1) -> VaRResult:
    """Variance-covariance (normal) VaR with square-root-of-time scaling.
    Tends to understate tail risk when returns are fat-tailed."""
    mu = r.mean() * horizon
    sigma = r.std() * np.sqrt(horizon)
    z = norm.ppf(1.0 - alpha)
    var = -(mu + z * sigma)
    cvar = -(mu - sigma * norm.pdf(z) / (1.0 - alpha))
    return VaRResult(float(var), float(cvar))


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.95,
    horizon: int = 1,
    n_sims: int = 10_000,
    method: str = "gaussian",
    seed: int = 42,
) -> VaRResult:
    """Simulate horizon returns and read VaR/CVaR off the simulated distribution.

    method="gaussian": draw daily asset returns from a multivariate normal fitted to the
    data (captures correlations, not fat tails).
    method="bootstrap": resample whole historical days, which keeps fat tails and the
    cross-asset dependence seen on each day.
    """
    rng = np.random.default_rng(seed)
    w = weights.reindex(returns.columns).fillna(0.0).values
    if method == "gaussian":
        mean, cov = returns.mean().values, returns.cov().values
        chunks, chunk = [], 5_000  # simulate in chunks to keep memory bounded
        for start in range(0, n_sims, chunk):
            size = min(chunk, n_sims - start)
            sims = rng.multivariate_normal(mean, cov, size=(size, horizon))
            chunks.append(sims @ w)
        daily = np.concatenate(chunks)
    elif method == "bootstrap":
        port = returns.values @ w
        daily = port[rng.integers(0, len(port), size=(n_sims, horizon))]
    else:
        raise ValueError(f"Unknown Monte Carlo method: {method!r}")
    total = np.prod(1.0 + daily, axis=1) - 1.0
    var, cvar = _from_sample(total, alpha)
    return VaRResult(var, cvar, total)


def var_table(
    returns: pd.DataFrame,
    weights: dict[str, pd.Series],
    alpha: float = 0.95,
    horizon: int = 1,
    n_sims: int = 10_000,
    mc_method: str = "gaussian",
) -> pd.DataFrame:
    """VaR and CVaR for each strategy under all three methods."""
    rows = {}
    for name, w in weights.items():
        r = portfolio_returns(returns, w)
        hist = historical_var(r, alpha, horizon)
        param = parametric_var(r, alpha, horizon)
        mc = monte_carlo_var(returns, w, alpha, horizon, n_sims, mc_method)
        rows[name] = {
            ("Historical", "VaR"): hist.var, ("Historical", "CVaR"): hist.cvar,
            ("Parametric", "VaR"): param.var, ("Parametric", "CVaR"): param.cvar,
            ("Monte Carlo", "VaR"): mc.var, ("Monte Carlo", "CVaR"): mc.cvar,
        }
    table = pd.DataFrame(rows).T
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


def var_backtest(r: pd.Series, alpha: float = 0.95, window: int = 252) -> dict[str, float]:
    """Rolling historical-VaR backtest: how often did the next day's loss exceed the VaR
    estimated from the previous ``window`` days? Should be close to 1 - alpha."""
    var = -r.rolling(window).quantile(1.0 - alpha).shift(1)
    valid = var.dropna()
    breaches = (r.loc[valid.index] < -valid).sum()
    n = len(valid)
    return {
        "observations": n,
        "breaches": int(breaches),
        "breach_rate": breaches / n if n else np.nan,
        "expected_rate": 1.0 - alpha,
    }


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1.0 + r).cumprod()
    return wealth / wealth.cummax() - 1.0


def performance_metrics(
    r: pd.Series, rf: float = 0.0, frequency: int = TRADING_DAYS
) -> dict[str, float]:
    """Realised performance statistics for a daily return series."""
    r = r.dropna()
    years = len(r) / frequency
    cagr = (1.0 + r).prod() ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = r.std() * np.sqrt(frequency)
    excess = r - rf / frequency
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean()) * np.sqrt(frequency)
    mdd = drawdown_series(r).min()
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": excess.mean() * frequency / vol if vol > 0 else np.nan,
        "Sortino": excess.mean() * frequency / downside if downside > 0 else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Worst day": r.min(),
        "Skew": float(skew(r)),
        "Excess kurtosis": float(kurtosis(r)),
    }
