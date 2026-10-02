"""Portfolio construction: mean-variance, risk parity and hierarchical risk parity.

All inputs are annualised (expected returns as a Series, covariance as a DataFrame) and
all functions return a weight Series indexed by asset that sums to 1 (long-only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from .estimators import correlation

Bounds = tuple[float, float]


# --------------------------------------------------------------------------- helpers
def check_bounds(n: int, bounds: Bounds) -> None:
    lo, hi = bounds
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError("Weight bounds must satisfy 0 <= min <= max <= 1.")
    if lo * n > 1 + 1e-9 or hi * n < 1 - 1e-9:
        raise ValueError(
            f"Weight bounds {lo:.0%}-{hi:.0%} are infeasible for {n} assets: "
            "they must allow the weights to add up to 100%."
        )


def clean_weights(w: np.ndarray, cutoff: float = 1e-5) -> np.ndarray:
    """Zero out numerical noise and renormalise."""
    w = np.asarray(w, dtype=float).copy()
    w[np.abs(w) < cutoff] = 0.0
    w[w < 0] = 0.0
    return w / w.sum()


def portfolio_performance(
    weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0
) -> tuple[float, float, float]:
    """Ex-ante (expected return, volatility, Sharpe ratio)."""
    w = weights.reindex(mu.index).fillna(0.0).values
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ cov.values @ w))
    return ret, vol, (ret - rf) / vol if vol > 0 else np.nan


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance contributed by each asset (sums to 1)."""
    w = weights.reindex(cov.index).fillna(0.0).values
    marginal = cov.values @ w
    total = w @ marginal
    return pd.Series(w * marginal / total, index=cov.index)


def _solve(objective, n: int, bounds: Bounds, constraints=(), jac=None) -> np.ndarray:
    check_bounds(n, bounds)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}]
    cons.extend(constraints)
    x0 = np.clip(np.full(n, 1.0 / n), *bounds)
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=jac,
        bounds=[bounds] * n,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not res.success:
        raise RuntimeError(f"Optimiser did not converge: {res.message}")
    return clean_weights(res.x)


# --------------------------------------------------------------------------- mean-variance
def equal_weight(assets) -> pd.Series:
    assets = list(assets)
    return pd.Series(1.0 / len(assets), index=assets)


def min_volatility(cov: pd.DataFrame, bounds: Bounds = (0.0, 1.0)) -> pd.Series:
    """Global minimum-variance portfolio."""
    s = cov.values
    w = _solve(lambda w: w @ s @ w, len(s), bounds, jac=lambda w: 2 * s @ w)
    return pd.Series(w, index=cov.index)


def max_sharpe(
    mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0, bounds: Bounds = (0.0, 1.0)
) -> pd.Series:
    """Tangency portfolio: maximise (return - rf) / volatility."""
    m, s = mu.values, cov.loc[mu.index, mu.index].values
    if np.all(m <= rf):
        raise ValueError(
            "Every expected return is below the risk-free rate, so the maximum-Sharpe "
            "portfolio is not meaningful. Try a longer window or a lower risk-free rate."
        )

    def neg_sharpe(w):
        vol = np.sqrt(w @ s @ w)
        return -(w @ m - rf) / vol

    def grad(w):
        vol = np.sqrt(w @ s @ w)
        excess = w @ m - rf
        return -(m * vol - excess * (s @ w) / vol) / vol**2

    w = _solve(neg_sharpe, len(m), bounds, jac=grad)
    return pd.Series(w, index=mu.index)


def efficient_return(
    mu: pd.Series, cov: pd.DataFrame, target: float, bounds: Bounds = (0.0, 1.0)
) -> pd.Series:
    """Minimum-volatility portfolio with a given expected return."""
    m, s = mu.values, cov.loc[mu.index, mu.index].values
    con = {"type": "eq", "fun": lambda w: w @ m - target, "jac": lambda w: m}
    w = _solve(lambda w: w @ s @ w, len(m), bounds, constraints=[con], jac=lambda w: 2 * s @ w)
    return pd.Series(w, index=mu.index)


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, bounds: Bounds = (0.0, 1.0), n_points: int = 40
) -> pd.DataFrame:
    """Points on the efficient frontier, from the minimum-variance portfolio to the
    highest achievable return under the bounds."""
    n = len(mu)
    check_bounds(n, bounds)
    w_min = min_volatility(cov.loc[mu.index, mu.index], bounds)
    r_low = float(w_min @ mu)
    lp = linprog(-mu.values, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[bounds] * n)
    r_high = -lp.fun if lp.success else float(mu.max())

    rows = []
    for target in np.linspace(r_low, r_high - 1e-6 * max(abs(r_high), 1), n_points):
        try:
            w = efficient_return(mu, cov, target, bounds)
        except RuntimeError:
            continue
        ret, vol, _ = portfolio_performance(w, mu, cov)
        rows.append({"return": ret, "volatility": vol})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- risk parity
def risk_parity(cov: pd.DataFrame, budgets: pd.Series | None = None) -> pd.Series:
    """Equal-risk-contribution portfolio (or custom risk budgets).

    Solved via the convex formulation min 0.5 y'Sy - sum(b_i log y_i), whose optimum
    satisfies y_i (S y)_i = b_i, i.e. risk contributions proportional to the budgets.
    Weight bounds are not applied: the point is to let risk, not caps, set the weights.
    """
    s = cov.values
    n = len(s)
    b = np.full(n, 1.0 / n) if budgets is None else budgets.reindex(cov.index).values
    b = b / b.sum()

    res = minimize(
        lambda y: 0.5 * y @ s @ y - b @ np.log(y),
        x0=1.0 / np.sqrt(np.diag(s)),
        jac=lambda y: s @ y - b / y,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * n,
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
    )
    if not res.success:
        raise RuntimeError(f"Risk parity did not converge: {res.message}")
    return pd.Series(res.x / res.x.sum(), index=cov.index)


# --------------------------------------------------------------------------- HRP
def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    sub = cov.loc[items, items].values
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_linkage(cov: pd.DataFrame, method: str = "single") -> np.ndarray:
    """Hierarchical clustering of assets on correlation distance sqrt((1 - rho) / 2)."""
    dist = np.sqrt(0.5 * (1.0 - correlation(cov).values))
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method=method)


def hrp(cov: pd.DataFrame, method: str = "single") -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    1. Cluster assets by correlation distance.
    2. Re-order them so similar assets sit next to each other (quasi-diagonalisation).
    3. Recursively split the ordered list in two, sharing capital between the halves in
       inverse proportion to their variance.
    Needs no matrix inversion, so it is robust when the covariance is noisy.
    """
    order = list(cov.index[leaves_list(hrp_linkage(cov, method))])
    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [
            half
            for c in clusters
            if len(c) > 1
            for half in (c[: len(c) // 2], c[len(c) // 2 :])
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_left, v_right = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - v_left / (v_left + v_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights.reindex(cov.index)
