"""Stress testing: replay historical crises and apply hypothetical shocks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "Global Financial Crisis (Oct 2007 - Mar 2009)": ("2007-10-09", "2009-03-09"),
    "Lehman collapse (Sep - Nov 2008)": ("2008-09-12", "2008-11-20"),
    "Euro debt crisis & US downgrade (Jul - Oct 2011)": ("2011-07-22", "2011-10-03"),
    "COVID-19 crash (Feb - Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Inflation & rate shock (Jan - Oct 2022)": ("2022-01-03", "2022-10-12"),
}


@dataclass
class ScenarioResult:
    name: str
    portfolio_return: float
    max_drawdown: float
    coverage: float                      # share of portfolio weight with data in the window
    asset_returns: pd.Series
    path: pd.Series                      # buy-and-hold value, starting at 1
    missing: list[str] = field(default_factory=list)


def historical_scenario(
    prices: pd.DataFrame, weights: pd.Series, start: str, end: str, name: str = ""
) -> ScenarioResult:
    """Buy-and-hold the portfolio through a past window.

    Assets that did not trade at the start of the window (e.g. listed later) are
    dropped and the remaining weights rescaled; ``coverage`` reports how much of the
    portfolio that leaves, so the result is never silently misleading.
    """
    w = weights[weights > 0]
    # Forward-fill from the full history so an exchange holiday on the start date still
    # finds the previous close, while an asset that listed later stays NaN at the start.
    filled = prices[list(w.index)].ffill().loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if filled.empty:
        return ScenarioResult(name, np.nan, np.nan, 0.0, pd.Series(dtype=float),
                              pd.Series(dtype=float), list(w.index))

    start_px, end_px = filled.iloc[0], filled.iloc[-1]
    available = start_px.notna() & end_px.notna()
    missing = list(w.index[~available.values])
    if not available.any():
        return ScenarioResult(name, np.nan, np.nan, 0.0, pd.Series(dtype=float),
                              pd.Series(dtype=float), missing)

    have = list(w.index[available.values])
    w_have = w[have] / w[have].sum()
    asset_returns = end_px[have] / start_px[have] - 1.0
    path = (filled[have] / start_px[have]).fillna(1.0) @ w_have
    drawdown = (path / path.cummax() - 1.0).min()
    return ScenarioResult(
        name=name,
        portfolio_return=float(path.iloc[-1] - 1.0),
        max_drawdown=float(drawdown),
        coverage=float(w[have].sum() / w.sum()),
        asset_returns=asset_returns,
        path=path,
        missing=missing,
    )


def run_historical_scenarios(
    prices: pd.DataFrame,
    weights: dict[str, pd.Series],
    scenarios: dict[str, tuple[str, str]] = HISTORICAL_SCENARIOS,
) -> dict[str, dict[str, ScenarioResult]]:
    """{scenario: {strategy: result}}"""
    return {
        scen: {name: historical_scenario(prices, w, s, e, scen) for name, w in weights.items()}
        for scen, (s, e) in scenarios.items()
    }


def betas(returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
    """Each asset's beta to the benchmark: Cov(r_i, r_m) / Var(r_m)."""
    joined = returns.join(benchmark_returns.rename("__bench__"), how="inner").dropna()
    bench = joined.pop("__bench__")
    return joined.apply(lambda col: col.cov(bench)) / bench.var()


def shock_impact(weights: pd.Series, asset_shocks: pd.Series) -> float:
    """Instant portfolio return if each asset moves by ``asset_shocks``."""
    return float(weights.reindex(asset_shocks.index).fillna(0.0) @ asset_shocks)


def beta_shock(weights: pd.Series, asset_betas: pd.Series, market_move: float) -> float:
    """Portfolio impact of a market move, transmitted to each asset through its beta."""
    return shock_impact(weights, asset_betas * market_move)
