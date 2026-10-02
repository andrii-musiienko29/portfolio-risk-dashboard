"""Walk-forward backtesting.

At each rebalance date a strategy only sees the trailing ``lookback`` days of returns,
sets target weights at that day's close, and holds them (letting them drift with prices)
until the next rebalance. This is an honest out-of-sample test: no future data is used,
unlike evaluating weights on the same data they were optimised on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

StrategyFn = Callable[[pd.DataFrame], pd.Series]

FREQUENCIES = {"Monthly": "M", "Quarterly": "Q", "Annually": "Y"}


@dataclass
class BacktestResult:
    returns: pd.DataFrame                 # daily strategy returns (net of costs)
    weights: dict[str, pd.DataFrame]      # target weights at each rebalance
    turnover: pd.Series                   # average one-way annual turnover
    fallbacks: pd.Series                  # rebalances where the optimiser failed


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "Quarterly") -> set[pd.Timestamp]:
    """Last trading day of each month / quarter / year in ``index``."""
    code = FREQUENCIES.get(frequency, frequency)
    s = pd.Series(index, index=index)
    return set(s.groupby(index.to_period(code)).max())


def walk_forward(
    returns: pd.DataFrame,
    strategies: dict[str, StrategyFn],
    lookback: int = 252,
    frequency: str = "Quarterly",
    cost_bps: float = 0.0,
) -> BacktestResult:
    idx, cols = returns.index, returns.columns
    values = returns.values
    n_assets = len(cols)
    rdates = rebalance_dates(idx, frequency)
    schedule = [i for i, d in enumerate(idx) if i >= lookback - 1 and d in rdates]
    if not schedule:
        raise ValueError(
            f"Not enough history: need more than {lookback} days of overlapping data."
        )
    schedule_set, first = set(schedule), schedule[0]
    years = (len(idx) - first - 1) / 252

    out, weight_hist, turnover, fallbacks = {}, {}, {}, {}
    for name, fn in strategies.items():
        daily = np.full(len(idx), np.nan)
        w: np.ndarray | None = None
        pending_cost, total_turnover, n_fail = 0.0, 0.0, 0
        hist = {}
        for i in range(first, len(idx)):
            if w is not None:
                r = values[i]
                gross = float(w @ r)
                daily[i] = gross - pending_cost
                pending_cost = 0.0
                w = w * (1.0 + r) / (1.0 + gross)
            if i in schedule_set:
                window = returns.iloc[i - lookback + 1 : i + 1]
                try:
                    target = fn(window).reindex(cols).fillna(0.0).values
                except Exception:  # noqa: BLE001 - keep the backtest running
                    n_fail += 1
                    target = w if w is not None else np.full(n_assets, 1.0 / n_assets)
                traded = np.abs(target - (w if w is not None else 0.0)).sum()
                if w is not None:
                    total_turnover += traded / 2
                pending_cost = traded * cost_bps / 10_000
                w = target.copy()
                hist[idx[i]] = target
        out[name] = daily
        weight_hist[name] = pd.DataFrame(hist, index=cols).T
        turnover[name] = total_turnover / years if years > 0 else np.nan
        fallbacks[name] = n_fail

    daily_df = pd.DataFrame(out, index=idx).iloc[first + 1 :]
    return BacktestResult(daily_df, weight_hist, pd.Series(turnover), pd.Series(fallbacks))
