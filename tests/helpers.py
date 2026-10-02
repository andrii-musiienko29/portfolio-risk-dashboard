"""Shared synthetic data for the tests (no network needed)."""

from __future__ import annotations

import pandas as pd

from portfolio_risk import data

ASSETS = ["EQ1", "EQ2", "EQ3", "BOND", "GOLD", "MKT"]


def sample_prices(start: str = "2006-01-02", end: str = "2023-12-29") -> pd.DataFrame:
    return data.demo_prices(ASSETS, start, end, seed=11, market_like=("MKT",))


def sample_returns(start: str = "2016-01-01") -> pd.DataFrame:
    return data.to_returns(data.clean_prices(sample_prices().loc[start:]))
