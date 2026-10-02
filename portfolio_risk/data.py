"""Loading price data and turning it into returns."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252

PRESETS: dict[str, list[str]] = {
    "Multi-asset ETFs": ["SPY", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "VNQ", "DBC"],
    "US large caps": ["AAPL", "MSFT", "AMZN", "NVDA", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"],
    "FTSE MIB leaders": [
        "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "G.MI",
        "RACE.MI", "STLAM.MI", "PRY.MI", "MONC.MI", "TIT.MI",
    ],
}

DEFAULT_BENCHMARKS: dict[str, str] = {
    "Multi-asset ETFs": "SPY",
    "US large caps": "SPY",
    "FTSE MIB leaders": "FTSEMIB.MI",
}


def parse_tickers(text: str) -> list[str]:
    """Split free text ("aapl, msft; ENI.MI") into a de-duplicated, upper-case ticker list."""
    tickers: list[str] = []
    for chunk in text.replace(";", ",").replace("\n", ",").split(","):
        for token in chunk.split():
            symbol = token.strip().upper()
            if symbol and symbol not in tickers:
                tickers.append(symbol)
    return tickers


def download_prices(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """Download dividend- and split-adjusted close prices from Yahoo Finance.

    Returns a DataFrame indexed by date with one column per ticker that returned data.
    Columns keep the order of ``tickers``.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("yfinance is needed for live data: pip install yfinance") from exc

    tickers = list(dict.fromkeys(tickers))
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )
    if raw is None or raw.empty:
        raise ValueError(
            "Yahoo Finance returned no data. Check the tickers and your internet connection."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            prices = raw.xs("Close", axis=1, level=1)
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices = prices.dropna(axis=1, how="all")
    prices.index = pd.DatetimeIndex(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    ordered = [t for t in tickers if t in prices.columns]
    return prices[ordered].sort_index().astype(float)


def fetch_market_caps(tickers: Iterable[str]) -> pd.Series:
    """Best-effort market capitalisations (NaN where Yahoo has none, e.g. most ETFs)."""
    import yfinance as yf

    caps = {}
    for ticker in tickers:
        try:
            caps[ticker] = float(yf.Ticker(ticker).fast_info["marketCap"])
        except Exception:  # noqa: BLE001 - Yahoo is unreliable, treat any failure as missing
            caps[ticker] = np.nan
    return pd.Series(caps, dtype=float)


# Calm drift with three crisis regimes, so the demo data has something to stress-test.
_DEMO_REGIMES = [
    ("2008-09-01", "2009-03-09", -0.0040, 0.030),
    ("2011-07-22", "2011-10-03", -0.0020, 0.020),
    ("2020-02-20", "2020-03-23", -0.0120, 0.045),
    ("2022-01-03", "2022-10-12", -0.0008, 0.014),
]


def demo_prices(
    tickers: Sequence[str],
    start: str,
    end: str,
    seed: int = 7,
    market_like: Sequence[str] = (),
) -> pd.DataFrame:
    """Synthetic prices from a one-factor model, for offline use and tests.

    Each asset gets a random market beta (some negative, bond-like), a small alpha and
    idiosyncratic noise. Tickers in ``market_like`` track the factor closely so they can
    serve as a benchmark.
    """
    dates = pd.bdate_range(start, end)
    n, t = len(tickers), len(dates)
    rng = np.random.default_rng(seed)

    f_mu = np.full(t, 0.0004)
    f_sd = np.full(t, 0.009)
    for s, e, mu, sd in _DEMO_REGIMES:
        mask = (dates >= s) & (dates <= e)
        f_mu[mask], f_sd[mask] = mu, sd
    factor = rng.normal(f_mu, f_sd)

    betas = rng.uniform(-0.3, 1.4, n)
    alphas = rng.uniform(-0.0001, 0.0003, n)
    idio = rng.uniform(0.004, 0.015, n)
    for i, ticker in enumerate(tickers):
        if ticker in market_like:
            betas[i], alphas[i], idio[i] = 1.0, 0.0, 0.002

    rets = alphas + np.outer(factor, betas) + rng.normal(0.0, 1.0, (t, n)) * idio
    rets = np.clip(rets, -0.5, 0.5)
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(prices, index=dates, columns=list(tickers))


def clean_prices(prices: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Forward-fill short gaps (holidays on different exchanges), then keep only dates
    where every asset has a price."""
    return prices.ffill(limit=max_gap).dropna(how="any")


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().iloc[1:]
