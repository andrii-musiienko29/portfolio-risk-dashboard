# Portfolio Optimisation & Risk Dashboard

![CI](https://github.com/andrii-musiienko29/portfolio-risk-dashboard/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An interactive Streamlit app that builds portfolios with four institutional allocation methods,
compares them on the same data, and measures their risk with Value-at-Risk, Conditional VaR,
historical crisis replays and an out-of-sample walk-forward backtest.

**[▶ Live demo](https://YOUR-APP.streamlit.app)**

![Dashboard screenshot](docs/screenshot.png)

## What it does

Pick any set of Yahoo Finance tickers (US stocks, ETFs, Borsa Italiana `.MI` stocks…) and the app:

| Area | Details |
|---|---|
| **Portfolio construction** | Equal weight · Max-Sharpe and min-volatility mean-variance (long-only, per-asset bounds) · Black-Litterman with your own views and confidence levels · Risk parity (equal risk contribution) · Hierarchical Risk Parity |
| **Estimation** | Historical or exponentially weighted returns · Sample or Ledoit-Wolf shrinkage covariance |
| **Analytics** | Efficient frontier · Risk contributions · Correlation heatmap · Prior vs posterior returns |
| **VaR & CVaR** | Historical, parametric (normal) and Monte Carlo (Gaussian or bootstrap), any confidence level and horizon, in % or € · Rolling VaR backtest (breach rate vs expected) |
| **Stress tests** | Replays of the 2008 GFC, Lehman, 2011 euro crisis, COVID crash and 2022 rate shock · Beta-transmitted market shock · Custom per-asset shocks |
| **Backtest** | Walk-forward rebalancing (monthly/quarterly/annual), trailing estimation window, transaction costs, turnover |

A **demo data** mode generates synthetic prices, so the app also works offline.

## Quick start

```bash
git clone https://github.com/andrii-musiienko29/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Project structure

```
portfolio-risk-dashboard/
├── app.py                     # Streamlit UI (layout and charts only)
├── portfolio_risk/            # All the maths, independent of the UI
│   ├── data.py                # Yahoo Finance download, demo data, returns
│   ├── estimators.py          # Expected returns, covariance (Ledoit-Wolf)
│   ├── optimizers.py          # MVO, efficient frontier, risk parity, HRP
│   ├── black_litterman.py     # Implied returns, views, posterior
│   ├── risk.py                # VaR/CVaR (3 methods), drawdowns, metrics
│   ├── stress.py              # Historical scenarios, beta and custom shocks
│   ├── backtest.py            # Walk-forward backtest with costs
│   └── analysis.py            # One-call pipeline used by the app
└── tests/                     # pytest suite (runs in GitHub Actions)
```

Keeping the analytics in a package separate from the UI means every number on screen is covered
by unit tests and the library can be reused from a notebook:

```python
from portfolio_risk import data, estimators, optimizers, risk

prices = data.download_prices(["SPY", "TLT", "GLD", "EFA"], "2015-01-01", "2025-01-01")
returns = data.to_returns(data.clean_prices(prices))
cov = estimators.covariance(returns, "ledoit_wolf")

weights = optimizers.hrp(cov)
print(risk.historical_var(risk.portfolio_returns(returns, weights), alpha=0.99))
```

## Methods in brief

- **Mean-variance**: maximise Sharpe or minimise variance with SLSQP under long-only bounds.
- **Black-Litterman**: implied equilibrium returns $\pi = \delta \Sigma w_{mkt}$ are blended with absolute views; view uncertainty $\Omega$ is set from user confidence (Idzorek-style), so a 50% confident view lands halfway between prior and view.
- **Risk parity**: solves $\min_y \tfrac12 y^\top \Sigma y - \sum_i b_i \log y_i$, whose optimum gives risk contributions equal to the budgets $b_i$.
- **HRP** (López de Prado, 2016): single-linkage clustering on $\sqrt{(1-\rho)/2}$, quasi-diagonalisation, recursive bisection by inverse cluster variance.
- **VaR/CVaR**: empirical quantiles of overlapping horizon returns; normal closed form; Monte Carlo from a multivariate normal or a bootstrap of historical days.

The tests check the theory, not just that code runs: risk parity gives equal risk contributions,
Black-Litterman with no views recovers the market portfolio, HRP on uncorrelated assets equals
inverse-variance weights, Gaussian Monte Carlo VaR converges to the parametric value, and the
backtest never sees future data.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest -v
ruff check .
```

## Deploying a live demo

1. Push the repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) with GitHub.
3. Click **New app**, choose the repo, branch `main` and file `app.py`, then deploy.
4. Paste the URL into the "Live demo" link at the top of this README.

## Limitations

- All estimates come from one historical window, and markets change regime.
- In-sample results flatter mean-variance portfolios; judge strategies on the walk-forward backtest.
- Crisis replays use today's weights and leave out assets that did not exist yet (the app reports coverage).
- The market-cap prior for Black-Litterman uses current caps, which adds look-ahead bias to the backtest.
- Yahoo Finance data is free and occasionally wrong.

This project is for education and demonstration. It is **not investment advice**.

## License

MIT
