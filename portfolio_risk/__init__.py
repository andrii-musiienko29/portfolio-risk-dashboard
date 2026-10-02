"""Portfolio optimisation and risk analytics.

Modules
-------
data            Price download (Yahoo Finance), offline demo data, returns.
estimators      Expected-return and covariance estimators.
optimizers      Mean-variance, risk parity and hierarchical risk parity.
black_litterman Black-Litterman prior/posterior returns.
risk            VaR / CVaR (historical, parametric, Monte Carlo) and performance metrics.
stress          Historical crisis replays and hypothetical shocks.
backtest        Walk-forward (out-of-sample) backtesting with rebalancing costs.
analysis        One-call pipeline used by the Streamlit app.
"""

__version__ = "1.0.0"
