"""Portfolio Optimisation & Risk Dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portfolio_risk import analysis, backtest, data, estimators, optimizers as opt, risk, stress
from portfolio_risk.analysis import STRATEGIES

st.set_page_config(
    page_title="Portfolio Optimisation & Risk Dashboard", page_icon="📈", layout="wide"
)

STRESS_HISTORY_START = dt.date(2007, 1, 1)
PALETTE = px.colors.qualitative.Set2
COLOR = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(STRATEGIES)}


# ============================================================================ helpers
def fmt_pct(df: pd.DataFrame, decimals: int = 2):
    return df.style.format(f"{{:.{decimals}%}}", na_rep="–")


def fmt_metrics(df: pd.DataFrame):
    pct_cols = [c for c in ["CAGR", "Volatility", "Max drawdown", "Worst day",
                            "Expected return", "Expected volatility"] if c in df.columns]
    num_cols = [c for c in df.columns if c not in pct_cols]
    return df.style.format("{:.2%}", subset=pct_cols, na_rep="–").format(
        "{:.2f}", subset=num_cols, na_rep="–")


@st.cache_data(show_spinner="Loading prices…", ttl=3600)
def load_prices(tickers: tuple[str, ...], start: str, end: str, demo: bool,
                market_like: tuple[str, ...]) -> pd.DataFrame:
    if demo:
        return data.demo_prices(list(tickers), start, end, market_like=market_like)
    return data.download_prices(list(tickers), start, end)


@st.cache_data(show_spinner="Fetching market caps…", ttl=86400)
def load_market_caps(tickers: tuple[str, ...]) -> pd.Series:
    return data.fetch_market_caps(tickers)


@st.cache_data(show_spinner="Computing VaR and CVaR…")
def cached_var_table(returns, weights_df, alpha, horizon, n_sims, mc_method):
    weights = {c: weights_df[c] for c in weights_df.columns}
    return risk.var_table(returns, weights, alpha, horizon, n_sims, mc_method)


@st.cache_data(show_spinner="Running walk-forward backtest…")
def cached_backtest(returns, settings_key: str, _settings, _custom_prior,
                    lookback: int, frequency: str, cost_bps: float):
    fns = analysis.strategy_functions(_settings, _custom_prior)
    return backtest.walk_forward(returns, fns, lookback, frequency, cost_bps)


# ============================================================================ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    source = st.radio(
        "Data source", ["Yahoo Finance (live)", "Demo data (offline)"], key="source",
        help="Demo data is synthetic and works without an internet connection.",
    )
    demo = source.startswith("Demo")

    preset = st.selectbox("Asset universe preset", list(data.PRESETS), key="preset")
    tickers_text = st.text_area(
        "Tickers (comma-separated)", ", ".join(data.PRESETS[preset]),
        key=f"tickers_{preset}", help="Any Yahoo Finance symbols, e.g. AAPL, ENI.MI, GLD.",
    )
    benchmark = st.text_input(
        "Benchmark (used for betas)", data.DEFAULT_BENCHMARKS[preset], key=f"bench_{preset}"
    ).strip().upper()

    today = dt.date.today()
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", today - dt.timedelta(days=3652), max_value=today)
    end = c2.date_input("End", today, max_value=today)

    st.subheader("Estimation")
    ret_label = st.selectbox("Expected returns", ["Historical mean", "Exponentially weighted"])
    cov_label = st.selectbox("Covariance", ["Ledoit-Wolf shrinkage", "Sample"])
    rf = st.number_input("Risk-free rate (annual %)", 0.0, 20.0, 2.0, 0.25) / 100

    st.subheader("Constraints")
    lo, hi = st.slider("Weight per asset (%)", 0, 100, (0, 40), step=5,
                       help="Applies to the mean-variance and Black-Litterman portfolios.")
    bounds = (lo / 100, hi / 100)

    st.subheader("Risk")
    alpha = st.select_slider("Confidence level", [0.90, 0.95, 0.975, 0.99], value=0.95,
                             format_func=lambda x: f"{x:.1%}")
    horizon = int(st.number_input("Horizon (trading days)", 1, 20, 1))
    n_sims = st.select_slider("Monte Carlo simulations", [1_000, 5_000, 10_000, 25_000, 50_000],
                              value=10_000)
    mc_label = st.radio("Monte Carlo method", ["Gaussian", "Bootstrap"], horizontal=True)
    port_value = st.number_input("Portfolio value (€)", 1_000, 1_000_000_000, 100_000, 10_000)

    st.subheader("Black-Litterman")
    prior_label = st.selectbox(
        "Prior (market) portfolio",
        ["Equal weight", "Inverse volatility", "Market cap (live data, stocks only)"],
    )
    delta = st.number_input("Risk aversion δ", 0.5, 10.0, 2.5, 0.1)
    tau = st.number_input("Uncertainty scaling τ", 0.01, 1.0, 0.05, 0.01)


# ============================================================================ data
st.title("📈 Portfolio Optimisation & Risk Dashboard")
st.caption(
    "Compare mean-variance, Black-Litterman, risk parity and hierarchical risk parity on the "
    "same data, then measure VaR/CVaR three ways and stress-test against past crises."
)

tickers = data.parse_tickers(tickers_text)
if len(tickers) < 2:
    st.warning("Enter at least two tickers.")
    st.stop()
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

download = tuple(dict.fromkeys(tickers + ([benchmark] if benchmark else [])))
try:
    prices_all = load_prices(download, min(start, STRESS_HISTORY_START).isoformat(),
                             (end + dt.timedelta(days=1)).isoformat(), demo, (benchmark,))
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load prices: {exc}\n\nTip: switch the data source to **Demo data**.")
    st.stop()

missing = [t for t in tickers if t not in prices_all.columns]
if missing:
    st.warning(f"No data found for {', '.join(missing)}; they were left out.")
assets = [t for t in tickers if t in prices_all.columns]
if len(assets) < 2:
    st.error("Fewer than two tickers returned data.")
    st.stop()

universe_all = prices_all[assets]
window_raw = universe_all.loc[pd.Timestamp(start): pd.Timestamp(end)]
window = data.clean_prices(window_raw)
if len(window) < 126:
    st.error("Less than six months of overlapping history. Pick an earlier start date or "
             "remove recently listed tickers.")
    st.stop()
first_valid = window_raw.apply(pd.Series.first_valid_index)
if window.index[0] > window_raw.index[0] + pd.Timedelta(days=10):
    youngest = first_valid.idxmax()
    st.info(f"The analysis starts on {window.index[0]:%d %b %Y} because **{youngest}** has no "
            "earlier data.")

returns = data.to_returns(window)
bench_returns = None
if benchmark in prices_all.columns:
    bench_returns = prices_all[benchmark].loc[window.index[0]: window.index[-1]].ffill().pct_change().dropna()

# ============================================================================ tabs
(tab_over, tab_opt, tab_bl, tab_risk, tab_stress, tab_bt, tab_method) = st.tabs([
    "📊 Overview", "⚖️ Optimisation", "🔭 Black-Litterman", "🛡️ VaR & CVaR",
    "🌪️ Stress tests", "🔁 Backtest", "ℹ️ Methodology",
])

# The views editor lives in the Black-Litterman tab, but its values are needed before any
# weights are computed, so it is rendered first.
with tab_bl:
    st.subheader("Your views")
    st.caption("Absolute views such as “I expect GLD to return 8% a year”, each with a "
               "confidence from 1% (barely trust it) to 99% (almost certain). Add or delete "
               "rows freely; leave the table empty to use the prior alone.")
    views_df = st.data_editor(
        pd.DataFrame({"Asset": [assets[0]], "Expected return (%)": [8.0],
                      "Confidence (%)": [50]}),
        num_rows="dynamic", hide_index=True, key=f"views_{'-'.join(assets)}",
        column_config={
            "Asset": st.column_config.SelectboxColumn(options=assets, required=True),
            "Expected return (%)": st.column_config.NumberColumn(
                min_value=-100.0, max_value=200.0, step=0.5, format="%.1f"),
            "Confidence (%)": st.column_config.NumberColumn(
                min_value=1, max_value=99, step=1, format="%d"),
        },
    )
views_clean = views_df.dropna().drop_duplicates("Asset", keep="last")
views = tuple(
    (str(r["Asset"]), float(r["Expected return (%)"]) / 100, float(r["Confidence (%)"]) / 100)
    for _, r in views_clean.iterrows()
)

prior_mode = {"Equal weight": "equal", "Inverse volatility": "inverse_vol"}.get(prior_label, "custom")
custom_prior = None
if prior_mode == "custom":
    custom_prior = (pd.Series(np.nan, index=assets) if demo
                    else load_market_caps(tuple(assets)))

settings = analysis.Settings(
    return_method="ema" if ret_label.startswith("Exp") else "historical",
    cov_method="sample" if cov_label == "Sample" else "ledoit_wolf",
    rf=rf, bounds=bounds, prior=prior_mode, tau=tau, risk_aversion=delta, views=views,
)

try:
    opt.check_bounds(len(assets), bounds)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

result = analysis.run_analysis(returns, settings, custom_prior)
mu, cov, weights = result.mu, result.cov, result.weights
for note in result.notes:
    st.info(note)
for name, msg in result.errors.items():
    st.warning(f"**{name}** could not be computed: {msg}")
if not weights:
    st.stop()

W = pd.DataFrame(weights)[[s for s in STRATEGIES if s in weights]]
port_rets = pd.DataFrame({name: risk.portfolio_returns(returns, w) for name, w in weights.items()})


# ============================================================================ overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets", len(assets))
    c2.metric("From", f"{window.index[0]:%d %b %Y}")
    c3.metric("To", f"{window.index[-1]:%d %b %Y}")
    c4.metric("Trading days", f"{len(returns):,}")

    st.subheader("Growth of 100")
    growth = window / window.iloc[0] * 100
    fig = px.line(growth, labels={"value": "Value", "index": "", "variable": "Asset"})
    fig.update_layout(height=420, legend_title_text="")
    st.plotly_chart(fig)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Asset statistics")
        stats = pd.DataFrame({a: risk.performance_metrics(returns[a], rf) for a in assets}).T
        stats.insert(0, "Expected return", mu)
        st.dataframe(fmt_metrics(stats[["Expected return", "Volatility", "Sharpe",
                                        "Max drawdown", "Skew", "Excess kurtosis"]]))
        st.caption("Expected return uses the estimator chosen in the sidebar. Excess kurtosis "
                   "above 0 means fatter tails than a normal distribution.")
    with right:
        st.subheader("Correlation")
        corr = estimators.correlation(cov)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig)


# ============================================================================ optimisation
with tab_opt:
    st.subheader("Portfolio weights")
    long = W.reset_index(names="Asset").melt(id_vars="Asset", var_name="Strategy",
                                             value_name="Weight")
    fig = px.bar(long, x="Asset", y="Weight", color="Strategy", barmode="group",
                 color_discrete_map=COLOR)
    fig.update_layout(height=420, yaxis_tickformat=".0%", legend_title_text="")
    st.plotly_chart(fig)
    with st.expander("Weights table"):
        st.dataframe(fmt_pct(W, 1))

    st.subheader("Efficient frontier")
    try:
        frontier = opt.efficient_frontier(mu, cov, bounds, 40)
    except Exception as exc:  # noqa: BLE001
        frontier = pd.DataFrame(columns=["return", "volatility"])
        st.warning(f"Could not trace the frontier: {exc}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["return"], mode="lines",
                             name="Efficient frontier", line=dict(color="#555", width=2)))
    asset_vol = np.sqrt(np.diag(cov))
    fig.add_trace(go.Scatter(x=asset_vol, y=mu, mode="markers+text", text=list(mu.index),
                             textposition="top center", name="Assets",
                             marker=dict(color="#bbb", size=8)))
    for name, w in weights.items():
        ret, vol, _ = opt.portfolio_performance(w, mu, cov, rf)
        fig.add_trace(go.Scatter(x=[vol], y=[ret], mode="markers", name=name,
                                 marker=dict(symbol="star", size=16, color=COLOR[name],
                                             line=dict(color="black", width=1))))
    fig.update_layout(height=500, xaxis_title="Volatility (annual)",
                      yaxis_title="Expected return (annual)",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig)
    st.caption("Every portfolio is plotted using the same historical estimates, so mean-variance "
               "portfolios sit on the frontier by construction. That is in-sample; the Backtest "
               "tab shows how the strategies do on data they have not seen.")

    left, right = st.columns(2)
    with left:
        st.subheader("Expected vs realised (in-sample)")
        rows = {}
        for name, w in weights.items():
            ret, vol, sharpe = opt.portfolio_performance(w, mu, cov, rf)
            m = risk.performance_metrics(port_rets[name], rf)
            rows[name] = {"Expected return": ret, "Expected volatility": vol,
                          "Expected Sharpe": sharpe, "CAGR": m["CAGR"],
                          "Max drawdown": m["Max drawdown"], "Sharpe": m["Sharpe"]}
        st.dataframe(fmt_metrics(pd.DataFrame(rows).T))
    with right:
        st.subheader("Risk contributions")
        rc = pd.DataFrame({n: opt.risk_contributions(w, cov) for n, w in weights.items()})
        rc_long = rc.reset_index(names="Asset").melt(id_vars="Asset", var_name="Strategy",
                                                     value_name="Share of risk")
        fig = px.bar(rc_long, y="Strategy", x="Share of risk", color="Asset", orientation="h")
        fig.update_layout(height=380, xaxis_tickformat=".0%", legend_title_text="")
        st.plotly_chart(fig)
        st.caption("Capital weights and risk weights differ: a 10% stake in a volatile asset "
                   "can carry far more than 10% of the risk. Risk parity equalises these bars.")


# ============================================================================ black-litterman
with tab_bl:
    bl_res = result.bl_result
    comparison = pd.DataFrame({
        "Prior weight": bl_res.market_weights,
        "Historical mean": mu,
        "Implied (prior)": bl_res.prior + rf,
        "Posterior (BL)": result.bl_returns,
    })
    st.subheader("Expected returns: history vs equilibrium vs your views")
    long = comparison.drop(columns="Prior weight").reset_index(names="Asset").melt(
        id_vars="Asset", var_name="Estimate", value_name="Return")
    fig = px.bar(long, x="Asset", y="Return", color="Estimate", barmode="group")
    fig.update_layout(height=400, yaxis_tickformat=".0%", legend_title_text="")
    st.plotly_chart(fig)
    st.dataframe(fmt_pct(comparison))
    if "Black-Litterman" in weights:
        st.subheader("Weights: prior vs Black-Litterman")
        wcmp = pd.DataFrame({"Prior": bl_res.market_weights,
                             "Black-Litterman": weights["Black-Litterman"]})
        fig = px.bar(wcmp.reset_index(names="Asset").melt(id_vars="Asset", var_name="Portfolio",
                                                          value_name="Weight"),
                     x="Asset", y="Weight", color="Portfolio", barmode="group")
        fig.update_layout(height=380, yaxis_tickformat=".0%", legend_title_text="")
        st.plotly_chart(fig)
    st.caption("Without views the posterior equals the implied returns, and an unconstrained "
               "optimiser would simply hold the prior portfolio. Views tilt away from it in "
               "proportion to their confidence.")


# ============================================================================ VaR / CVaR
with tab_risk:
    mc_method = mc_label.lower()
    table = cached_var_table(returns, W, alpha, horizon, n_sims, mc_method)
    unit = st.radio("Show losses as", ["% of portfolio", "€"], horizontal=True)
    st.subheader(f"{alpha:.1%} {horizon}-day VaR and CVaR")
    if unit == "€":
        st.dataframe((table * port_value).style.format("€{:,.0f}"))
    else:
        st.dataframe(fmt_pct(table))
    st.caption("VaR: the loss not exceeded with the chosen confidence. CVaR (expected "
               "shortfall): the average loss when VaR *is* exceeded. When historical CVaR is "
               "well above parametric CVaR, returns have fatter tails than a normal distribution.")

    pick = st.selectbox("Inspect a strategy", list(weights), key="risk_pick")
    r = port_rets[pick]
    hist = risk.historical_var(r, alpha, horizon)
    param = risk.parametric_var(r, alpha, horizon)
    mc = risk.monte_carlo_var(returns, weights[pick], alpha, horizon, n_sims, mc_method)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=hist.sample, histnorm="probability density", nbinsx=120,
                               name="Historical", opacity=0.6, marker_color="#4C78A8"))
    fig.add_trace(go.Histogram(x=mc.sample, histnorm="probability density", nbinsx=120,
                               name="Monte Carlo", opacity=0.45, marker_color="#F58518"))
    for label, res_, color in [("Historical", hist, "#4C78A8"), ("Parametric", param, "#54A24B"),
                               ("Monte Carlo", mc, "#F58518")]:
        fig.add_vline(x=-res_.var, line_dash="dash", line_color=color,
                      annotation_text=f"{label} VaR", annotation_position="top left")
    fig.update_layout(barmode="overlay", height=420, xaxis_tickformat=".1%",
                      xaxis_title=f"{horizon}-day return", yaxis_title="Density")
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        dd = risk.drawdown_series(r)
        fig = px.area(dd, labels={"value": "Drawdown", "index": ""})
        fig.update_layout(height=320, showlegend=False, yaxis_tickformat=".0%",
                          title="Drawdown")
        fig.update_traces(line_color="#E45756")
        st.plotly_chart(fig)
    with c2:
        roll = r.rolling(63).std() * np.sqrt(252)
        fig = px.line(roll, labels={"value": "Volatility", "index": ""})
        fig.update_layout(height=320, showlegend=False, yaxis_tickformat=".0%",
                          title="Rolling 3-month volatility")
        st.plotly_chart(fig)

    bt_var = risk.var_backtest(r, alpha)
    st.markdown(
        f"**VaR model check.** Using a rolling one-year historical VaR, the next day's loss "
        f"exceeded the {alpha:.1%} VaR on **{bt_var['breaches']}** of {bt_var['observations']:,} "
        f"days (**{bt_var['breach_rate']:.2%}**, expected {bt_var['expected_rate']:.2%})."
    )


# ============================================================================ stress tests
with tab_stress:
    st.subheader("Historical crisis replays")
    st.caption("Buy-and-hold each portfolio through past crises using today's weights. Assets "
               "without data at the start of a crisis are left out and the rest rescaled.")
    scen = stress.run_historical_scenarios(universe_all, weights)
    table = pd.DataFrame({s: {n: r_.portfolio_return for n, r_ in res.items()}
                          for s, res in scen.items()})
    coverage = pd.DataFrame({s: {n: r_.coverage for n, r_ in res.items()}
                             for s, res in scen.items()})
    if unit == "€":
        st.dataframe((table * port_value).style.format("€{:,.0f}", na_rep="no data"))
    else:
        st.dataframe(fmt_pct(table))
    low = coverage.stack()
    low = low[low < 0.999]
    if not low.empty:
        gaps = sorted({m for res in scen.values() for r_ in res.values() for m in r_.missing})
        st.warning(f"Some assets had no data for some crises ({', '.join(gaps)}). Affected "
                   f"results cover as little as {low.min():.0%} of the portfolio.")

    pick_scen = st.selectbox("Scenario path", list(scen), key="scen_pick")
    paths = pd.DataFrame({n: r_.path for n, r_ in scen[pick_scen].items() if not r_.path.empty})
    if paths.empty:
        st.info("None of the assets have data for this period.")
    else:
        fig = px.line(paths - 1, color_discrete_map=COLOR,
                      labels={"value": "Return", "index": "", "variable": ""})
        fig.update_layout(height=420, yaxis_tickformat=".0%")
        st.plotly_chart(fig)
        dd_row = pd.DataFrame({n: {"Return": r_.portfolio_return,
                                   "Max drawdown in window": r_.max_drawdown,
                                   "Coverage": r_.coverage}
                               for n, r_ in scen[pick_scen].items()}).T
        st.dataframe(fmt_pct(dd_row, 1))

    st.subheader("Hypothetical shocks")
    left, right = st.columns(2)
    with left:
        st.markdown("**Market shock via beta**")
        if bench_returns is None or bench_returns.empty:
            st.info("Add a benchmark ticker with data to use this test.")
        else:
            move = st.slider(f"{benchmark} move (%)", -50, 30, -20) / 100
            b = stress.betas(returns, bench_returns)
            impact = pd.Series({n: stress.beta_shock(w, b, move) for n, w in weights.items()})
            out = pd.DataFrame({"Impact": impact, "€ impact": impact * port_value})
            st.dataframe(out.style.format({"Impact": "{:.2%}", "€ impact": "€{:,.0f}"}))
            with st.expander("Asset betas"):
                st.dataframe(b.to_frame("Beta").style.format("{:.2f}"))
    with right:
        st.markdown("**Custom shock per asset**")
        shocks = st.data_editor(
            pd.DataFrame({"Asset": assets, "Shock (%)": [0.0] * len(assets)}),
            hide_index=True, disabled=["Asset"], key=f"shocks_{'-'.join(assets)}",
            column_config={"Shock (%)": st.column_config.NumberColumn(
                min_value=-100.0, max_value=500.0, step=1.0, format="%.1f")},
        )
        s_series = shocks.set_index("Asset")["Shock (%)"].fillna(0.0) / 100
        impact = pd.Series({n: stress.shock_impact(w, s_series) for n, w in weights.items()})
        st.dataframe(pd.DataFrame({"Impact": impact, "€ impact": impact * port_value})
                     .style.format({"Impact": "{:.2%}", "€ impact": "€{:,.0f}"}))


# ============================================================================ backtest
with tab_bt:
    st.subheader("Walk-forward backtest (out-of-sample)")
    st.caption("At each rebalance date every strategy is re-estimated using only the trailing "
               "lookback window, then held until the next rebalance. No future data is used.")
    c1, c2, c3 = st.columns(3)
    lookback = c1.select_slider("Lookback window", [126, 252, 504, 756], value=252,
                                format_func=lambda d: f"{d} days (~{d / 252:.1f}y)")
    freq = c2.selectbox("Rebalance", list(backtest.FREQUENCIES), index=1)
    cost = c3.number_input("Transaction cost (bps per trade)", 0.0, 100.0, 10.0, 1.0)

    if len(returns) <= lookback + 63:
        st.info("Not enough history for this lookback. Choose a shorter window or an "
                "earlier start date.")
    elif st.toggle("Run backtest", key="run_bt"):
        bt = cached_backtest(returns, repr(settings), settings, custom_prior,
                             lookback, freq, cost)
        growth = (1 + bt.returns).cumprod()
        fig = px.line(growth, color_discrete_map=COLOR,
                      labels={"value": "Growth of 1", "index": "", "variable": ""})
        fig.update_layout(height=450)
        st.plotly_chart(fig)

        metrics = pd.DataFrame({n: risk.performance_metrics(bt.returns[n], rf)
                                for n in bt.returns}).T
        metrics["Turnover / yr"] = bt.turnover
        st.dataframe(fmt_metrics(metrics[["CAGR", "Volatility", "Sharpe", "Sortino",
                                          "Max drawdown", "Calmar", "Turnover / yr"]]))
        if bt.fallbacks.sum():
            st.caption("Rebalances where an optimiser failed and kept the previous weights: "
                       + ", ".join(f"{k} ({v})" for k, v in bt.fallbacks.items() if v))

        pick_bt = st.selectbox("Weights over time", list(bt.weights), key="bt_pick")
        wh = bt.weights[pick_bt]
        fig = px.area(wh, labels={"value": "Weight", "index": "", "variable": "Asset"})
        fig.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig)
    else:
        st.info("Switch on **Run backtest** (it takes a few seconds).")


# ============================================================================ methodology
with tab_method:
    st.markdown(
        r"""
### Portfolio construction
- **Equal weight.** $1/N$ in each asset. A surprisingly hard benchmark to beat out of sample.
- **Mean-variance (Markowitz).** *Max Sharpe* maximises $(\mu^\top w - r_f)/\sqrt{w^\top\Sigma w}$;
  *Min volatility* minimises $w^\top \Sigma w$. Both are long-only with per-asset bounds, solved with SLSQP.
  They are very sensitive to expected-return estimates, which is why the other methods exist.
- **Black-Litterman.** Reverse-optimises the prior portfolio to get implied equilibrium returns
  $\pi = \delta\Sigma w_{mkt}$, then blends in views:
  $\mu_{BL} = \pi + \tau\Sigma P^\top(P\tau\Sigma P^\top + \Omega)^{-1}(Q - P\pi)$.
  View uncertainty $\Omega$ is set from your confidence levels (Idzorek-style).
- **Risk parity.** Each asset contributes the same share of portfolio variance. Solved via the convex
  problem $\min_y \tfrac12 y^\top\Sigma y - \sum_i b_i \log y_i$. Ignores expected returns entirely.
- **Hierarchical risk parity (López de Prado, 2016).** Clusters assets by correlation distance, orders
  them so similar assets sit together, then splits capital top-down by inverse cluster variance.
  Needs no matrix inversion, so it copes well with noisy covariance estimates.

### Risk measures
- **Historical VaR/CVaR.** Empirical quantile of past returns, with overlapping compounded returns for horizons over one day.
- **Parametric VaR/CVaR.** Normal distribution, square-root-of-time scaling. Understates fat tails.
- **Monte Carlo VaR/CVaR.** Either a multivariate normal fitted to asset returns (captures correlations)
  or a bootstrap of historical days (keeps fat tails and same-day dependence).
- **Stress tests.** Historical replays of five crises, a beta-transmitted market shock, and custom per-asset shocks.

### Limitations (read before trusting any number)
- Estimates come from one historical window; regimes change.
- In-sample results flatter mean-variance portfolios. Use the walk-forward backtest to judge strategies.
- Constant-weight portfolio returns assume daily rebalancing; the backtest uses realistic drift and periodic rebalancing.
- Crisis replays use today's weights and skip assets that did not exist yet.
- The Black-Litterman market-cap prior uses current caps, which adds look-ahead bias to the backtest.
"""
    )
