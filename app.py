from __future__ import annotations

from datetime import date
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics.performance import calculate_performance, metrics_frame
from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.data.data_loader import DataDownloadError, download_history, load_csv
from src.screening import BatchRunConfig, RANKING_METRICS, rank_results, run_batch, select_top_bottom
from src.strategies import MomentumStrategy, MovingAverageCrossover, RSIMeanReversion
from src.universe import get_universe
from src.visualization.charts import allocation_chart, drawdown_chart, equity_chart, monthly_heatmap, price_signals_chart, returns_chart

st.set_page_config(page_title="Modular Backtesting Lab", page_icon="📈", layout="wide")


def fmt_currency(value: Any) -> str:
    return "N/A" if not np.isfinite(value) else f"${value:,.2f}"


def fmt_pct(value: Any) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2%}"


def fmt_number(value: Any) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2f}"


def make_strategy(name: str, params: dict[str, Any]):
    if name == "Moving Average Crossover":
        return MovingAverageCrossover(params["ma_short"], params["ma_long"])
    if name == "RSI Mean Reversion":
        return RSIMeanReversion(params["rsi_period"], params["rsi_oversold"], params["rsi_overbought"])
    return MomentumStrategy(params["mom_lookback"], params["mom_entry"] / 100, params["mom_exit"] / 100)


def config_from(params: dict[str, Any]) -> BacktestConfig:
    return BacktestConfig(params["capital"], params["commission_pct"] / 100, params["fixed_fee"], params["slippage_bps"], params["fractional"])


def run_one(data: pd.DataFrame, name: str, params: dict[str, Any]) -> tuple[BacktestResult, dict[str, Any]]:
    strategy = make_strategy(name, params)
    result = BacktestEngine(config_from(params)).run(data, strategy.generate_signals(data))
    return result, calculate_performance(result.history, result.trades, params["capital"], params["risk_free"] / 100)


def run_buy_hold(data: pd.DataFrame, params: dict[str, Any]) -> tuple[BacktestResult, dict[str, Any]]:
    signals = pd.Series(0, index=data.index, dtype=int)
    result = BacktestEngine(config_from(params)).run(data, signals, initial_buy=True)
    return result, calculate_performance(result.history, result.trades, params["capital"], params["risk_free"] / 100)


st.title("Modular Backtesting Lab")
st.caption("Transparent long-or-cash strategy research with next-session execution.")
st.warning("This application is for educational and research purposes only and does not constitute investment advice.")

with st.sidebar:
    st.header("Backtest setup")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    start = st.date_input("Start date", date(2018, 1, 1))
    end = st.date_input("End date", date(2025, 12, 31))
    uploaded = st.file_uploader("CSV fallback (optional)", type=["csv"], help="Columns: Date, Open, High, Low, Close, Volume")
    capital = st.number_input("Initial capital ($)", min_value=100.0, value=100_000.0, step=1_000.0)
    risk_free = st.number_input("Annual risk-free rate (%)", min_value=0.0, value=0.0, step=0.1)
    fractional = st.checkbox("Allow fractional shares", value=True)
    commission_pct = st.number_input("Commission (%)", min_value=0.0, value=0.05, step=0.01, format="%.3f")
    fixed_fee = st.number_input("Fixed commission per trade ($)", min_value=0.0, value=0.0, step=0.5)
    slippage_bps = st.number_input("Slippage (basis points)", min_value=0.0, value=5.0, step=1.0)
    strategy_name = st.selectbox("Strategy", ["Moving Average Crossover", "RSI Mean Reversion", "Momentum"])
    st.subheader("Strategy parameters")
    ma_short = st.number_input("MA short window", min_value=2, value=20)
    ma_long = st.number_input("MA long window", min_value=3, value=50)
    rsi_period = st.number_input("RSI lookback", min_value=2, value=14)
    rsi_oversold = st.number_input("RSI oversold", min_value=0.0, max_value=99.0, value=30.0)
    rsi_overbought = st.number_input("RSI overbought", min_value=1.0, max_value=100.0, value=70.0)
    mom_lookback = st.number_input("Momentum lookback", min_value=1, value=20)
    mom_entry = st.number_input("Momentum entry threshold (%)", value=3.0, step=0.5)
    mom_exit = st.number_input("Momentum exit threshold (%)", value=0.0, step=0.5)
    clear_cache = st.button("Clear market-data cache", use_container_width=True)
    run_selected = st.button("Run Backtest", type="primary", use_container_width=True)
    run_all = st.button("Run All Strategies", use_container_width=True)

params = {"capital": capital, "risk_free": risk_free, "fractional": fractional, "commission_pct": commission_pct,
          "fixed_fee": fixed_fee, "slippage_bps": slippage_bps, "ma_short": int(ma_short), "ma_long": int(ma_long),
          "rsi_period": int(rsi_period), "rsi_oversold": rsi_oversold, "rsi_overbought": rsi_overbought,
          "mom_lookback": int(mom_lookback), "mom_entry": mom_entry, "mom_exit": mom_exit}

if clear_cache:
    # Failed requests raise before returning, so they are never cached; this also clears prior successful data.
    download_history.clear()
    st.cache_data.clear()
    st.session_state.pop("bt_data", None)
    st.success("Market-data cache cleared. The next run will download fresh data.")

if run_selected or run_all:
    if start >= end:
        st.error("End date must be after start date.")
    elif ma_short >= ma_long or rsi_oversold >= rsi_overbought or mom_exit >= mom_entry:
        st.error("Check strategy parameters: MA short < long; RSI oversold < overbought; momentum exit < entry.")
    else:
        try:
            with st.spinner("Loading data and running the simulation..."):
                data = load_csv(uploaded) if uploaded is not None else download_history(ticker, str(start), str(end))
                if len(data) < 2:
                    raise ValueError("At least two valid daily bars are needed to run a backtest.")
                selected, selected_metrics = run_one(data, strategy_name, params)
                buy_hold, buy_hold_metrics = run_buy_hold(data, params)
                comparisons: dict[str, tuple[BacktestResult, dict[str, Any]]] = {strategy_name: (selected, selected_metrics), "Buy and Hold": (buy_hold, buy_hold_metrics)}
                if run_all:
                    comparisons = {name: run_one(data, name, params) for name in ["Moving Average Crossover", "RSI Mean Reversion", "Momentum"]}
                    comparisons["Buy and Hold"] = (buy_hold, buy_hold_metrics)
                st.session_state["bt_data"] = data
                st.session_state["selected_name"] = strategy_name
                st.session_state["selected_result"] = selected
                st.session_state["selected_metrics"] = selected_metrics
                st.session_state["buy_hold"] = buy_hold
                st.session_state["buy_hold_metrics"] = buy_hold_metrics
                st.session_state["comparison"] = comparisons
        except (DataDownloadError, ValueError, KeyError) as exc:
            st.error(f"Could not run backtest: {exc}")
            st.caption("Download diagnostics were written to the Streamlit server log.")
        except (OSError, pd.errors.ParserError) as exc:
            st.error(f"Data retrieval failed: {exc}")

tabs = st.tabs(["Overview", "Price and Signals", "Risk Analysis", "Trade Log", "Strategy Comparison", "Stock Ranking", "Methodology"])
ready = "selected_result" in st.session_state
if not ready:
    tabs[0].info("Configure the sidebar and select **Run Backtest**. Yahoo data is cached; you may instead upload a valid OHLCV CSV.")

if ready:
    data = st.session_state["bt_data"]
    result: BacktestResult = st.session_state["selected_result"]
    metrics = st.session_state["selected_metrics"]
    buy_hold: BacktestResult = st.session_state["buy_hold"]
    buy_hold_metrics = st.session_state["buy_hold_metrics"]
    name = st.session_state["selected_name"]
    with tabs[0]:
        st.subheader(f"{name} overview")
        cards = [("Final value", fmt_currency(metrics["Final Portfolio Value"])), ("Total return", fmt_pct(metrics["Total Return"])),
                 ("CAGR", fmt_pct(metrics["CAGR"])), ("Sharpe ratio", fmt_number(metrics["Sharpe Ratio"])),
                 ("Max drawdown", fmt_pct(metrics["Maximum Drawdown"])), ("Total costs", fmt_currency(metrics["Total Transaction Costs"]))]
        cols = st.columns(6)
        for col, (label, value) in zip(cols, cards): col.metric(label, value)
        st.plotly_chart(equity_chart({name: result, "Buy and Hold": buy_hold}), use_container_width=True, key="overview_equity")
        metric_table = pd.DataFrame({"Strategy": metrics, "Buy and Hold": buy_hold_metrics})
        st.dataframe(metric_table, use_container_width=True)
        if result.warnings:
            st.warning("\n".join(sorted(set(result.warnings))))
    with tabs[1]:
        st.plotly_chart(price_signals_chart(data, result.trades), use_container_width=True, key="price_signals")
        st.caption("Signals use the close of day t; plotted trade markers show execution at day t+1's open. The final exit is a required close-of-test liquidation.")
    with tabs[2]:
        left, right = st.columns(2)
        left.plotly_chart(drawdown_chart(result.history), use_container_width=True, key="drawdown")
        right.plotly_chart(returns_chart(result.history), use_container_width=True, key="daily_returns")
        left, right = st.columns(2)
        left.plotly_chart(monthly_heatmap(result.history), use_container_width=True, key="monthly_heatmap")
        right.plotly_chart(allocation_chart(result.history), use_container_width=True, key="allocation")
    with tabs[3]:
        st.subheader("Complete trade log")
        st.dataframe(result.trades, use_container_width=True, hide_index=True)
        st.download_button("Download daily portfolio history CSV", result.history.reset_index().to_csv(index=False).encode(), "daily_portfolio_history.csv", "text/csv")
        st.download_button("Download trade log CSV", result.trades.to_csv(index=False).encode(), "trade_log.csv", "text/csv")
        st.download_button("Download performance summary CSV", pd.DataFrame([metrics]).to_csv(index=False).encode(), "performance_summary.csv", "text/csv")
    with tabs[4]:
        comparison = st.session_state["comparison"]
        metrics_by_name = {key: value[1] for key, value in comparison.items()}
        table = metrics_frame(metrics_by_name)
        st.dataframe(table.style.format({"Total Return": "{:.2%}", "CAGR": "{:.2%}", "Maximum Drawdown": "{:.2%}", "Win Rate": "{:.2%}", "Total Transaction Costs": "${:,.2f}"}), use_container_width=True)
        st.plotly_chart(equity_chart({key: value[0] for key, value in comparison.items()}), use_container_width=True, key="comparison_equity")
        return_winner = table["Total Return"].idxmax()
        sharpe_winner = table["Sharpe Ratio"].idxmax()
        st.info(f"Strongest historical return: **{return_winner}**. Strongest historical Sharpe ratio: **{sharpe_winner}**. Historical performance does not guarantee future results.")

with tabs[5]:
    st.subheader("Cross-Sectional Strategy Ranking")
    st.info("Rankings describe historical strategy performance over the selected period. They are not forward-looking recommendations.")
    universe_name = st.selectbox("Stock universe", ["S&P 100 (current constituents)", "Nasdaq-100 (current constituents)", "Custom ticker list", "Uploaded universe CSV"], key="rank_universe")
    custom_universe = st.text_area("Custom tickers (comma-separated)", "AAPL, MSFT, NVDA, JPM, XOM, JNJ", key="rank_custom") if universe_name == "Custom ticker list" else ""
    universe_upload = st.file_uploader("Universe CSV (Ticker column)", type=["csv"], key="rank_universe_upload") if universe_name == "Uploaded universe CSV" else None
    rank_start_col, rank_end_col, rank_metric_col = st.columns(3)
    rank_start = rank_start_col.date_input("Ranking start date", date(2018, 1, 1), key="rank_start")
    rank_end = rank_end_col.date_input("Ranking end date", date(2025, 12, 31), key="rank_end")
    ranking_metric = rank_metric_col.selectbox("Ranking metric", list(RANKING_METRICS), key="rank_metric")
    rank_strategy = st.selectbox("Ranking strategy", ["Moving Average Crossover", "RSI Mean Reversion", "Momentum"], key="rank_strategy")
    param_a, param_b, param_c = st.columns(3)
    rank_ma_short = param_a.number_input("Ranking MA short", min_value=2, value=20, key="rank_ma_short")
    rank_ma_long = param_a.number_input("Ranking MA long", min_value=3, value=50, key="rank_ma_long")
    rank_rsi_period = param_b.number_input("Ranking RSI lookback", min_value=2, value=14, key="rank_rsi_period")
    rank_rsi_oversold = param_b.number_input("Ranking RSI oversold", min_value=0.0, max_value=99.0, value=30.0, key="rank_rsi_oversold")
    rank_rsi_overbought = param_b.number_input("Ranking RSI overbought", min_value=1.0, max_value=100.0, value=70.0, key="rank_rsi_overbought")
    rank_mom_lookback = param_c.number_input("Ranking momentum lookback", min_value=1, value=20, key="rank_mom_lookback")
    rank_mom_entry = param_c.number_input("Ranking momentum entry (%)", value=3.0, step=0.5, key="rank_mom_entry")
    rank_mom_exit = param_c.number_input("Ranking momentum exit (%)", value=0.0, step=0.5, key="rank_mom_exit")
    cost_a, cost_b, cost_c, cost_d = st.columns(4)
    rank_capital = cost_a.number_input("Ranking initial capital ($)", min_value=100.0, value=100_000.0, step=1_000.0, key="rank_capital")
    rank_commission = cost_b.number_input("Ranking commission (%)", min_value=0.0, value=0.05, step=0.01, key="rank_commission")
    rank_fixed_fee = cost_c.number_input("Ranking fixed fee ($)", min_value=0.0, value=0.0, step=0.5, key="rank_fixed_fee")
    rank_slippage = cost_d.number_input("Ranking slippage (bps)", min_value=0.0, value=5.0, step=1.0, key="rank_slippage")
    rank_risk_free, minimum_coverage = st.columns(2)
    rank_rf_value = rank_risk_free.number_input("Ranking risk-free rate (%)", min_value=0.0, value=0.0, step=0.1, key="rank_rf")
    rank_coverage = minimum_coverage.slider("Minimum data coverage", min_value=0.50, max_value=1.00, value=0.90, step=0.05, key="rank_coverage")
    st.caption("Fractional shares are enabled for every ranked ticker so nominal share prices do not distort the comparison.")
    run_ranking = st.button("Run Stock Ranking", type="primary", key="run_ranking")
    if run_ranking:
        if rank_start >= rank_end:
            st.error("Ranking end date must be after start date.")
        elif rank_ma_short >= rank_ma_long or rank_rsi_oversold >= rank_rsi_overbought or rank_mom_exit >= rank_mom_entry:
            st.error("Check ranking parameters: MA short < long; RSI oversold < overbought; momentum exit < entry.")
        else:
            try:
                universe = get_universe(universe_name, custom_universe, universe_upload)
                progress_bar = st.progress(0, text="Preparing controlled sequential batch...")
                def update_progress(position: int, total: int, symbol: str) -> None:
                    progress_bar.progress(position / max(total, 1), text=f"Processing {position}/{total}: {symbol}")
                batch_config = BatchRunConfig(
                    start=str(rank_start), end=str(rank_end), strategy_name=rank_strategy,
                    strategy_params={"ma_short": int(rank_ma_short), "ma_long": int(rank_ma_long), "rsi_period": int(rank_rsi_period),
                                     "rsi_oversold": rank_rsi_oversold, "rsi_overbought": rank_rsi_overbought,
                                     "mom_lookback": int(rank_mom_lookback), "mom_entry": rank_mom_entry, "mom_exit": rank_mom_exit},
                    backtest_config=BacktestConfig(rank_capital, rank_commission / 100, rank_fixed_fee, rank_slippage, True),
                    risk_free_rate=rank_rf_value / 100, minimum_coverage=rank_coverage,
                )
                batch_result = run_batch(universe.tickers, batch_config, progress_callback=update_progress)
                progress_bar.empty()
                st.session_state["ranking_universe"] = universe
                st.session_state["ranking_metric"] = ranking_metric
                st.session_state["ranking_results"] = rank_results(batch_result.results, ranking_metric)
                st.session_state["ranking_excluded"] = batch_result.excluded
                st.session_state["ranking_submitted"] = batch_result.submitted_tickers
                st.session_state["ranking_eligible"] = batch_result.eligible_stocks
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                st.error(f"Could not run stock ranking: {exc}")
    if "ranking_results" in st.session_state:
        ranked = st.session_state["ranking_results"]
        excluded = st.session_state["ranking_excluded"]
        universe = st.session_state["ranking_universe"]
        ranking_metric = st.session_state["ranking_metric"]
        summary_cols = st.columns(5)
        summary_cols[0].metric("Universe", universe.name)
        summary_cols[1].metric("Snapshot date", universe.snapshot_date)
        summary_cols[2].metric("Submitted", st.session_state["ranking_submitted"])
        summary_cols[3].metric("Eligible", st.session_state["ranking_eligible"])
        summary_cols[4].metric("Excluded", len(excluded))
        if universe.current_constituents:
            st.warning("Built-in universes use current constituents. Historical rankings can therefore contain survivorship bias.")
        top, bottom = select_top_bottom(ranked, ranking_metric)
        display_columns = ["Rank", "Ticker", "Final Value", "Strategy Total Return", "Buy and Hold Return", "Excess Return", "CAGR", "Sharpe Ratio", "Calmar Ratio", "Maximum Drawdown", "Completed Trades", "Win Rate", "Total Transaction Costs", "Data Coverage"]
        st.markdown("#### Top 3")
        st.dataframe(top.reindex(columns=display_columns), use_container_width=True, hide_index=True)
        st.markdown("#### Bottom 3")
        st.dataframe(bottom.reindex(columns=display_columns), use_container_width=True, hide_index=True)
        metric_column = RANKING_METRICS[ranking_metric]
        chart_rows = ranked[ranked["Eligible for Ranking"] & ranked[metric_column].notna()].sort_values(metric_column)
        if not chart_rows.empty:
            bar_chart = px.bar(chart_rows, x=metric_column, y="Ticker", orientation="h", title=f"{ranking_metric} across ranking-eligible stocks")
            st.plotly_chart(bar_chart, use_container_width=True, key="ranking_metric_chart")
            comparison_rows = pd.concat([top.assign(Group="Top 3"), bottom.assign(Group="Bottom 3")]).drop_duplicates("Ticker")
            comparison_chart = px.bar(comparison_rows, x="Ticker", y=["Strategy Total Return", "Buy and Hold Return"], barmode="group", title="Top and Bottom: Strategy versus Buy and Hold")
            st.plotly_chart(comparison_chart, use_container_width=True, key="ranking_comparison_chart")
        st.markdown("#### Full ranking results")
        st.dataframe(ranked, use_container_width=True, hide_index=True)
        st.markdown("#### Excluded stocks")
        st.dataframe(excluded, use_container_width=True, hide_index=True)
        top_bottom = pd.concat([top.assign(List="Top 3"), bottom.assign(List="Bottom 3")])
        st.download_button("Download full ranking results CSV", ranked.to_csv(index=False).encode(), "stock_ranking_results.csv", "text/csv", key="ranking_results_download")
        st.download_button("Download Top and Bottom results CSV", top_bottom.to_csv(index=False).encode(), "stock_ranking_top_bottom.csv", "text/csv", key="ranking_top_bottom_download")
        st.download_button("Download excluded stocks CSV", excluded.to_csv(index=False).encode(), "stock_ranking_excluded.csv", "text/csv", key="ranking_excluded_download")
        st.caption("Data coverage filters incomplete histories. Strategy Return is the strategy portfolio return; Excess Return subtracts the same-cost Buy and Hold return.")

with tabs[6]:
    st.subheader("Methodology and limitations")
    st.markdown("""
**Signals and execution.** Moving Average buys/sells on SMA crossovers. RSI uses Wilder smoothing: it buys when yesterday's RSI was at or below oversold and today's RSI rises, then sells as RSI crosses into overbought. Momentum is the trailing percentage return; sidebar thresholds are percentages (3 means 3%). Every strategy uses information through the close of day *t*, and orders execute at day *t+1* open. This timing prevents look-ahead bias. Missing opens fall back to the same day's close and produce a warning.

**Costs and accounting.** Buy price = market open × (1 + slippage bps / 10,000); sell price = market open × (1 − slippage bps / 10,000). Percentage and fixed commissions are deducted from cash. Entries use available cash only; leverage and shorting are not allowed. Remaining shares are sold at the final valid close for complete reporting.

**Metrics.** CAGR, annualized volatility and Sharpe use 252 trading days; maximum drawdown is measured from each running equity peak. Undefined ratios are shown as N/A.

**Limitations.** Backtests can be distorted by parameter overfitting, survivorship bias, corporate-action/data-quality issues, and the simplification of fills at daily open prices. They are research tools, not investment advice.
""")
