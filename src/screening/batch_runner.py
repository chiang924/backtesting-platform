from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import logging
import pandas as pd

from src.analytics.performance import calculate_performance
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.data.data_loader import download_history
from src.strategies import MomentumStrategy, MovingAverageCrossover, RSIMeanReversion
from src.universe.universe_validator import deduplicate_tickers

logger = logging.getLogger("backtesting_lab.screening")
RESULT_COLUMNS = [
    "Ticker", "Final Value", "Strategy Total Return", "Buy and Hold Return", "Excess Return", "CAGR",
    "Sharpe Ratio", "Calmar Ratio", "Maximum Drawdown", "Completed Trades", "Win Rate",
    "Total Transaction Costs", "Data Coverage",
]


@dataclass(frozen=True)
class BatchRunConfig:
    start: str
    end: str
    strategy_name: str
    strategy_params: dict[str, float | int]
    backtest_config: BacktestConfig
    risk_free_rate: float = 0.0
    minimum_coverage: float = 0.90
    retry_attempts: int = 2


@dataclass
class BatchRunResult:
    results: pd.DataFrame
    excluded: pd.DataFrame
    submitted_tickers: int
    eligible_stocks: int


def build_strategy(name: str, params: dict[str, float | int]):
    if name == "Moving Average Crossover":
        return MovingAverageCrossover(int(params["ma_short"]), int(params["ma_long"]))
    if name == "RSI Mean Reversion":
        return RSIMeanReversion(int(params["rsi_period"]), float(params["rsi_oversold"]), float(params["rsi_overbought"]))
    if name == "Momentum":
        return MomentumStrategy(int(params["mom_lookback"]), float(params["mom_entry"]) / 100, float(params["mom_exit"]) / 100)
    raise ValueError(f"Unsupported strategy: {name}")


def required_observations(name: str, params: dict[str, float | int]) -> int:
    if name == "Moving Average Crossover":
        return int(params["ma_long"]) + 1
    if name == "RSI Mean Reversion":
        return int(params["rsi_period"]) + 2
    return int(params["mom_lookback"]) + 2


def _expected_sessions(start: str, end: str) -> int:
    return max(len(pd.bdate_range(start, end)), 1)


def _is_temporary_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(word in message for word in ("timeout", "connection", "rate", "temporar", "network"))


def _excluded(ticker: str, stage: str, reason: str, rows: int = 0, coverage: float = 0.0) -> dict[str, object]:
    return {"Ticker": ticker, "Failure Stage": stage, "Exclusion Reason": reason, "Data Rows": rows, "Data Coverage": coverage}


def run_batch(tickers: list[str], config: BatchRunConfig,
              data_fetcher: Callable[[str, str, str], pd.DataFrame] = download_history,
              progress_callback: Callable[[int, int, str], None] | None = None) -> BatchRunResult:
    """Run controlled, sequential backtests and isolate every ticker failure."""
    submitted = deduplicate_tickers(tickers)
    expected = _expected_sessions(config.start, config.end)
    warmup = required_observations(config.strategy_name, config.strategy_params)
    rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for position, ticker in enumerate(submitted, start=1):
        if progress_callback:
            progress_callback(position, len(submitted), ticker)
        data: pd.DataFrame | None = None
        for attempt in range(config.retry_attempts):
            try:
                data = data_fetcher(ticker, config.start, config.end)
                break
            except Exception as exc:
                if attempt + 1 >= config.retry_attempts or not _is_temporary_error(exc):
                    logger.warning("%s download failed: %s", ticker, exc)
                    excluded.append(_excluded(ticker, "Download", str(exc)))
                    break
        if data is None:
            continue
        coverage = len(data) / expected
        if coverage < config.minimum_coverage:
            excluded.append(_excluded(ticker, "Coverage", f"Coverage {coverage:.1%} is below {config.minimum_coverage:.1%}.", len(data), coverage))
            continue
        if len(data) < warmup:
            excluded.append(_excluded(ticker, "History", f"{len(data)} rows is below the {warmup}-row strategy warm-up.", len(data), coverage))
            continue
        if data[["Open", "Close"]].isna().any().any() or (data[["Open", "Close"]] <= 0).any().any():
            excluded.append(_excluded(ticker, "Data validation", "Open and Close prices must be present and positive.", len(data), coverage))
            continue
        try:
            strategy = build_strategy(config.strategy_name, config.strategy_params)
            engine = BacktestEngine(config.backtest_config)
            strategy_result = engine.run(data, strategy.generate_signals(data))
            buy_hold_result = engine.run(data, pd.Series(0, index=data.index, dtype=int), initial_buy=True)
            metrics = calculate_performance(strategy_result.history, strategy_result.trades, config.backtest_config.initial_capital, config.risk_free_rate)
            buy_hold = calculate_performance(buy_hold_result.history, buy_hold_result.trades, config.backtest_config.initial_capital, config.risk_free_rate)
            rows.append({
                "Ticker": ticker, "Final Value": metrics["Final Portfolio Value"], "Strategy Total Return": metrics["Total Return"],
                "Buy and Hold Return": buy_hold["Total Return"], "Excess Return": metrics["Total Return"] - buy_hold["Total Return"],
                "CAGR": metrics["CAGR"], "Sharpe Ratio": metrics["Sharpe Ratio"], "Calmar Ratio": metrics["Calmar Ratio"],
                "Maximum Drawdown": metrics["Maximum Drawdown"], "Completed Trades": metrics["Completed Trades"],
                "Win Rate": metrics["Win Rate"], "Total Transaction Costs": metrics["Total Transaction Costs"], "Data Coverage": coverage,
            })
        except Exception as exc:
            logger.warning("%s backtest failed: %s", ticker, exc)
            excluded.append(_excluded(ticker, "Backtest", str(exc), len(data), coverage))
    result_frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    excluded_frame = pd.DataFrame(excluded, columns=["Ticker", "Failure Stage", "Exclusion Reason", "Data Rows", "Data Coverage"])
    return BatchRunResult(result_frame, excluded_frame, len(submitted), len(result_frame))
