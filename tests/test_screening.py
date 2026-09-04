import io
import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.screening.batch_runner import BatchRunConfig, build_strategy, required_observations, run_batch
from src.screening.ranking import RANKING_METRICS, rank_results, select_top_bottom
from src.universe.universe_validator import parse_custom_tickers, validate_universe_csv
from src.strategies import CustomRuleStrategy, RegimeAdaptiveBreakout


def ranking_frame():
    return pd.DataFrame({
        "Ticker": ["A", "B", "C", "D", "E", "F", "Z"],
        "Strategy Total Return": [.40, .30, .20, .10, .05, -.10, .99],
        "CAGR": [.21, .20, .19, .18, .17, .16, np.nan], "Sharpe Ratio": [1.0, .9, .8, .7, .6, .5, np.nan],
        "Calmar Ratio": [1.6, 1.5, 1.4, 1.3, 1.2, 1.1, np.nan], "Maximum Drawdown": [-.1, -.2, -.3, -.4, -.5, -.6, np.nan],
        "Excess Return": [.1, .2, .3, .4, .5, .6, np.nan],
        "Completed Trades": [1, 1, 1, 1, 1, 1, 0],
    })


def test_top_and_bottom_selection_and_undefined_metrics():
    ranked = rank_results(ranking_frame(), "Total Return")
    top, bottom = select_top_bottom(ranked, "Total Return")
    assert top["Ticker"].tolist() == ["A", "B", "C"]
    assert bottom["Ticker"].tolist() == ["F", "E", "D"]
    assert "Z" not in top["Ticker"].tolist()
    assert pd.isna(ranked.loc[ranked["Ticker"] == "Z", "Rank"].iloc[0])


def test_ranking_direction_for_drawdown_and_excess_return():
    ranked_drawdown = rank_results(ranking_frame(), "Maximum Drawdown")
    ranked_excess = rank_results(ranking_frame(), "Excess Return versus Buy and Hold")
    assert ranked_drawdown.iloc[0]["Ticker"] == "A"  # -10% is less bad than -20%.
    assert ranked_excess.iloc[0]["Ticker"] == "F"


@pytest.mark.parametrize("metric", list(RANKING_METRICS))
def test_each_ranking_metric_prefers_the_higher_value_including_less_negative_drawdown(metric):
    ranked = rank_results(ranking_frame(), metric)
    assert ranked.iloc[0]["Ticker"] == ("F" if metric == "Excess Return versus Buy and Hold" else "A")


def bars(values=(10, 9, 8, 10, 12, 11, 8)):
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"Open": values, "High": values, "Low": values, "Close": values, "Volume": 1}, index=index)


def batch_config(minimum_coverage=.9):
    return BatchRunConfig(
        start="2024-01-01", end="2024-01-09", strategy_name="Moving Average Crossover",
        strategy_params={"ma_short": 2, "ma_long": 3, "rsi_period": 2, "rsi_oversold": 30., "rsi_overbought": 70., "mom_lookback": 2, "mom_entry": 3., "mom_exit": 0.},
        backtest_config=BacktestConfig(100., commission_rate=.01, fixed_commission=1., slippage_bps=10., fractional_shares=True),
        minimum_coverage=minimum_coverage,
    )


def test_insufficient_history_and_failed_ticker_do_not_stop_batch():
    def fetch(ticker, _start, _end):
        if ticker == "BAD":
            raise RuntimeError("network unavailable")
        return bars() if ticker == "GOOD" else bars((10, 11))
    result = run_batch(["GOOD", "SHORT", "BAD"], batch_config(.1), data_fetcher=fetch)
    assert result.results["Ticker"].tolist() == ["GOOD"]
    assert set(result.excluded["Ticker"]) == {"SHORT", "BAD"}
    assert set(result.excluded["Failure Stage"]) == {"History", "Download"}


def test_zero_trade_stocks_are_unranked_but_retained():
    frame = ranking_frame().iloc[:1].copy()
    frame["Completed Trades"] = 0
    ranked = rank_results(frame, "Total Return")
    top, bottom = select_top_bottom(ranked, "Total Return")
    assert not ranked["Eligible for Ranking"].iloc[0]
    assert top.empty and bottom.empty


def test_all_failed_batch_produces_renderable_empty_ranking():
    result = run_batch(["BAD"], batch_config(), data_fetcher=lambda *_args: (_ for _ in ()).throw(RuntimeError("bad data")))
    ranked = rank_results(result.results, "Total Return")
    top, bottom = select_top_bottom(ranked, "Total Return")
    assert result.results.empty and len(result.excluded) == 1
    assert {"Eligible for Ranking", "Rank", "Strategy Total Return"}.issubset(ranked.columns)
    assert top.empty and bottom.empty


def test_custom_tickers_are_stripped_uppercased_and_deduplicated():
    assert parse_custom_tickers(" aapl,MSFT, aapl , brk.b ") == ["AAPL", "MSFT", "BRK-B"]


def test_excess_return_and_shared_backtest_assumptions():
    result = run_batch(["ONE", "TWO"], batch_config(), data_fetcher=lambda *_args: bars())
    assert len(result.results) == 2
    assert np.allclose(result.results["Excess Return"], result.results["Strategy Total Return"] - result.results["Buy and Hold Return"])
    assert result.results["Total Transaction Costs"].ge(0).all()


def test_fractional_buy_and_hold_is_nominal_price_independent():
    index = pd.date_range("2024-01-01", periods=2, freq="B")
    low = pd.DataFrame({"Open": [10., 11.], "Close": [10., 11.]}, index=index)
    high = pd.DataFrame({"Open": [1000., 1100.], "Close": [1000., 1100.]}, index=index)
    config = BacktestConfig(100., fractional_shares=True)
    low_result = BacktestEngine(config).run(low, pd.Series(0, index=index), initial_buy=True)
    high_result = BacktestEngine(config).run(high, pd.Series(0, index=index), initial_buy=True)
    assert low_result.history["Cumulative Return"].iloc[-1] == pytest.approx(high_result.history["Cumulative Return"].iloc[-1])


def test_universe_csv_validation():
    assert validate_universe_csv(io.BytesIO(b"Ticker\n aapl \nMSFT\naapl\n")) == ["AAPL", "MSFT"]
    with pytest.raises(ValueError, match="Ticker column"):
        validate_universe_csv(io.BytesIO(b"Symbol\nAAPL\n"))


def test_regime_adaptive_strategy_is_available_to_batch_ranking():
    params = {
        "rab_ema": 200,
        "rab_efficiency_window": 20,
        "rab_efficiency_threshold": .30,
        "rab_breakout": 55,
        "rab_exit": 20,
        "rab_rsi_period": 14,
        "rab_rsi_oversold": 30.,
        "rab_rsi_exit": 55.,
    }
    strategy = build_strategy("Regime-Adaptive Breakout", params)
    assert isinstance(strategy, RegimeAdaptiveBreakout)
    assert required_observations("Regime-Adaptive Breakout", params) == 200


def test_custom_rule_strategy_is_available_to_batch_ranking():
    params = {"custom_entry": "close > sma(20)", "custom_exit": "close < sma(10)"}
    strategy = build_strategy("Custom Rule Strategy", params)
    assert isinstance(strategy, CustomRuleStrategy)
    assert required_observations("Custom Rule Strategy", params) == 21
