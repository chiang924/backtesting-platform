import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig, BacktestEngine


def bars(opens=(10, 11, 12, 13), closes=(10, 11, 12, 13)):
    index = pd.date_range("2024-01-01", periods=len(opens), freq="B")
    return pd.DataFrame({"Open": opens, "High": opens, "Low": opens, "Close": closes, "Volume": 100}, index=index)


def test_signal_executes_next_day_not_same_day():
    data = bars()
    result = BacktestEngine(BacktestConfig(100)).run(data, pd.Series([1, 0, 0, 0], index=data.index))
    buy = result.trades.iloc[0]
    assert buy["Signal Date"] == data.index[0]
    assert buy["Execution Date"] == data.index[1]
    assert buy["Execution Price"] == 11


def test_commission_reduces_final_equity():
    data = bars((10, 10, 10), (10, 10, 10))
    signals = pd.Series([1, 0, 0], index=data.index)
    no_fee = BacktestEngine(BacktestConfig(100)).run(data, signals)
    with_fee = BacktestEngine(BacktestConfig(100, commission_rate=0.01)).run(data, signals)
    assert with_fee.history["Total Equity"].iloc[-1] < no_fee.history["Total Equity"].iloc[-1]
    assert with_fee.history["Total Transaction Costs"].iloc[-1] == pytest.approx(1.980198, rel=1e-5)


def test_slippage_direction_and_cash_limit():
    data = bars((10, 10, 10), (10, 10, 10))
    result = BacktestEngine(BacktestConfig(100, slippage_bps=100)).run(data, pd.Series([1, 0, 0], index=data.index))
    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy["Execution Price"] == pytest.approx(10.1)
    assert sell["Execution Price"] == pytest.approx(9.9)
    assert (result.history["Cash"] >= -1e-8).all()


def test_duplicate_entries_and_empty_exits_are_ignored():
    data = bars((10, 10, 10, 10, 10), (10, 10, 10, 10, 10))
    signals = pd.Series([-1, 1, 1, -1, -1], index=data.index)
    result = BacktestEngine(BacktestConfig(100)).run(data, signals)
    assert list(result.trades["Action"]) == ["BUY", "SELL"]


def test_final_liquidation_is_logged():
    data = bars((10, 10), (10, 12))
    result = BacktestEngine(BacktestConfig(100)).run(data, pd.Series([1, 0], index=data.index))
    assert result.trades.iloc[-1]["Exit Reason"] == "End of test liquidation"
    assert result.history["Shares"].iloc[-1] == 0
