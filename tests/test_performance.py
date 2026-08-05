import pandas as pd
import pytest

from src.analytics.performance import calculate_performance, max_drawdown
from src.data.data_loader import normalize_ohlcv


def test_max_drawdown_and_return_are_correct():
    idx = pd.date_range("2023-01-01", periods=4, freq="B")
    history = pd.DataFrame({"Total Equity": [100, 120, 90, 110], "Daily Return": [0, .2, -.25, .2222], "Shares": [0, 1, 1, 0], "Total Transaction Costs": [0, 0, 0, 0]}, index=idx)
    metrics = calculate_performance(history, pd.DataFrame(), 100)
    assert max_drawdown(history["Total Equity"]) == pytest.approx(-.25)
    assert metrics["Total Return"] == pytest.approx(.10)


def test_empty_and_insufficient_data_are_safe():
    with pytest.raises(ValueError):
        normalize_ohlcv(pd.DataFrame())
    one = pd.DataFrame({"Total Equity": [100], "Daily Return": [0.0], "Shares": [0], "Total Transaction Costs": [0] }, index=pd.DatetimeIndex(["2024-01-01"]))
    metrics = calculate_performance(one, pd.DataFrame(), 100)
    assert pd.isna(metrics["CAGR"])


def test_cagr_uses_elapsed_calendar_time():
    history = pd.DataFrame({"Total Equity": [100, 110], "Daily Return": [0.0, 0.1], "Shares": [0, 0], "Total Transaction Costs": [0, 0]},
                           index=pd.DatetimeIndex(["2024-01-01", "2025-01-01"]))
    metrics = calculate_performance(history, pd.DataFrame(), 100)
    assert metrics["CAGR"] == pytest.approx(1.1 ** (365.25 / 366) - 1)
