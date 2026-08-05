import pandas as pd
import pytest

from src.strategies import MomentumStrategy, MovingAverageCrossover, RSIMeanReversion


def frame(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"Open": values, "High": values, "Low": values, "Close": values, "Volume": 1}, index=idx)


def test_moving_average_cross_generates_entry_and_exit():
    signals = MovingAverageCrossover(2, 3).generate_signals(frame([10, 9, 8, 10, 12, 11, 8]))
    assert 1 in signals.values
    assert -1 in signals.values


def test_rsi_is_bounded_and_rebound_can_signal():
    strategy = RSIMeanReversion(2, 40, 70)
    data = frame([10, 8, 6, 7, 8, 10, 12])
    rsi = strategy.rsi(data["Close"])
    assert rsi.dropna().between(0, 100).all()
    assert 1 in strategy.generate_signals(data).values


def test_momentum_uses_only_trailing_values():
    strategy = MomentumStrategy(2, 0.05, -0.01)
    data = frame([10, 10, 11, 12, 13])
    assert strategy.momentum(data["Close"]).iloc[2] == pytest.approx(0.1)
    assert strategy.momentum(data["Close"]).iloc[0] != strategy.momentum(data["Close"]).iloc[0]  # NaN
