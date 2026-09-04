import pandas as pd
import pytest

from src.strategies import CustomRuleStrategy, MomentumStrategy, MovingAverageCrossover, RegimeAdaptiveBreakout, RSIMeanReversion, RuleSyntaxError


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


def test_regime_adaptive_strategy_generates_entry_exit_and_is_prefix_stable():
    data = frame([10, 10, 9, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9])
    strategy = RegimeAdaptiveBreakout(
        ema_window=3,
        efficiency_window=2,
        efficiency_threshold=.5,
        breakout_window=3,
        exit_window=2,
        rsi_period=2,
        rsi_oversold=40,
        rsi_exit=60,
    )
    signals = strategy.generate_signals(data)
    assert 1 in signals.values
    assert -1 in signals.values
    prefix = strategy.generate_signals(data.iloc[:10])
    pd.testing.assert_series_equal(prefix, signals.iloc[:10])


def test_efficiency_ratio_is_bounded_and_detects_directional_prices():
    trend = frame([10, 11, 12, 13, 14, 15, 16])
    noise = frame([10, 11, 10, 11, 10, 11, 10])
    strategy = RegimeAdaptiveBreakout(ema_window=3, efficiency_window=3, breakout_window=4, exit_window=2)
    trend_ratio = strategy.efficiency_ratio(trend["Close"]).dropna()
    noise_ratio = strategy.efficiency_ratio(noise["Close"]).dropna()
    assert trend_ratio.between(0, 1).all()
    assert noise_ratio.between(0, 1).all()
    assert trend_ratio.mean() > noise_ratio.mean()


def test_regime_adaptive_parameter_validation():
    with pytest.raises(ValueError, match="smaller than the breakout"):
        RegimeAdaptiveBreakout(breakout_window=20, exit_window=20)
    with pytest.raises(ValueError, match="Efficiency threshold"):
        RegimeAdaptiveBreakout(efficiency_threshold=0)
    with pytest.raises(ValueError, match="RSI settings"):
        RegimeAdaptiveBreakout(rsi_oversold=60, rsi_exit=55)


def test_custom_rule_strategy_compiles_indicators_crosses_and_boolean_logic():
    data = frame([10, 9, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9])
    strategy = CustomRuleStrategy(
        "close > ema(3) and close > highest(3)",
        "crosses_below(close, ema(3)) or rsi(2) < 20",
    )
    signals = strategy.generate_signals(data)
    assert 1 in signals.values
    assert -1 in signals.values
    assert strategy.minimum_history == 4


def test_custom_rule_strategy_is_prefix_stable_and_excludes_current_breakout_close():
    data = frame([10, 10, 11, 12, 13, 14, 12, 11])
    strategy = CustomRuleStrategy("close > highest(3)", "close < lowest(2)")
    full = strategy.generate_signals(data)
    prefix = strategy.generate_signals(data.iloc[:6])
    pd.testing.assert_series_equal(prefix, full.iloc[:6])
    assert full.iloc[3] == 1


@pytest.mark.parametrize(
    "rule",
    [
        "__import__('os').system('whoami')",
        "close.__class__",
        "close[0] > 1",
        "sma(2.5) > close",
        "close and ema(5)",
    ],
)
def test_custom_rule_language_rejects_unsafe_or_non_boolean_expressions(rule):
    if rule == "close and ema(5)":
        strategy = CustomRuleStrategy(rule, "close < ema(5)")
        with pytest.raises(RuleSyntaxError):
            strategy.generate_signals(frame([1, 2, 3, 4, 5, 6]))
    else:
        with pytest.raises(RuleSyntaxError):
            CustomRuleStrategy(rule, "close < ema(5)")
