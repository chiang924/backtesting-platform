from __future__ import annotations

import pandas as pd

from .base_strategy import BaseStrategy


class RSIMeanReversion(BaseStrategy):
    """Wilder RSI strategy that buys a rebound from oversold and sells at overbought."""

    name = "RSI Mean Reversion"
    description = "Buy when RSI was at/below oversold and rises; sell when RSI reaches overbought."

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        if period < 2 or not 0 <= oversold < overbought <= 100:
            raise ValueError("RSI period must be at least 2 and thresholds must satisfy 0 <= oversold < overbought <= 100.")
        self.period, self.oversold, self.overbought = period, oversold, overbought

    def rsi(self, close: pd.Series) -> pd.Series:
        delta = close.astype(float).diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1 / self.period, adjust=False, min_periods=self.period).mean()
        avg_loss = losses.ewm(alpha=1 / self.period, adjust=False, min_periods=self.period).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        result = 100 - (100 / (1 + rs))
        result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
        result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
        return result.rename("RSI")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = self._empty_signals(data)
        rsi = self.rsi(data["Close"])
        # A rebound is unambiguous: yesterday was oversold and today's RSI rose.
        enter = (rsi.shift(1) <= self.oversold) & (rsi > rsi.shift(1))
        exit_ = (rsi >= self.overbought) & (rsi.shift(1) < self.overbought)
        signals.loc[enter.fillna(False)] = 1
        signals.loc[exit_.fillna(False)] = -1
        return signals
