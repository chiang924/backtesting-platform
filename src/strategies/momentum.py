from __future__ import annotations

import pandas as pd

from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Long-only strategy based on trailing, not forward, percentage return."""

    name = "Momentum"
    description = "Buy when trailing momentum crosses above entry; sell when it crosses below exit."

    def __init__(self, lookback: int = 20, entry_threshold: float = 0.03, exit_threshold: float = 0.0) -> None:
        if lookback < 1 or exit_threshold >= entry_threshold:
            raise ValueError("Lookback must be positive and exit threshold must be below entry threshold.")
        self.lookback, self.entry_threshold, self.exit_threshold = lookback, entry_threshold, exit_threshold

    def momentum(self, close: pd.Series) -> pd.Series:
        return close.astype(float).pct_change(self.lookback).rename("Momentum")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = self._empty_signals(data)
        momentum = self.momentum(data["Close"])
        enter = (momentum > self.entry_threshold) & (momentum.shift(1) <= self.entry_threshold)
        exit_ = (momentum < self.exit_threshold) & (momentum.shift(1) >= self.exit_threshold)
        signals.loc[enter.fillna(False)] = 1
        signals.loc[exit_.fillna(False)] = -1
        return signals
