from __future__ import annotations

import pandas as pd

from .base_strategy import BaseStrategy


class MovingAverageCrossover(BaseStrategy):
    """Enter/exit when short simple moving average crosses the long average."""

    name = "Moving Average Crossover"
    description = "Buy on a short-SMA cross above the long SMA; sell on a cross below."

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if short_window < 1 or long_window < 2 or short_window >= long_window:
            raise ValueError("Short moving-average window must be positive and smaller than long window.")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = self._empty_signals(data)
        close = data["Close"].astype(float)
        short = close.rolling(self.short_window, min_periods=self.short_window).mean()
        long = close.rolling(self.long_window, min_periods=self.long_window).mean()
        valid = short.notna() & long.notna()
        above = short > long
        crossed_up = valid & above & ~above.shift(1, fill_value=False)
        crossed_down = valid & ~above & above.shift(1, fill_value=False)
        signals.loc[crossed_up] = 1
        signals.loc[crossed_down] = -1
        return signals
