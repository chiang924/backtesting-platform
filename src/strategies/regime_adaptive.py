from __future__ import annotations

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from .rsi_strategy import RSIMeanReversion


class RegimeAdaptiveBreakout(BaseStrategy):
    """Long-or-cash strategy that adapts its entry trigger to the market regime.

    The long-term EMA is the risk filter. Kaufman's efficiency ratio separates
    directional markets from noisy ranges. Trend regimes use a Donchian-style
    breakout; range regimes use an RSI rebound. Exits combine a long-term risk
    stop, a shorter price channel, and an RSI normalization exit in range mode.
    """

    name = "Regime-Adaptive Breakout"
    description = (
        "Risk-on above the long-term EMA; buy Donchian breakouts in efficient trends "
        "or RSI rebounds in ranges, then return to cash on regime-aware exits."
    )

    def __init__(
        self,
        ema_window: int = 200,
        efficiency_window: int = 20,
        efficiency_threshold: float = 0.30,
        breakout_window: int = 55,
        exit_window: int = 20,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_exit: float = 55.0,
    ) -> None:
        if ema_window < 2:
            raise ValueError("EMA window must be at least 2.")
        if efficiency_window < 2:
            raise ValueError("Efficiency-ratio window must be at least 2.")
        if not 0 < efficiency_threshold <= 1:
            raise ValueError("Efficiency threshold must be in (0, 1].")
        if exit_window < 2 or breakout_window < 3 or exit_window >= breakout_window:
            raise ValueError("Exit window must be at least 2 and smaller than the breakout window.")
        if rsi_period < 2 or not 0 <= rsi_oversold < rsi_exit <= 100:
            raise ValueError("RSI settings must satisfy 0 <= oversold < exit <= 100.")
        self.ema_window = ema_window
        self.efficiency_window = efficiency_window
        self.efficiency_threshold = efficiency_threshold
        self.breakout_window = breakout_window
        self.exit_window = exit_window
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_exit = rsi_exit

    def efficiency_ratio(self, close: pd.Series) -> pd.Series:
        """Kaufman efficiency ratio: net movement divided by path length."""
        values = close.astype(float)
        direction = values.diff(self.efficiency_window).abs()
        path_length = values.diff().abs().rolling(
            self.efficiency_window, min_periods=self.efficiency_window
        ).sum()
        ratio = direction / path_length.replace(0, np.nan)
        return ratio.clip(0, 1).rename("Efficiency Ratio")

    def indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return auditable indicator and regime state used by the UI."""
        close = data["Close"].astype(float)
        ema = close.ewm(
            span=self.ema_window, adjust=False, min_periods=self.ema_window
        ).mean()
        efficiency = self.efficiency_ratio(close)
        breakout_high = close.rolling(
            self.breakout_window, min_periods=self.breakout_window
        ).max().shift(1)
        exit_low = close.rolling(
            self.exit_window, min_periods=self.exit_window
        ).min().shift(1)
        rsi = RSIMeanReversion(
            self.rsi_period, self.rsi_oversold, self.rsi_exit
        ).rsi(close)

        ready = ema.notna() & efficiency.notna() & breakout_high.notna() & exit_low.notna() & rsi.notna()
        risk_on = close > ema
        trending = efficiency >= self.efficiency_threshold
        regime = pd.Series("Warm-up", index=data.index, dtype="object", name="Regime")
        regime.loc[ready & ~risk_on] = "Risk-Off"
        regime.loc[ready & risk_on & trending] = "Trend"
        regime.loc[ready & risk_on & ~trending] = "Range"

        return pd.DataFrame(
            {
                "Close": close,
                "Long-Term EMA": ema,
                "Efficiency Ratio": efficiency,
                "Breakout High": breakout_high,
                "Exit Low": exit_low,
                "RSI": rsi,
                "Regime": regime,
            },
            index=data.index,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        indicators = self.indicators(data)
        close = indicators["Close"]
        ema = indicators["Long-Term EMA"]
        breakout_high = indicators["Breakout High"]
        exit_low = indicators["Exit Low"]
        rsi = indicators["RSI"]
        regime = indicators["Regime"]

        # The channel already excludes today's close, so a close above it is a fresh breakout.
        breakout = close > breakout_high
        rsi_rebound = (rsi.shift(1) <= self.rsi_oversold) & (rsi > rsi.shift(1))
        trend_entry = (regime == "Trend") & breakout
        range_entry = (regime == "Range") & rsi_rebound

        risk_exit = (close < ema) & (close.shift(1) >= ema.shift(1))
        channel_exit = (close < exit_low) & (close.shift(1) >= exit_low.shift(1))
        range_exit = (
            (regime == "Range")
            & (rsi >= self.rsi_exit)
            & (rsi.shift(1) < self.rsi_exit)
        )

        signals = self._empty_signals(data)
        signals.loc[(trend_entry | range_entry).fillna(False)] = 1
        # An exit always wins if entry and exit conditions collide on one close.
        signals.loc[(risk_exit | channel_exit | range_exit).fillna(False)] = -1
        signals.attrs["strategy"] = self.name
        signals.attrs["efficiency_threshold"] = self.efficiency_threshold
        return signals
