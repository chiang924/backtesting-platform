from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Shared interface for long-or-cash daily trading strategies."""

    name: str
    description: str

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return an indexed series: 1=enter, -1=exit, 0=no new order."""

    @staticmethod
    def _empty_signals(data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index, dtype="int64", name="signal")
