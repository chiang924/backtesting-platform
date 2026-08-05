from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Portfolio:
    """Mutable long-only portfolio state used by the execution engine."""

    cash: float
    shares: float = 0.0
    cost_basis: float = 0.0
    entry_date: object | None = None
    realized_pnl: float = 0.0
    total_costs: float = 0.0

    @property
    def invested(self) -> bool:
        return self.shares > 1e-12
