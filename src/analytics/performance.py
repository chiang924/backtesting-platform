from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and np.isfinite(denominator) else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    """Return the most negative peak-to-trough drawdown."""
    values = equity.dropna().astype(float)
    if values.empty:
        return float("nan")
    return float((values / values.cummax() - 1).min())


def calculate_performance(history: pd.DataFrame, trades: pd.DataFrame, initial_capital: float,
                          risk_free_rate: float = 0.0) -> dict[str, Any]:
    """Calculate annualized and trade-level metrics from the engine's audit trail."""
    if history.empty:
        return {"Initial Capital": initial_capital, "Final Portfolio Value": float("nan")}
    equity = history["Total Equity"].astype(float)
    daily = history["Daily Return"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    final = float(equity.iloc[-1])
    total_return = _safe_divide(final, initial_capital) - 1
    years = max((history.index[-1] - history.index[0]).days / 365.25, 0)
    cagr = (final / initial_capital) ** (1 / years) - 1 if years > 0 and final > 0 else float("nan")
    volatility = float(daily.std(ddof=1) * math.sqrt(252)) if len(daily) > 1 else float("nan")
    excess_daily = daily - risk_free_rate / 252
    sharpe = float(excess_daily.mean() / excess_daily.std(ddof=1) * math.sqrt(252)) if len(daily) > 1 and excess_daily.std(ddof=1) > 0 else float("nan")
    mdd = max_drawdown(equity)
    calmar = _safe_divide(cagr, abs(mdd))
    sells = trades[trades["Action"] == "SELL"] if not trades.empty else pd.DataFrame()
    pnl = sells["Realized P&L"].astype(float) if not sells.empty else pd.Series(dtype=float)
    gains, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_value = trades["Gross Transaction Value"].sum() if not trades.empty else 0.0
    avg_equity = equity.mean()
    exposure = float((history["Shares"] > 1e-12).mean())
    return {
        "Initial Capital": float(initial_capital), "Final Portfolio Value": final, "Total Return": total_return,
        "CAGR": cagr, "Annualized Volatility": volatility, "Sharpe Ratio": sharpe,
        "Maximum Drawdown": mdd, "Calmar Ratio": calmar, "Completed Trades": int(len(sells)),
        "Win Rate": float((pnl > 0).mean()) if len(pnl) else float("nan"),
        "Average Gain": float(gains.mean()) if len(gains) else float("nan"),
        "Average Loss": float(losses.mean()) if len(losses) else float("nan"),
        "Profit Factor": _safe_divide(gains.sum(), abs(losses.sum())),
        "Average Holding Period": float(sells["Holding Period (days)"].mean()) if len(sells) else float("nan"),
        "Portfolio Turnover": _safe_divide(gross_value, avg_equity),
        "Total Transaction Costs": float(history["Total Transaction Costs"].iloc[-1]),
        "Exposure Percentage": exposure,
    }


def metrics_frame(metrics_by_name: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the compact table used by strategy comparison."""
    desired = ["Total Return", "CAGR", "Sharpe Ratio", "Maximum Drawdown", "Completed Trades", "Win Rate", "Total Transaction Costs"]
    return pd.DataFrame({name: {key: values.get(key, float("nan")) for key in desired}
                         for name, values in metrics_by_name.items()}).T
