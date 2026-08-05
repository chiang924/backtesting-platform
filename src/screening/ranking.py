from __future__ import annotations

import pandas as pd

RANKING_METRICS = {
    "Total Return": "Strategy Total Return",
    "CAGR": "CAGR",
    "Sharpe Ratio": "Sharpe Ratio",
    "Calmar Ratio": "Calmar Ratio",
    "Maximum Drawdown": "Maximum Drawdown",
    "Excess Return versus Buy and Hold": "Excess Return",
}


def ranking_ascending(metric: str) -> bool:
    """Return whether lower values are better for the selected metric."""
    if metric not in RANKING_METRICS:
        raise ValueError(f"Unsupported ranking metric: {metric}")
    # Drawdowns are negative, so the less negative (higher) value is better.
    return False


def rank_results(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Sort valid round-trip strategies and leave undefined/zero-trade rows unranked."""
    if results.empty:
        empty = results.copy()
        empty["Eligible for Ranking"] = pd.Series(dtype="bool")
        empty["Rank"] = pd.Series(dtype="Int64")
        return empty
    column = RANKING_METRICS[metric]
    ranked = results.copy()
    eligible = (ranked["Completed Trades"] > 0) & ranked[column].notna()
    ranked["Eligible for Ranking"] = eligible
    ranked["Rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    eligible_index = ranked.loc[eligible].sort_values(column, ascending=ranking_ascending(metric), kind="stable").index
    ranked.loc[eligible_index, "Rank"] = range(1, len(eligible_index) + 1)
    return ranked.sort_values(["Eligible for Ranking", "Rank", "Ticker"], ascending=[False, True, True], na_position="last")


def select_top_bottom(ranked: pd.DataFrame, metric: str, count: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select ranking-eligible top and bottom slices without promoting undefined values."""
    if ranked.empty:
        return ranked.copy(), ranked.copy()
    eligible = ranked[ranked["Eligible for Ranking"]].sort_values("Rank")
    return eligible.head(count).copy(), eligible.tail(count).sort_values("Rank", ascending=False).copy()
