from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def equity_chart(results: dict[str, object]) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        history = result.history
        fig.add_trace(go.Scatter(x=history.index, y=history["Total Equity"], mode="lines", name=name))
    fig.update_layout(title="Portfolio Equity", xaxis_title="Date", yaxis_title="Value", hovermode="x unified", height=430)
    return fig


def price_signals_chart(data: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="Price"))
    if not trades.empty:
        for action, color, symbol in [("BUY", "#22c55e", "triangle-up"), ("SELL", "#ef4444", "triangle-down")]:
            subset = trades[trades["Action"] == action]
            fig.add_trace(go.Scatter(x=subset["Execution Date"], y=subset["Execution Price"], mode="markers", name=action,
                                     marker={"color": color, "size": 11, "symbol": symbol},
                                     hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>" + action + "</extra>"))
    fig.update_layout(title="Price and Executed Trades", xaxis_rangeslider_visible=False, yaxis_title="Price", height=500)
    return fig


def drawdown_chart(history: pd.DataFrame) -> go.Figure:
    drawdown = history["Total Equity"] / history["Total Equity"].cummax() - 1
    fig = go.Figure(go.Scatter(x=history.index, y=drawdown, fill="tozeroy", line={"color": "#ef4444"}, name="Drawdown"))
    fig.update_layout(title="Portfolio Drawdown", yaxis_tickformat=".1%", height=320)
    return fig


def returns_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=history.index, y=history["Daily Return"], name="Daily return", marker_color="#3b82f6"))
    fig.update_layout(title="Daily Portfolio Returns", yaxis_tickformat=".1%", height=300)
    return fig


def monthly_heatmap(history: pd.DataFrame) -> go.Figure:
    monthly = (1 + history["Daily Return"]).resample("ME").prod() - 1
    grid = pd.DataFrame({"Month": monthly.index.month, "Year": monthly.index.year, "Return": monthly.values})
    pivot = grid.pivot(index="Month", columns="Year", values="Return").reindex(range(1, 13))
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                               colorscale="RdYlGn", zmid=0, colorbar_tickformat=".0%", hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>"))
    fig.update_layout(title="Monthly Returns", height=400)
    return fig


def allocation_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history["Cash"], stackgroup="one", name="Cash"))
    fig.add_trace(go.Scatter(x=history.index, y=history["Position Value"], stackgroup="one", name="Position"))
    fig.update_layout(title="Cash and Position Allocation", yaxis_title="Value", height=350)
    return fig
