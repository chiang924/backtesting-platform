from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def adaptive_regime_chart(
    data: pd.DataFrame,
    indicators: pd.DataFrame,
    trades: pd.DataFrame,
    efficiency_threshold: float,
) -> go.Figure:
    """Two-panel diagnostic chart for the regime-adaptive strategy."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28],
        subplot_titles=("Price, risk filter and channels", "Kaufman Efficiency Ratio"),
    )
    custom = indicators[["Regime", "RSI"]].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close",
            line={"color": "#e2e8f0", "width": 1.5},
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Close: %{y:.2f}<br>Regime: %{customdata[0]}"
                "<br>RSI: %{customdata[1]:.1f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    overlays = [
        ("Long-Term EMA", "#38bdf8", "solid"),
        ("Breakout High", "#22c55e", "dash"),
        ("Exit Low", "#f59e0b", "dot"),
    ]
    for column, color, dash in overlays:
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators[column],
                mode="lines",
                name=column,
                line={"color": color, "width": 1.2, "dash": dash},
            ),
            row=1,
            col=1,
        )
    if not trades.empty:
        for action, color, symbol in [
            ("BUY", "#22c55e", "triangle-up"),
            ("SELL", "#ef4444", "triangle-down"),
        ]:
            subset = trades[trades["Action"] == action]
            fig.add_trace(
                go.Scatter(
                    x=subset["Execution Date"],
                    y=subset["Execution Price"],
                    mode="markers",
                    name=action,
                    marker={"color": color, "size": 10, "symbol": symbol},
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>" + action + "</extra>",
                ),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Scatter(
            x=indicators.index,
            y=indicators["Efficiency Ratio"],
            mode="lines",
            name="Efficiency Ratio",
            line={"color": "#a78bfa", "width": 1.5},
        ),
        row=2,
        col=1,
    )
    fig.add_hrect(
        y0=efficiency_threshold,
        y1=1,
        fillcolor="#22c55e",
        opacity=0.08,
        line_width=0,
        row=2,
        col=1,
        annotation_text="Trend regime",
        annotation_position="top left",
    )
    fig.add_hline(
        y=efficiency_threshold,
        line_dash="dash",
        line_color="#a78bfa",
        row=2,
        col=1,
        annotation_text=f"Threshold {efficiency_threshold:.2f}",
        annotation_position="bottom right",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        title="Regime-Adaptive Breakout Diagnostics",
        hovermode="x unified",
        height=700,
        xaxis_rangeslider_visible=False,
    )
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
