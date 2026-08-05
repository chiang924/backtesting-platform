from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import math
import pandas as pd

from .portfolio import Portfolio


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_rate: float = 0.0  # decimal: 0.001 = 0.1%
    fixed_commission: float = 0.0
    slippage_bps: float = 0.0
    fractional_shares: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        if self.commission_rate < 0 or self.fixed_commission < 0 or self.slippage_bps < 0:
            raise ValueError("Transaction costs cannot be negative.")


@dataclass
class BacktestResult:
    history: pd.DataFrame
    trades: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


class BacktestEngine:
    """Transparent daily long/cash simulator. Signals execute at next session's open."""

    TRADE_COLUMNS = [
        "Trade #", "Signal Date", "Execution Date", "Action", "Execution Price", "Quantity",
        "Gross Transaction Value", "Commission", "Slippage Cost", "Cash After Transaction",
        "Realized P&L", "Holding Period (days)", "Exit Reason",
    ]

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(self, data: pd.DataFrame, signals: pd.Series, initial_buy: bool = False) -> BacktestResult:
        """Run daily accounting. A row's signal becomes an order on the next row's open."""
        if data is None or len(data) < 1:
            raise ValueError("At least one valid trading day is required.")
        required = {"Open", "Close"}
        if not required.issubset(data.columns):
            raise ValueError("Data must include Open and Close columns.")
        prices = data.copy().sort_index()
        prices = prices[~prices.index.duplicated(keep="last")]
        prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
        prices["Open"] = pd.to_numeric(prices["Open"], errors="coerce")
        prices = prices.dropna(subset=["Close"])
        if prices.empty:
            raise ValueError("No valid Close prices are available.")
        aligned = signals.reindex(prices.index, fill_value=0).fillna(0).astype(int).clip(-1, 1)
        portfolio = Portfolio(cash=float(self.config.initial_capital))
        warnings: list[str] = []
        history: list[dict[str, float | pd.Timestamp]] = []
        trades: list[dict[str, object]] = []
        pending: tuple[int, pd.Timestamp] | None = None
        prior_equity = self.config.initial_capital

        for position, (date, bar) in enumerate(prices.iterrows()):
            market_open = float(bar["Open"]) if pd.notna(bar["Open"]) and bar["Open"] > 0 else None
            close = float(bar["Close"])
            if pending is not None:
                action, signal_date = pending
                execution_price = market_open
                fallback = False
                if execution_price is None:
                    execution_price, fallback = close, True
                    warnings.append(f"{date.date()}: Open was unavailable; executed signal at Close.")
                if action == 1 and not portfolio.invested:
                    trade = self._buy(portfolio, date, signal_date, execution_price)
                    if trade:
                        trades.append(trade)
                elif action == -1 and portfolio.invested:
                    trades.append(self._sell(portfolio, date, signal_date, execution_price, "Strategy exit"))
                pending = None

            # Buy-and-hold is intentionally entered at the first available opening price.
            if position == 0 and initial_buy and not portfolio.invested:
                execution_price = market_open if market_open is not None else close
                if market_open is None:
                    warnings.append(f"{date.date()}: Open was unavailable; Buy and Hold entered at Close.")
                trade = self._buy(portfolio, date, date, execution_price)
                if trade:
                    trades.append(trade)

            # Reporting is marked to the close, except the mandatory final liquidation below.
            if position == len(prices) - 1 and portfolio.invested:
                trades.append(self._sell(portfolio, date, date, close, "End of test liquidation"))

            market_value = portfolio.shares * close
            equity = portfolio.cash + market_value
            unrealized = market_value - portfolio.cost_basis if portfolio.invested else 0.0
            daily_return = equity / prior_equity - 1 if prior_equity else 0.0
            history.append({
                "Date": date, "Cash": portfolio.cash, "Shares": portfolio.shares,
                "Position Value": market_value, "Total Equity": equity, "Daily Return": daily_return,
                "Cumulative Return": equity / self.config.initial_capital - 1,
                "Realized P&L": portfolio.realized_pnl, "Unrealized P&L": unrealized,
                "Total Transaction Costs": portfolio.total_costs,
            })
            prior_equity = equity
            # Queue only after closing accounting: this prevents same-day/look-ahead execution.
            signal = int(aligned.loc[date])
            if position < len(prices) - 1 and signal in (-1, 1):
                pending = (signal, date)

        trades_frame = pd.DataFrame(trades, columns=self.TRADE_COLUMNS)
        if not trades_frame.empty:
            trades_frame["Trade #"] = range(1, len(trades_frame) + 1)
        return BacktestResult(pd.DataFrame(history).set_index("Date"), trades_frame, warnings)

    def _buy(self, portfolio: Portfolio, execution_date: pd.Timestamp, signal_date: pd.Timestamp,
             market_price: float) -> dict[str, object] | None:
        effective = market_price * (1 + self.config.slippage_bps / 10_000)
        denominator = effective * (1 + self.config.commission_rate)
        available = portfolio.cash - self.config.fixed_commission
        if available <= 0 or denominator <= 0:
            return None
        quantity = available / denominator
        if not self.config.fractional_shares:
            quantity = math.floor(quantity)
        if quantity <= 0:
            return None
        gross = quantity * effective
        commission = gross * self.config.commission_rate + self.config.fixed_commission
        # Guard rounding: never permit negative cash due to a computed order size.
        if gross + commission > portfolio.cash + 1e-8:
            return None
        portfolio.cash -= gross + commission
        # Floating-point order sizing can otherwise leave an immaterial negative residue.
        if abs(portfolio.cash) < 1e-9:
            portfolio.cash = 0.0
        portfolio.shares = quantity
        portfolio.cost_basis = gross + commission
        portfolio.entry_date = execution_date
        portfolio.total_costs += commission + (effective - market_price) * quantity
        return self._trade_row(len_placeholder=0, signal_date=signal_date, execution_date=execution_date,
                               action="BUY", price=effective, quantity=quantity, gross=gross,
                               commission=commission, slippage=abs(effective - market_price) * quantity,
                               cash=portfolio.cash, realized=0.0, holding=None, reason="Entry")

    def _sell(self, portfolio: Portfolio, execution_date: pd.Timestamp, signal_date: pd.Timestamp,
              market_price: float, reason: str) -> dict[str, object]:
        effective = market_price * (1 - self.config.slippage_bps / 10_000)
        quantity = portfolio.shares
        gross = quantity * effective
        commission = gross * self.config.commission_rate + self.config.fixed_commission
        proceeds = gross - commission
        realized = proceeds - portfolio.cost_basis
        holding = (execution_date - portfolio.entry_date).days if portfolio.entry_date is not None else None
        portfolio.cash += proceeds
        portfolio.realized_pnl += realized
        portfolio.total_costs += commission + (market_price - effective) * quantity
        portfolio.shares, portfolio.cost_basis, portfolio.entry_date = 0.0, 0.0, None
        return self._trade_row(len_placeholder=0, signal_date=signal_date, execution_date=execution_date,
                               action="SELL", price=effective, quantity=quantity, gross=gross,
                               commission=commission, slippage=abs(effective - market_price) * quantity,
                               cash=portfolio.cash, realized=realized, holding=holding, reason=reason)

    @staticmethod
    def _trade_row(len_placeholder: int, **values: object) -> dict[str, object]:
        return {
            "Trade #": len_placeholder, "Signal Date": values["signal_date"], "Execution Date": values["execution_date"],
            "Action": values["action"], "Execution Price": values["price"], "Quantity": values["quantity"],
            "Gross Transaction Value": values["gross"], "Commission": values["commission"],
            "Slippage Cost": values["slippage"], "Cash After Transaction": values["cash"],
            "Realized P&L": values["realized"], "Holding Period (days)": values["holding"], "Exit Reason": values["reason"],
        }
