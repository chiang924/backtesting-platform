# Modular Backtesting Lab

A transparent Streamlit application for researching long-or-cash stock strategies with daily Yahoo Finance data or a local CSV. It implements its own execution and portfolio accounting rather than using a backtesting framework.

> Educational and research use only — this is not investment advice, and historical results do not guarantee future performance.

## Features

- Download and cache daily OHLCV data from Yahoo Finance, with robust MultiIndex normalization.
- CSV fallback, transparent next-trading-day execution, commissions, fixed fees, and slippage.
- Fractional or whole-share sizing, no leverage/shorting, trade audit log, and final-day liquidation.
- A safe text-to-backtest rule language, plus Moving Average Crossover, Wilder RSI Mean Reversion, Momentum, Regime-Adaptive Breakout, Buy & Hold, and multi-strategy comparison.
- Cross-sectional Stock Ranking across current S&P 100/Nasdaq-100 constituents, a custom list, or an uploaded ticker CSV.
- Interactive Plotly charts, downloadable CSV reports, and performance/risk statistics.
- A dedicated regime diagnostic chart showing the long-term EMA, breakout/exit channels, Efficiency Ratio, RSI state, and executed trades.

## Screenshots

_Add screenshots of your configured dashboard here._

## Architecture

```
app.py                      Streamlit UI and orchestration
src/data/data_loader.py     Yahoo download, CSV parsing, validation
src/strategies/             Shared interface, rule compiler, and signal generators
src/backtest/engine.py      Next-open execution and portfolio accounting
src/backtest/portfolio.py   Portfolio state
src/analytics/performance.py Metrics and comparison table
src/visualization/charts.py Plotly figures
src/universe/              Universe providers and ticker validation
src/screening/             Controlled batch runner and ranking rules
tests/                      Deterministic unit tests
```

## Strategies

- **Moving Average Crossover:** buy when the short simple moving average crosses above the long average; sell on a cross below. The short window must be smaller than the long window.
- **RSI Mean Reversion:** RSI uses Wilder smoothing. Buy when the prior day's RSI is at/below oversold and today's RSI rises; sell when RSI crosses into the overbought level.
- **Momentum:** trailing percentage return over the lookback. Buy when it crosses above the entry threshold and sell when it crosses below the exit threshold. The UI thresholds are entered as percentages: `3` means 3% (stored as 0.03 internally).
- **Regime-Adaptive Breakout:** a long-or-cash composite research strategy. Price must first be above its long-term EMA. Kaufman's Efficiency Ratio then labels the risk-on market as directional or range-bound. Directional regimes enter on a prior Donchian-channel breakout; range regimes enter on a Wilder-RSI rebound. A long-term EMA break, a shorter price-channel break, or an RSI normalization exit returns the portfolio to cash. Default parameters are EMA 200, Efficiency Ratio 20 / 0.30, breakout 55, exit 20, and RSI 14 / 30 / 55. This combines established concepts and does not claim academic novelty.
- **Custom Rule Strategy:** type entry and exit formulas directly in the sidebar. The rules are parsed into vectorized Pandas conditions by a restricted syntax tree; Python `eval`, attributes, indexing, imports, assignments, and arbitrary function calls are not permitted. This strategy also works in cross-sectional stock ranking.

Example:

```text
BUY WHEN  close > ema(200) and er(20) >= 0.30 and close > highest(55)
SELL WHEN crosses_below(close, ema(200)) or crosses_below(close, lowest(20))
```

Available price fields are `open`, `high`, `low`, `close`, and `volume`. Indicators are `sma(n)`, `ema(n)`, `rsi(n)`, `momentum(n)`, `er(n)`, `highest(n)`, `lowest(n)`, `atr(n)`, and `zscore(n)`. Event helpers are `crosses_above(a, b)` and `crosses_below(a, b)`. Formulas support numeric comparisons, `+ - * /`, and the lowercase boolean operators `and`, `or`, and `not`. `highest(n)` and `lowest(n)` use the prior *n* closes and exclude today's close.

## Backtesting assumptions

Indicators use information available at close on day *t*. A strategy signal created then is queued and executes at the next valid trading day's Open, avoiding look-ahead bias. If the Open is unavailable, the engine executes at that session's Close and reports a warning.

Buy execution price is `market price × (1 + slippage bps / 10,000)`; sell execution price is `market price × (1 - slippage bps / 10,000)`. Percentage and fixed commission are deducted from cash. Entries use all available cash without allowing negative balances or leverage. Any open position is liquidated at the final valid Close for a complete report.

Metrics use 252 trading days per year. `N/A` is shown when a ratio is undefined.

## Installation and launch

Requires Python 3.11 or newer.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Enter a symbol such as `AAPL`, choose dates and costs, then select **Run Backtest**. **Run All Strategies** additionally compares every built-in and custom strategy with Buy & Hold under the same settings.

## Streamlit Community Cloud

Deploy this repository with `app.py` as the main file. Community Cloud installs the Linux-compatible runtime dependencies from `requirements.txt`; the Yahoo Finance cache is stored in the worker's temporary directory rather than the source checkout. The app requires Python 3.11 or newer and does not require secrets or environment variables.

## Stock Ranking

The **Stock Ranking** tab applies one strategy, parameter set, capital amount, and trading-cost model to every ticker in the selected universe. It uses fractional shares by default so nominal share price does not distort comparisons. A stock needs at least 90% data coverage, adequate strategy warm-up history, valid Open/Close data, and a completed backtest; stocks without a completed round-trip remain visible in the full table but are not ranked in Top 3 or Bottom 3.

Built-in universes are live lists of current constituents, stamped with the retrieval date. They can therefore have survivorship bias for historical periods. Rankings describe historical strategy performance, not forward-looking recommendations. Strategy Return is the selected strategy's net return; Excess Return is its net return minus the Buy & Hold return produced with the same accounting and costs.

Run the deterministic test suite from the repository root with:

```bash
python -m pytest -q
```

## CSV upload format

CSV files must have this header and daily rows. Dates must be parseable and Open/High/Low/Close/Volume must be numeric.

```csv
Date,Open,High,Low,Close,Volume
2024-01-02,185.64,188.44,183.89,185.64,82488700
2024-01-03,184.22,185.88,183.43,184.25,58414500
```

## Known limitations

Daily OHLCV fills are simplified and cannot reproduce intraday order priority, bid/ask spread, liquidity constraints, partial fills, taxes, dividends, or all corporate-action effects. Yahoo Finance availability and data quality can vary. Historical optimization may overfit; survivorship bias and data errors can materially affect conclusions.
