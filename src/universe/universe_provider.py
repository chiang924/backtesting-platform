from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import BinaryIO
import logging
import pandas as pd
import requests

from .universe_validator import deduplicate_tickers, parse_custom_tickers, validate_universe_csv

logger = logging.getLogger("backtesting_lab.universe")

BUILTIN_UNIVERSES = {
    "S&P 100 (current constituents)": "https://en.wikipedia.org/wiki/S%26P_100",
    # The index overview page no longer contains its constituent table.  This
    # dedicated list page does, and is updated as constituents change.
    "Nasdaq-100 (current constituents)": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
}


@dataclass(frozen=True)
class UniverseDefinition:
    name: str
    snapshot_date: str
    tickers: list[str]
    current_constituents: bool = False


def _find_ticker_column(table: pd.DataFrame) -> object | None:
    for column in table.columns:
        name = str(column).strip().lower()
        if name in {"symbol", "ticker", "ticker symbol"}:
            # Preserve the native label: pandas can expose tuple labels for
            # multi-level HTML table headers.
            return column
    return None


def _fetch_current_constituents(name: str) -> UniverseDefinition:
    """Fetch the published current-constituent table for a built-in universe."""
    url = BUILTIN_UNIVERSES[name]
    response = requests.get(url, headers={"User-Agent": "ModularBacktestingLab/1.0"}, timeout=20)
    response.raise_for_status()
    for table in pd.read_html(BytesIO(response.content)):
        column = _find_ticker_column(table)
        if column:
            tickers = deduplicate_tickers(table[column].dropna().tolist())
            if tickers:
                logger.info("Fetched %s current constituents: %d tickers", name, len(tickers))
                return UniverseDefinition(name, date.today().isoformat(), tickers, current_constituents=True)
    raise ValueError(f"Could not find a ticker table for {name}.")


def get_universe(name: str, custom_tickers: str = "", uploaded: BinaryIO | BytesIO | None = None) -> UniverseDefinition:
    """Return a validated universe definition independently of the Streamlit UI."""
    if name in BUILTIN_UNIVERSES:
        return _fetch_current_constituents(name)
    if name == "Custom ticker list":
        tickers = parse_custom_tickers(custom_tickers)
        if not tickers:
            raise ValueError("Enter at least one comma-separated ticker.")
        return UniverseDefinition(name, date.today().isoformat(), tickers)
    if name == "Uploaded universe CSV":
        if uploaded is None:
            raise ValueError("Upload a universe CSV with a Ticker column.")
        return UniverseDefinition(name, date.today().isoformat(), validate_universe_csv(uploaded))
    raise ValueError(f"Unsupported universe: {name}")
