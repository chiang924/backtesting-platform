from __future__ import annotations

from io import BytesIO
from typing import BinaryIO
import pandas as pd


def normalize_ticker(value: object) -> str:
    """Normalize a user ticker to Yahoo Finance's conventional symbol form."""
    return str(value).strip().upper().replace(".", "-")


def deduplicate_tickers(values: list[object]) -> list[str]:
    """Strip, normalize, and preserve first occurrence order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        ticker = normalize_ticker(value)
        if ticker and ticker not in seen:
            seen.add(ticker)
            cleaned.append(ticker)
    return cleaned


def parse_custom_tickers(value: str) -> list[str]:
    """Parse comma-separated ticker input without accepting empty values."""
    return deduplicate_tickers(value.split(","))


def validate_universe_csv(uploaded: BinaryIO | BytesIO) -> list[str]:
    """Read an uploaded universe CSV that must expose a Ticker column."""
    frame = pd.read_csv(uploaded)
    ticker_column = next((column for column in frame.columns if str(column).strip().lower() == "ticker"), None)
    if ticker_column is None:
        raise ValueError("Universe CSV must include a Ticker column.")
    tickers = deduplicate_tickers(frame[ticker_column].dropna().tolist())
    if not tickers:
        raise ValueError("Universe CSV contains no usable ticker values.")
    return tickers
