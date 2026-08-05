from __future__ import annotations

from io import BytesIO
import logging
from pathlib import Path
import tempfile
from typing import BinaryIO
import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFException
from peewee import OperationalError as PeeweeOperationalError

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
logger = logging.getLogger("backtesting_lab.data")
if not logger.handlers:
    _data_handler = logging.StreamHandler()
    _data_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_data_handler)
logger.setLevel(logging.INFO)
logger.propagate = False
# Streamlit Community Cloud's source checkout is not a suitable cache target. The system
# temporary directory is writable on Windows and Linux and is safe to discard between runs.
YFINANCE_CACHE_DIR = Path(tempfile.gettempdir()) / "modular_backtesting_lab" / "yfinance"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


class DataDownloadError(RuntimeError):
    """A Yahoo Finance request that could not produce valid market data."""


def _describe_frame(frame: pd.DataFrame | None) -> str:
    """Return safe diagnostics for downloads without emitting market data rows."""
    if frame is None:
        return "frame=None"
    return f"shape={frame.shape}, columns={list(frame.columns)}"


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize yfinance or CSV data to daily OHLCV columns."""
    logger.info("Normalizing OHLCV input: %s", _describe_frame(frame))
    if frame is None or frame.empty:
        raise ValueError("No historical data was returned.")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance can return (field, ticker) or (ticker, field).
        if set(REQUIRED_COLUMNS[1:]).intersection(data.columns.get_level_values(0)):
            data.columns = data.columns.get_level_values(0)
        elif set(REQUIRED_COLUMNS[1:]).intersection(data.columns.get_level_values(-1)):
            data.columns = data.columns.get_level_values(-1)
        else:
            raise ValueError("Could not recognize OHLCV columns in multi-index response.")
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.set_index("Date")
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[~data.index.isna()]
    missing = set(REQUIRED_COLUMNS[1:]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")
    data = data[REQUIRED_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data.dropna(subset=["Close"])
    logger.info("OHLCV rows remaining after normalization: %d; columns=%s", len(data), list(data.columns))
    if data.empty:
        raise ValueError("No rows with a valid Close price are available.")
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def download_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download valid daily data; only successful normalized results enter the cache."""
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Enter a ticker symbol.")
    start_date = pd.Timestamp(start).date().isoformat()
    # Yahoo treats the end parameter as exclusive, while the UI treats it as inclusive.
    yahoo_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    logger.info("Yahoo request ticker=%s start=%s end=%s (Yahoo exclusive end=%s)", symbol, start_date, end, yahoo_end)
    try:
        # Avoid the threaded yfinance path in Streamlit's long-running process.
        raw = yf.download(symbol, start=start_date, end=yahoo_end, interval="1d", auto_adjust=False,
                          progress=False, group_by="column", threads=False, multi_level_index=True)
    except (OSError, ValueError, YFException, PeeweeOperationalError) as exc:
        logger.exception("yfinance.download raised for %s", symbol)
        raise DataDownloadError(f"Yahoo Finance download for {symbol} failed: {exc}") from exc
    logger.info("Yahoo response ticker=%s %s", symbol, _describe_frame(raw))
    if raw is None or raw.empty:
        # download() records some Yahoo errors internally and returns an empty frame. History with
        # raise_errors=True exposes the underlying Yahoo exception instead of hiding it.
        logger.warning("yfinance.download returned no rows for %s; retrying with exception-enabled history().", symbol)
        try:
            raw = yf.Ticker(symbol).history(start=start_date, end=yahoo_end, interval="1d", auto_adjust=False,
                                             actions=False, raise_errors=True)
        except (OSError, ValueError, YFException, PeeweeOperationalError) as exc:
            logger.exception("yfinance history fallback raised for %s", symbol)
            raise DataDownloadError(
                f"Yahoo Finance returned no rows for {symbol} ({start_date} through {end}); "
                f"history() reported: {exc}"
            ) from exc
        logger.info("Yahoo history fallback ticker=%s %s", symbol, _describe_frame(raw))
    try:
        return normalize_ohlcv(raw)
    except ValueError as exc:
        logger.exception("Yahoo data validation failed for %s: %s", symbol, _describe_frame(raw))
        raise DataDownloadError(
            f"Yahoo Finance returned unusable data for {symbol} ({start_date} through {end}): {exc}. "
            f"Raw response: {_describe_frame(raw)}"
        ) from exc


def load_csv(uploaded: BinaryIO | BytesIO) -> pd.DataFrame:
    """Load a user CSV containing Date, Open, High, Low, Close, Volume columns."""
    return normalize_ohlcv(pd.read_csv(uploaded))
