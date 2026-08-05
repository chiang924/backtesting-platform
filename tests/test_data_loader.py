import pandas as pd
import pytest

from src.data import data_loader


def test_empty_yahoo_response_is_not_cached_and_retry_uses_inclusive_dates(monkeypatch):
    """Regression: Streamlit must retry after an empty server-process Yahoo response."""
    calls: list[tuple[str, str, str, bool]] = []
    history_attempts = 0
    index = pd.date_range("2018-01-01", periods=2, freq="B")
    raw = pd.DataFrame(
        [[10, 11], [11, 12]], index=index,
        columns=pd.MultiIndex.from_tuples([("Open", "AAPL"), ("Close", "AAPL")]),
    )
    raw[("High", "AAPL")] = [11, 12]
    raw[("Low", "AAPL")] = [9, 10]
    raw[("Volume", "AAPL")] = [100, 100]

    def fake_download(symbol, start, end, threads, **_kwargs):
        calls.append((symbol, start, end, threads))
        return pd.DataFrame()

    class FakeTicker:
        def __init__(self, symbol):
            assert symbol == "AAPL"

        def history(self, **_kwargs):
            nonlocal history_attempts
            history_attempts += 1
            if history_attempts == 1:
                raise OSError("temporary Yahoo failure")
            return raw

    monkeypatch.setattr(data_loader.yf, "download", fake_download)
    monkeypatch.setattr(data_loader.yf, "Ticker", FakeTicker)
    data_loader.download_history.clear()
    with pytest.raises(data_loader.DataDownloadError, match="temporary Yahoo failure"):
        data_loader.download_history(" aapl ", "2018-01-01", "2025-12-31")
    received = data_loader.download_history(" aapl ", "2018-01-01", "2025-12-31")
    data_loader.download_history.clear()

    assert len(calls) == 2  # The failed empty response was not cached.
    assert calls[0] == ("AAPL", "2018-01-01", "2026-01-01", False)
    assert list(received.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(received) == 2
