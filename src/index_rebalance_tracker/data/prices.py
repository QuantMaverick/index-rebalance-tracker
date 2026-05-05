"""yfinance wrapper with disk cache, retry, and rate-limit etiquette.

Cache layout::

    ~/.cache/index-rebalance-tracker/prices/{TICKER}.parquet

One file per ticker, columns ``[date, open, high, low, close, adj_close,
volume]``. On hit, we read the cached file and slice; on miss or partial
miss we fetch the missing range and merge.

The wrapper is the only network surface for prices. Tests inject a
``YFinanceFetcher`` protocol implementation to avoid hitting yfinance.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "index-rebalance-tracker" / "prices"
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class YFinanceFetcher(Protocol):
    """Protocol for the yfinance-shaped callable, for dependency injection in tests."""

    def __call__(self, ticker: str, start: date, end: date) -> pd.DataFrame: ...


def _default_fetcher(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Default implementation hitting yfinance.

    Imported lazily so unit tests don't pay the import cost.
    """
    import yfinance as yf  # noqa: PLC0415

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    # yfinance can return a MultiIndex when multiple tickers are passed; flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index = pd.Index(pd.to_datetime(df.index).tz_localize(None).date, name="date")
    typed: pd.DataFrame = df[PRICE_COLUMNS].astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "adj_close": "float64",
            "volume": "int64",
        }
    )
    return typed


class PriceCache:
    """Thin disk cache around a price-fetcher.

    The cache is always-true-on-hit: once a date is written, we trust it. To
    invalidate, delete the parquet file. Splits and dividends arrive via
    yfinance's ``adj_close`` already; raw OHLC is preserved separately.

    Args:
        root: Cache root directory. Defaults to
            ``~/.cache/index-rebalance-tracker/prices``.
        fetcher: Injected fetcher callable conforming to ``YFinanceFetcher``.
            If ``None``, the default yfinance fetcher is used.
        rate_limit_seconds: Sleep between live fetches to avoid throttling.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        fetcher: YFinanceFetcher | None = None,
        rate_limit_seconds: float = 0.5,
    ) -> None:
        self.root = root or DEFAULT_CACHE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self._fetcher: Callable[[str, date, date], pd.DataFrame] = fetcher or _default_fetcher
        self.rate_limit_seconds = rate_limit_seconds

    # ── public API ─────────────────────────────────────────────────────────

    def fetch_prices(
        self,
        ticker: str,
        start: date,
        end: date,
        *,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Return OHLCV bars for ``ticker`` from ``start`` to ``end`` inclusive.

        Args:
            ticker: Trading symbol.
            start: First date to include.
            end: Last date to include.
            max_retries: Network retries with exponential backoff (1, 2, 4 s).

        Returns:
            DataFrame indexed by ``date`` with columns ``[open, high, low,
            close, adj_close, volume]``. Empty DataFrame if no data.

        Example:
            >>> from datetime import date
            >>> cache = PriceCache(fetcher=lambda t, s, e: ...)  # doctest: +SKIP
            >>> df = cache.fetch_prices("AAPL", date(2020, 1, 1), date(2020, 12, 31))  # doctest: +SKIP
        """
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        cached = self._read_cache(ticker)
        missing_ranges = _missing_ranges(cached.index.tolist(), start, end)
        if not missing_ranges:
            return _slice(cached, start, end)

        new_frames: list[pd.DataFrame] = []
        for ms, me in missing_ranges:
            new = self._fetch_with_retry(ticker, ms, me, max_retries=max_retries)
            if not new.empty:
                new_frames.append(new)

        if new_frames:
            merged = pd.concat([cached, *new_frames])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self._write_cache(ticker, merged)
            cached = merged
        return _slice(cached, start, end)

    # ── internals ──────────────────────────────────────────────────────────

    def _fetch_with_retry(
        self, ticker: str, start: date, end: date, *, max_retries: int
    ) -> pd.DataFrame:
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                time.sleep(self.rate_limit_seconds)
                df = self._fetcher(ticker, start, end)
                logger.info("fetched %s [%s, %s]: %d rows", ticker, start, end, len(df))
                return df
            except Exception as exc:  # noqa: BLE001 - yfinance raises diverse types
                last_exc = exc
                logger.warning(
                    "fetch %s attempt %d/%d failed: %s", ticker, attempt + 1, max_retries, exc
                )
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"failed to fetch {ticker} after {max_retries} attempts") from last_exc

    def _path(self, ticker: str) -> Path:
        return self.root / f"{ticker.upper()}.parquet"

    def _read_cache(self, ticker: str) -> pd.DataFrame:
        path = self._path(ticker)
        if not path.exists():
            return pd.DataFrame(columns=PRICE_COLUMNS).rename_axis("date")
        df = pd.read_parquet(path)
        df.index = pd.Index(pd.to_datetime(df.index).date, name="date")
        return df

    def _write_cache(self, ticker: str, df: pd.DataFrame) -> None:
        path = self._path(ticker)
        out = df.copy()
        out.index = pd.to_datetime(out.index)  # parquet wants datetime
        out.to_parquet(path)


# ── Helpers (pure, testable) ────────────────────────────────────────────────


def _slice(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask]


def _missing_ranges(
    cached_dates: list[date], want_start: date, want_end: date
) -> list[tuple[date, date]]:
    """Compute the (start, end) pairs of date ranges not yet cached.

    Conservative: even if cache covers most of the want-range, a single gap
    triggers a fetch of the full uncovered tail. We don't try to surgical-
    fetch holes mid-range — yfinance is fast enough that simplicity wins.

    Args:
        cached_dates: Sorted list of dates already on disk for this ticker.
        want_start: Requested start (inclusive).
        want_end: Requested end (inclusive).

    Returns:
        List of (start, end) ranges to fetch. Empty if cache already covers.
    """
    if not cached_dates:
        return [(want_start, want_end)]
    cached_set = set(cached_dates)
    cached_min = min(cached_set)
    cached_max = max(cached_set)
    ranges: list[tuple[date, date]] = []
    if want_start < cached_min:
        ranges.append((want_start, min(cached_min - timedelta(days=1), want_end)))
    if want_end > cached_max:
        ranges.append((max(cached_max + timedelta(days=1), want_start), want_end))
    return ranges


def now_iso() -> str:
    """UTC timestamp in ISO-8601 (used for cache stamping)."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
