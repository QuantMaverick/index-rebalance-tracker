"""Tests for the PriceCache wrapper and missing-range computation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from index_rebalance_tracker.data.prices import (
    PRICE_COLUMNS,
    PriceCache,
    _missing_ranges,
    _slice,
)


def _make_df(dates: list[date]) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame indexed by ``date``."""
    n = len(dates)
    df = pd.DataFrame(
        {
            "open": list(range(n)),
            "high": list(range(n)),
            "low": list(range(n)),
            "close": list(range(n)),
            "adj_close": list(range(n)),
            "volume": [1_000_000] * n,
        },
        index=pd.Index(dates, name="date"),
    )
    return df.astype({"volume": "int64"})


# ── _missing_ranges ─────────────────────────────────────────────────────────


def test_missing_ranges_empty_cache() -> None:
    out = _missing_ranges([], date(2020, 1, 1), date(2020, 1, 5))
    assert out == [(date(2020, 1, 1), date(2020, 1, 5))]


def test_missing_ranges_full_coverage() -> None:
    cached = [date(2020, 1, d) for d in range(1, 11)]
    out = _missing_ranges(cached, date(2020, 1, 2), date(2020, 1, 9))
    assert out == []


def test_missing_ranges_extend_right() -> None:
    cached = [date(2020, 1, d) for d in range(1, 6)]
    out = _missing_ranges(cached, date(2020, 1, 1), date(2020, 1, 10))
    assert out == [(date(2020, 1, 6), date(2020, 1, 10))]


def test_missing_ranges_extend_left() -> None:
    cached = [date(2020, 1, d) for d in range(5, 11)]
    out = _missing_ranges(cached, date(2020, 1, 1), date(2020, 1, 10))
    assert out == [(date(2020, 1, 1), date(2020, 1, 4))]


def test_missing_ranges_extend_both_sides() -> None:
    cached = [date(2020, 1, d) for d in range(5, 8)]
    out = _missing_ranges(cached, date(2020, 1, 1), date(2020, 1, 10))
    assert out == [(date(2020, 1, 1), date(2020, 1, 4)), (date(2020, 1, 8), date(2020, 1, 10))]


# ── _slice ──────────────────────────────────────────────────────────────────


def test_slice_inclusive() -> None:
    df = _make_df([date(2020, 1, d) for d in range(1, 11)])
    sliced = _slice(df, date(2020, 1, 3), date(2020, 1, 5))
    assert list(sliced.index) == [date(2020, 1, 3), date(2020, 1, 4), date(2020, 1, 5)]


def test_slice_empty_df() -> None:
    df = pd.DataFrame(columns=PRICE_COLUMNS)
    sliced = _slice(df, date(2020, 1, 1), date(2020, 1, 5))
    assert sliced.empty


# ── PriceCache (with injected fetcher — no network) ─────────────────────────


def test_pricecache_miss_then_hit(tmp_path: Path) -> None:
    """First call fetches; second call hits cache (fetcher not called again)."""
    fetcher_calls: list[tuple[str, date, date]] = []

    def fake_fetcher(ticker: str, start: date, end: date) -> pd.DataFrame:
        fetcher_calls.append((ticker, start, end))
        return _make_df([date(2020, 1, d) for d in range(1, 6)])

    cache = PriceCache(root=tmp_path, fetcher=fake_fetcher, rate_limit_seconds=0)

    df1 = cache.fetch_prices("AAPL", date(2020, 1, 1), date(2020, 1, 5))
    assert len(df1) == 5
    assert len(fetcher_calls) == 1

    df2 = cache.fetch_prices("AAPL", date(2020, 1, 2), date(2020, 1, 4))
    assert len(df2) == 3
    assert len(fetcher_calls) == 1, "second call should be served from cache"


def test_pricecache_partial_extends(tmp_path: Path) -> None:
    """Initial fetch then a wider range; only the gap should be re-fetched."""
    fetcher_calls: list[tuple[date, date]] = []

    def fake_fetcher(ticker: str, start: date, end: date) -> pd.DataFrame:
        fetcher_calls.append((start, end))
        if (start, end) == (date(2020, 1, 1), date(2020, 1, 5)):
            return _make_df([date(2020, 1, d) for d in range(1, 6)])
        return _make_df([date(2020, 1, d) for d in range(6, 11)])

    cache = PriceCache(root=tmp_path, fetcher=fake_fetcher, rate_limit_seconds=0)
    cache.fetch_prices("AAPL", date(2020, 1, 1), date(2020, 1, 5))
    cache.fetch_prices("AAPL", date(2020, 1, 1), date(2020, 1, 10))
    assert fetcher_calls == [
        (date(2020, 1, 1), date(2020, 1, 5)),
        (date(2020, 1, 6), date(2020, 1, 10)),
    ]


def test_pricecache_rejects_inverted_range(tmp_path: Path) -> None:
    cache = PriceCache(root=tmp_path, fetcher=lambda *_: pd.DataFrame(), rate_limit_seconds=0)
    with pytest.raises(ValueError, match=r"start.*<=.*end"):
        cache.fetch_prices("AAPL", date(2020, 1, 5), date(2020, 1, 1))


def test_pricecache_retry_on_failure(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def flaky(ticker: str, start: date, end: date) -> pd.DataFrame:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("simulated network blip")
        return _make_df([date(2020, 1, 1)])

    cache = PriceCache(root=tmp_path, fetcher=flaky, rate_limit_seconds=0)
    df = cache.fetch_prices("AAPL", date(2020, 1, 1), date(2020, 1, 1), max_retries=3)
    assert len(df) == 1
    assert attempts["n"] == 3


def test_pricecache_persists_across_instances(tmp_path: Path) -> None:
    """Round-trip via parquet: one cache writes, a second reads."""

    def fake(ticker: str, start: date, end: date) -> pd.DataFrame:
        return _make_df([date(2020, 1, d) for d in range(1, 4)])

    cache1 = PriceCache(root=tmp_path, fetcher=fake, rate_limit_seconds=0)
    cache1.fetch_prices("AAPL", date(2020, 1, 1), date(2020, 1, 3))

    calls = {"n": 0}

    def watch(ticker: str, start: date, end: date) -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame()

    cache2 = PriceCache(root=tmp_path, fetcher=watch, rate_limit_seconds=0)
    df = cache2.fetch_prices("AAPL", date(2020, 1, 1), date(2020, 1, 3))
    assert len(df) == 3
    assert calls["n"] == 0
