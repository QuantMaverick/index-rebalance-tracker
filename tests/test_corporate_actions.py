"""Tests for split / dividend / total-return helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from index_rebalance_tracker.data.corporate_actions import apply_split, total_return


def _ohlcv(dates: list[str], closes: list[float], volumes: list[int]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )


def test_apple_2014_seven_for_one_split() -> None:
    """AAPL did a 7-for-1 split on 2014-06-09. Pre-split price 700 → 100,
    pre-split volume 1M → 7M-equivalent."""
    df = _ohlcv(
        ["2014-06-06", "2014-06-09"],
        closes=[700.0, 100.0],
        volumes=[1_000_000, 7_000_000],
    )
    adj = apply_split(df, pd.Timestamp("2014-06-09"), ratio=7.0)
    assert adj.loc["2014-06-06", "close"] == pytest.approx(100.0)
    assert int(adj.loc["2014-06-06", "volume"]) == 7_000_000
    # post-split row untouched
    assert adj.loc["2014-06-09", "close"] == pytest.approx(100.0)
    assert int(adj.loc["2014-06-09", "volume"]) == 7_000_000


def test_apply_split_does_not_mutate_input() -> None:
    df = _ohlcv(["2020-01-01", "2020-01-02"], [100.0, 50.0], [1_000_000, 2_000_000])
    snapshot = df.copy()
    _ = apply_split(df, pd.Timestamp("2020-01-02"), ratio=2.0)
    pd.testing.assert_frame_equal(df, snapshot)


def test_apply_split_rejects_non_positive_ratio() -> None:
    df = _ohlcv(["2020-01-01"], [100.0], [1_000_000])
    with pytest.raises(ValueError, match=r"ratio must be > 0"):
        apply_split(df, pd.Timestamp("2020-01-01"), ratio=0.0)


def test_total_return_first_value_is_nan() -> None:
    df = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 99.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    r = total_return(df)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.01)
    assert r.iloc[2] == pytest.approx(99.0 / 101.0 - 1)


def test_total_return_requires_adj_close() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    with pytest.raises(KeyError):
        total_return(df)
