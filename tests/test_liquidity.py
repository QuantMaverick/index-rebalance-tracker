"""Tests for liquidity diagnostics: Amihud, Corwin-Schultz, Kyle's lambda."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from index_rebalance_tracker.analysis.liquidity import (
    amihud_illiquidity,
    corwin_schultz_spread,
    daily_return_vol,
    kyle_lambda,
)

# ── helpers ─────────────────────────────────────────────────────────────────


def _ohlcv(
    n: int,
    *,
    base_price: float = 100.0,
    daily_vol: float = 0.02,
    daily_volume: int = 1_000_000,
    spread_pct: float = 0.001,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV with a known proportional spread.

    The intraday range is ``base × (1 + spread/2)`` for high and
    ``base × (1 − spread/2)`` for low, plus light random-walk drift on
    close. Volume is constant.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    log_returns = rng.normal(0, daily_vol, n)
    close = base_price * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + spread_pct / 2)
    low = close * (1.0 - spread_pct / 2)
    open_ = close * (1.0 + rng.normal(0, daily_vol / 4, n))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": [daily_volume] * n,
        },
        index=idx,
    )


# ── Amihud ──────────────────────────────────────────────────────────────────


def test_amihud_positive_for_normal_data() -> None:
    df = _ohlcv(80, daily_vol=0.02, daily_volume=1_000_000)
    ratio = amihud_illiquidity(df, window=60)
    assert ratio > 0
    assert ratio == ratio  # not NaN


def test_amihud_doubles_when_returns_double() -> None:
    """Amihud ratio is linear in |return| at fixed dollar volume."""
    df1 = _ohlcv(80, daily_vol=0.02, seed=1)
    df2 = df1.copy()
    # Stretch returns by 2x by stretching adj_close path around its first value
    base = df1["adj_close"].iloc[0]
    log_diff = np.log(df1["adj_close"] / base)
    df2["adj_close"] = base * np.exp(log_diff * 2)
    df2["close"] = df2["adj_close"]
    r1 = amihud_illiquidity(df1, window=60)
    r2 = amihud_illiquidity(df2, window=60)
    # Ratio of ratios should be ~2 within Monte-Carlo noise
    assert 1.5 < r2 / r1 < 2.5


def test_amihud_halves_when_volume_doubles() -> None:
    """Amihud ratio is inverse in dollar volume."""
    df1 = _ohlcv(80, daily_volume=1_000_000)
    df2 = df1.copy()
    df2["volume"] = df2["volume"] * 2
    r1 = amihud_illiquidity(df1, window=60)
    r2 = amihud_illiquidity(df2, window=60)
    assert r2 == pytest.approx(r1 / 2.0, rel=1e-6)


def test_amihud_returns_nan_on_thin_data() -> None:
    df = _ohlcv(3)
    assert math.isnan(amihud_illiquidity(df, window=60))


# ── Corwin-Schultz ─────────────────────────────────────────────────────────


def test_corwin_schultz_positive_for_normal_data() -> None:
    df = _ohlcv(80, spread_pct=0.005)
    s = corwin_schultz_spread(df, window=60)
    assert s > 0
    # Should be in a sensible range for a 50bps quoted spread
    assert s < 0.05


def test_corwin_schultz_increases_with_quoted_spread() -> None:
    """Wider quoted spread → wider estimated CS spread."""
    df_tight = _ohlcv(80, spread_pct=0.001, seed=2)
    df_wide = _ohlcv(80, spread_pct=0.02, seed=2)
    s_tight = corwin_schultz_spread(df_tight, window=60)
    s_wide = corwin_schultz_spread(df_wide, window=60)
    assert s_wide > s_tight


def test_corwin_schultz_returns_nan_when_no_high_low() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert math.isnan(corwin_schultz_spread(df))


def test_corwin_schultz_handles_short_series() -> None:
    df = _ohlcv(3)
    assert math.isnan(corwin_schultz_spread(df, window=60))


# ── Kyle's lambda ──────────────────────────────────────────────────────────


def test_kyle_lambda_positive_when_buys_push_price_up() -> None:
    """Construct a series where positive return ↔ high volume by design.
    Lambda must be positive."""
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)
    rets = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    # Volume strictly proportional to |return|, sign agrees with return →
    # tick-rule signed volume goes the same way as ΔP → lambda > 0
    vol = (np.abs(rets) * 1e9).astype(int) + 100
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "adj_close": close,
            "volume": vol,
        },
        index=idx,
    )
    lam = kyle_lambda(df)
    assert lam > 0


def test_kyle_lambda_returns_nan_on_thin_data() -> None:
    df = _ohlcv(3)
    assert math.isnan(kyle_lambda(df))


def test_kyle_lambda_window_caps_sample() -> None:
    """The window parameter controls how many trailing rows enter the regression."""
    df = _ohlcv(200, seed=4)
    lam_full = kyle_lambda(df)
    lam_60 = kyle_lambda(df, window=60)
    # Both should be finite; the windowed version reflects only recent flow
    assert not math.isnan(lam_full)
    assert not math.isnan(lam_60)


# ── daily_return_vol ───────────────────────────────────────────────────────


def test_daily_return_vol_recovers_input_vol() -> None:
    """At n=200, sample std should be within 15% of the DGP daily vol."""
    df = _ohlcv(200, daily_vol=0.02, seed=5)
    vol = daily_return_vol(df, window=60)
    assert 0.015 < vol < 0.025


def test_daily_return_vol_returns_nan_on_thin_data() -> None:
    df = _ohlcv(5)
    assert math.isnan(daily_return_vol(df))
