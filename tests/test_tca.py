"""Tests for TCA: estimated demand and square-root impact shortfall."""

from __future__ import annotations

import math

import pytest

from index_rebalance_tracker.analysis.tca import (
    estimated_demand_usd,
    implementation_shortfall_summary,
    square_root_impact_bps,
)

# ── estimated_demand_usd ───────────────────────────────────────────────────


def test_estimated_demand_simple_proportion() -> None:
    """1% weight in $1T AUM → $10B demand."""
    demand = estimated_demand_usd(1_000_000_000_000, 1_000_000_000, 100_000_000_000)
    assert demand == pytest.approx(10_000_000_000)


def test_estimated_demand_zero_weight() -> None:
    """Zero stock cap → zero demand (the stock contributes nothing to the index)."""
    demand = estimated_demand_usd(1e12, 0.0, 1e13)
    assert demand == 0.0


def test_estimated_demand_zero_index_returns_nan() -> None:
    assert math.isnan(estimated_demand_usd(1e12, 1e9, 0.0))


# ── square_root_impact_bps ─────────────────────────────────────────────────


def test_square_root_impact_quarter_adv_at_2pct_vol() -> None:
    """For Q/ADV = 0.25, σ=0.02, k=1 → cost = 1 × 0.02 × 0.5 × 10000 = 100 bps."""
    bps = square_root_impact_bps(2.5e8, 1e9, 0.02, n_days=1)
    assert bps == pytest.approx(100.0)


def test_square_root_impact_scales_as_sqrt_q() -> None:
    """Doubling demand should multiply cost by √2."""
    base = square_root_impact_bps(1e8, 1e9, 0.02, n_days=1)
    doubled = square_root_impact_bps(2e8, 1e9, 0.02, n_days=1)
    assert doubled / base == pytest.approx(math.sqrt(2.0), rel=1e-9)


def test_square_root_impact_scales_inversely_with_sqrt_n() -> None:
    """Spreading over 4 days halves the cost."""
    forced = square_root_impact_bps(1e8, 1e9, 0.02, n_days=1)
    spread4 = square_root_impact_bps(1e8, 1e9, 0.02, n_days=4)
    assert spread4 == pytest.approx(forced / 2.0, rel=1e-9)


def test_square_root_impact_scales_with_vol() -> None:
    """Doubling daily vol should double the cost."""
    base = square_root_impact_bps(1e8, 1e9, 0.02, n_days=1)
    high_vol = square_root_impact_bps(1e8, 1e9, 0.04, n_days=1)
    assert high_vol == pytest.approx(2 * base, rel=1e-9)


def test_square_root_impact_returns_nan_on_degenerate_inputs() -> None:
    assert math.isnan(square_root_impact_bps(0, 1e9, 0.02))
    assert math.isnan(square_root_impact_bps(1e8, 0, 0.02))
    assert math.isnan(square_root_impact_bps(1e8, 1e9, 0))


# ── implementation_shortfall_summary ───────────────────────────────────────


def test_shortfall_savings_equal_forced_minus_spread() -> None:
    out = implementation_shortfall_summary(1e8, 1e9, 0.02, spread_n_days=5)
    assert out["savings_bps"] == pytest.approx(out["forced_bps"] - out["spread_bps"])


def test_shortfall_savings_positive_when_n_gt_1() -> None:
    """Spreading always reduces impact when N > 1, so savings_bps > 0."""
    out = implementation_shortfall_summary(1e8, 1e9, 0.02, spread_n_days=5)
    assert out["savings_bps"] > 0


def test_shortfall_savings_zero_when_n_equals_1() -> None:
    """If we 'spread' over 1 day, savings = 0."""
    out = implementation_shortfall_summary(1e8, 1e9, 0.02, spread_n_days=1)
    assert out["savings_bps"] == pytest.approx(0.0)


def test_shortfall_returns_nan_on_degenerate_inputs() -> None:
    out = implementation_shortfall_summary(0, 1e9, 0.02)
    assert math.isnan(out["forced_bps"])
    assert math.isnan(out["spread_bps"])
    assert math.isnan(out["savings_bps"])


def test_shortfall_realistic_passive_fund_scenario() -> None:
    """Sanity check: $10B demand, $5B ADV, 2% vol, 5-day window.
    Cost should be in plausible 100-300 bps range for forced execution."""
    out = implementation_shortfall_summary(10e9, 5e9, 0.02, spread_n_days=5)
    assert 100 < out["forced_bps"] < 500
    assert out["spread_bps"] < out["forced_bps"]
