"""Tests for sector + size matching."""

from __future__ import annotations

import pandas as pd
import pytest

from index_rebalance_tracker.analysis.sector_match import (
    adv_dollars_60d,
    find_matched_controls,
)

# ── find_matched_controls ──────────────────────────────────────────────────


@pytest.fixture
def universe_5() -> tuple[list[str], dict[str, str], dict[str, float]]:
    universe = ["AAPL", "MSFT", "NVDA", "JPM", "BAC"]
    sector = {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "NVDA": "Information Technology",
        "JPM": "Financials",
        "BAC": "Financials",
    }
    adv = {"AAPL": 8e9, "MSFT": 7e9, "NVDA": 6e9, "JPM": 1e9, "BAC": 5e8}
    return universe, sector, adv


def test_match_picks_same_sector_only(
    universe_5: tuple[list[str], dict[str, str], dict[str, float]],
) -> None:
    universe, sector, adv = universe_5
    out = find_matched_controls(
        "AAPL", "Information Technology", universe, sector, adv, n_matches=4
    )
    # Only 2 same-sector peers exist (MSFT, NVDA); JPM/BAC are Financials
    assert out == ["MSFT", "NVDA"]


def test_match_orders_by_log_adv_proximity(
    universe_5: tuple[list[str], dict[str, str], dict[str, float]],
) -> None:
    universe, sector, adv = universe_5
    out = find_matched_controls(
        "AAPL", "Information Technology", universe, sector, adv, n_matches=2
    )
    # MSFT (7e9) closer to AAPL (8e9) than NVDA (6e9) in log-space
    assert out == ["MSFT", "NVDA"]


def test_match_excludes_event_ticker(
    universe_5: tuple[list[str], dict[str, str], dict[str, float]],
) -> None:
    universe, sector, adv = universe_5
    out = find_matched_controls(
        "MSFT", "Information Technology", universe, sector, adv, n_matches=5
    )
    assert "MSFT" not in out


def test_match_n_matches_caps_results(
    universe_5: tuple[list[str], dict[str, str], dict[str, float]],
) -> None:
    universe, sector, adv = universe_5
    out = find_matched_controls(
        "AAPL", "Information Technology", universe, sector, adv, n_matches=1
    )
    assert len(out) == 1
    assert out == ["MSFT"]  # closest match


def test_match_skips_non_positive_adv(
    universe_5: tuple[list[str], dict[str, str], dict[str, float]],
) -> None:
    universe, sector, adv = universe_5
    adv_with_zero = {**adv, "MSFT": 0.0}
    out = find_matched_controls(
        "AAPL", "Information Technology", universe, sector, adv_with_zero, n_matches=5
    )
    assert "MSFT" not in out
    assert out == ["NVDA"]


def test_match_raises_when_event_missing_adv() -> None:
    with pytest.raises(KeyError, match=r"adv_dollars missing"):
        find_matched_controls("AAPL", "Information Technology", ["AAPL"], {}, {}, n_matches=5)


def test_match_raises_when_event_adv_non_positive() -> None:
    with pytest.raises(ValueError, match=r"non-positive ADV"):
        find_matched_controls(
            "AAPL",
            "Information Technology",
            ["AAPL"],
            {"AAPL": "Information Technology"},
            {"AAPL": 0.0},
        )


def test_match_returns_empty_when_no_same_sector_peers() -> None:
    universe = ["AAPL", "JPM"]
    sector = {"AAPL": "Information Technology", "JPM": "Financials"}
    adv = {"AAPL": 1e9, "JPM": 5e8}
    out = find_matched_controls(
        "AAPL", "Information Technology", universe, sector, adv, n_matches=5
    )
    assert out == []


# ── adv_dollars_60d ────────────────────────────────────────────────────────


def test_adv_dollars_60d_handles_short_series() -> None:
    """A ticker with fewer than 60 rows still gets a value (mean of available)."""
    df = pd.DataFrame(
        {"close": [100.0, 101.0, 99.0], "volume": [1_000_000, 1_500_000, 800_000]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    out = adv_dollars_60d({"AAPL": df})
    expected = (1_000_000 * 100 + 1_500_000 * 101 + 800_000 * 99) / 3
    assert out["AAPL"] == pytest.approx(expected)


def test_adv_dollars_60d_uses_trailing_window() -> None:
    """For 100 rows of constant 1M volume × $100 price, ADV = $100M."""
    n = 100
    df = pd.DataFrame(
        {"close": [100.0] * n, "volume": [1_000_000] * n},
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )
    out = adv_dollars_60d({"AAPL": df}, window=60)
    assert out["AAPL"] == pytest.approx(100_000_000)


def test_adv_dollars_60d_skips_empty_or_zero() -> None:
    empty = pd.DataFrame(columns=["close", "volume"])
    zero = pd.DataFrame({"close": [100.0], "volume": [0]}, index=pd.to_datetime(["2024-01-01"]))
    out = adv_dollars_60d({"EMPTY": empty, "ZERO": zero, "OK": _make_ok_df()})
    assert "EMPTY" not in out
    assert "ZERO" not in out
    assert "OK" in out


def _make_ok_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [100.0, 101.0], "volume": [1_000_000, 2_000_000]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
