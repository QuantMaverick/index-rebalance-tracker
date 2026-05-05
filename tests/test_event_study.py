"""Unit tests for the event-study core: market-model AR, sector-matched AR,
CAR aggregation, and the top-level driver.

Strategy: build synthetic returns with known α and β, then verify the
estimator recovers them and that AR ≈ 0 within Monte-Carlo noise on
non-event days. Hand-computed CAR examples for window aggregation.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from index_rebalance_tracker.analysis.event_study import (
    cumulative_abnormal_return,
    estimate_market_model,
    market_model_abnormal_returns,
    run_event_study,
    sector_matched_abnormal_returns,
    standard_windows,
)
from index_rebalance_tracker.models import IndexEvent

# ── synthetic generators ────────────────────────────────────────────────────


def _synthetic_returns(
    n: int,
    *,
    alpha: float,
    beta: float,
    market_vol: float = 0.01,
    idio_vol: float = 0.005,
    seed: int = 0,
    start: str = "2020-01-01",
) -> tuple[pd.Series, pd.Series]:
    """Generate (event_returns, market_returns) under a known DGP."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B")
    r_m = pd.Series(rng.normal(0, market_vol, n), index=idx, name="r_m")
    r_i = pd.Series(
        alpha + beta * r_m.to_numpy() + rng.normal(0, idio_vol, n),
        index=idx,
        name="r_i",
    )
    return r_i, r_m


# ── estimate_market_model ──────────────────────────────────────────────────


def test_market_model_recovers_known_alpha_beta() -> None:
    n = 400
    r_i, r_m = _synthetic_returns(n, alpha=0.0003, beta=1.2, idio_vol=0.001)
    ann_date = r_i.index[-1].date()
    alpha, beta, n_obs = estimate_market_model(r_i, r_m, ann_date)
    assert alpha == pytest.approx(0.0003, abs=5e-4)
    assert beta == pytest.approx(1.2, abs=0.05)
    assert n_obs == 250


def test_market_model_respects_estimation_gap() -> None:
    """The 30-day gap window must NOT be in the regression sample."""
    n = 400
    r_i, r_m = _synthetic_returns(n, alpha=0.0, beta=1.0, idio_vol=0.001)
    ann_date = r_i.index[-1].date()
    _, _, n_obs = estimate_market_model(
        r_i, r_m, ann_date, estimation_window_days=250, estimation_gap_days=30
    )
    assert n_obs == 250  # 250 prior to the 30-day gap


def test_market_model_respects_estimation_window_length() -> None:
    n = 200
    r_i, r_m = _synthetic_returns(n, alpha=0.0, beta=1.0, idio_vol=0.001)
    ann_date = r_i.index[-1].date()
    _, _, n_obs = estimate_market_model(
        r_i, r_m, ann_date, estimation_window_days=100, estimation_gap_days=30
    )
    assert n_obs == 100


def test_market_model_raises_when_too_few_observations() -> None:
    n = 50
    r_i, r_m = _synthetic_returns(n, alpha=0.0, beta=1.0, idio_vol=0.001)
    ann_date = r_i.index[-1].date()
    with pytest.raises(ValueError, match=r"insufficient estimation data"):
        estimate_market_model(
            r_i, r_m, ann_date, estimation_window_days=250, estimation_gap_days=30
        )


# ── market_model_abnormal_returns ──────────────────────────────────────────


def test_ar_zero_under_correctly_specified_model() -> None:
    """If returns follow R = α + β·R_m + ε exactly with no shock, mean AR = 0."""
    n = 400
    r_i, r_m = _synthetic_returns(n, alpha=0.0002, beta=1.1, idio_vol=0.001, seed=42)
    ann_date = r_i.index[-1].date()
    ar = market_model_abnormal_returns(r_i, r_m, ann_date)
    # Mean AR over the post-estimation window should be close to zero
    post = ar.loc[ar.index > pd.Timestamp(ann_date) - pd.Timedelta(days=60)]
    assert abs(post.mean()) < 0.001


def test_ar_picks_up_shock_on_event_day() -> None:
    """Inject a +5% shock on event day; AR for that day should be ≈ +5%."""
    n = 400
    r_i, r_m = _synthetic_returns(n, alpha=0.0, beta=1.0, idio_vol=0.0001, seed=7)
    event_idx = r_i.index[-15]
    r_i.loc[event_idx] = r_i.loc[event_idx] + 0.05
    ann_date = r_i.index[-1].date()
    ar = market_model_abnormal_returns(r_i, r_m, ann_date)
    assert ar.loc[event_idx] == pytest.approx(0.05, abs=0.005)


# ── sector_matched_abnormal_returns ────────────────────────────────────────


def test_sector_matched_ar_is_event_minus_median() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    event = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=idx, name="event")
    controls = pd.DataFrame(
        {
            "C1": [0.005, -0.01, 0.02, 0.001, 0.005],
            "C2": [0.0, -0.015, 0.025, 0.002, 0.003],
            "C3": [0.002, -0.012, 0.022, 0.0, 0.004],
        },
        index=idx,
    )
    ar = sector_matched_abnormal_returns(event, controls)
    median = controls.median(axis=1)
    expected = event - median
    pd.testing.assert_series_equal(ar.rename("event"), expected, check_names=False)


def test_sector_matched_ar_rejects_zero_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    event = pd.Series([0.01, 0.02, 0.03], index=idx)
    controls = pd.DataFrame(index=idx)
    with pytest.raises(ValueError, match=r"at least one column"):
        sector_matched_abnormal_returns(event, controls)


# ── cumulative_abnormal_return ─────────────────────────────────────────────


def test_car_simple_sum() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    ar = pd.Series([0.01, 0.02, -0.005, 0.01, -0.015], index=idx)
    car, n = cumulative_abnormal_return(ar, idx[1].date(), idx[3].date())
    assert car == pytest.approx(0.025)  # 0.02 + -0.005 + 0.01
    assert n == 3


def test_car_drops_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    ar = pd.Series([0.01, np.nan, 0.02, np.nan, 0.03], index=idx)
    car, n = cumulative_abnormal_return(ar, idx[0].date(), idx[4].date())
    assert car == pytest.approx(0.06)
    assert n == 3


def test_car_empty_window_returns_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    ar = pd.Series([0.01, 0.02, 0.03], index=idx)
    car, n = cumulative_abnormal_return(ar, date(2030, 1, 1), date(2030, 12, 31))
    assert np.isnan(car)
    assert n == 0


# ── standard_windows ───────────────────────────────────────────────────────


def test_standard_windows_produces_four_labels() -> None:
    windows = standard_windows(
        announcement_date=date(2024, 6, 10),
        effective_date=date(2024, 6, 17),
        pre_run_up_days=5,
        post_drift_days=5,
    )
    labels = [w[0] for w in windows]
    assert len(labels) == 4
    assert "[T-A-5, T-A-1]" in labels
    assert "[T-A, T-E-1]" in labels
    assert "[T-E, T-E+5]" in labels


def test_standard_windows_dates_are_calendar_days() -> None:
    windows = standard_windows(
        date(2024, 6, 10), date(2024, 6, 17), pre_run_up_days=5, post_drift_days=5
    )
    pre = next(w for w in windows if w[0] == "[T-A-5, T-A-1]")
    assert pre[1] == date(2024, 6, 5)
    assert pre[2] == date(2024, 6, 9)


# ── run_event_study (end-to-end on synthetic) ──────────────────────────────


def test_run_event_study_emits_observations_for_both_models() -> None:
    """With one event and minimum data, expect: 4 windows × 2 models = 8 rows."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-01-01", periods=400, freq="B")
    r_m = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    universe = ["EVENT", "C1", "C2", "C3", "C4", "C5", "OTHER"]
    returns: dict[str, pd.Series] = {
        t: pd.Series(0.0001 + 1.0 * r_m + rng.normal(0, 0.002, 400), index=idx, name=t)
        for t in universe
    }
    sector_map = dict.fromkeys(universe[:6], "Information Technology")
    sector_map["OTHER"] = "Financials"
    adv_map = dict.fromkeys(universe, 1000000000.0)

    event = IndexEvent(
        event_id="test-add-EVENT",
        index="sp500",
        ticker="EVENT",
        action="add",
        effective_date=idx[-15].date(),
        announcement_date=idx[-20].date(),
        reason="test",
        source_url="test",
    )

    obs = run_event_study(
        events=[event],
        returns=returns,
        market_returns=r_m,
        sector_map=sector_map,
        adv_dollars_60d=adv_map,
        model="both",
    )
    models = {o.model for o in obs}
    assert models == {"market", "sector_matched"}
    # Each model emits one row per window with non-zero days
    assert len(obs) >= 6  # at least 3 windows × 2 models in the synthetic data


def test_run_event_study_skips_event_with_no_returns() -> None:
    """An event whose ticker has no return series should be silently skipped."""
    event = IndexEvent(
        event_id="test-add-MISSING",
        index="sp500",
        ticker="MISSING",
        action="add",
        effective_date=date(2024, 6, 17),
        announcement_date=date(2024, 6, 10),
        reason="test",
        source_url="test",
    )
    obs = run_event_study(
        events=[event],
        returns={},
        market_returns=pd.Series(dtype=float),
        sector_map={},
        adv_dollars_60d={},
        model="market",
    )
    assert obs == []


def test_run_event_study_model_market_only_skips_sector_matched() -> None:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2023-01-01", periods=400, freq="B")
    r_m = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    returns = {
        "EVENT": pd.Series(0.0 + 1.0 * r_m + rng.normal(0, 0.002, 400), index=idx),
    }
    event = IndexEvent(
        event_id="test-add-EVENT",
        index="sp500",
        ticker="EVENT",
        action="add",
        effective_date=idx[-15].date(),
        announcement_date=idx[-20].date(),
        reason="test",
        source_url="test",
    )
    obs = run_event_study(
        events=[event],
        returns=returns,
        market_returns=r_m,
        sector_map={"EVENT": "Information Technology"},
        adv_dollars_60d={"EVENT": 1e9},
        model="market",
    )
    assert all(o.model == "market" for o in obs)
