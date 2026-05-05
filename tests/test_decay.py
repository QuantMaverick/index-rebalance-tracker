"""Tests for cohort decay aggregation."""

from __future__ import annotations

from datetime import date

import pytest

from index_rebalance_tracker.analysis.decay import _cohort_label, cohort_decay
from index_rebalance_tracker.models import CARObservation, IndexEvent


def _ev(event_id: str, year: int, action: str = "add") -> IndexEvent:
    return IndexEvent(
        event_id=event_id,
        index="sp500",
        ticker=f"T{event_id}",
        action=action,  # type: ignore[arg-type]
        effective_date=date(year, 6, 1),
        announcement_date=None,
        reason=None,
        source_url=None,
    )


def _obs(
    event_id: str,
    car: float,
    *,
    window_label: str = "[T-A, T-E-1]",
    model: str = "market",
) -> CARObservation:
    return CARObservation(
        event_id=event_id,
        window_label=window_label,
        model=model,  # type: ignore[arg-type]
        car=car,
        n_days=4,
        benchmark_return=0.0,
    )


def test_cohort_label_yearly() -> None:
    assert _cohort_label(date(2020, 6, 1), "yearly") == "2020"
    assert _cohort_label(date(2024, 12, 31), "yearly") == "2024"


def test_cohort_label_biannual() -> None:
    assert _cohort_label(date(2020, 6, 1), "biannual") == "2020-2021"
    assert _cohort_label(date(2021, 1, 1), "biannual") == "2020-2021"
    assert _cohort_label(date(2022, 6, 1), "biannual") == "2022-2023"


def test_cohort_decay_groups_by_year() -> None:
    events = {
        "e1": _ev("e1", 2020),
        "e2": _ev("e2", 2020),
        "e3": _ev("e3", 2024),
    }
    obs = [_obs("e1", 0.05), _obs("e2", 0.07), _obs("e3", 0.02)]
    cohorts = cohort_decay(obs, events, grouping="yearly")
    by_year = {c.cohort: c for c in cohorts}
    assert by_year["2020"].n_events == 2
    assert by_year["2020"].mean_car == pytest.approx(0.06)
    assert by_year["2024"].n_events == 1
    assert by_year["2024"].mean_car == pytest.approx(0.02)


def test_cohort_decay_filters_by_window() -> None:
    events = {"e1": _ev("e1", 2020)}
    obs = [
        _obs("e1", 0.05, window_label="[T-A, T-E-1]"),
        _obs("e1", 0.99, window_label="[T-A-5, T-A-1]"),
    ]
    cohorts = cohort_decay(obs, events, window_label="[T-A, T-E-1]")
    assert cohorts[0].mean_car == pytest.approx(0.05)


def test_cohort_decay_filters_by_model() -> None:
    events = {"e1": _ev("e1", 2020)}
    obs = [
        _obs("e1", 0.05, model="market"),
        _obs("e1", 0.99, model="sector_matched"),
    ]
    cohorts = cohort_decay(obs, events, model="market")
    assert cohorts[0].mean_car == pytest.approx(0.05)


def test_cohort_decay_filters_by_action() -> None:
    events = {
        "e1": _ev("e1", 2020, action="add"),
        "e2": _ev("e2", 2020, action="delete"),
    }
    obs = [_obs("e1", 0.05), _obs("e2", -0.10)]
    cohorts = cohort_decay(obs, events, action="add")
    assert len(cohorts) == 1
    assert cohorts[0].mean_car == pytest.approx(0.05)


def test_cohort_decay_biannual_groups() -> None:
    events = {
        "e1": _ev("e1", 2020),
        "e2": _ev("e2", 2021),
        "e3": _ev("e3", 2022),
    }
    obs = [_obs("e1", 0.04), _obs("e2", 0.08), _obs("e3", 0.02)]
    cohorts = cohort_decay(obs, events, grouping="biannual")
    by_label = {c.cohort: c for c in cohorts}
    assert by_label["2020-2021"].n_events == 2
    assert by_label["2020-2021"].mean_car == pytest.approx(0.06)
    assert by_label["2022-2023"].n_events == 1


def test_cohort_decay_iqr_correctly_computed() -> None:
    events = {f"e{i}": _ev(f"e{i}", 2020) for i in range(1, 5)}
    cars = [0.02, 0.04, 0.06, 0.08]
    obs = [_obs(f"e{i + 1}", c) for i, c in enumerate(cars)]
    cohorts = cohort_decay(obs, events)
    c = cohorts[0]
    assert c.median_car == pytest.approx(0.05)
    assert c.car_p25 == pytest.approx(0.035)
    assert c.car_p75 == pytest.approx(0.065)


def test_cohort_decay_empty_when_no_matching_obs() -> None:
    events = {"e1": _ev("e1", 2020)}
    obs = [_obs("e1", 0.05, model="sector_matched")]
    cohorts = cohort_decay(obs, events, model="market")
    assert cohorts == []


def test_cohort_decay_skips_observations_with_unknown_event_id() -> None:
    """An obs whose event_id isn't in `events` is silently dropped."""
    events = {"e1": _ev("e1", 2020)}
    obs = [_obs("e1", 0.05), _obs("ghost", 999.0)]
    cohorts = cohort_decay(obs, events)
    assert cohorts[0].n_events == 1
    assert cohorts[0].mean_car == pytest.approx(0.05)


def test_cohort_decay_sorted_ascending_by_label() -> None:
    events = {
        "a": _ev("a", 2024),
        "b": _ev("b", 2018),
        "c": _ev("c", 2021),
    }
    obs = [_obs(eid, 0.01) for eid in ("a", "b", "c")]
    cohorts = cohort_decay(obs, events)
    assert [c.cohort for c in cohorts] == ["2018", "2021", "2024"]
