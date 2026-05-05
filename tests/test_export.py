"""Tests for the JSON / dashboard contract writers."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from index_rebalance_tracker.export import (
    annual_tca_summary,
    build_event_results,
    write_decay_file,
    write_events_file,
    write_methodology_file,
    write_tca_summary_file,
)
from index_rebalance_tracker.models import (
    SCHEMA_VERSION,
    CARObservation,
    CohortDecay,
    DecayFile,
    EventsFile,
    IndexEvent,
    LiquidityMetrics,
    MethodologyConstants,
    TCAEstimate,
    TCASummaryFile,
)

# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def example_event() -> IndexEvent:
    return IndexEvent(
        event_id="sp500-add-TSLA-2020-12-21",
        index="sp500",
        ticker="TSLA",
        action="add",
        effective_date=date(2020, 12, 21),
        announcement_date=date(2020, 12, 16),
        reason="market cap inclusion",
        source_url="https://example.com",
    )


@pytest.fixture
def example_car(example_event: IndexEvent) -> CARObservation:
    return CARObservation(
        event_id=example_event.event_id,
        window_label="[T-A, T-E-1]",
        model="market",
        car=0.072,
        n_days=4,
        benchmark_return=0.005,
    )


@pytest.fixture
def example_liquidity(example_event: IndexEvent) -> LiquidityMetrics:
    return LiquidityMetrics(
        event_id=example_event.event_id,
        adv_60d_shares=50_000_000,
        adv_60d_dollars=2_000_000_000,
        corwin_schultz_spread=0.0008,
        amihud_illiquidity=1.2e-12,
        kyle_lambda=3.4e-9,
    )


@pytest.fixture
def example_tca(example_event: IndexEvent) -> TCAEstimate:
    return TCAEstimate(
        event_id=example_event.event_id,
        passive_aum_usd=6.5e12,
        estimated_demand_usd=2.0e10,
        forced_execution_cost_bps=120.5,
        spread_execution_cost_bps=53.9,
        savings_bps=66.6,
    )


# ── build_event_results ────────────────────────────────────────────────────


def test_build_event_results_joins_all_three(
    example_event: IndexEvent,
    example_car: CARObservation,
    example_liquidity: LiquidityMetrics,
    example_tca: TCAEstimate,
) -> None:
    results = build_event_results(
        [example_event], [example_car], [example_liquidity], [example_tca]
    )
    assert len(results) == 1
    r = results[0]
    assert r.event == example_event
    assert r.car_observations == [example_car]
    assert r.liquidity == example_liquidity
    assert r.tca == example_tca


def test_build_event_results_handles_missing_optional_blocks(
    example_event: IndexEvent, example_car: CARObservation
) -> None:
    """An event with no liquidity / TCA row should still be emitted with nulls."""
    results = build_event_results([example_event], [example_car], [], [])
    assert results[0].liquidity is None
    assert results[0].tca is None


def test_build_event_results_groups_multiple_car_rows_per_event(
    example_event: IndexEvent, example_car: CARObservation
) -> None:
    """The driver emits 4 windows × 2 models = 8 rows per event; we keep them all."""
    car2 = example_car.model_copy(update={"window_label": "[T-A-5, T-A-1]"})
    car3 = example_car.model_copy(update={"model": "sector_matched"})
    results = build_event_results([example_event], [example_car, car2, car3], [], [])
    assert len(results[0].car_observations) == 3


# ── annual_tca_summary ─────────────────────────────────────────────────────


def test_annual_tca_summary_aggregates_by_year(
    example_event: IndexEvent, example_tca: TCAEstimate
) -> None:
    other = example_event.model_copy(
        update={
            "event_id": "sp500-add-OTHER-2024-06-01",
            "ticker": "OTHER",
            "effective_date": date(2024, 6, 1),
        }
    )
    other_tca = example_tca.model_copy(
        update={
            "event_id": other.event_id,
            "estimated_demand_usd": 5e9,
            "forced_execution_cost_bps": 80.0,
            "spread_execution_cost_bps": 36.0,
            "savings_bps": 44.0,
        }
    )
    annual = annual_tca_summary(
        [example_tca, other_tca],
        {example_event.event_id: example_event, other.event_id: other},
    )
    by_year = {row.year: row for row in annual}
    assert by_year[2020].n_events == 1
    assert by_year[2020].mean_forced_bps == pytest.approx(120.5)
    assert by_year[2024].n_events == 1
    assert by_year[2024].mean_forced_bps == pytest.approx(80.0)


def test_annual_tca_summary_skips_events_not_in_event_map(
    example_event: IndexEvent, example_tca: TCAEstimate
) -> None:
    """A TCA row whose event_id is unknown is silently dropped."""
    annual = annual_tca_summary([example_tca], {})
    assert annual == []


# ── JSON writers (round-trip) ──────────────────────────────────────────────


def test_write_events_file_round_trips_to_pydantic(
    tmp_path: Path,
    example_event: IndexEvent,
    example_car: CARObservation,
    example_liquidity: LiquidityMetrics,
    example_tca: TCAEstimate,
) -> None:
    results = build_event_results(
        [example_event], [example_car], [example_liquidity], [example_tca]
    )
    path = tmp_path / "events_sp500.json"
    write_events_file(path, index="sp500", events=results, as_of=datetime(2026, 5, 5))

    raw = json.loads(path.read_text())
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["index"] == "sp500"
    parsed = EventsFile.model_validate_json(path.read_text())
    assert parsed.events[0].event.ticker == "TSLA"


def test_write_decay_file_round_trips(tmp_path: Path) -> None:
    cohorts = [
        CohortDecay(
            cohort="2020", n_events=2, mean_car=0.06, median_car=0.06, car_p25=0.05, car_p75=0.07
        ),
        CohortDecay(
            cohort="2024", n_events=1, mean_car=0.02, median_car=0.02, car_p25=0.02, car_p75=0.02
        ),
    ]
    path = tmp_path / "decay_sp500.json"
    write_decay_file(
        path,
        index="sp500",
        cohorts=cohorts,
        grouping="yearly",
        window_label="[T-A, T-E-1]",
        model_used="market",
        action="add",
    )
    parsed = DecayFile.model_validate_json(path.read_text())
    assert len(parsed.cohorts) == 2
    assert parsed.grouping == "yearly"
    assert parsed.model_used == "market"


def test_write_tca_summary_file_round_trips(tmp_path: Path) -> None:
    rows = [
        # build a minimal valid row inline
    ]
    path = tmp_path / "tca_summary.json"
    write_tca_summary_file(path, index="sp500", passive_aum_usd=6.5e12, annual=rows)
    parsed = TCASummaryFile.model_validate_json(path.read_text())
    assert parsed.passive_aum_usd == 6.5e12
    assert parsed.annual == []


def test_write_methodology_file_includes_citations(tmp_path: Path) -> None:
    path = tmp_path / "methodology_constants.json"
    write_methodology_file(path, passive_aum_usd=6.5e12)
    parsed = MethodologyConstants.model_validate_json(path.read_text())
    assert parsed.passive_aum_usd == 6.5e12
    assert "BrownWarner1985" in parsed.sources
    assert "Petajisto2011" in parsed.sources
    assert "AlmgrenEtAl2005" in parsed.sources


def test_write_creates_parent_directory(tmp_path: Path) -> None:
    """Writers should mkdir -p the parent dir without complaint."""
    path = tmp_path / "nested" / "path" / "events_sp500.json"
    write_events_file(path, index="sp500", events=[])
    assert path.exists()
