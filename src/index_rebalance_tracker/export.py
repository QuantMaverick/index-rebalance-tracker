"""JSON / Parquet writers for the dashboard contract.

The output contract is the project's public surface — the downstream
dashboard validates every file against the matching pydantic model in
:mod:`index_rebalance_tracker.models`. Drift here is silent corruption,
so all writes go through ``model.model_dump_json(indent=2)`` rather
than ``json.dump`` of a dict.

Files produced (per ``build-dashboard``):

* ``events_<index>.json``   — :class:`EventsFile`  per-event combined payload
* ``decay_<index>.json``    — :class:`DecayFile`   cohort aggregation
* ``tca_summary.json``      — :class:`TCASummaryFile` annual TCA roll-up
* ``methodology_constants.json`` — :class:`MethodologyConstants`
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .models import (
    CARObservation,
    CohortDecay,
    DecayFile,
    EventResult,
    EventsFile,
    IndexEvent,
    IndexName,
    LiquidityMetrics,
    MethodologyConstants,
    TCAEstimate,
    TCASummaryFile,
    TCASummaryRow,
)

# ── per-event combiner ─────────────────────────────────────────────────────


def build_event_results(
    events: list[IndexEvent],
    car_obs: list[CARObservation],
    liquidity: list[LiquidityMetrics],
    tca: list[TCAEstimate],
) -> list[EventResult]:
    """Join per-event records into the combined ``EventResult`` payload.

    Args:
        events: Source events (from M1).
        car_obs: CAR rows (from M2). Multiple per event allowed.
        liquidity: Liquidity rows (from M3). At most one per event.
        tca: TCA rows (from M3). At most one per event.

    Returns:
        One ``EventResult`` per input event, with empty / null sub-fields
        when the analysis was filtered out for that event.
    """
    car_by_event: dict[str, list[CARObservation]] = {}
    for obs in car_obs:
        car_by_event.setdefault(obs.event_id, []).append(obs)
    liq_by_event: dict[str, LiquidityMetrics] = {x.event_id: x for x in liquidity}
    tca_by_event: dict[str, TCAEstimate] = {x.event_id: x for x in tca}

    return [
        EventResult(
            event=e,
            car_observations=car_by_event.get(e.event_id, []),
            liquidity=liq_by_event.get(e.event_id),
            tca=tca_by_event.get(e.event_id),
        )
        for e in events
    ]


# ── annual TCA roll-up ─────────────────────────────────────────────────────


def annual_tca_summary(
    tca: list[TCAEstimate], events: dict[str, IndexEvent]
) -> list[TCASummaryRow]:
    """Aggregate TCA rows by calendar year of the event's effective date.

    Args:
        tca: Per-event TCA rows.
        events: Map of event_id → IndexEvent (for the effective date).

    Returns:
        Sorted list of ``TCASummaryRow``, one per year with at least one
        matching event.
    """
    by_year: dict[int, list[TCAEstimate]] = {}
    for row in tca:
        ev = events.get(row.event_id)
        if ev is None:
            continue
        by_year.setdefault(ev.effective_date.year, []).append(row)

    out: list[TCASummaryRow] = []
    for year in sorted(by_year):
        rows = by_year[year]
        forced = np.array([r.forced_execution_cost_bps for r in rows], dtype=float)
        spread = np.array([r.spread_execution_cost_bps for r in rows], dtype=float)
        savings = np.array([r.savings_bps for r in rows], dtype=float)
        demand = np.array([r.estimated_demand_usd for r in rows], dtype=float)
        out.append(
            TCASummaryRow(
                year=year,
                n_events=len(rows),
                total_demand_usd=float(np.sum(demand)),
                mean_forced_bps=float(np.mean(forced)),
                mean_spread_bps=float(np.mean(spread)),
                mean_savings_bps=float(np.mean(savings)),
            )
        )
    return out


# ── JSON writers ───────────────────────────────────────────────────────────


def write_events_file(
    path: Path,
    *,
    index: IndexName,
    events: list[EventResult],
    as_of: datetime | None = None,
) -> None:
    """Write the combined per-event JSON.

    Args:
        path: Destination file path. Parent directory is created if missing.
        index: Index identifier (``sp500`` or ``msci_sg``).
        events: Pre-built ``EventResult`` rows.
        as_of: Snapshot timestamp; defaults to current UTC.
    """
    payload = EventsFile(
        as_of=as_of or datetime.utcnow(),
        index=index,
        events=events,
    )
    _write_json(path, payload.model_dump_json(indent=2))


def write_decay_file(
    path: Path,
    *,
    index: IndexName,
    cohorts: list[CohortDecay],
    grouping: str,
    window_label: str,
    model_used: str,
    action: str,
    as_of: datetime | None = None,
) -> None:
    """Write the cohort-decay JSON."""
    payload = DecayFile(
        as_of=as_of or datetime.utcnow(),
        index=index,
        grouping=grouping,  # type: ignore[arg-type]
        window_label=window_label,
        model_used=model_used,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        cohorts=cohorts,
    )
    _write_json(path, payload.model_dump_json(indent=2))


def write_tca_summary_file(
    path: Path,
    *,
    index: IndexName,
    passive_aum_usd: float,
    annual: list[TCASummaryRow],
    as_of: datetime | None = None,
) -> None:
    """Write the annual TCA summary JSON."""
    payload = TCASummaryFile(
        as_of=as_of or datetime.utcnow(),
        index=index,
        passive_aum_usd=passive_aum_usd,
        annual=annual,
    )
    _write_json(path, payload.model_dump_json(indent=2))


def write_methodology_file(
    path: Path,
    *,
    passive_aum_usd: float,
    market_model_estimation_window_days: int = 250,
    market_model_gap_days: int = 30,
    sector_match_pool_size: int = 5,
    decay_cohort_grouping: str = "yearly",
) -> None:
    """Write methodology constants the dashboard renders on its About page."""
    payload = MethodologyConstants(
        passive_aum_usd=passive_aum_usd,
        market_model_estimation_window_days=market_model_estimation_window_days,
        market_model_gap_days=market_model_gap_days,
        sector_match_pool_size=sector_match_pool_size,
        decay_cohort_grouping=decay_cohort_grouping,  # type: ignore[arg-type]
        sources={
            "BrownWarner1985": "https://doi.org/10.1016/0304-405X(85)90042-X",
            "Petajisto2011": "https://doi.org/10.1016/j.jempfin.2010.10.002",
            "CorwinSchultz2012": "https://doi.org/10.1111/j.1540-6261.2012.01729.x",
            "Amihud2002": "https://doi.org/10.1016/S1386-4181(01)00024-6",
            "AlmgrenEtAl2005": "https://www.courant.nyu.edu/~almgren/papers/costestim.pdf",
            "Perold1988": "https://doi.org/10.3905/jpm.1988.409150",
        },
    )
    _write_json(path, payload.model_dump_json(indent=2))


# ── helpers ────────────────────────────────────────────────────────────────


def _write_json(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
