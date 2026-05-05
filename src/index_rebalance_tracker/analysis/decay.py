"""Cohort-aggregated decay of the index-effect CAR.

The "index effect" — the abnormal return of a stock newly added to the
S&P 500 between announcement and effective date — was estimated at +8.8 %
on average for 1990–2005 by Petajisto (2011). Subsequent literature
(Greenwood-Sammon 2022, Bennett-Stulz-Wang 2023) suggests this premium has
compressed materially as more arbitrage capital chases the trade.

This module groups historical CAR observations by year (or two-year
buckets) and aggregates with mean / median / IQR per cohort. The output
is the "headline result" rendered prominently in the dashboard.

References:
    Petajisto, A. (2011). The index premium and its hidden cost for
    index funds. *Journal of Empirical Finance* 18(2), 271–288.

    Greenwood, R. and Sammon, M. (2022). The disappearing index effect.
    *Working paper, Harvard Business School.*
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np

from ..models import CARObservation, CohortDecay, IndexEvent

Grouping = Literal["yearly", "biannual"]
ModelLabel = Literal["market", "sector_matched"]
ActionLabel = Literal["add", "delete"]


def cohort_decay(
    car_observations: list[CARObservation],
    events: dict[str, IndexEvent],
    *,
    grouping: Grouping = "yearly",
    window_label: str = "[T-A, T-E-1]",
    model: ModelLabel = "market",
    action: ActionLabel = "add",
) -> list[CohortDecay]:
    """Aggregate CAR observations into year (or biannual) cohorts.

    Args:
        car_observations: Flat list emitted by ``run_event_study``.
        events: Map of event_id → IndexEvent (we need the effective_date
            to determine the cohort).
        grouping: "yearly" (one cohort per calendar year) or "biannual"
            (two-year buckets, ``"2010-2011"`` etc.).
        window_label: Filter to one specific window label
            (default ``"[T-A, T-E-1]"`` — the announcement-to-effective
            window, the canonical "index effect" period).
        model: Filter to "market" or "sector_matched" (default "market").
        action: Filter to "add" or "delete" (default "add" — additions
            are the textbook subject; deletions can be aggregated
            separately).

    Returns:
        List of CohortDecay rows, one per cohort with at least one
        matching observation. Sorted ascending by cohort label.

    Example:
        >>> from index_rebalance_tracker.models import CARObservation, IndexEvent
        >>> events = {
        ...     "e1": IndexEvent(event_id="e1", index="sp500", ticker="A",
        ...                       action="add", effective_date=date(2020, 6, 1),
        ...                       announcement_date=None, reason=None, source_url=None),
        ... }
        >>> obs = [CARObservation(event_id="e1", window_label="[T-A, T-E-1]",
        ...                        model="market", car=0.05, n_days=4,
        ...                        benchmark_return=0.0)]
        >>> cohorts = cohort_decay(obs, events, grouping="yearly")
        >>> cohorts[0].cohort
        '2020'
        >>> round(cohorts[0].mean_car, 2)
        0.05
    """
    by_cohort: dict[str, list[float]] = {}
    for obs in car_observations:
        if obs.window_label != window_label or obs.model != model:
            continue
        event = events.get(obs.event_id)
        if event is None or event.action != action:
            continue
        cohort = _cohort_label(event.effective_date, grouping)
        by_cohort.setdefault(cohort, []).append(obs.car)

    out: list[CohortDecay] = []
    for cohort, cars in sorted(by_cohort.items()):
        arr = np.array(cars, dtype=float)
        if arr.size == 0:
            continue
        out.append(
            CohortDecay(
                cohort=cohort,
                n_events=int(arr.size),
                mean_car=float(np.mean(arr)),
                median_car=float(np.median(arr)),
                car_p25=float(np.percentile(arr, 25)),
                car_p75=float(np.percentile(arr, 75)),
            )
        )
    return out


def _cohort_label(d: date, grouping: Grouping) -> str:
    """Map a date to its cohort label."""
    if grouping == "yearly":
        return str(d.year)
    # biannual: 2010-2011, 2012-2013, ...
    start = d.year - (d.year % 2)
    return f"{start}-{start + 1}"
