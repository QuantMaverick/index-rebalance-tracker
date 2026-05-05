"""Cumulative-abnormal-return (CAR) computation for index rebalance events.

Two abnormal-return models are implemented; the user picks one or both via
the CLI.

**Market model** (Brown & Warner 1985)
    Estimate :math:`R_{i,t} = \\alpha_i + \\beta_i R_{m,t} + \\epsilon_{i,t}`
    over a 250-trading-day window ending ``estimation_gap_days`` (default 30)
    before the announcement date. Abnormal return on each event-window day is
    :math:`AR_{i,t} = R_{i,t} - (\\hat\\alpha_i + \\hat\\beta_i R_{m,t})`.

**Sector-matched control** (Petajisto 2011, Beneish-Whaley 1996 lineage)
    For each event stock, identify five non-event peers in the same GICS
    sector and closest in 60-day pre-event average dollar volume (an ADV-
    based size proxy — see methodology limit below). Abnormal return is the
    event return minus the median of the five control returns on the same
    calendar day.

Both methods produce a per-day :math:`AR` series; the cumulative-abnormal-
return over a window is the simple sum (event-time aggregation, not
calendar-time).

Methodology limits (surfaced in :func:`run_event_study` docstring):

* Announcement date for SP500 changes is approximated as ``T_E -
  announcement_offset_days`` because Wikipedia only carries the effective
  date. The default offset (5 trading days) matches the typical S&P press-
  release lead time observed in 2010–2024.
* Size proxy is 60-day ADV in dollars rather than market cap, because
  market-cap timeseries are not in our free data feed and we want the
  matching to be as-of-T_A, not current.
* Sector membership uses the *current* Wikipedia GICS sector — see README §
  Limitations for the documented drift impact.

References:
    Brown, S. J. and Warner, J. B. (1985). Using daily stock returns: The
    case of event studies. *Journal of Financial Economics* 14(1), 3–31.

    Petajisto, A. (2011). The index premium and its hidden cost for index
    funds. *Journal of Empirical Finance* 18(2), 271–288.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from ..models import CARObservation, IndexEvent
from .sector_match import find_matched_controls

logger = logging.getLogger(__name__)

DEFAULT_ESTIMATION_WINDOW = 250
DEFAULT_ESTIMATION_GAP = 30
DEFAULT_SECTOR_MATCH_POOL = 5
DEFAULT_ANNOUNCEMENT_OFFSET = 5  # trading days


# ── Market-model AR ─────────────────────────────────────────────────────────


def estimate_market_model(
    event_returns: pd.Series,
    market_returns: pd.Series,
    announcement_date: date,
    *,
    estimation_window_days: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap_days: int = DEFAULT_ESTIMATION_GAP,
) -> tuple[float, float, int]:
    """OLS-fit ``alpha`` and ``beta`` from the pre-event estimation window.

    Args:
        event_returns: Daily simple returns indexed by date.
        market_returns: Daily simple returns of the market proxy, same index.
        announcement_date: T_A. The estimation window ends
            ``estimation_gap_days`` trading days before this.
        estimation_window_days: Length of the regression sample (default 250).
        estimation_gap_days: Gap between estimation window end and T_A
            (default 30) — protects against pre-announcement leakage.

    Returns:
        Tuple ``(alpha, beta, n_obs)``. ``n_obs`` is the actual number of
        joint observations used, which may be less than
        ``estimation_window_days`` if the event return series is short.

    Raises:
        ValueError: if fewer than 30 joint observations are available — the
            regression would be too noisy to publish.
    """
    joined = pd.concat([event_returns.rename("r_i"), market_returns.rename("r_m")], axis=1).dropna()
    pre_event = joined.loc[joined.index < pd.Timestamp(announcement_date)]
    # Drop the gap zone closest to T_A
    if estimation_gap_days > 0:
        pre_event = (
            pre_event.iloc[:-estimation_gap_days]
            if len(pre_event) > estimation_gap_days
            else pre_event.iloc[0:0]
        )
    # Take the trailing N observations of what remains
    sample = pre_event.iloc[-estimation_window_days:]

    n_obs = len(sample)
    if n_obs < 30:
        raise ValueError(
            f"insufficient estimation data: n_obs={n_obs} for announcement {announcement_date}"
        )

    x = sample["r_m"].to_numpy()
    y = sample["r_i"].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    return float(alpha), float(beta), n_obs


def market_model_abnormal_returns(
    event_returns: pd.Series,
    market_returns: pd.Series,
    announcement_date: date,
    *,
    estimation_window_days: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap_days: int = DEFAULT_ESTIMATION_GAP,
) -> pd.Series:
    """Compute the per-day abnormal-return series under the market model.

    Args:
        event_returns: Daily returns of the event stock.
        market_returns: Daily returns of the market proxy.
        announcement_date: T_A.
        estimation_window_days: Days in the regression sample.
        estimation_gap_days: Gap between sample end and T_A.

    Returns:
        Series indexed by date with column name ``ar``. Values on dates
        with missing event or market return are NaN.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2020-01-01", periods=300, freq="B")
        >>> rng = __import__("numpy").random.default_rng(0)
        >>> r_m = pd.Series(rng.normal(0, 0.01, 300), index=idx)
        >>> r_i = 0.0002 + 1.2 * r_m + pd.Series(rng.normal(0, 0.005, 300), index=idx)
        >>> ar = market_model_abnormal_returns(r_i, r_m, idx[280].date())
        >>> isinstance(ar, pd.Series)
        True
    """
    alpha, beta, _ = estimate_market_model(
        event_returns,
        market_returns,
        announcement_date,
        estimation_window_days=estimation_window_days,
        estimation_gap_days=estimation_gap_days,
    )
    expected = alpha + beta * market_returns
    return (event_returns - expected).rename("ar")


# ── Sector-matched AR ──────────────────────────────────────────────────────


def sector_matched_abnormal_returns(
    event_returns: pd.Series,
    control_returns: pd.DataFrame,
) -> pd.Series:
    """Compute AR as event return minus median of control returns.

    Args:
        event_returns: Daily returns of the event stock.
        control_returns: DataFrame indexed by date, columns are tickers of
            the matched controls. Must align with ``event_returns`` index.

    Returns:
        Series of abnormal returns named ``ar``. NaN where the control
        median is undefined (all controls missing for that day).

    Raises:
        ValueError: if ``control_returns`` has zero columns.
    """
    if control_returns.shape[1] == 0:
        raise ValueError("control_returns must have at least one column")
    median = control_returns.median(axis=1)
    return (event_returns - median).rename("ar")


# ── CAR aggregation ────────────────────────────────────────────────────────


def cumulative_abnormal_return(
    abnormal_returns: pd.Series,
    window_start: date,
    window_end: date,
) -> tuple[float, int]:
    """Sum daily abnormal returns over a closed window.

    Args:
        abnormal_returns: Daily AR series, indexed by date.
        window_start: Inclusive start.
        window_end: Inclusive end.

    Returns:
        Tuple ``(car, n_days)``. CAR is NaN if all days in the window are
        NaN; ``n_days`` counts non-NaN observations included.
    """
    mask = (abnormal_returns.index >= pd.Timestamp(window_start)) & (
        abnormal_returns.index <= pd.Timestamp(window_end)
    )
    sliced = abnormal_returns.loc[mask].dropna()
    if sliced.empty:
        return float("nan"), 0
    return float(sliced.sum()), int(len(sliced))


# ── Window construction ────────────────────────────────────────────────────


def standard_windows(
    announcement_date: date,
    effective_date: date,
    *,
    pre_run_up_days: int = 5,
    post_drift_days: int = 5,
    full_pre_run_up_days: int = 5,
    full_post_drift_days: int = 5,
) -> list[tuple[str, date, date]]:
    """Generate the four canonical event-study windows.

    Returns:
        List of ``(label, start_date, end_date)`` tuples:

        * ``[T-A-N, T-A-1]`` — pre-announcement run-up
        * ``[T-A, T-E-1]`` — announcement-to-effective
        * ``[T-E, T-E+M]`` — post-effective drift / reversal
        * ``[T-A-N, T-E+M]`` — full event window

        Calendar-day arithmetic — for fine slicing, callers should use
        the AR series index directly.
    """
    pre_start = announcement_date - timedelta(days=pre_run_up_days)
    pre_end = announcement_date - timedelta(days=1)
    main_start = announcement_date
    main_end = effective_date - timedelta(days=1)
    post_start = effective_date
    post_end = effective_date + timedelta(days=post_drift_days)
    full_start = announcement_date - timedelta(days=full_pre_run_up_days)
    full_end = effective_date + timedelta(days=full_post_drift_days)
    return [
        (f"[T-A-{pre_run_up_days}, T-A-1]", pre_start, pre_end),
        ("[T-A, T-E-1]", main_start, main_end),
        (f"[T-E, T-E+{post_drift_days}]", post_start, post_end),
        (f"[T-A-{full_pre_run_up_days}, T-E+{full_post_drift_days}]", full_start, full_end),
    ]


# ── Top-level driver ───────────────────────────────────────────────────────


def run_event_study(
    events: list[IndexEvent],
    returns: dict[str, pd.Series],
    market_returns: pd.Series,
    sector_map: dict[str, str],
    adv_dollars_60d: dict[str, float],
    *,
    model: Literal["market", "sector_matched", "both"] = "both",
    pre_run_up_days: int = 5,
    post_drift_days: int = 20,
    estimation_window_days: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap_days: int = DEFAULT_ESTIMATION_GAP,
    sector_match_pool: int = DEFAULT_SECTOR_MATCH_POOL,
    announcement_offset_days: int = DEFAULT_ANNOUNCEMENT_OFFSET,
) -> list[CARObservation]:
    """Run the full event study and emit ``CARObservation`` rows.

    For each event in ``events`` we produce up to three windows × up to two
    models worth of CAR rows (market and/or sector-matched). Events with
    insufficient pre-event data or that fail sector matching are skipped
    with a warning log line.

    Args:
        events: Event records sourced from M1.
        returns: Per-ticker daily simple-return Series, indexed by date.
        market_returns: Market proxy daily-return Series (e.g. SPY).
        sector_map: Ticker → GICS sector. Used for sector matching.
        adv_dollars_60d: Ticker → 60-day average dollar volume. Used as size
            proxy for sector matching (see methodology note).
        model: Which AR model(s) to run.
        pre_run_up_days: Days pre-announcement included in the run-up window.
        post_drift_days: Days post-effective included in the drift window.
        estimation_window_days: Market-model estimation sample length.
        estimation_gap_days: Trading-day gap between estimation end and T_A.
        sector_match_pool: Number of matched controls to take.
        announcement_offset_days: T_A = T_E − offset (calendar days).

    Returns:
        Flat list of ``CARObservation`` records, suitable for tabular
        export or aggregation by cohort.
    """
    obs: list[CARObservation] = []

    universe = list(returns.keys())

    for event in events:
        if event.ticker not in returns:
            logger.debug("event %s: no returns for %s, skipping", event.event_id, event.ticker)
            continue
        ann_date = event.announcement_date or _approximate_announcement(
            event.effective_date, announcement_offset_days
        )
        windows = standard_windows(
            ann_date,
            event.effective_date,
            pre_run_up_days=pre_run_up_days,
            post_drift_days=post_drift_days,
            full_pre_run_up_days=pre_run_up_days,
            full_post_drift_days=post_drift_days,
        )

        event_returns = returns[event.ticker]

        if model in ("market", "both"):
            try:
                ar = market_model_abnormal_returns(
                    event_returns,
                    market_returns,
                    ann_date,
                    estimation_window_days=estimation_window_days,
                    estimation_gap_days=estimation_gap_days,
                )
                obs.extend(_car_rows(event.event_id, "market", ar, market_returns, windows))
            except ValueError as exc:
                logger.warning("market-model skip %s: %s", event.event_id, exc)

        if model in ("sector_matched", "both"):
            try:
                controls = find_matched_controls(
                    event_ticker=event.ticker,
                    event_sector=sector_map[event.ticker],
                    universe=universe,
                    sector_map=sector_map,
                    adv_dollars=adv_dollars_60d,
                    n_matches=sector_match_pool,
                )
            except (KeyError, ValueError) as exc:
                logger.warning("sector match skip %s: %s", event.event_id, exc)
                continue
            control_df = pd.concat({t: returns[t] for t in controls if t in returns}, axis=1)
            if control_df.shape[1] == 0:
                logger.warning("no control returns for %s — skipping", event.event_id)
                continue
            ar = sector_matched_abnormal_returns(event_returns, control_df)
            benchmark_series = control_df.median(axis=1)
            obs.extend(_car_rows(event.event_id, "sector_matched", ar, benchmark_series, windows))

    return obs


# ── Helpers ────────────────────────────────────────────────────────────────


def _approximate_announcement(effective_date: date, offset_days: int) -> date:
    """T_A = T_E − offset (calendar days). Documented assumption."""
    return effective_date - timedelta(days=offset_days)


def _car_rows(
    event_id: str,
    model_label: Literal["market", "sector_matched"],
    ar: pd.Series,
    benchmark: pd.Series,
    windows: list[tuple[str, date, date]],
) -> list[CARObservation]:
    out: list[CARObservation] = []
    for label, start, end in windows:
        car, n_days = cumulative_abnormal_return(ar, start, end)
        if n_days == 0:
            continue
        bench_car, _ = cumulative_abnormal_return(benchmark, start, end)
        out.append(
            CARObservation(
                event_id=event_id,
                window_label=label,
                model=model_label,
                car=car,
                n_days=n_days,
                benchmark_return=0.0 if pd.isna(bench_car) else bench_car,
            )
        )
    return out
