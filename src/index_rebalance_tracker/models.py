"""Pydantic data models that cross module boundaries.

Every structure exchanged between scrapers, analytics, and exporters lives
here. The downstream dashboard project copies this file verbatim to validate
inputs — keep models stable; bump ``schema_version`` on breaking change.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: str = "0.1.0"

IndexName = Literal["sp500", "msci_sg"]
Action = Literal["add", "delete"]


class Constituent(BaseModel):
    """A single index member as of a snapshot date.

    Source: Wikipedia "List of S&P 500 companies" main constituents table
    for SP500. GICS sector reflects current Wikipedia state — see README §
    Limitations on classification drift over historical events.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(..., description="Trading symbol, e.g. 'AAPL'.")
    name: str = Field(..., description="Issuer name, e.g. 'Apple Inc.'.")
    gics_sector: str | None = Field(None, description="GICS Sector (level 1).")
    gics_sub_industry: str | None = Field(None, description="GICS Sub-Industry (level 4).")
    headquarters: str | None = None
    date_added: date | None = Field(None, description="First date in the index per Wikipedia.")
    cik: str | None = Field(None, description="SEC Central Index Key, zero-padded.")
    founded: str | None = None


class IndexEvent(BaseModel):
    """A single addition or deletion event.

    For SP500 sourced from the Wikipedia "Selected changes" table, which lists
    only the *effective* date. Announcement date for SP500 additions is
    typically 3–5 trading days before effective date; we approximate by
    sourcing from S&P press release archives where available, otherwise back
    out from the effective date with a documented offset.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(..., description="Stable hash of (index, ticker, action, effective).")
    index: IndexName
    ticker: str
    action: Action
    effective_date: date
    announcement_date: date | None = Field(
        None, description="Press-release date if known; otherwise null."
    )
    reason: str | None = Field(None, description="Free-text reason from source.")
    source_url: str | None = None


class PriceBar(BaseModel):
    """Daily OHLCV bar.

    All fields are split- and dividend-adjusted unless explicitly noted in
    the caller. ``adj_close`` is dividend-adjusted close (yfinance's
    ``Adj Close``); ``close`` is raw last trade.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


class CARObservation(BaseModel):
    """One row of the event-study output: cumulative abnormal return for a
    single (event, window, model) triple."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    window_label: str = Field(..., description='e.g. "[T-A, T-E]" or "[T-A-5, T-A-1]".')
    model: Literal["market", "sector_matched"]
    car: float = Field(..., description="Cumulative abnormal return over the window, decimal.")
    n_days: int = Field(..., ge=1)
    benchmark_return: float = Field(..., description="Cumulative benchmark/control return.")


class LiquidityMetrics(BaseModel):
    """Pre-event liquidity diagnostics for one event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    adv_60d_shares: float = Field(..., description="Avg daily volume, 60d pre-announcement.")
    adv_60d_dollars: float
    corwin_schultz_spread: float | None = Field(
        None, description="Bid-ask spread proxy via Corwin-Schultz (2012); null if not estimable."
    )
    amihud_illiquidity: float | None = Field(
        None, description="Amihud (2002) ratio: mean(|R| / dollar_volume), 60d pre-event."
    )
    kyle_lambda: float | None = Field(
        None, description="Slope from ΔP regressed on signed volume over [T-A, T-E]."
    )


class TCAEstimate(BaseModel):
    """Hypothetical implementation-shortfall estimate for a passive fund."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    passive_aum_usd: float
    estimated_demand_usd: float
    forced_execution_cost_bps: float = Field(
        ..., description="Cost in bps if all rebalance flow hits T-E close."
    )
    spread_execution_cost_bps: float = Field(
        ..., description="Cost in bps if flow is split evenly across [T-A+1, T-E]."
    )
    savings_bps: float = Field(..., description="forced - spread; positive = spreading is cheaper.")


class CohortDecay(BaseModel):
    """One row of decay analysis output: average CAR for a year cohort."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cohort: str = Field(..., description='e.g. "2010-2014" or "2024".')
    n_events: int
    mean_car: float
    median_car: float
    car_p25: float
    car_p75: float


class UpcomingEvent(BaseModel):
    """One forward-looking event surfaced by the live monitor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: IndexName
    ticker: str
    action: Action
    announcement_date: date
    effective_date: date
    estimated_demand_usd: float | None = None
    source_url: str


class UpcomingEventsFile(BaseModel):
    """Schema for ``output/upcoming.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    as_of: datetime
    events: list[UpcomingEvent]


class EventResult(BaseModel):
    """Combined per-event payload: meta + CAR rows + liquidity + TCA.

    This is the unit of analysis the dashboard renders for any single
    addition or deletion event. ``car_observations`` may have multiple
    rows (one per window × model); ``liquidity`` and ``tca`` are
    null when the event was filtered out (insufficient pre-event data).
    """

    model_config = ConfigDict(extra="forbid")

    event: IndexEvent
    car_observations: list[CARObservation] = Field(default_factory=list)
    liquidity: LiquidityMetrics | None = None
    tca: TCAEstimate | None = None


class EventsFile(BaseModel):
    """Schema for ``output/events_<index>.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    as_of: datetime
    index: IndexName
    events: list[EventResult]


class DecayFile(BaseModel):
    """Schema for ``output/decay_<index>.json``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    schema_version: str = SCHEMA_VERSION
    as_of: datetime
    index: IndexName
    grouping: Literal["yearly", "biannual"]
    window_label: str
    model_used: Literal["market", "sector_matched"]
    action: Literal["add", "delete"]
    cohorts: list[CohortDecay]


class TCASummaryRow(BaseModel):
    """Annual aggregate of TCA estimates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int
    n_events: int
    total_demand_usd: float
    mean_forced_bps: float
    mean_spread_bps: float
    mean_savings_bps: float


class TCASummaryFile(BaseModel):
    """Schema for ``output/tca_summary.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    as_of: datetime
    index: IndexName
    passive_aum_usd: float
    annual: list[TCASummaryRow]


class MethodologyConstants(BaseModel):
    """Schema for ``output/methodology_constants.json``.

    Surfaces every assumption the dashboard renders. Keep this rigorously
    in sync with the analysis modules — drift here is silent corruption.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    passive_aum_usd: float
    market_model_estimation_window_days: int
    market_model_gap_days: int
    sector_match_pool_size: int
    decay_cohort_grouping: Literal["yearly", "biannual"]
    sources: dict[str, str] = Field(
        default_factory=dict, description='Citation key → URL, e.g. {"BrownWarner1985": "..."}.'
    )
