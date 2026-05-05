"""Typer-based CLI entry point for the ``index-rebalance`` console script.

Subcommands:

* ``pull-history`` — scrape Wikipedia for SP500 (M1 wired) or MSCI SG (M5).
* ``event-study`` — wired in M2.
* ``tca`` — wired in M3.
* ``monitor`` — wired in M5.
* ``build-dashboard`` — wired in M4.

Only ``pull-history --index sp500`` is fully implemented at M1; other
commands return a clear "not yet implemented" exit code so CI never silently
green-lights an empty workflow.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .analysis.decay import cohort_decay
from .analysis.event_study import run_event_study
from .analysis.liquidity import (
    amihud_illiquidity,
    corwin_schultz_spread,
    daily_return_vol,
    kyle_lambda,
)
from .analysis.sector_match import adv_dollars_60d
from .analysis.tca import estimated_demand_usd, implementation_shortfall_summary
from .data.msci_sg import discover_review_urls, fetch_msci_sg_changes
from .data.prices import PriceCache
from .data.sp500_history import fetch_changes, fetch_constituents
from .export import (
    annual_tca_summary,
    build_event_results,
    write_decay_file,
    write_events_file,
    write_methodology_file,
    write_tca_summary_file,
)
from .models import (
    CARObservation,
    IndexEvent,
    LiquidityMetrics,
    TCAEstimate,
)
from .monitor.announcements import MSCI_REVIEWS_URL, fetch_upcoming

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Index Rebalance Tracker — event-study + TCA for SP500 and MSCI Singapore.",
)

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


# ── pull-history ────────────────────────────────────────────────────────────


@app.command("pull-history")
def pull_history(
    index: Annotated[str, typer.Option(help='Which index: "sp500" or "msci-sg" (M5).')] = "sp500",
    start: Annotated[
        str | None,
        typer.Option(help='Lower bound on effective_date, ISO format (e.g. "2010-01-01").'),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for the parquet outputs."),
    ] = Path("data"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Pull constituents + historical changes for the chosen index.

    Writes ``constituents_<index>.parquet`` and ``events_<index>.parquet`` to
    the output directory. Idempotent; safe to re-run.
    """
    _setup_logging(verbose=verbose)
    if index == "sp500":
        _pull_sp500(start_date=_parse_iso_date(start), output_dir=output_dir)
    elif index == "msci-sg":
        _pull_msci_sg(output_dir=output_dir, verbose=verbose)
    else:
        console.print(f"[red]Unknown index: {index!r}[/red]")
        raise typer.Exit(code=2)


def _pull_msci_sg(output_dir: Path, verbose: bool) -> None:
    """Best-effort MSCI Singapore history fetch.

    The free-data path (MSCI reviews page) does not expose constituent-
    level events. We surface what we can find and write an empty events
    parquet with a clear warning. Production deployments should wire a
    paid MSCI subscription here.
    """
    import httpx  # local import to keep cold-CLI import light

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print("[cyan]Fetching MSCI reviews index…[/cyan]")
    headers = {"User-Agent": "index-rebalance-tracker (+research)"}
    try:
        with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
            resp = client.get(MSCI_REVIEWS_URL)
            resp.raise_for_status()
            html = resp.text
    except (httpx.HTTPError, OSError, RuntimeError) as exc:  # noqa: BLE001
        console.print(f"[red]MSCI fetch failed: {type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    review_urls = discover_review_urls(html)
    console.print(f"  found {len(review_urls)} Singapore-mentioning review links")
    if verbose:
        for url in review_urls:
            console.print(f"  • {url}")

    events = fetch_msci_sg_changes(html)
    ev_df = pd.DataFrame(
        [e.model_dump() for e in events]
        or [{"event_id": "", "index": "", "ticker": "", "action": "", "effective_date": None}],
    ).iloc[: len(events)]  # iloc trims the placeholder row when events is empty
    ev_path = output_dir / "events_msci_sg.parquet"
    ev_df.to_parquet(ev_path)
    console.print(f"  wrote {len(ev_df)} events → {ev_path}")
    if len(ev_df) == 0:
        console.print(
            "[yellow]Note: free-data MSCI scraper cannot extract ticker-level events. "
            "See README §Limitations.[/yellow]"
        )


def _pull_sp500(start_date: date | None, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print("[cyan]Fetching constituents…[/cyan]")
    constituents = fetch_constituents()
    cons_df = pd.DataFrame([c.model_dump() for c in constituents])
    cons_path = output_dir / "constituents_sp500.parquet"
    cons_df.to_parquet(cons_path)
    console.print(f"  wrote {len(cons_df)} rows → {cons_path}")

    console.print("[cyan]Fetching changes…[/cyan]")
    events = fetch_changes(start=start_date)
    ev_df = pd.DataFrame([e.model_dump() for e in events])
    ev_path = output_dir / "events_sp500.parquet"
    ev_df.to_parquet(ev_path)
    console.print(f"  wrote {len(ev_df)} rows → {ev_path}")

    _print_summary(cons_df, ev_df)


def _print_summary(cons_df: pd.DataFrame, ev_df: pd.DataFrame) -> None:
    table = Table(title="SP500 history pull — summary", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Constituents (current)", str(len(cons_df)))
    table.add_row("Total events", str(len(ev_df)))
    if not ev_df.empty:
        table.add_row("Adds", str((ev_df["action"] == "add").sum()))
        table.add_row("Deletes", str((ev_df["action"] == "delete").sum()))
        table.add_row("Earliest event", str(ev_df["effective_date"].min()))
        table.add_row("Latest event", str(ev_df["effective_date"].max()))
    console.print(table)


# ── stubs for later milestones ──────────────────────────────────────────────


def _stub(milestone: str) -> None:
    console.print(f"[yellow]Not yet implemented — lands in milestone {milestone}.[/yellow]")
    raise typer.Exit(code=2)


@app.command("event-study")
def event_study(
    index: Annotated[str, typer.Option(help='Currently only "sp500" supported.')] = "sp500",
    window_pre: Annotated[int, typer.Option(help="Days before announcement.")] = 5,
    window_post: Annotated[int, typer.Option(help="Days after effective.")] = 20,
    model: Annotated[str, typer.Option(help='"market" | "sector_matched" | "both"')] = "both",
    start: Annotated[
        str | None,
        typer.Option(help="Filter events with effective_date >= this ISO date."),
    ] = "2010-01-01",
    data_dir: Annotated[Path, typer.Option(help="Where M1 wrote parquets.")] = Path("data"),
    output_dir: Annotated[Path, typer.Option(help="CAR observations output dir.")] = Path("output"),
    market_proxy: Annotated[str, typer.Option(help="Market-return ticker.")] = "SPY",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the event study and write CAR observations.

    Reads ``data/{events,constituents}_sp500.parquet`` (from
    ``pull-history``), fetches prices for every event ticker plus the market
    proxy via the cached :class:`PriceCache`, computes daily simple returns,
    runs the chosen model(s), and writes a parquet of ``CARObservation``
    rows to ``output_dir/car_<index>.parquet``.

    Methodology surfaced in the output JSON contract; see
    :mod:`index_rebalance_tracker.analysis.event_study` and the
    Methodology page for the underlying derivations.
    """
    _setup_logging(verbose=verbose)
    if index != "sp500":
        console.print(
            f"[yellow]event-study currently only supports sp500. (got {index!r})[/yellow]"
        )
        raise typer.Exit(code=2)
    if model not in {"market", "sector_matched", "both"}:
        console.print(f"[red]Unknown model: {model!r}[/red]")
        raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    start_date = _parse_iso_date(start)

    console.print("[cyan]Loading M1 outputs…[/cyan]")
    events_df = pd.read_parquet(data_dir / "events_sp500.parquet")
    cons_df = pd.read_parquet(data_dir / "constituents_sp500.parquet")
    sector_map = dict(zip(cons_df["ticker"], cons_df["gics_sector"], strict=False))

    if start_date is not None:
        events_df = events_df[events_df["effective_date"] >= start_date]

    events = [IndexEvent.model_validate(row) for row in events_df.to_dict("records")]
    console.print(f"  {len(events)} events to study (model={model})")

    console.print(f"[cyan]Fetching prices for {len(events)} events + {market_proxy} proxy…[/cyan]")
    cache = PriceCache()
    price_start = _to_date(events_df["effective_date"].min()) - timedelta(days=400)
    price_end_d = _to_date(events_df["effective_date"].max()) + timedelta(days=60)

    market_df = cache.fetch_prices(market_proxy, price_start, price_end_d)
    market_returns = market_df["adj_close"].pct_change().rename("r_m")
    market_returns.index = pd.to_datetime(market_returns.index)

    universe_tickers = sorted({event.ticker for event in events} | set(cons_df["ticker"]))
    returns: dict[str, pd.Series] = {}
    prices_for_adv: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(universe_tickers):
        if i % 50 == 0:
            console.print(f"  fetched {i}/{len(universe_tickers)} tickers")
        try:
            df = cache.fetch_prices(ticker, price_start, price_end_d)
        except Exception as exc:  # noqa: BLE001 — yfinance raises mixed types
            logger = logging.getLogger(__name__)
            logger.warning("price fetch failed for %s: %s", ticker, exc)
            continue
        if df.empty:
            continue
        prices_for_adv[ticker] = df
        s = df["adj_close"].pct_change()
        s.index = pd.to_datetime(s.index)
        returns[ticker] = s

    adv_map = adv_dollars_60d(prices_for_adv)
    console.print(f"  ADV computed for {len(adv_map)} tickers")

    console.print("[cyan]Running event study…[/cyan]")
    obs = run_event_study(
        events=events,
        returns=returns,
        market_returns=market_returns,
        sector_map=sector_map,
        adv_dollars_60d=adv_map,
        model=model,  # type: ignore[arg-type]
        pre_run_up_days=window_pre,
        post_drift_days=window_post,
    )
    console.print(f"  {len(obs)} CAR observations")
    out_df = pd.DataFrame([o.model_dump() for o in obs])
    out_path = output_dir / f"car_{index}.parquet"
    out_df.to_parquet(out_path)
    console.print(f"  wrote {out_path}")


@app.command("tca")
def tca(
    index: Annotated[str, typer.Option(help='Currently only "sp500" supported.')] = "sp500",
    aum_billions: Annotated[
        float, typer.Option(help="Passive AUM in USD billions (~6500 for SP500 in 2024).")
    ] = 6500.0,
    spread_days: Annotated[int, typer.Option(help="Days the spread-execution path uses.")] = 5,
    data_dir: Annotated[Path, typer.Option(help="Where M1 wrote parquets.")] = Path("data"),
    output_dir: Annotated[Path, typer.Option(help="Output dir for liquidity + TCA.")] = Path(
        "output"
    ),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Compute liquidity diagnostics + TCA implementation shortfall per event.

    Writes ``output_dir/liquidity_<index>.parquet`` and
    ``output_dir/tca_<index>.parquet``. The estimated demand for each
    event uses the latest available price × shares-outstanding-proxy
    against the sum of all constituents' market caps as the index
    denominator.
    """
    _setup_logging(verbose=verbose)
    if index != "sp500":
        console.print(f"[yellow]tca currently only supports sp500. (got {index!r})[/yellow]")
        raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    passive_aum_usd = aum_billions * 1e9

    console.print("[cyan]Loading M1 outputs…[/cyan]")
    events_df = pd.read_parquet(data_dir / "events_sp500.parquet")
    cons_df = pd.read_parquet(data_dir / "constituents_sp500.parquet")
    events = [IndexEvent.model_validate(row) for row in events_df.to_dict("records")]
    console.print(f"  {len(events)} events; {len(cons_df)} current constituents")

    cache = PriceCache()
    universe_tickers = sorted({e.ticker for e in events} | set(cons_df["ticker"]))
    price_start = _to_date(events_df["effective_date"].min()) - timedelta(days=120)
    price_end_d = _to_date(events_df["effective_date"].max()) + timedelta(days=10)

    console.print("[cyan]Fetching prices for universe…[/cyan]")
    prices: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(universe_tickers):
        if i % 50 == 0:
            console.print(f"  fetched {i}/{len(universe_tickers)} tickers")
        try:
            df = cache.fetch_prices(ticker, price_start, price_end_d)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("price fetch failed for %s: %s", ticker, exc)
            continue
        if not df.empty:
            prices[ticker] = df

    # Aggregate index market cap via close × volume-as-shares-proxy is wrong;
    # instead use sum of (close × constant_shares_proxy) — simplest defensible
    # proxy: assume each constituent contributes equally to index cap. This
    # is rough; the M4 notebook documents the limitation.
    # For the demand estimate we use stock_cap = close × adv_60d / close_avg
    # as a "size proxy" — really we want shares_outstanding which we don't have.
    # Compromise: use 60-day ADV-dollars as a relative size proxy.
    adv_dollars_map = adv_dollars_60d(prices)
    total_adv = sum(adv_dollars_map.values())

    console.print("[cyan]Computing per-event liquidity + TCA…[/cyan]")
    liq_rows: list[LiquidityMetrics] = []
    tca_rows: list[TCAEstimate] = []
    for event in events:
        if event.ticker not in prices:
            continue
        df = prices[event.ticker]
        adv_d = adv_dollars_map.get(event.ticker, 0.0)
        # Rough shares-ADV: dollar ADV ÷ recent close
        recent_close = float(df["close"].iloc[-1]) if not df.empty else 0.0
        adv_shares = adv_d / recent_close if recent_close > 0 else 0.0

        liq = LiquidityMetrics(
            event_id=event.event_id,
            adv_60d_shares=adv_shares,
            adv_60d_dollars=adv_d,
            corwin_schultz_spread=_finite_or_none(corwin_schultz_spread(df)),
            amihud_illiquidity=_finite_or_none(amihud_illiquidity(df)),
            kyle_lambda=_finite_or_none(kyle_lambda(df, window=60)),
        )
        liq_rows.append(liq)

        if total_adv <= 0 or adv_d <= 0:
            continue
        # Use ADV-share-of-index as float-weight proxy
        demand = estimated_demand_usd(passive_aum_usd, adv_d, total_adv)
        vol = daily_return_vol(df, window=60)
        if vol != vol or demand != demand:  # NaN guard
            continue
        summary = implementation_shortfall_summary(demand, adv_d, vol, spread_n_days=spread_days)
        if summary["forced_bps"] != summary["forced_bps"]:
            continue
        tca_rows.append(
            TCAEstimate(
                event_id=event.event_id,
                passive_aum_usd=passive_aum_usd,
                estimated_demand_usd=demand,
                forced_execution_cost_bps=summary["forced_bps"],
                spread_execution_cost_bps=summary["spread_bps"],
                savings_bps=summary["savings_bps"],
            )
        )

    liq_df = pd.DataFrame([r.model_dump() for r in liq_rows])
    tca_df = pd.DataFrame([r.model_dump() for r in tca_rows])
    liq_path = output_dir / f"liquidity_{index}.parquet"
    tca_path = output_dir / f"tca_{index}.parquet"
    liq_df.to_parquet(liq_path)
    tca_df.to_parquet(tca_path)
    console.print(f"  wrote {len(liq_df)} liquidity rows → {liq_path}")
    console.print(f"  wrote {len(tca_df)} TCA rows → {tca_path}")


def _finite_or_none(x: float) -> float | None:
    """Return None for NaN / inf so pydantic Optional fields serialize as null."""
    import math  # noqa: PLC0415

    return None if math.isnan(x) or math.isinf(x) else x


@app.command("monitor")
def monitor(
    emit_json: Annotated[
        Path, typer.Option("--emit-json", help="Where to write the upcoming.json contract.")
    ] = Path("output/upcoming.json"),
    backward_days: Annotated[
        int, typer.Option(help="Days back from now to include in the window.")
    ] = 14,
    forward_days: Annotated[
        int, typer.Option(help="Days forward from now to include in the window.")
    ] = 30,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the live announcement scraper and write ``upcoming.json``.

    Concurrent ``httpx`` async fetches for both SP500 (Wikipedia) and
    MSCI Singapore (best-effort). One scraper failing does not block
    the other — the output file is always written, even if it contains
    only one source's events.

    Schema is :class:`UpcomingEventsFile`. Cron-friendly: idempotent and
    safe to run hourly.
    """
    _setup_logging(verbose=verbose)
    emit_json.parent.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[cyan]Scraping window: [{backward_days}d back, {forward_days}d forward]…[/cyan]"
    )
    payload = asyncio.run(fetch_upcoming(backward_days=backward_days, forward_days=forward_days))
    emit_json.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]✓ wrote {len(payload.events)} upcoming events → {emit_json}[/green]")
    if len(payload.events) == 0:
        console.print(
            "[yellow]Empty result is normal between announcement cycles. "
            "Check back during the SP500 quarterly review windows (March/June/Sept/Dec).[/yellow]"
        )


@app.command("build-dashboard")
def build_dashboard(
    aum_billions: Annotated[
        float, typer.Option(help="Passive AUM in USD billions for TCA estimates.")
    ] = 6500.0,
    spread_days: Annotated[int, typer.Option(help="Days the spread-execution path uses.")] = 5,
    start: Annotated[
        str | None,
        typer.Option(help="Filter events with effective_date >= this ISO date."),
    ] = "2010-01-01",
    data_dir: Annotated[Path, typer.Option(help="Where M1 wrote parquets.")] = Path("data"),
    output_dir: Annotated[Path, typer.Option(help="Dashboard JSON output dir.")] = Path("output"),
    market_proxy: Annotated[str, typer.Option(help="Market-return ticker.")] = "SPY",
    cohort_grouping: Annotated[
        str, typer.Option(help='"yearly" or "biannual" cohort buckets.')
    ] = "yearly",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """End-to-end runner: scrape → fetch → analyze → write dashboard JSON.

    Reads M1 parquets, fetches prices for the constituent universe via
    the cached :class:`PriceCache`, runs the M2 event study + M3
    liquidity/TCA analysis, computes the M4 cohort decay, and writes:

    * ``events_sp500.json``           — combined per-event payload
    * ``decay_sp500.json``            — cohort-aggregated CAR
    * ``tca_summary.json``            — annual TCA roll-up
    * ``methodology_constants.json``  — assumptions surfaced for the dashboard

    The downstream dashboard project validates each file against the
    pydantic models in :mod:`index_rebalance_tracker.models`.
    """
    _setup_logging(verbose=verbose)
    if cohort_grouping not in {"yearly", "biannual"}:
        console.print(f"[red]Unknown cohort grouping: {cohort_grouping!r}[/red]")
        raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    passive_aum_usd = aum_billions * 1e9
    start_date = _parse_iso_date(start)

    # ── M1 parquets ────────────────────────────────────────────────────────
    console.print("[cyan]Loading M1 outputs…[/cyan]")
    events_df = pd.read_parquet(data_dir / "events_sp500.parquet")
    cons_df = pd.read_parquet(data_dir / "constituents_sp500.parquet")
    sector_map = dict(zip(cons_df["ticker"], cons_df["gics_sector"], strict=False))
    if start_date is not None:
        events_df = events_df[events_df["effective_date"] >= start_date]
    events = [IndexEvent.model_validate(row) for row in events_df.to_dict("records")]
    events_by_id = {e.event_id: e for e in events}
    console.print(f"  {len(events)} events × full price universe")

    # ── price universe fetch ───────────────────────────────────────────────
    console.print("[cyan]Fetching prices for universe + market proxy…[/cyan]")
    cache = PriceCache()
    universe_tickers = sorted({e.ticker for e in events} | set(cons_df["ticker"]))
    price_start = _to_date(events_df["effective_date"].min()) - timedelta(days=400)
    price_end_d = _to_date(events_df["effective_date"].max()) + timedelta(days=60)

    market_df = cache.fetch_prices(market_proxy, price_start, price_end_d)
    market_returns = market_df["adj_close"].pct_change().rename("r_m")
    market_returns.index = pd.to_datetime(market_returns.index)

    returns: dict[str, pd.Series] = {}
    prices: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(universe_tickers):
        if i % 50 == 0:
            console.print(f"  fetched {i}/{len(universe_tickers)}")
        try:
            df = cache.fetch_prices(ticker, price_start, price_end_d)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("price fetch failed for %s: %s", ticker, exc)
            continue
        if df.empty:
            continue
        prices[ticker] = df
        s = df["adj_close"].pct_change()
        s.index = pd.to_datetime(s.index)
        returns[ticker] = s

    adv_map = adv_dollars_60d(prices)
    total_adv = sum(adv_map.values())
    console.print(f"  ADV computed for {len(adv_map)} tickers")

    # ── M2: event study ────────────────────────────────────────────────────
    console.print("[cyan]Running event study (market + sector_matched)…[/cyan]")
    car_obs: list[CARObservation] = run_event_study(
        events=events,
        returns=returns,
        market_returns=market_returns,
        sector_map=sector_map,
        adv_dollars_60d=adv_map,
        model="both",
        pre_run_up_days=5,
        post_drift_days=20,
    )
    console.print(f"  {len(car_obs)} CAR observations")

    # ── M3: liquidity + TCA ────────────────────────────────────────────────
    console.print("[cyan]Running liquidity + TCA…[/cyan]")
    liq_rows: list[LiquidityMetrics] = []
    tca_rows: list[TCAEstimate] = []
    for event in events:
        if event.ticker not in prices:
            continue
        df = prices[event.ticker]
        adv_d = adv_map.get(event.ticker, 0.0)
        recent_close = float(df["close"].iloc[-1]) if not df.empty else 0.0
        adv_shares = adv_d / recent_close if recent_close > 0 else 0.0

        liq_rows.append(
            LiquidityMetrics(
                event_id=event.event_id,
                adv_60d_shares=adv_shares,
                adv_60d_dollars=adv_d,
                corwin_schultz_spread=_finite_or_none(corwin_schultz_spread(df)),
                amihud_illiquidity=_finite_or_none(amihud_illiquidity(df)),
                kyle_lambda=_finite_or_none(kyle_lambda(df, window=60)),
            )
        )
        if total_adv <= 0 or adv_d <= 0:
            continue
        demand = estimated_demand_usd(passive_aum_usd, adv_d, total_adv)
        vol = daily_return_vol(df, window=60)
        if vol != vol or demand != demand:
            continue
        shortfall = implementation_shortfall_summary(demand, adv_d, vol, spread_n_days=spread_days)
        if shortfall["forced_bps"] != shortfall["forced_bps"]:
            continue
        tca_rows.append(
            TCAEstimate(
                event_id=event.event_id,
                passive_aum_usd=passive_aum_usd,
                estimated_demand_usd=demand,
                forced_execution_cost_bps=shortfall["forced_bps"],
                spread_execution_cost_bps=shortfall["spread_bps"],
                savings_bps=shortfall["savings_bps"],
            )
        )
    console.print(f"  liquidity rows: {len(liq_rows)}, TCA rows: {len(tca_rows)}")

    # ── M4: cohort decay ───────────────────────────────────────────────────
    console.print("[cyan]Aggregating decay cohorts…[/cyan]")
    cohorts = cohort_decay(
        car_obs,
        events_by_id,
        grouping=cohort_grouping,  # type: ignore[arg-type]
        window_label="[T-A, T-E-1]",
        model="market",
        action="add",
    )
    console.print(f"  {len(cohorts)} cohorts")

    # ── write JSON contract ────────────────────────────────────────────────
    console.print("[cyan]Writing dashboard JSON…[/cyan]")
    event_results = build_event_results(events, car_obs, liq_rows, tca_rows)
    write_events_file(output_dir / "events_sp500.json", index="sp500", events=event_results)
    write_decay_file(
        output_dir / "decay_sp500.json",
        index="sp500",
        cohorts=cohorts,
        grouping=cohort_grouping,
        window_label="[T-A, T-E-1]",
        model_used="market",
        action="add",
    )
    annual = annual_tca_summary(tca_rows, events_by_id)
    write_tca_summary_file(
        output_dir / "tca_summary.json",
        index="sp500",
        passive_aum_usd=passive_aum_usd,
        annual=annual,
    )
    write_methodology_file(
        output_dir / "methodology_constants.json",
        passive_aum_usd=passive_aum_usd,
        decay_cohort_grouping=cohort_grouping,
    )
    console.print("[green]✓ Dashboard JSON contract written.[/green]")


# ── helpers ─────────────────────────────────────────────────────────────────

ALLOWED_INDEX: tuple[Literal["sp500", "msci-sg"], ...] = ("sp500", "msci-sg")


def _parse_iso_date(s: str | None) -> date | None:
    if s is None:
        return None
    return date.fromisoformat(s)


def _to_date(x: object) -> date:
    """Normalize either ``pd.Timestamp``, ``datetime``, or ``date`` to ``date``."""
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    return pd.Timestamp(x).date()  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover
    app()
