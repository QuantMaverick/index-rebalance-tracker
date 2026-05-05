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

import logging
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .analysis.event_study import run_event_study
from .analysis.liquidity import (
    amihud_illiquidity,
    corwin_schultz_spread,
    daily_return_vol,
    kyle_lambda,
)
from .analysis.sector_match import adv_dollars_60d
from .analysis.tca import estimated_demand_usd, implementation_shortfall_summary
from .data.prices import PriceCache
from .data.sp500_history import fetch_changes, fetch_constituents
from .models import IndexEvent, LiquidityMetrics, TCAEstimate

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
        console.print("[yellow]MSCI Singapore scraper lands in M5.[/yellow]")
        raise typer.Exit(code=2)
    else:
        console.print(f"[red]Unknown index: {index!r}[/red]")
        raise typer.Exit(code=2)


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
    price_start = (events_df["effective_date"].min() - pd.Timedelta(days=400)).date()
    price_end = events_df["effective_date"].max() + pd.Timedelta(days=60)
    price_end_d = price_end.date() if hasattr(price_end, "date") else price_end

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
    price_start = (events_df["effective_date"].min() - pd.Timedelta(days=120)).date()
    price_end = events_df["effective_date"].max() + pd.Timedelta(days=10)
    price_end_d = price_end.date() if hasattr(price_end, "date") else price_end

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
    emit_json: Annotated[Path, typer.Option("--emit-json")] = Path("output/upcoming.json"),
) -> None:
    """Live announcement scraper. Writes upcoming.json. (M5)"""
    _ = emit_json
    _stub("M5")


@app.command("build-dashboard")
def build_dashboard() -> None:
    """Run all analytics and write the full dashboard JSON contract. (M4)"""
    _stub("M4")


# ── helpers ─────────────────────────────────────────────────────────────────

ALLOWED_INDEX: tuple[Literal["sp500", "msci-sg"], ...] = ("sp500", "msci-sg")


def _parse_iso_date(s: str | None) -> date | None:
    if s is None:
        return None
    return date.fromisoformat(s)


if __name__ == "__main__":  # pragma: no cover
    app()
