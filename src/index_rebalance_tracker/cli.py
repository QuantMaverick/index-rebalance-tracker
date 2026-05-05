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

from .data.sp500_history import fetch_changes, fetch_constituents

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
    index: Annotated[str, typer.Option()] = "sp500",
    window_pre: Annotated[int, typer.Option(help="Days before announcement.")] = 5,
    window_post: Annotated[int, typer.Option(help="Days after effective.")] = 20,
    model: Annotated[str, typer.Option(help='"market" | "sector_matched" | "both"')] = "both",
) -> None:
    """Run the event study and write CAR observations. (M2)"""
    _ = (index, window_pre, window_post, model)
    _stub("M2")


@app.command("tca")
def tca(
    index: Annotated[str, typer.Option()] = "sp500",
    aum_billions: Annotated[float, typer.Option(help="Passive AUM in USD billions.")] = 6500.0,
) -> None:
    """Compute TCA estimates for the hypothetical passive fund. (M3)"""
    _ = (index, aum_billions)
    _stub("M3")


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
