"""CLI smoke tests using typer's CliRunner.

We monkey-patch ``fetch_constituents`` and ``fetch_changes`` so the CLI runs
fully offline. The test verifies wiring (typer routing, output paths,
parquet writing), not parser correctness — that's covered in
test_data_parsing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from index_rebalance_tracker.cli import app
from index_rebalance_tracker.models import Constituent, IndexEvent

runner = CliRunner()


def _fake_constituents() -> list[Constituent]:
    return [
        Constituent(
            ticker="AAPL",
            name="Apple Inc.",
            gics_sector="Information Technology",
            gics_sub_industry="Technology Hardware, Storage and Peripherals",
            headquarters="Cupertino, California",
            date_added=date(1982, 11, 30),
            cik="0000320193",
            founded="1976",
        ),
        Constituent(
            ticker="MSFT",
            name="Microsoft Corp.",
            gics_sector="Information Technology",
            gics_sub_industry="Systems Software",
            headquarters="Redmond, Washington",
            date_added=date(1994, 6, 1),
            cik="0000789019",
            founded="1975",
        ),
    ]


def _fake_changes() -> list[IndexEvent]:
    return [
        IndexEvent(
            event_id="sp500-add-TSLA-2020-12-21-deadbeef",
            index="sp500",
            ticker="TSLA",
            action="add",
            effective_date=date(2020, 12, 21),
            announcement_date=None,
            reason="market cap inclusion",
            source_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        ),
        IndexEvent(
            event_id="sp500-delete-AIV-2020-12-21-cafebabe",
            index="sp500",
            ticker="AIV",
            action="delete",
            effective_date=date(2020, 12, 21),
            announcement_date=None,
            reason="replaced by TSLA",
            source_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        ),
    ]


def test_pull_history_sp500_writes_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "index_rebalance_tracker.cli.fetch_constituents", lambda: _fake_constituents()
    )
    monkeypatch.setattr(
        "index_rebalance_tracker.cli.fetch_changes", lambda start=None: _fake_changes()
    )

    result = runner.invoke(
        app,
        ["pull-history", "--index", "sp500", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output

    cons = pd.read_parquet(tmp_path / "constituents_sp500.parquet")
    assert list(cons["ticker"]) == ["AAPL", "MSFT"]
    assert cons.iloc[0]["gics_sector"] == "Information Technology"

    events = pd.read_parquet(tmp_path / "events_sp500.parquet")
    assert len(events) == 2
    assert set(events["action"]) == {"add", "delete"}


def test_pull_history_msci_sg_returns_exit_2(tmp_path: Path) -> None:
    """MSCI SG land in M5 — should exit 2 (not implemented), not 0."""
    result = runner.invoke(
        app, ["pull-history", "--index", "msci-sg", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 2


def test_pull_history_unknown_index_returns_exit_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["pull-history", "--index", "ftse100", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 2


def test_event_study_stub_returns_exit_2() -> None:
    result = runner.invoke(app, ["event-study"])
    assert result.exit_code == 2
    assert "M2" in result.output


def test_tca_stub_returns_exit_2() -> None:
    result = runner.invoke(app, ["tca"])
    assert result.exit_code == 2
    assert "M3" in result.output


def test_build_dashboard_stub_returns_exit_2() -> None:
    result = runner.invoke(app, ["build-dashboard"])
    assert result.exit_code == 2
    assert "M4" in result.output
