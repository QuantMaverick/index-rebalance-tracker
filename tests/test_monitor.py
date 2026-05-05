"""Tests for the live announcement monitor.

The async ``fetch_upcoming`` is exercised via :class:`httpx.MockTransport`
so unit tests run offline. The pure parser functions take HTML strings
directly.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from index_rebalance_tracker.monitor.announcements import (
    fetch_upcoming,
    parse_msci_sg_upcoming,
    parse_sp500_upcoming,
)

# ── parse_sp500_upcoming on the committed fixture ──────────────────────────


def test_parse_sp500_upcoming_filters_to_window(sp500_wikipedia_html: str) -> None:
    """Constrain to 2024 — should pick up 2024 events only."""
    events = parse_sp500_upcoming(sp500_wikipedia_html, date(2024, 1, 1), date(2024, 12, 31))
    for e in events:
        assert date(2024, 1, 1) <= e.effective_date <= date(2024, 12, 31)
    assert events, "expected at least one 2024 event in the committed fixture"


def test_parse_sp500_upcoming_uses_default_announcement_offset(
    sp500_wikipedia_html: str,
) -> None:
    """Wikipedia carries only effective_date; we approximate
    announcement_date as effective_date − 5 days."""
    events = parse_sp500_upcoming(sp500_wikipedia_html, date(2024, 1, 1), date(2024, 12, 31))
    sample = events[0]
    assert sample.announcement_date == sample.effective_date - timedelta(days=5)


def test_parse_sp500_upcoming_returns_sp500_index(sp500_wikipedia_html: str) -> None:
    events = parse_sp500_upcoming(sp500_wikipedia_html, date(2024, 1, 1), date(2024, 12, 31))
    assert all(e.index == "sp500" for e in events)


def test_parse_sp500_upcoming_returns_empty_outside_window(sp500_wikipedia_html: str) -> None:
    """Wikipedia 'Selected changes' rarely has 1980 entries; expect empty."""
    events = parse_sp500_upcoming(sp500_wikipedia_html, date(1980, 1, 1), date(1980, 12, 31))
    assert events == []


def test_parse_sp500_upcoming_handles_malformed_html() -> None:
    """No 'changes' table → empty list, no crash."""
    events = parse_sp500_upcoming(
        "<html><body>No table here</body></html>", date(2024, 1, 1), date(2024, 12, 31)
    )
    assert events == []


# ── parse_msci_sg_upcoming ─────────────────────────────────────────────────


def test_parse_msci_sg_upcoming_returns_empty_when_no_singapore_links() -> None:
    """Page without Singapore-mentioning links → [] with logged warning."""
    html = "<html><body><a href='/foo'>Index Reviews</a></body></html>"
    events = parse_msci_sg_upcoming(html, date(2024, 1, 1), date(2024, 12, 31))
    assert events == []


def test_parse_msci_sg_upcoming_extracts_dated_singapore_link() -> None:
    """Anchor mentions Singapore + a date in window → one placeholder event."""
    html = """
    <html><body>
      <a href="/x.pdf">MSCI Singapore Quarterly Review November 30, 2024</a>
      <a href="/y.pdf">MSCI Japan Review</a>
    </body></html>
    """
    events = parse_msci_sg_upcoming(html, date(2024, 11, 1), date(2024, 12, 31))
    assert len(events) == 1
    e = events[0]
    assert e.index == "msci_sg"
    assert e.effective_date == date(2024, 11, 30)
    # Announcement is approximated as 14 days before effective for MSCI
    assert e.announcement_date == date(2024, 11, 30) - timedelta(days=14)


def test_parse_msci_sg_upcoming_filters_to_window() -> None:
    html = """
    <html><body>
      <a href="/old.pdf">MSCI Singapore Review January 15, 2020</a>
      <a href="/new.pdf">MSCI Singapore Review November 30, 2024</a>
    </body></html>
    """
    events = parse_msci_sg_upcoming(html, date(2024, 1, 1), date(2025, 6, 30))
    assert len(events) == 1
    assert events[0].effective_date.year == 2024


# ── fetch_upcoming end-to-end with mocked HTTP ─────────────────────────────


def _make_mock_transport(*, wiki_html: str = "", msci_html: str = "") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "wikipedia" in url:
            return httpx.Response(200, text=wiki_html)
        if "msci.com" in url:
            return httpx.Response(200, text=msci_html)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
@pytest.mark.mock_http
async def test_fetch_upcoming_happy_path(
    sp500_wikipedia_html: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns a populated EventsFile with both source paths fired."""
    transport = _make_mock_transport(wiki_html=sp500_wikipedia_html, msci_html="")

    real_async_client = httpx.AsyncClient

    def patched(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("index_rebalance_tracker.monitor.announcements.httpx.AsyncClient", patched)
    payload = await fetch_upcoming(
        backward_days=365 * 5,
        forward_days=0,
        nowts=datetime(2024, 12, 31),
    )
    assert payload.schema_version
    assert isinstance(payload.as_of, datetime)
    # SP500 events should land; MSCI returns empty
    assert any(e.index == "sp500" for e in payload.events)


@pytest.mark.asyncio
@pytest.mark.mock_http
async def test_fetch_upcoming_one_scraper_failing_does_not_blank_the_other(
    sp500_wikipedia_html: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 500 on MSCI must not block the SP500 path."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "msci.com" in str(request.url):
            return httpx.Response(500)
        return httpx.Response(200, text=sp500_wikipedia_html)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("index_rebalance_tracker.monitor.announcements.httpx.AsyncClient", patched)
    payload = await fetch_upcoming(
        backward_days=365 * 5,
        forward_days=0,
        nowts=datetime(2024, 12, 31),
    )
    assert any(e.index == "sp500" for e in payload.events)
    # No MSCI events because the fetch failed
    assert not any(e.index == "msci_sg" for e in payload.events)


@pytest.mark.asyncio
@pytest.mark.mock_http
async def test_fetch_upcoming_writes_valid_schema_on_empty_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even with zero events, the file must validate against UpcomingEventsFile."""
    transport = _make_mock_transport(wiki_html="<html></html>", msci_html="<html></html>")
    real_async_client = httpx.AsyncClient

    def patched(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("index_rebalance_tracker.monitor.announcements.httpx.AsyncClient", patched)
    payload = await fetch_upcoming(backward_days=14, forward_days=30)
    out = tmp_path / "upcoming.json"
    out.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    from index_rebalance_tracker.models import UpcomingEventsFile  # noqa: PLC0415

    parsed = UpcomingEventsFile.model_validate_json(out.read_text())
    assert parsed.events == []
