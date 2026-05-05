"""Live announcement monitor.

Two scrapers run side-by-side, both async via ``httpx``:

* **SP500** — sources the Wikipedia "Selected changes" table, filters to
  events whose effective date falls in a configurable forward / backward
  window. Wikipedia typically updates within 24h of an S&P press release
  and is the most stable free source. The table carries effective date
  only; we approximate ``announcement_date`` as ``effective_date − 5``
  calendar days, matching the typical S&P press-release lead time.

* **MSCI Singapore** — sources the MSCI quarterly review page. This
  scraper is *brittle by design*: MSCI's review pages don't expose a
  stable HTML structure, link patterns change between quarters, and
  much of the data lives in PDFs. This module returns best-effort
  results and logs a clear warning when parsing fails — the caller
  receives an empty list, never a stale or hallucinated event.

The two scrapers run concurrently with ``asyncio.gather`` so a slow
MSCI fetch doesn't block the SP500 path. Both share an ``httpx``
``AsyncClient`` for connection reuse.

Output schema is :class:`UpcomingEventsFile` in ``models.py``; the
downstream dashboard validates against that.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime, timedelta

import httpx
from bs4 import BeautifulSoup, Tag

from ..data.sp500_history import (
    USER_AGENT,
    WIKIPEDIA_URL,
    _clean_ticker,
    _parse_date,
)
from ..models import IndexName, UpcomingEvent, UpcomingEventsFile

logger = logging.getLogger(__name__)

MSCI_REVIEWS_URL = (
    "https://www.msci.com/our-solutions/index-quality-policies-research/index-reviews"
)
DEFAULT_WINDOW_BACKWARD_DAYS = 14
DEFAULT_WINDOW_FORWARD_DAYS = 30
DEFAULT_ANNOUNCEMENT_OFFSET = 5


async def fetch_upcoming(
    *,
    backward_days: int = DEFAULT_WINDOW_BACKWARD_DAYS,
    forward_days: int = DEFAULT_WINDOW_FORWARD_DAYS,
    timeout: float = 30.0,
    nowts: datetime | None = None,
) -> UpcomingEventsFile:
    """Run both scrapers concurrently and return the merged file payload.

    Args:
        backward_days: Include effective dates back N days from now (catches
            announcements that already came out and are about to be
            effective).
        forward_days: Include effective dates up to N days in the future.
        timeout: HTTPX request timeout per call.
        nowts: Override "now" for deterministic tests (default: UTC now).

    Returns:
        A populated :class:`UpcomingEventsFile`. Empty event list on
        scraper failure rather than raising.
    """
    now = nowts or datetime.utcnow()
    cutoff_back = now.date() - timedelta(days=backward_days)
    cutoff_forward = now.date() + timedelta(days=forward_days)

    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        sp500_task = asyncio.create_task(_fetch_sp500_upcoming(client, cutoff_back, cutoff_forward))
        msci_task = asyncio.create_task(
            _fetch_msci_sg_upcoming(client, cutoff_back, cutoff_forward)
        )
        results = await asyncio.gather(sp500_task, msci_task, return_exceptions=True)

    events: list[UpcomingEvent] = []
    for source, result in zip(("sp500", "msci_sg"), results, strict=False):
        if isinstance(result, BaseException):
            logger.warning("%s scraper failed: %s: %s", source, type(result).__name__, result)
            continue
        events.extend(result)

    return UpcomingEventsFile(as_of=now, events=events)


# ── SP500 (Wikipedia) ──────────────────────────────────────────────────────


async def _fetch_sp500_upcoming(
    client: httpx.AsyncClient,
    cutoff_back: date,
    cutoff_forward: date,
) -> list[UpcomingEvent]:
    """Scrape Wikipedia 'Selected changes' for events in the window."""
    try:
        resp = await client.get(WIKIPEDIA_URL)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Wikipedia fetch failed: %s", exc)
        return []
    return parse_sp500_upcoming(resp.text, cutoff_back, cutoff_forward)


def parse_sp500_upcoming(html: str, cutoff_back: date, cutoff_forward: date) -> list[UpcomingEvent]:
    """Parse the changes table and filter to the upcoming window.

    Args:
        html: Raw Wikipedia HTML containing ``id="changes"``.
        cutoff_back: Lower bound on effective_date.
        cutoff_forward: Upper bound on effective_date.

    Returns:
        Sorted list of :class:`UpcomingEvent`. Empty if no events in window.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="changes")
    if not isinstance(table, Tag):
        logger.warning("Wikipedia: changes table not found")
        return []

    out: list[UpcomingEvent] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 5:
            continue
        eff_date = _parse_date(cells[0].get_text(strip=True))
        if eff_date is None:
            continue
        if not (cutoff_back <= eff_date <= cutoff_forward):
            continue

        announcement = eff_date - timedelta(days=DEFAULT_ANNOUNCEMENT_OFFSET)
        added_ticker = _clean_ticker(cells[1].get_text(strip=True))
        removed_ticker = _clean_ticker(cells[3].get_text(strip=True))

        if added_ticker:
            out.append(
                _make_upcoming("sp500", added_ticker, "add", announcement, eff_date, WIKIPEDIA_URL)
            )
        if removed_ticker:
            out.append(
                _make_upcoming(
                    "sp500", removed_ticker, "delete", announcement, eff_date, WIKIPEDIA_URL
                )
            )

    out.sort(key=lambda e: (e.effective_date, e.ticker))
    return out


# ── MSCI Singapore ─────────────────────────────────────────────────────────


async def _fetch_msci_sg_upcoming(
    client: httpx.AsyncClient,
    cutoff_back: date,
    cutoff_forward: date,
) -> list[UpcomingEvent]:
    """Scrape MSCI quarterly reviews page.

    MSCI publishes review summaries as PDFs; the index page links to them
    but has no stable structure or schema. We do a best-effort parse of
    visible review-link text, returning an empty list (with a warning)
    when we cannot find dated review entries.

    The honest engineering note: a production deployment would replace
    this with a paid MSCI data subscription or a vendor like Solactive /
    FactSet that publishes structured event tables.
    """
    try:
        resp = await client.get(MSCI_REVIEWS_URL)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("MSCI reviews fetch failed: %s", exc)
        return []
    return parse_msci_sg_upcoming(resp.text, cutoff_back, cutoff_forward)


_MSCI_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},\s*\d{4}\b"
)


def parse_msci_sg_upcoming(
    html: str, cutoff_back: date, cutoff_forward: date
) -> list[UpcomingEvent]:
    """Best-effort scrape of the MSCI reviews page.

    Looks for links whose anchor text mentions "Singapore" and parses
    any date strings nearby. Because MSCI's HTML structure is unstable
    and the actual constituent changes typically live inside linked
    PDFs, this parser:

    * Returns an empty list if no Singapore-mentioning entries are
      found in the configured date window.
    * Does NOT attempt to extract individual ticker-level events from
      PDFs — that requires a brittle PDF parsing pipeline outside the
      scope of this free-tier scraper.
    * Logs a clear warning so operators know they need to manually
      verify upcoming MSCI Singapore events.

    Returns:
        List of :class:`UpcomingEvent`. Will return an empty list with
        a warning logged in the typical case.
    """
    soup = BeautifulSoup(html, "lxml")
    sg_anchors = [
        a for a in soup.find_all("a") if "singapore" in a.get_text(" ", strip=True).lower()
    ]
    if not sg_anchors:
        logger.warning(
            "MSCI: no Singapore-mentioning links found on reviews page — "
            "MSCI HTML may have changed; manual verification required for upcoming MSCI SG events."
        )
        return []

    out: list[UpcomingEvent] = []
    for anchor in sg_anchors:
        text = anchor.get_text(" ", strip=True)
        date_match = _MSCI_DATE_RE.search(text)
        if not date_match:
            continue
        eff_date = _parse_date(date_match.group(0))
        if eff_date is None or not (cutoff_back <= eff_date <= cutoff_forward):
            continue
        href = anchor.get("href", "")
        source_url = href if href.startswith("http") else f"https://www.msci.com{href}"
        # Cannot extract individual tickers from index page; emit a placeholder
        # event so the dashboard surfaces "MSCI SG review on date X" without
        # claiming specific add/delete tickers we can't verify.
        out.append(
            _make_upcoming(
                "msci_sg",
                "REVIEW",
                "add",  # placeholder; the actual review may include both
                eff_date - timedelta(days=14),  # MSCI typically announces 1–2 weeks ahead
                eff_date,
                source_url,
            )
        )

    if not out:
        logger.warning(
            "MSCI: Singapore links found but no parseable dates in window "
            "[%s, %s] — review HTML structure may have changed.",
            cutoff_back,
            cutoff_forward,
        )

    out.sort(key=lambda e: (e.effective_date, e.ticker))
    return out


# ── helpers ────────────────────────────────────────────────────────────────


def _make_upcoming(
    index: IndexName,
    ticker: str,
    action: str,
    announcement_date: date,
    effective_date: date,
    source_url: str,
) -> UpcomingEvent:
    return UpcomingEvent(
        index=index,
        ticker=ticker,
        action=action,  # type: ignore[arg-type]
        announcement_date=announcement_date,
        effective_date=effective_date,
        estimated_demand_usd=None,  # computed downstream when prices arrive
        source_url=source_url,
    )
