"""Wikipedia scraper for S&P 500 constituents and additions/deletions history.

The "List of S&P 500 companies" Wikipedia article maintains two tables:

* ``id="constituents"`` — current members with GICS sector + sub-industry.
* ``id="changes"`` — chronological additions and deletions, typically
  back to ~1995. The table has a nested-header structure: top-level
  columns are ``Effective Date | Added | Removed | Reason`` with
  ``Added`` and ``Removed`` each split into ``Ticker`` + ``Security``.

Scraping policy:
* 1 req/sec hard rate limit.
* User-Agent identifies the project per Wikipedia bot etiquette.
* HTML is cached on disk so unit tests run against fixtures without network.
* Parser targets table ``id`` attributes — robust to column renames as long
  as the table identity persists.

References:
    Wikipedia. "List of S&P 500 companies."
    https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

from ..models import Action, Constituent, IndexEvent

logger = logging.getLogger(__name__)

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = (
    "index-rebalance-tracker (research; +https://github.com/QuantMaverick/index-rebalance-tracker)"
)
RATE_LIMIT_SECONDS = 1.0


# ── HTTP fetch ──────────────────────────────────────────────────────────────


def fetch_html(url: str = WIKIPEDIA_URL, *, timeout: float = 30.0) -> str:
    """Fetch a Wikipedia article's HTML, respecting rate limits.

    Args:
        url: Full Wikipedia article URL.
        timeout: HTTPX timeout in seconds.

    Returns:
        Raw HTML body.

    Raises:
        httpx.HTTPStatusError: on non-2xx response.

    Example:
        >>> html = fetch_html()  # doctest: +SKIP
        >>> len(html) > 100_000  # doctest: +SKIP
        True
    """
    time.sleep(RATE_LIMIT_SECONDS)
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


# ── Parsers ─────────────────────────────────────────────────────────────────


def parse_constituents(html: str) -> list[Constituent]:
    """Parse the current S&P 500 members table.

    Args:
        html: Raw Wikipedia HTML containing a table with ``id="constituents"``.

    Returns:
        List of Constituent records, one per member. Order is the source-table
        order (alphabetical by ticker as of the snapshot).

    Raises:
        ValueError: if the constituents table is not found or has no rows.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="constituents")
    if not isinstance(table, Tag):
        raise ValueError("constituents table not found in Wikipedia HTML")

    rows = table.find_all("tr")
    if len(rows) <= 1:
        raise ValueError("constituents table has no data rows")

    headers = _row_headers(rows[0])
    out: list[Constituent] = []
    for tr in rows[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) < 6:
            continue
        record = dict(zip(headers, cells, strict=False))
        out.append(
            Constituent(
                ticker=_clean_ticker(record.get("Symbol", "")),
                name=record.get("Security", ""),
                gics_sector=_normalize(record.get("GICSSector") or record.get("GICS Sector")),
                gics_sub_industry=_normalize(record.get("GICS Sub-Industry")),
                headquarters=_normalize(record.get("Headquarters Location")),
                date_added=_parse_date(record.get("Date added")),
                cik=_normalize(record.get("CIK")),
                founded=_normalize(record.get("Founded")),
            )
        )
    return out


def parse_changes(
    html: str,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[IndexEvent]:
    """Parse the historical additions/deletions table.

    Args:
        html: Raw Wikipedia HTML containing a table with ``id="changes"``.
        start: Filter events with ``effective_date >= start`` if provided.
        end: Filter events with ``effective_date <= end`` if provided.

    Returns:
        List of IndexEvent records. Each effective date generates 0–2 events
        (an Add and/or a Delete). Sorted ascending by ``effective_date``.

    Raises:
        ValueError: if the changes table is not found.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="changes")
    if not isinstance(table, Tag):
        raise ValueError("changes table not found in Wikipedia HTML")

    events: list[IndexEvent] = []
    body_rows = [tr for tr in table.find_all("tr") if tr.find("td")]

    for tr in body_rows:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 5:
            continue
        # Schema: [Effective Date, Added Ticker, Added Security, Removed Ticker,
        #          Removed Security, Reason]
        eff_date = _parse_date(cells[0].get_text(strip=True))
        if eff_date is None:
            continue
        if start is not None and eff_date < start:
            continue
        if end is not None and eff_date > end:
            continue

        added_ticker = _clean_ticker(cells[1].get_text(strip=True))
        removed_ticker = _clean_ticker(cells[3].get_text(strip=True))
        reason = cells[5].get_text(" ", strip=True) if len(cells) > 5 else ""

        if added_ticker:
            events.append(_make_event("sp500", added_ticker, "add", eff_date, reason))
        if removed_ticker:
            events.append(_make_event("sp500", removed_ticker, "delete", eff_date, reason))

    events.sort(key=lambda e: (e.effective_date, e.ticker))
    return events


# ── Public convenience ──────────────────────────────────────────────────────


def fetch_constituents(html: str | None = None) -> list[Constituent]:
    """Fetch + parse current S&P 500 members.

    Args:
        html: Pre-fetched HTML; if None, makes a network request.

    Returns:
        List of Constituent records.
    """
    if html is None:
        html = fetch_html()
    return parse_constituents(html)


def fetch_changes(
    start: date | None = None,
    end: date | None = None,
    *,
    html: str | None = None,
) -> list[IndexEvent]:
    """Fetch + parse the additions/deletions history, optionally filtered.

    Args:
        start: Lower bound on effective_date (inclusive).
        end: Upper bound on effective_date (inclusive).
        html: Pre-fetched HTML; if None, makes a network request.

    Returns:
        List of IndexEvent records sorted ascending by effective_date.
    """
    if html is None:
        html = fetch_html()
    return parse_changes(html, start=start, end=end)


# ── Cache helpers (used by CLI) ─────────────────────────────────────────────


def cache_html(html: str, cache_dir: Path) -> Path:
    """Write fetched HTML to a timestamped cache file.

    Args:
        html: HTML body.
        cache_dir: Directory in which to write; created if missing.

    Returns:
        Path of the written file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = cache_dir / f"sp500_wikipedia_{stamp}.html"
    path.write_text(html, encoding="utf-8")
    return path


# ── Internal ────────────────────────────────────────────────────────────────


def _row_headers(tr: Tag) -> list[str]:
    return [th.get_text(strip=True) for th in tr.find_all(["th", "td"])]


def _clean_ticker(s: str) -> str:
    """Normalize a ticker cell.

    Wikipedia sometimes appends footnote markers like ``[1]`` or class-share
    suffixes like ``BRK.B``. Strip footnotes; preserve dot-share notation.
    """
    cleaned = re.sub(r"\[[^\]]*\]", "", s).strip()
    return cleaned.upper()


def _normalize(s: str | None) -> str | None:
    if s is None:
        return None
    cleaned = re.sub(r"\[[^\]]*\]", "", s).strip()
    return cleaned or None


_DATE_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y")


def _parse_date(s: str | None) -> date | None:
    """Parse a Wikipedia-style date string. Returns None if unparseable."""
    if not s:
        return None
    cleaned = re.sub(r"\[[^\]]*\]", "", s).strip()
    if not cleaned:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    logger.debug("could not parse date: %r", s)
    return None


def _make_event(
    index: str,
    ticker: str,
    action: Action,
    effective: date,
    reason: str,
) -> IndexEvent:
    seed = f"{index}|{ticker}|{action}|{effective.isoformat()}".encode()
    digest = hashlib.sha1(seed, usedforsecurity=False).hexdigest()[:12]
    return IndexEvent(
        event_id=f"{index}-{action}-{ticker}-{effective.isoformat()}-{digest}",
        index="sp500",
        ticker=ticker,
        action=action,
        effective_date=effective,
        announcement_date=None,
        reason=reason or None,
        source_url=WIKIPEDIA_URL,
    )
