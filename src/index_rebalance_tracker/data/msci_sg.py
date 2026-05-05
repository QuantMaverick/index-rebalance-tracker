"""MSCI Singapore Free index historical scraper.

**Honest scope statement:** MSCI does not publish a free, machine-readable
historical-changes feed for the MSCI Singapore Free Index. Their public
review pages link to PDF announcements with no consistent schema across
quarters. A production-grade parallel of the SP500 pipeline therefore
requires either:

1. A paid MSCI Index data subscription (which provides a clean event
   table via API), or
2. A vendor data product (FactSet, Refinitiv, Bloomberg) that publishes
   structured constituent changes.

This module provides a *best-effort* free-data parser that:

* Fetches the MSCI quarterly reviews index page.
* Returns an empty list (with a clear warning log) when it cannot
  identify Singapore-specific announcements in the page.
* Does NOT silently fabricate constituent-level events — if the parser
  cannot pin a ticker to an action and date, it simply returns nothing
  for that quarter.

The CLI surfaces this state plainly: ``pull-history --index msci-sg``
prints how many events were extracted, and the README's Limitations
section explains why the count may be 0 for free-data users.

References:
    MSCI Index Reviews:
    https://www.msci.com/our-solutions/index-quality-policies-research/index-reviews
"""

from __future__ import annotations

import logging
import re
from datetime import date

from bs4 import BeautifulSoup, Tag

from ..models import IndexEvent

logger = logging.getLogger(__name__)

MSCI_REVIEWS_URL = (
    "https://www.msci.com/our-solutions/index-quality-policies-research/index-reviews"
)

_QUARTER_RE = re.compile(
    r"\b(?:Q[1-4]\s+\d{4}|(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4})\b"
)


def parse_msci_sg_changes(html: str) -> list[IndexEvent]:
    """Best-effort extraction of MSCI Singapore historical events from the
    reviews index page HTML.

    Returns:
        List of :class:`IndexEvent`. Empty when no parseable Singapore
        review entries are found — the recommended interpretation is
        "no free-data events available; see Limitations in README".

    Note:
        This function deliberately does not raise — it logs a warning
        and returns ``[]`` so the CLI can carry on and write an empty
        events file rather than fail the whole pipeline.
    """
    soup = BeautifulSoup(html, "lxml")
    sg_anchors = [
        a for a in soup.find_all("a") if "singapore" in a.get_text(" ", strip=True).lower()
    ]
    if not sg_anchors:
        logger.warning(
            "MSCI Singapore: no Singapore-mentioning links found on reviews index. "
            "Free-data history is unavailable; populate via paid MSCI subscription "
            "or vendor data feed. See README Limitations."
        )
        return []

    events: list[IndexEvent] = []
    seen: set[str] = set()
    for anchor in sg_anchors:
        if not isinstance(anchor, Tag):
            continue
        href_raw = anchor.get("href", "")
        if not isinstance(href_raw, str) or not href_raw or href_raw in seen:
            continue
        seen.add(href_raw)
        # Without a stable schema we can't pin individual ticker actions
        # — log the discovered URLs so an operator can manually inspect.
        logger.info("MSCI Singapore review URL discovered: %s", href_raw)

    # No ticker-level events parseable from the index page alone.
    return events


def fetch_msci_sg_changes(html: str | None = None) -> list[IndexEvent]:
    """Public entry point. Honest about its limitations — see module
    docstring for the full scope statement.

    Args:
        html: Pre-fetched HTML; if ``None``, the caller must supply
            it (we don't fetch synchronously here — use the async
            monitor for live fetches).

    Returns:
        List of :class:`IndexEvent`. Empty in the typical free-data case.
    """
    if html is None:
        logger.warning(
            "MSCI Singapore: no HTML supplied. Pass pre-fetched HTML or "
            "use the monitor module's async fetcher for live data."
        )
        return []
    return parse_msci_sg_changes(html)


def discover_review_urls(html: str) -> list[str]:
    """Return distinct URLs of Singapore-mentioning review links on the page.

    Useful diagnostic for understanding what's available — surfaced via
    the CLI in verbose mode so an operator can manually inspect the
    underlying PDFs.

    Args:
        html: Raw MSCI reviews page HTML.

    Returns:
        Sorted list of unique URLs. Relative URLs are normalized to
        absolute against ``msci.com``.
    """
    soup = BeautifulSoup(html, "lxml")
    out: set[str] = set()
    for a in soup.find_all("a"):
        if not isinstance(a, Tag):
            continue
        text = a.get_text(" ", strip=True)
        if "singapore" not in text.lower():
            continue
        href_raw = a.get("href", "")
        if not isinstance(href_raw, str) or not href_raw:
            continue
        url = href_raw if href_raw.startswith("http") else f"https://www.msci.com{href_raw}"
        out.add(url)
    return sorted(out)


def _parse_quarter_label(s: str) -> date | None:
    """Coarse parser for "Q2 2024" or "May 2024" → first day of quarter/month.

    Used for diagnostics only; not currently part of the public API.
    """
    s = s.strip()
    qmatch = re.match(r"Q([1-4])\s+(\d{4})", s)
    if qmatch:
        q, year = int(qmatch.group(1)), int(qmatch.group(2))
        month = (q - 1) * 3 + 1
        return date(year, month, 1)
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    parts = s.lower().split()
    if len(parts) == 2 and parts[0] in months:
        try:
            return date(int(parts[1]), months[parts[0]], 1)
        except ValueError:
            return None
    return None
