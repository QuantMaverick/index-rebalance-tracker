"""Tests for the MSCI Singapore historical scraper.

The MSCI scraper is *brittle by design* — the goal of these tests is to
verify it never silently fabricates events, returns clean empty lists
on missing input, and surfaces the discovered review URLs for operator
diagnostic.
"""

from __future__ import annotations

from datetime import date

from index_rebalance_tracker.data.msci_sg import (
    _parse_quarter_label,
    discover_review_urls,
    fetch_msci_sg_changes,
    parse_msci_sg_changes,
)

# ── parse_msci_sg_changes ──────────────────────────────────────────────────


def test_parse_returns_empty_with_no_singapore_links() -> None:
    html = "<html><body><a href='/foo'>Other</a></body></html>"
    assert parse_msci_sg_changes(html) == []


def test_parse_returns_empty_when_singapore_links_present() -> None:
    """The free-data path can never extract ticker-level events. Confirmed."""
    html = """
    <html><body>
      <a href="/q1.pdf">MSCI Singapore Q1 2024 Review</a>
      <a href="/q2.pdf">MSCI Singapore Q2 2024 Review</a>
    </body></html>
    """
    # The parser logs the URLs but emits no events
    assert parse_msci_sg_changes(html) == []


def test_parse_does_not_raise_on_malformed_html() -> None:
    """Empty / broken HTML → empty list, no crash."""
    assert parse_msci_sg_changes("") == []
    assert parse_msci_sg_changes("<html>") == []


# ── fetch_msci_sg_changes ──────────────────────────────────────────────────


def test_fetch_with_no_html_returns_empty() -> None:
    """Calling without HTML logs a warning and returns []."""
    assert fetch_msci_sg_changes(None) == []


def test_fetch_with_html_routes_to_parser() -> None:
    html = "<html><body><a href='/x'>MSCI Singapore Review</a></body></html>"
    assert fetch_msci_sg_changes(html) == []


# ── discover_review_urls ───────────────────────────────────────────────────


def test_discover_review_urls_returns_singapore_links_only() -> None:
    html = """
    <html><body>
      <a href="/sg.pdf">MSCI Singapore Review</a>
      <a href="/jp.pdf">MSCI Japan Review</a>
      <a href="/hk.pdf">MSCI Hong Kong Review</a>
      <a href="https://example.com/sg-q2.pdf">Singapore Q2 2024</a>
    </body></html>
    """
    urls = discover_review_urls(html)
    assert urls == [
        "https://example.com/sg-q2.pdf",
        "https://www.msci.com/sg.pdf",
    ]


def test_discover_review_urls_deduplicates() -> None:
    html = """
    <html><body>
      <a href="/sg.pdf">MSCI Singapore Review (PDF)</a>
      <a href="/sg.pdf">Singapore index review</a>
    </body></html>
    """
    urls = discover_review_urls(html)
    assert urls == ["https://www.msci.com/sg.pdf"]


def test_discover_review_urls_returns_empty_when_no_singapore() -> None:
    html = "<html><body><a href='/jp.pdf'>Japan Review</a></body></html>"
    assert discover_review_urls(html) == []


# ── _parse_quarter_label ───────────────────────────────────────────────────


def test_parse_quarter_label_q_format() -> None:
    assert _parse_quarter_label("Q1 2024") == date(2024, 1, 1)
    assert _parse_quarter_label("Q2 2024") == date(2024, 4, 1)
    assert _parse_quarter_label("Q4 2023") == date(2023, 10, 1)


def test_parse_quarter_label_month_format() -> None:
    assert _parse_quarter_label("May 2024") == date(2024, 5, 1)
    assert _parse_quarter_label("November 2023") == date(2023, 11, 1)


def test_parse_quarter_label_invalid_returns_none() -> None:
    assert _parse_quarter_label("not-a-date") is None
    assert _parse_quarter_label("") is None
    assert _parse_quarter_label("Q5 2024") is None  # invalid quarter
