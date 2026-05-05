"""Unit tests for the SP500 Wikipedia parser.

Uses a committed snapshot of the Wikipedia article (May 2026) so tests are
deterministic and offline. The fixture is large (~830 KB) but in exchange we
get a real-world test surface — column-rename or table-structure changes on
Wikipedia will fail loudly instead of silently corrupting downstream data.
"""

from __future__ import annotations

from datetime import date

import pytest

from index_rebalance_tracker.data.sp500_history import (
    _clean_ticker,
    _parse_date,
    parse_changes,
    parse_constituents,
)

# ── Constituents ────────────────────────────────────────────────────────────


def test_constituents_count_in_realistic_range(sp500_wikipedia_html: str) -> None:
    """SP500 has 500-505 members at any time; assert that range."""
    cs = parse_constituents(sp500_wikipedia_html)
    assert 500 <= len(cs) <= 510, f"unexpected constituent count: {len(cs)}"


def test_constituents_include_anchor_tickers(sp500_wikipedia_html: str) -> None:
    """AAPL, MSFT, GOOG, JPM should always be in the index — sanity probe."""
    cs = parse_constituents(sp500_wikipedia_html)
    tickers = {c.ticker for c in cs}
    for must_have in ("AAPL", "MSFT", "JPM"):
        assert must_have in tickers, f"{must_have} missing from parsed constituents"


def test_constituents_have_gics_sector(sp500_wikipedia_html: str) -> None:
    """Every parsed row should have a non-empty GICS sector."""
    cs = parse_constituents(sp500_wikipedia_html)
    missing = [c.ticker for c in cs if not c.gics_sector]
    assert not missing, f"missing GICS sector for: {missing[:5]}…"


def test_apple_classified_as_information_technology(sp500_wikipedia_html: str) -> None:
    """AAPL must land in Information Technology — sanity check sector parsing."""
    cs = parse_constituents(sp500_wikipedia_html)
    aapl = next((c for c in cs if c.ticker == "AAPL"), None)
    assert aapl is not None
    assert aapl.gics_sector == "Information Technology"


def test_constituents_no_footnote_brackets_in_ticker(sp500_wikipedia_html: str) -> None:
    """Tickers like ``BRK.B`` should survive; ``BRK.B[1]`` should be cleaned."""
    cs = parse_constituents(sp500_wikipedia_html)
    for c in cs:
        assert "[" not in c.ticker, f"footnote leak in ticker: {c.ticker!r}"


# ── Changes ─────────────────────────────────────────────────────────────────


def test_changes_parsed_at_all(sp500_wikipedia_html: str) -> None:
    events = parse_changes(sp500_wikipedia_html)
    assert len(events) > 100, f"only {len(events)} events — parser likely broken"


def test_changes_sorted_ascending(sp500_wikipedia_html: str) -> None:
    events = parse_changes(sp500_wikipedia_html)
    dates = [e.effective_date for e in events]
    assert dates == sorted(dates)


def test_changes_actions_are_add_or_delete(sp500_wikipedia_html: str) -> None:
    events = parse_changes(sp500_wikipedia_html)
    actions = {e.action for e in events}
    assert actions <= {"add", "delete"}


def test_changes_filter_by_start(sp500_wikipedia_html: str) -> None:
    cutoff = date(2020, 1, 1)
    events = parse_changes(sp500_wikipedia_html, start=cutoff)
    assert all(e.effective_date >= cutoff for e in events)
    assert events, "expected at least one event since 2020"


def test_changes_filter_by_window(sp500_wikipedia_html: str) -> None:
    start, end = date(2020, 1, 1), date(2020, 12, 31)
    events = parse_changes(sp500_wikipedia_html, start=start, end=end)
    for e in events:
        assert start <= e.effective_date <= end


def test_event_id_is_stable(sp500_wikipedia_html: str) -> None:
    """Re-parsing the same HTML must produce the same event_ids — they're
    keys downstream and a hash drift would fragment the database."""
    first = parse_changes(sp500_wikipedia_html, start=date(2020, 1, 1))
    second = parse_changes(sp500_wikipedia_html, start=date(2020, 1, 1))
    assert [e.event_id for e in first] == [e.event_id for e in second]


# ── Helpers ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AAPL", "AAPL"),
        ("BRK.B", "BRK.B"),
        ("BF.B[1]", "BF.B"),
        ("AAPL  ", "AAPL"),
        ("aapl", "AAPL"),
    ],
)
def test_clean_ticker(raw: str, expected: str) -> None:
    assert _clean_ticker(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2020-12-21", date(2020, 12, 21)),
        ("December 21, 2020", date(2020, 12, 21)),
        ("Dec 21, 2020", date(2020, 12, 21)),
        ("21 December 2020", date(2020, 12, 21)),
        ("2020-12-21[1]", date(2020, 12, 21)),
        ("", None),
        (None, None),
        ("not-a-date", None),
    ],
)
def test_parse_date(raw: str | None, expected: date | None) -> None:
    assert _parse_date(raw) == expected
