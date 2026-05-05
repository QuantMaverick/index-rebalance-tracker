"""Find sector + size-matched controls for the sector-matched-AR model.

For each event stock at the announcement date, we want N peers that share
the same GICS sector and are closest in pre-event size. Free data sources
don't carry historical market-cap timeseries, so we use 60-day average
dollar volume (ADV) as the size proxy. ADV is highly correlated with market
cap within sector — the largest names trade the most dollars — and has the
operational advantage of being directly computable from the OHLCV parquet
we cache in M1.

Matching algorithm:

1. Filter the universe to ``ticker != event_ticker`` and same GICS sector.
2. Drop tickers whose ADV is missing or non-positive.
3. Rank remaining peers by ``|log(ADV) - log(event_ADV)|`` ascending.
4. Take the top ``n_matches`` (default 5).

Edge cases:

* Sector universe smaller than ``n_matches`` after filtering → return all
  available peers (caller can detect via length and decide how to handle).
* Event ticker has no ADV in ``adv_dollars`` → raise ``KeyError`` (caller
  is expected to skip the event).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def find_matched_controls(
    event_ticker: str,
    event_sector: str,
    universe: list[str],
    sector_map: dict[str, str],
    adv_dollars: dict[str, float],
    *,
    n_matches: int = 5,
) -> list[str]:
    """Return up to ``n_matches`` tickers in the same sector closest to
    ``event_ticker`` by log-ADV-dollars.

    Args:
        event_ticker: Symbol of the event stock.
        event_sector: GICS sector of the event stock.
        universe: Pool of candidate tickers (typically all SP500 members).
        sector_map: Ticker → GICS sector mapping.
        adv_dollars: Ticker → 60-day average dollar volume (USD).
        n_matches: Max number of controls to return (default 5).

    Returns:
        List of matched-control tickers, length ``min(n_matches, n_eligible)``.
        Empty if no eligible peers exist.

    Raises:
        KeyError: if ``event_ticker`` is missing from ``adv_dollars``.

    Example:
        >>> universe = ["AAPL", "MSFT", "NVDA", "JPM", "BAC"]
        >>> sector = {
        ...     "AAPL": "Information Technology",
        ...     "MSFT": "Information Technology",
        ...     "NVDA": "Information Technology",
        ...     "JPM": "Financials",
        ...     "BAC": "Financials",
        ... }
        >>> adv = {"AAPL": 8e9, "MSFT": 7e9, "NVDA": 6e9, "JPM": 1e9, "BAC": 5e8}
        >>> find_matched_controls("AAPL", "Information Technology", universe, sector, adv,
        ...                        n_matches=2)
        ['MSFT', 'NVDA']
    """
    if event_ticker not in adv_dollars:
        raise KeyError(f"adv_dollars missing for event ticker {event_ticker!r}")
    event_adv = adv_dollars[event_ticker]
    if event_adv <= 0:
        raise ValueError(f"event ticker {event_ticker!r} has non-positive ADV: {event_adv}")
    log_event = math.log(event_adv)

    candidates: list[tuple[float, str]] = []
    for t in universe:
        if t == event_ticker:
            continue
        if sector_map.get(t) != event_sector:
            continue
        adv = adv_dollars.get(t, 0.0)
        if adv <= 0:
            continue
        dist = abs(math.log(adv) - log_event)
        candidates.append((dist, t))

    candidates.sort()
    return [t for _, t in candidates[:n_matches]]


def adv_dollars_60d(prices: dict[str, pd.DataFrame], window: int = 60) -> dict[str, float]:
    """Compute 60-day average dollar volume from a prices dict.

    For each ticker, takes the trailing ``window`` rows of ``volume * close``
    and averages. Tickers with fewer than ``window`` rows return the mean
    over what's available.

    Args:
        prices: Ticker → OHLCV DataFrame.
        window: Number of trailing days (default 60).

    Returns:
        Ticker → mean dollar volume.
    """
    import pandas as pd  # noqa: PLC0415

    out: dict[str, float] = {}
    for ticker, df in prices.items():
        if df.empty or "volume" not in df.columns or "close" not in df.columns:
            continue
        tail: pd.DataFrame = df.tail(window)
        dollar_vol = (tail["volume"] * tail["close"]).mean()
        if pd.notna(dollar_vol) and dollar_vol > 0:
            out[ticker] = float(dollar_vol)
    return out
