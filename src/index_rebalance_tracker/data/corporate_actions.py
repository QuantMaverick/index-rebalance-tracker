"""Split and dividend handling for raw OHLCV → adjusted-price computation.

yfinance's ``Adj Close`` already incorporates splits + dividends, but for
some analyses (TCA, Kyle's lambda where signed-volume must be in shares of
the *historical* book) we need raw shares × historical-price. This module
does both: applies forward-split adjustment when needed, and computes
total-return series for use in the event study.
"""

from __future__ import annotations

import pandas as pd

DAILY_RETURN_FROM_ADJ_CLOSE = "adj_close"


def total_return(prices: pd.DataFrame) -> pd.Series:
    """Total-return series from an adjusted-close column.

    Uses the standard log-of-ratio for additive aggregation in
    event-study windows.

    Args:
        prices: DataFrame with at least ``adj_close`` column, indexed by date.

    Returns:
        Series of daily simple returns, named ``return``. First value is NaN.

    Raises:
        KeyError: if ``adj_close`` column is missing.
    """
    if DAILY_RETURN_FROM_ADJ_CLOSE not in prices.columns:
        raise KeyError(f"prices must contain '{DAILY_RETURN_FROM_ADJ_CLOSE}' column")
    s = prices[DAILY_RETURN_FROM_ADJ_CLOSE].pct_change()
    s.name = "return"
    return s


def apply_split(prices: pd.DataFrame, split_date: pd.Timestamp, ratio: float) -> pd.DataFrame:
    """Apply a forward stock split to raw OHLCV.

    For a 7-for-1 split on date D, raw prices on dates < D should be divided
    by 7 and raw volumes should be multiplied by 7 to make the series
    comparable to post-split data.

    Args:
        prices: DataFrame with [open, high, low, close, volume] indexed by date.
        split_date: First trading day on which the split is reflected.
        ratio: Split ratio (e.g. 7.0 for a 7-for-1 split).

    Returns:
        New DataFrame; original is not modified.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"open": [700, 100], "high": [710, 105], "low": [690, 95],
        ...      "close": [700, 100], "volume": [1_000_000, 7_000_000]},
        ...     index=pd.to_datetime(["2014-06-06", "2014-06-09"]),
        ... )
        >>> adj = apply_split(df, pd.Timestamp("2014-06-09"), ratio=7.0)
        >>> float(adj.loc["2014-06-06", "close"])
        100.0
        >>> int(adj.loc["2014-06-06", "volume"])
        7000000
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be > 0, got {ratio}")
    out = prices.copy()
    pre_mask = out.index < split_date
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out.loc[pre_mask, col] = out.loc[pre_mask, col] / ratio
    if "volume" in out.columns:
        out.loc[pre_mask, "volume"] = (out.loc[pre_mask, "volume"] * ratio).astype(
            out["volume"].dtype
        )
    return out
