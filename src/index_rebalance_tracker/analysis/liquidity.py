"""Liquidity diagnostics for index-rebalance events.

Four metrics are computed per event, all from daily OHLCV (no intraday
data needed). Each function is pure: input DataFrame in, scalar out.

References:
    Amihud, Y. (2002). Illiquidity and stock returns: cross-section and
    time-series effects. *Journal of Financial Markets* 5(1), 31–56.

    Corwin, S. A. and Schultz, P. (2012). A simple way to estimate bid-ask
    spreads from daily high and low prices. *Journal of Finance* 67(2),
    719–760.

    Kyle, A. S. (1985). Continuous auctions and insider trading.
    *Econometrica* 53(6), 1315–1335. (Empirical lambda variant: regress
    daily price change on tick-rule signed volume — see Hasbrouck 2009 for
    the daily-data discussion.)

Methodology limits:

* All four metrics use daily OHLCV — true intraday spread / impact would
  need tick-by-tick data. We document this prominently in the README.
* Kyle's lambda from daily data uses the tick rule (sign by sign of
  return) for signed volume; this is a known approximation and we
  surface the slope coefficient as-is for downstream interpretation.
* Corwin-Schultz floor: when the formula produces α < 0 we floor to 0
  per the published convention (negative spreads are impossible).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_LIQUIDITY_WINDOW_DAYS = 60


# ── Amihud illiquidity ─────────────────────────────────────────────────────


def amihud_illiquidity(
    prices: pd.DataFrame, *, window: int = DEFAULT_LIQUIDITY_WINDOW_DAYS
) -> float:
    """Amihud (2002) illiquidity ratio: ``mean(|R_t| / dollar_volume_t)``.

    The Amihud ratio is the average daily price impact per dollar of
    volume — higher means more illiquid. Uses simple daily returns from
    ``adj_close`` so dividend re-investment is handled.

    Args:
        prices: DataFrame indexed by date with at least ``adj_close``,
            ``volume``, and ``close`` columns.
        window: Number of trailing days to include (default 60).

    Returns:
        Amihud ratio (float). NaN if window has < 5 valid observations.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"adj_close": [100, 101, 99, 100], "close": [100, 101, 99, 100],
        ...      "volume": [1_000_000] * 4},
        ...     index=pd.date_range("2024-01-01", periods=4, freq="B"),
        ... )
        >>> ratio = amihud_illiquidity(df, window=4)
        >>> isinstance(ratio, float)
        True
    """
    if len(prices) < 2:
        return float("nan")
    sample = prices.tail(window + 1)
    returns = sample["adj_close"].pct_change()
    dollar_vol = sample["volume"] * sample["close"]
    valid = pd.concat({"r": returns.abs(), "dv": dollar_vol}, axis=1).dropna()
    valid = valid[valid["dv"] > 0]
    if len(valid) < 5:
        return float("nan")
    return float((valid["r"] / valid["dv"]).mean())


# ── Corwin-Schultz spread ──────────────────────────────────────────────────


_CS_DENOM = 3.0 - 2.0 * math.sqrt(2.0)


def corwin_schultz_spread(
    prices: pd.DataFrame, *, window: int = DEFAULT_LIQUIDITY_WINDOW_DAYS
) -> float:
    """High-low bid-ask spread proxy of Corwin & Schultz (2012).

    Two-day overlapping pairs, with the formula (negative α floored to 0):

    .. math::

        \\beta = (\\ln H_t/L_t)^2 + (\\ln H_{t+1}/L_{t+1})^2

        \\gamma = (\\ln \\max(H_t, H_{t+1}) / \\min(L_t, L_{t+1}))^2

        \\alpha = \\frac{\\sqrt{2\\beta} - \\sqrt{\\beta}}{3 - 2\\sqrt{2}}
                  - \\sqrt{\\frac{\\gamma}{3 - 2\\sqrt{2}}}

        S = \\frac{2(e^\\alpha - 1)}{1 + e^\\alpha}

    Args:
        prices: DataFrame with ``high`` and ``low`` columns indexed by date.
        window: Trailing-day window (default 60).

    Returns:
        Time-averaged proportional spread. NaN if fewer than 5 valid pairs.
    """
    if "high" not in prices.columns or "low" not in prices.columns:
        return float("nan")

    sample = prices.tail(window + 1)
    if len(sample) < 6:
        return float("nan")

    h = sample["high"].to_numpy()
    low = sample["low"].to_numpy()
    if np.any(h <= 0) or np.any(low <= 0):
        return float("nan")

    h_pair_max = np.maximum(h[:-1], h[1:])
    low_pair_min = np.minimum(low[:-1], low[1:])

    beta = np.log(h[:-1] / low[:-1]) ** 2 + np.log(h[1:] / low[1:]) ** 2
    gamma = np.log(h_pair_max / low_pair_min) ** 2

    with np.errstate(invalid="ignore"):
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DENOM - np.sqrt(gamma / _CS_DENOM)
    # Floor to zero per Corwin-Schultz convention
    alpha = np.where(alpha > 0, alpha, 0.0)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    valid = spread[~np.isnan(spread)]
    if len(valid) < 5:
        return float("nan")
    return float(np.mean(valid))


# ── Kyle's lambda (empirical, daily) ───────────────────────────────────────


def kyle_lambda(prices: pd.DataFrame, *, window: int | None = None) -> float:
    """Daily-data Kyle's-lambda proxy: slope of ΔP on signed volume.

    Signed volume uses the tick rule: ``sign(close_t - close_{t-1}) * volume_t``.
    The regression has no intercept (price change is zero on zero-flow days
    by construction), so:

    .. math::

        \\hat\\lambda = \\frac{\\sum_t Q_t \\Delta P_t}{\\sum_t Q_t^2}

    Units: dollars-per-share-traded. Higher means more sensitive to flow.

    Args:
        prices: OHLCV DataFrame.
        window: Trailing-day window (None → use all rows).

    Returns:
        Lambda estimate. NaN if fewer than 5 non-zero signed-volume observations.
    """
    sample = prices if window is None else prices.tail(window + 1)
    if len(sample) < 5 or "volume" not in sample.columns or "close" not in sample.columns:
        return float("nan")

    delta_p = sample["close"].diff()
    return_sign = np.sign(sample["close"].pct_change())
    signed_vol = return_sign * sample["volume"]

    valid = pd.concat({"dp": delta_p, "q": signed_vol}, axis=1).dropna()
    valid = valid[valid["q"] != 0]
    if len(valid) < 5:
        return float("nan")

    q = valid["q"].to_numpy()
    dp = valid["dp"].to_numpy()
    denom = float((q * q).sum())
    if denom == 0:
        return float("nan")
    return float((q * dp).sum() / denom)


# ── Convenience: daily return volatility ───────────────────────────────────


def daily_return_vol(prices: pd.DataFrame, *, window: int = DEFAULT_LIQUIDITY_WINDOW_DAYS) -> float:
    """Trailing-window standard deviation of daily simple returns.

    Used by the TCA module as the volatility input to square-root impact.
    Computed from ``adj_close`` to incorporate dividend re-investment.

    Args:
        prices: OHLCV DataFrame with ``adj_close``.
        window: Trailing-day window (default 60).

    Returns:
        Daily-return std as a decimal (e.g. 0.020 = 2 % daily vol). NaN
        if fewer than 10 valid observations.
    """
    if len(prices) < 2:
        return float("nan")
    returns = prices["adj_close"].tail(window + 1).pct_change().dropna()
    if len(returns) < 10:
        return float("nan")
    return float(returns.std(ddof=1))
