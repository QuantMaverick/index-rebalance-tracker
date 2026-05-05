"""Implementation-shortfall TCA for index-rebalance demand.

Estimates how much a hypothetical passive fund pays in market-impact costs
when forced to trade an addition (buy) or deletion (sell) on the effective
date, vs spreading the same demand evenly across the announcement-to-
effective window.

Cost model — square-root impact (Almgren et al. 2005):

.. math::

    \\text{cost\\_bps} = k \\cdot \\sigma_{\\text{daily}} \\cdot
    \\sqrt{q / \\text{ADV}} \\cdot 10^4

where:

* :math:`q` is the dollar amount being traded,
* :math:`\\text{ADV}` is the 60-day average daily dollar volume,
* :math:`\\sigma_{\\text{daily}}` is the daily-return standard deviation,
* :math:`k` is an empirically-calibrated coefficient (default 1.0 per
  Almgren et al. 2005, Appendix A).

For execution split evenly over :math:`N` days, the cost scales as
:math:`1/\\sqrt{N}` — splitting reduces impact.

Estimated demand:

.. math::

    \\text{demand} = \\text{passive\\_AUM} \\times
    \\frac{\\text{stock\\_market\\_cap}}{\\text{index\\_market\\_cap}}

Methodology limits:

* The ``k = 1.0`` constant is institutional-flow average. Different desks
  estimate different :math:`k` from their own fills; we use the published
  default and surface it in ``methodology_constants.json``.
* Spreading the trade evenly assumes the announcement-to-effective window
  is fully tradeable. In practice some passive fund mandates require
  trading on T-E close only — that's the "forced" benchmark.
* Market cap proxy: we use ``shares_outstanding × close`` from the most
  recent available price. For deep history this drifts from true market
  cap but is the best free-data approximation; documented.

References:
    Almgren, R., Thum, C., Hauptmann, E., and Li, H. (2005). Direct
    estimation of equity market impact. *Risk* 18(7), 58–62.

    Perold, A. F. (1988). The implementation shortfall: paper versus
    reality. *Journal of Portfolio Management* 14(3), 4–9.
"""

from __future__ import annotations

import math

# Empirical impact coefficient. Almgren et al. 2005 estimate k ∈ [0.6, 1.4]
# across NYSE-listed names; we use 1.0 as documented default.
DEFAULT_IMPACT_COEFFICIENT = 1.0


def estimated_demand_usd(
    passive_aum_usd: float,
    stock_market_cap_usd: float,
    index_market_cap_usd: float,
) -> float:
    """Dollar value of trading demand from passive funds for one stock.

    Args:
        passive_aum_usd: Total assets in passive index funds tracking the
            index (e.g. ~$6.5T for SP500 as of 2024).
        stock_market_cap_usd: Float-adjusted market cap of the event stock.
        index_market_cap_usd: Sum of float-adjusted market caps of all
            index members.

    Returns:
        Demand in USD. Negative for deletions only if caller passes
        negative AUM (which we don't); always positive here. NaN if
        index market cap is non-positive.

    Example:
        >>> # SP500 add of $50bn stock with $50T total index cap
        >>> demand = estimated_demand_usd(6_500_000_000_000, 50_000_000_000, 50_000_000_000_000)
        >>> round(demand / 1e9, 1)
        6.5
    """
    if index_market_cap_usd <= 0:
        return float("nan")
    return passive_aum_usd * stock_market_cap_usd / index_market_cap_usd


def square_root_impact_bps(
    demand_usd: float,
    adv_dollars: float,
    daily_vol: float,
    *,
    n_days: int = 1,
    impact_coefficient: float = DEFAULT_IMPACT_COEFFICIENT,
) -> float:
    """Square-root market-impact cost in basis points.

    For a trade of ``demand_usd`` split evenly over ``n_days``, the
    average impact in bps is:

    .. math::

        \\text{cost\\_bps} = \\frac{k \\cdot \\sigma \\cdot \\sqrt{q / \\text{ADV}}}
                                  {\\sqrt{n}} \\cdot 10^4

    Args:
        demand_usd: Total dollar amount to trade.
        adv_dollars: 60-day average daily dollar volume.
        daily_vol: Daily-return std as a decimal (e.g. 0.02 for 2 %).
        n_days: Trading days the order is split over (1 = forced T-E
            execution).
        impact_coefficient: ``k`` in the formula (default 1.0).

    Returns:
        Cost in basis points (bps of notional). NaN if any input is
        non-positive.

    Example:
        >>> # $100M trade, $1B ADV, 2% daily vol, single-day execution
        >>> cost = square_root_impact_bps(100e6, 1e9, 0.02, n_days=1)
        >>> round(cost, 1)
        63.2
        >>> # Split over 4 days: cost halved
        >>> cost4 = square_root_impact_bps(100e6, 1e9, 0.02, n_days=4)
        >>> round(cost4, 1)
        31.6
    """
    if demand_usd <= 0 or adv_dollars <= 0 or daily_vol <= 0 or n_days < 1:
        return float("nan")
    bps = impact_coefficient * daily_vol * math.sqrt(demand_usd / adv_dollars) * 10_000
    return bps / math.sqrt(n_days)


def implementation_shortfall_summary(
    demand_usd: float,
    adv_dollars: float,
    daily_vol: float,
    *,
    spread_n_days: int = 5,
    impact_coefficient: float = DEFAULT_IMPACT_COEFFICIENT,
) -> dict[str, float]:
    """Compare forced (1-day) vs spread (N-day) execution costs.

    Returns a dict with keys ``forced_bps``, ``spread_bps``, ``savings_bps``.
    All values NaN if any input is degenerate.

    Args:
        demand_usd: Total demand in dollars.
        adv_dollars: 60-day ADV in dollars.
        daily_vol: Daily-return std (decimal).
        spread_n_days: Days the spread-execution path uses (default 5,
            matching the typical SP500 T-A → T-E gap).
        impact_coefficient: ``k`` in the square-root model.

    Returns:
        Dict with three keys; ``savings_bps = forced_bps - spread_bps``.
    """
    forced = square_root_impact_bps(
        demand_usd, adv_dollars, daily_vol, n_days=1, impact_coefficient=impact_coefficient
    )
    spread = square_root_impact_bps(
        demand_usd,
        adv_dollars,
        daily_vol,
        n_days=spread_n_days,
        impact_coefficient=impact_coefficient,
    )
    if math.isnan(forced) or math.isnan(spread):
        return {"forced_bps": float("nan"), "spread_bps": float("nan"), "savings_bps": float("nan")}
    return {
        "forced_bps": forced,
        "spread_bps": spread,
        "savings_bps": forced - spread,
    }
