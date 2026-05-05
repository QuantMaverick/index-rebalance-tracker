"""Analytical modules: event study, liquidity, TCA, decay.

All modules in this subpackage are pure: they consume DataFrames from
:mod:`index_rebalance_tracker.data` and return DataFrames or pydantic models.
No I/O, no globals, no configuration state.

Land sequence:

* event_study (M2) — CAR with market-model and sector-matched controls
* liquidity (M3) — ADV, Corwin-Schultz, Amihud, Kyle's lambda
* tca (M3) — implementation-shortfall comparison
* decay (M4) — cohort-aggregated CAR over time
"""
