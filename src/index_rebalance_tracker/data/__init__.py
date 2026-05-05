"""Data ingestion: scrapers and price fetchers.

All side-effects (network, disk) live in this subpackage. Analytical modules
in :mod:`index_rebalance_tracker.analysis` consume DataFrames from here and
must remain pure.
"""
