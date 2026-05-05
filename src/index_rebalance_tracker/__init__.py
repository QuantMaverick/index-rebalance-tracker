"""Index Rebalance Tracker — event study, liquidity, TCA for SP500 + MSCI SG."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("index-rebalance-tracker")
except PackageNotFoundError:  # pragma: no cover - source checkout, not installed
    __version__ = "0.1.0+dev"

__all__ = ["__version__"]
