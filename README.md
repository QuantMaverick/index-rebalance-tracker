# Index Rebalance Tracker

Event-study, liquidity, and transaction-cost analytics for **S&P 500** and **MSCI Singapore Free** index rebalances. Free data sources only, fully reproducible, deterministic offline tests.

[![CI](https://github.com/QuantMaverick/index-rebalance-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantMaverick/index-rebalance-tracker/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What this is

A Python toolkit that:

1. **Reconstructs** the historical event study around index additions and deletions — cumulative abnormal returns (CAR) computed via market-model and sector-matched-control benchmarks.
2. **Quantifies** liquidity and transaction-cost proxies (ADV, Corwin-Schultz spread, Amihud illiquidity, Kyle's lambda) for each event.
3. **Live-monitors** upcoming announcements from S&P Dow Jones Indices and MSCI quarterly review calendars.
4. **Emits** a clean JSON / Parquet contract that downstream dashboards consume.

Built with `uv`, `ruff`, `mypy --strict`, `pytest --cov`, and `pre-commit`. CI runs on Ubuntu and macOS against Python 3.11 and 3.12.

## Headline result

The decay-of-the-index-effect chart is the project's headline. Run the notebook (`notebooks/01_methodology_walkthrough.ipynb`) end-to-end to reproduce it locally — the script writes `assets/decay_headline.png` plus `assets/tca_headline.png`, and `output/decay_sp500.json` with the same numbers in machine-readable form.

The expected pattern (Petajisto 2011 + Greenwood-Sammon 2022): SP500 addition CAR compressed from ~7-9% in the 1990s/2000s to ~1-3% in the 2020s as more arbitrage capital chases the trade. The notebook computes this on **your** data; whatever it shows is the headline. We report what the data says, not what the literature predicts.

To populate the chart files, run from a fully-installed environment:
```bash
uv run index-rebalance pull-history --index sp500 --start 2010-01-01
uv run index-rebalance build-dashboard --aum-billions 6500
uv run jupyter nbconvert --to notebook --execute notebooks/01_methodology_walkthrough.ipynb
```

## Status

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Wikipedia scraper + price fetcher + corporate-actions handler + CLI `pull-history` | ✅ |
| **M2** | Event study (market model + sector-matched controls + CLI `event-study`) | ✅ |
| **M3** | Liquidity + TCA (Corwin-Schultz, Amihud, Kyle, Implementation Shortfall) | ✅ |
| **M4** | Decay analysis + dashboard JSON exports + methodology notebook | ✅ |
| **M5** | Live monitor + MSCI Singapore parallel pipeline | ⏳ |

## Quickstart

```bash
# Clone and install (uv handles Python + venv + lockfile)
git clone https://github.com/QuantMaverick/index-rebalance-tracker
cd index-rebalance-tracker
uv sync --all-extras

# Pull S&P 500 history → data/{constituents,events}_sp500.parquet
uv run index-rebalance pull-history --index sp500 --start 2010-01-01

# Run the tests
uv run pytest

# (M2+) Run the event study and emit dashboard JSON
uv run index-rebalance event-study --index sp500 --window-pre 5 --window-post 20
uv run index-rebalance build-dashboard
```

## Methodology

Each formula on the dashboard is sourced from peer-reviewed literature; primary citations live in module docstrings.

* **Event study** — market-model AR per Brown & Warner (1985); 250-day estimation window ending 30 days before announcement. Sector-matched alternative uses median of 5 GICS-sector + size-decile-matched controls.
* **Liquidity** — Corwin & Schultz (2012) high-low spread proxy; Amihud (2002) illiquidity ratio; Kyle (1985) lambda estimated as Δprice / signed-volume regression slope.
* **TCA** — comparative implementation shortfall (Perold 1988): forced T-E close vs spread-across-window for a hypothetical $1B passive fund.
* **Decay** — yearly cohort means + IQR of CAR. Petajisto (2011) reported 8.8% premium for SP500 additions 1990–2005; we test whether this has compressed in recent data.

Detailed derivations land in `notebooks/01_methodology_walkthrough.ipynb` (M4).

## Limitations (read this before drawing conclusions)

* **Daily data only.** True intraday TCA needs trade-by-trade data; we use OHLCV-derived proxies. Documented at the formula level.
* **GICS sector reflects current Wikipedia state** — not historical-as-of-event. Sector reclassifications affect ~5–10% of stocks over 15 years; we accept this drift and document it.
* **MSCI Singapore scraper is brittle** — MSCI publishes review summaries as PDFs/HTML with no stable API. The scraper can break on layout changes; tests catch this loudly.
* **Float-weight estimation** uses yfinance `floatShares` → `heldPercentInsiders` fallback → `0.85` default. Documented per assumption.
* **Survivorship**: yfinance returns adjusted prices for delisted stocks but coverage is not perfect; we surface gaps in the output JSON.
* **Sample size for recent SP500 deletions is small** — typical year has 20–30 changes. Bootstrap / robust standard errors used in M2.

## Output contract (M4)

After `build-dashboard`, `output/` contains:

| File | Schema | Purpose |
|---|---|---|
| `events_sp500.json` | `IndexEvent` + analytics | Historical events with CAR, AR, liquidity, TCA |
| `events_msci_sg.json` | same | Same for MSCI Singapore |
| `decay_sp500.json` | `CohortDecay[]` | Cohort-aggregated CAR by year |
| `tca_summary.json` | `TCAEstimate[]` | Annual TCA cost for hypothetical $1B fund |
| `upcoming.json` | `UpcomingEventsFile` | Forward-looking events from live monitor |
| `methodology_constants.json` | `MethodologyConstants` | Assumptions surfaced for the dashboard |

Schemas are pydantic v2 models in [`models.py`](src/index_rebalance_tracker/models.py); the dashboard project copies that file verbatim to validate inputs.

## Citations

* Brown, S. J. and Warner, J. B. (1985). _Using daily stock returns: The case of event studies_. Journal of Financial Economics 14(1).
* Beneish, M. D. and Whaley, R. E. (1996). _An anatomy of the "S&P game": The effects of changing the rules_. Journal of Finance 51(5).
* Chen, H., Noronha, G., and Singal, V. (2004). _The price response to S&P 500 index additions and deletions_. Journal of Finance 59(4).
* Petajisto, A. (2011). _The index premium and its hidden cost for index funds_. Journal of Empirical Finance 18(2).
* Corwin, S. A. and Schultz, P. (2012). _A simple way to estimate bid-ask spreads from daily high and low prices_. Journal of Finance 67(2).
* Amihud, Y. (2002). _Illiquidity and stock returns: cross-section and time-series effects_. Journal of Financial Markets 5(1).
* Kyle, A. S. (1985). _Continuous auctions and insider trading_. Econometrica 53(6).
* Perold, A. F. (1988). _The implementation shortfall: paper versus reality_. Journal of Portfolio Management 14(3).

## License

MIT — see [LICENSE](LICENSE).

## Contact

QuantMaverick — [github.com/QuantMaverick](https://github.com/QuantMaverick)
