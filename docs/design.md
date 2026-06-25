# Value-Strategy repo restructuring — Design

**Date**: 2026-06-24
**Goal**: turn a monolithic notebook (`Value_Strategy_GARP.ipynb`, ~3,500 lines of code,
48 cells) into a structured, readable and presentable Python repo — a portfolio/CV showcase.

## Context

Long/short equity strategy on US small/mid caps. Value signal = intangible-adjusted B/M
(KC + OC, Peters & Taylor 2017), ranked by sector, filtered by quality + momentum. Short = bottom
quality of the growth bucket. ML detection of "junk rally" regimes (HMM/GMM + walk-forward) to
dynamically reduce the short. Data: CRSP + Compustat via **WRDS** (paid, online database) +
Fama-French + FRED.

The notebook currently runs a *live WRDS fetch* on every execution, with shared state through
global variables (`panel`, `panel_is`, `perf_is`, `perf_oos`, `mkt_df`, `results_df`...).

## Validated decisions

1. **Data**: *WRDS fetch -> parquet cache (overwrites the previous one on each run) -> downstream
   steps read the cache.* Reproducible starting point. A `--use-cache` flag allows re-running
   without WRDS from the latest parquet.
2. **Goal**: portfolio/CV showcase -> priority on readability, a polished README, a clear
   structure, clean code. Light tests (pure logic only).
3. **Orchestration**: a stage-by-stage `main.py` **AND** a thin narrative notebook that imports
   the modules.

## Architecture

```
value-strategy/
├── README.md                  # strategy pitch, results, pipeline diagram, how-to-run
├── requirements.txt
├── pyproject.toml             # package metadata (lightweight)
├── .gitignore                 # excludes data cache, .ipynb_checkpoints, anaconda_projects...
├── main.py                    # orchestrator: data -> signals -> backtest -> ML -> figures
├── src/value_strategy/
│   ├── __init__.py
│   ├── config.py              # FIXED params, paths, IS/OOS dates, thresholds
│   ├── data/
│   │   ├── wrds_loader.py      # connection + Compustat / CRSP / delistings / CCM / FRED queries
│   │   ├── panel.py           # merge CRSP+Compustat -> monthly panel (anti look-ahead)
│   │   ├── intangibles.py     # KC/OC Peters & Taylor (2017)
│   │   ├── cleaning.py        # variable cleaning + Amihud illiquidity / dynamic costs
│   │   └── cache.py           # save/load parquet (overwrites on each run)
│   ├── signals.py             # value + sector neutralization + quality + momentum + score
│   ├── portfolio.py           # L/S construction + performance net of costs
│   ├── factors.py             # Fama-French + perf stats + FF4 alpha + info ratio
│   ├── ml_regime/
│   │   ├── features.py        # feature engineering (CRSP+Compustat+FRED)
│   │   ├── labeling.py        # HMM/GMM regimes (euphoria / junk rally) + smoothing
│   │   ├── model.py           # walk-forward expanding window + adaptive threshold
│   │   ├── evaluation.py      # OOS metrics + feature importance
│   │   └── signals.py         # SHORT_REDUCE / FULL_SHORT
│   ├── dynamic_short.py       # ML-driven dynamic short strategy
│   └── plots.py               # all figures -> results/charts
├── notebooks/
│   └── value_strategy_narrative.ipynb   # thin notebook that imports the modules
├── results/charts/
└── tests/                     # light tests on the pure logic
```

## Splitting principles

- **One responsibility per module**, testable and readable in isolation.
- **Business logic extracted as-is** — the computations do not change. The refactor removes the
  global variables (explicit passing of DataFrames between functions) and **factors out the
  IS/OOS duplication** (cells 14-20 and 24 run the same signal pipeline twice -> a single
  `build_signals(panel, start, end)` function).
- **`config.py`** centralizes all the notebook's fixed parameters (IS/OOS dates, small/mid cap
  thresholds, `SECTOR_MIN_COUNT`, `REBAL_MONTHS`, `RF_ANNUAL`, exit thresholds, short weights).

## Data flow (`main.py`)

1. **data**: `wrds_loader` downloads -> `panel.merge_panel()` -> `intangibles` + `cleaning` ->
   `cache.save_frame()` (parquet, overwrite). `--use-cache` skips the fetch.
2. **signals**: `build_signals(panel, IS)` and `build_signals(panel, OOS)`.
3. **portfolio**: `construct_portfolios` + `compute_performance` -> `perf_is`, `perf_oos`.
4. **factors**: FF4, alpha/stats tables.
5. **ml_regime**: features -> labels -> walk-forward -> euphoria signals.
6. **dynamic_short**: combine ML signal + OOS perf -> `perf_h`.
7. **plots**: regenerate all figures in `results/charts`.

## Execution

The full pipeline was run end-to-end on real WRDS data (Parts 1 through 5 + generation of the
9 figures) and validated. Tests cover the pure functions that do not need WRDS (momentum, sector
ranking, quality score, regime smoothing, performance stats on synthetic series).

## Light tests (pytest)

- `test_signals.py`: skip-2 6M momentum, sector ranking with global fallback, composite quality
  score on a small synthetic DataFrame.
- `test_factors.py`: `performance_stats`, `compute_metrics` on known series.
- `test_ml_regime.py`: `smooth_regimes` (bridge + min duration), `find_optimal_threshold`,
  anti-degeneracy guard.

## Out of scope (YAGNI)

No rich CLI (minimal argparse is enough), no YAML config, no CI, no PyPI packaging, no refactor of
the computations themselves. The original research notebooks (`Code_Value.ipynb`,
`Value_Strategy_GARP.ipynb`) are moved to `artefact/` (out of the repo, ignored by git), kept
locally as reference but removed from the repo to keep a clean tree.
