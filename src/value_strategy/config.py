"""Global strategy parameters and project paths.

All parameters are FIXED — they are not recalibrated on the in-sample period.
The thresholds (top/bottom 20%, bottom 25% quality) follow the literature
(Fama-French: quintiles, Piotroski: tertiles).
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

# Cache of the final monthly panel (overwritten on each run with a WRDS fetch)
PANEL_CACHE = CACHE_DIR / "panel.parquet"
# Raw data reused by Part 4 (ML)
COMPUSTAT_CACHE = CACHE_DIR / "compustat.parquet"
CRSP_ML_CACHE = CACHE_DIR / "crsp_ml.parquet"
MACRO_CACHE = CACHE_DIR / "macro.parquet"


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for d in (RESULTS_DIR, CHARTS_DIR, DATA_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Time windows — In-Sample / Out-Of-Sample split
# 11 years each: statistical balance, the OOS covers the "value lost decade".
# ----------------------------------------------------------------------------
IS_START = "2003-01-01"
IS_END = "2013-12-31"
OOS_START = "2014-01-01"
OOS_END = "2024-12-31"

# ----------------------------------------------------------------------------
# Universe & construction
# ----------------------------------------------------------------------------
SMALL_MID_MIN_MCAP = 300          # $M — excludes illiquid micro-caps
SMALL_MID_MAX_MCAP = 10_000       # $M — excludes mega-caps (weak value premium)
SECTOR_MIN_COUNT = 8              # min stocks / sector for intra-sector ranking
REBAL_MONTHS = [6, 12]            # semi-annual rebalancing (June, December)
RF_ANNUAL = 0.02                  # annual risk-free rate
RF_MONTHLY = (1 + RF_ANNUAL) ** (1 / 12) - 1

# Value eligibility (sector B/M ranks)
LONG_BM_RANK = 0.80               # top 20% B/M = potential longs
SHORT_BM_RANK = 0.20              # bottom 20% B/M = potential shorts

# ----------------------------------------------------------------------------
# Long position hold / exit rules
# ----------------------------------------------------------------------------
MAX_SEJOUR_REBALS = 6             # 3 years max = 6 semi-annual rebalancings
SCORE_MIN_MAINTIEN = 0.2          # minimum quality score to stay in the portfolio
SHORT_QUALITY_QUANTILE = 0.25     # bottom 25% quality among growth = short

# ----------------------------------------------------------------------------
# Intangible capitalization (Peters & Taylor 2017)
# ----------------------------------------------------------------------------
KC_DEPRECIATION = 0.15            # delta Knowledge Capital (R&D)
OC_DEPRECIATION = 0.20            # delta Organization Capital (SG&A)
OC_CAPITALIZED_FRACTION = 0.30    # fraction of SG&A capitalized

# ----------------------------------------------------------------------------
# ML-driven dynamic short (Part 5)
# ----------------------------------------------------------------------------
SHORT_WEIGHT_REDUCE = 0.50        # short weight in the euphoria regime
SHORT_WEIGHT_FULL = 1.00          # short weight in the normal regime
TC_PER_TRANSITION = 0.0015        # 15 bp to reposition 50% of the short book

# Regime detection (Part 4) — guard against clustering degeneracy.
# If the HMM/GMM produces a minority regime below this share, fall back to a
# robust stress-score split (see ml_regime.labeling).
MIN_REGIME_SHARE = 0.20

# ----------------------------------------------------------------------------
# Consistent plotting palette (presentation)
# ----------------------------------------------------------------------------
C_IS = "#2196F3"      # blue   — in-sample
C_OOS = "#4CAF50"     # green  — out-of-sample
C_HML = "#F44336"     # red    — passive HML FF
C_MKT = "#9E9E9E"     # grey   — US market
C_VLINE = "#212121"   # black  — IS/OOS separation
C_ADJ = "#FF9800"     # orange — dynamic short