"""Paramètres globaux de la stratégie et chemins du projet.

Tous les paramètres sont FIXES — ils ne sont pas recalibrés sur l'in-sample.
Les seuils (top/bottom 20 %, bottom 25 % qualité) sont justifiés par la
littérature (Fama-French : quintiles, Piotroski : tertiles).
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Chemins
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

# Fichier de cache du panel mensuel final (écrasé à chaque run avec fetch WRDS)
PANEL_CACHE = CACHE_DIR / "panel.parquet"
# Données brutes réutilisées par la Partie 4 (ML)
COMPUSTAT_CACHE = CACHE_DIR / "compustat.parquet"
CRSP_ML_CACHE = CACHE_DIR / "crsp_ml.parquet"
MACRO_CACHE = CACHE_DIR / "macro.parquet"


def ensure_dirs() -> None:
    """Crée les répertoires de sortie s'ils n'existent pas."""
    for d in (RESULTS_DIR, CHARTS_DIR, DATA_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Fenêtres temporelles — split In-Sample / Out-Of-Sample
# 11 ans chacun : équilibre statistique, l'OOS inclut la "value lost decade".
# ----------------------------------------------------------------------------
IS_START = "2003-01-01"
IS_END = "2013-12-31"
OOS_START = "2014-01-01"
OOS_END = "2024-12-31"

# ----------------------------------------------------------------------------
# Univers & construction
# ----------------------------------------------------------------------------
SMALL_MID_MIN_MCAP = 300          # M$ — exclut micro-caps illiquides
SMALL_MID_MAX_MCAP = 10_000       # M$ — exclut mega-caps (prime value faible)
SECTOR_MIN_COUNT = 8              # min titres / secteur pour rank intra-secteur
REBAL_MONTHS = [6, 12]            # rebalancement semestriel (juin, décembre)
RF_ANNUAL = 0.02                  # taux sans risque annuel
RF_MONTHLY = (1 + RF_ANNUAL) ** (1 / 12) - 1

# Éligibilité value (rangs B/M sectoriels)
LONG_BM_RANK = 0.80               # top 20 % B/M = longs potentiels
SHORT_BM_RANK = 0.20              # bottom 20 % B/M = shorts potentiels

# ----------------------------------------------------------------------------
# Règles de maintien / sortie du long
# ----------------------------------------------------------------------------
MAX_SEJOUR_REBALS = 6             # 3 ans max = 6 rebalancements semestriels
SCORE_MIN_MAINTIEN = 0.2          # score qualité minimum pour rester en portefeuille
SHORT_QUALITY_QUANTILE = 0.25     # bottom 25 % qualité parmi growth = short

# ----------------------------------------------------------------------------
# Capitalisation des intangibles (Peters & Taylor 2017)
# ----------------------------------------------------------------------------
KC_DEPRECIATION = 0.15            # δ Knowledge Capital (R&D)
OC_DEPRECIATION = 0.20            # δ Organization Capital (SG&A)
OC_CAPITALIZED_FRACTION = 0.30    # part du SG&A capitalisée

# ----------------------------------------------------------------------------
# Short dynamique piloté par le ML (Partie 5)
# ----------------------------------------------------------------------------
SHORT_WEIGHT_REDUCE = 0.50        # poids short en régime euphoria
SHORT_WEIGHT_FULL = 1.00          # poids short en régime normal
TC_PER_TRANSITION = 0.0015        # 15 bp pour repositionner 50 % du short book

# Détection de régime (Partie 4) — garde-fou anti-dégénérescence du clustering.
# Si le HMM/GMM produit un régime minoritaire sous ce seuil, on retombe sur un
# découpage robuste par score de stress (cf. ml_regime.labeling).
MIN_REGIME_SHARE = 0.20

# ----------------------------------------------------------------------------
# Palette graphique cohérente (présentation)
# ----------------------------------------------------------------------------
C_IS = "#2196F3"      # bleu  — in-sample
C_OOS = "#4CAF50"     # vert  — out-of-sample
C_HML = "#F44336"     # rouge — HML passif FF
C_MKT = "#9E9E9E"     # gris  — marché US
C_VLINE = "#212121"   # noir  — séparation IS/OOS
C_ADJ = "#FF9800"     # orange — short dynamique
