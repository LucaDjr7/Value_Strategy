"""Partie 2/3 — Construction des signaux value + qualité + momentum.

Ce module factorise le pipeline de signaux appliqué **à l'identique** en
in-sample (2003-2013) et out-of-sample (2014-2024). Étapes :

  2.3  variables de valorisation (B/M ajusté intangibles)
  2.4  signal value avec neutralisation sectorielle
  2.5  ratios de qualité fondamentale (QARP)
  2.6  univers small/mid caps + winsorisation
  2.7  momentum 6 mois (skip-2)
  2.8  score qualité composite

``build_signals(panel, start, end)`` renvoie ``(panel_sm, panel_filtered)`` où
``panel_sm`` est l'univers small/mid (support des rendements et coûts) et
``panel_filtered`` l'univers final éligible (extrêmes value/growth, qualité
calculable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# (métrique qualité, ascending) — ascending=True : "plus bas = mieux"
QUALITY_SPECS = [
    ("ROCE", False),        # plus haut = mieux
    ("ROE", False),
    ("OM", False),
    ("NetD/OIBDP", True),   # plus bas = mieux (moins de dette)
]


def add_valuation(df: pd.DataFrame) -> pd.DataFrame:
    """2.3  Variables de valorisation : B/M ajusté intangibles -> signal_raw."""
    df = df.copy()
    df["Market Cap"] = df["prc"].abs() * df["shrout"] / 1_000
    df["net_debt"] = (
        df["dltt"].fillna(0) + df["dlc"].fillna(0) - df["che"].fillna(0)
    )
    df["EV"] = df["Market Cap"] + df["net_debt"]
    df["book_adj"] = df["ceq"] + df["KC"].fillna(0) + df["OC"].fillna(0)
    df["BM_adj"] = df["book_adj"] / df["Market Cap"].where(df["Market Cap"] > 0)
    df["BM_adj"] = df["BM_adj"].where(df["BM_adj"] > 0)
    df["signal_raw"] = np.log(df["BM_adj"])
    return df


def add_sector_value_rank(df: pd.DataFrame) -> pd.DataFrame:
    """2.4  Rank value intra-secteur (fallback global pour petits secteurs)."""
    df = df.copy()
    df["sector"] = (df["siccd"] // 100).astype("Int64")

    df["BM_rank_global"] = df.groupby("date")["signal_raw"].rank(
        pct=True, method="first"
    )
    df["BM_rank_sector"] = df.groupby(["date", "sector"])["signal_raw"].rank(
        pct=True, method="first"
    )
    sector_count = df.groupby(["date", "sector"])["permno"].transform("count")
    df["BM_rank"] = np.where(
        sector_count >= config.SECTOR_MIN_COUNT,
        df["BM_rank_sector"],
        df["BM_rank_global"],
    )

    df["long_eligible"] = df["BM_rank"] >= config.LONG_BM_RANK
    df["short_eligible"] = df["BM_rank"] <= config.SHORT_BM_RANK
    return df


def add_quality_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """2.5  Ratios de qualité fondamentale (ROCE, ROE, OM, levier, multiples)."""
    df = df.copy()
    capital_employed = (
        df["at"] - df["lct"] + df["KC"].fillna(0) + df["OC"].fillna(0)
    )
    df["ROCE"] = df["ebit"] / capital_employed.where(capital_employed > 0)
    df["ROE"] = df["ni"] / df["ceq"].where(df["ceq"] > 0)
    rev = df["sale"].where(df["sale"].notna(), df["revt"])
    df["OM"] = df["ebit"] / rev.where(rev > 0)
    df["NetD/OIBDP"] = df["net_debt"] / df["oibdp"].where(df["oibdp"] != 0)
    df["PER"] = df["Market Cap"] / df["ni"].where(df["ni"] > 0)
    df["EV/EBITDA"] = df["EV"] / df["oibdp"].where(df["oibdp"] > 0)
    return df


def filter_small_mid(df: pd.DataFrame) -> pd.DataFrame:
    """2.6  Univers small/mid caps + winsorisation économique des outliers."""
    sm = df.loc[
        (df["Market Cap"] >= config.SMALL_MID_MIN_MCAP)
        & (df["Market Cap"] < config.SMALL_MID_MAX_MCAP)
    ].copy()

    sm.loc[sm["PER"] > 500, "PER"] = np.nan
    sm.loc[sm["EV/EBITDA"] > 100, "EV/EBITDA"] = np.nan
    sm.loc[sm["OM"].abs() > 1.0, "OM"] = np.nan
    sm.loc[sm["ROE"].abs() > 5.0, "ROE"] = np.nan
    sm.loc[sm["NetD/OIBDP"].abs() > 30, "NetD/OIBDP"] = np.nan
    return sm


def add_momentum(panel_sm: pd.DataFrame) -> pd.DataFrame:
    """2.7  Momentum 6 mois (cumul t-6..t-2, skip-2) sur l'univers small/mid."""
    panel_sm = panel_sm.sort_values(["permno", "date"]).copy()
    panel_sm["mom_6m"] = panel_sm.groupby("permno")["ret"].transform(
        lambda x: (1 + x.shift(2)).rolling(5).apply(np.prod, raw=True) - 1
    )
    return panel_sm


def add_quality_score(value: pd.DataFrame) -> pd.DataFrame:
    """2.8  Score qualité composite (moyenne des rangs percentiles)."""
    value = value.copy()
    rank_cols = []
    for col, ascending in QUALITY_SPECS:
        rank_col = f"rank_{col.replace('/', '_')}"
        value[rank_col] = value.groupby("date")[col].rank(
            pct=True, ascending=ascending, method="first"
        )
        rank_cols.append(rank_col)

    value["n_quality"] = value[rank_cols].notna().sum(axis=1)
    value["score"] = value[rank_cols].mean(axis=1)
    return value


def build_signals(panel: pd.DataFrame, start: str, end: str):
    """Pipeline complet de signaux sur une fenêtre [start, end].

    Returns
    -------
    panel_sm : univers small/mid avec momentum (support rendements/coûts).
    panel_filtered : univers final éligible (extrêmes value/growth, n_quality>=2).
    """
    window = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    window["date"] = pd.to_datetime(window["date"])
    window = window.sort_values(["permno", "date"]).reset_index(drop=True)
    print(f"Panel {start[:4]}-{end[:4]} : {window.shape[0]:,} obs — "
          f"{window['permno'].nunique():,} titres")

    window = add_valuation(window)
    window = add_sector_value_rank(window)
    window = add_quality_ratios(window)

    panel_sm = filter_small_mid(window)

    # Univers investissable (signal value + prix > 1$ anti penny stocks)
    base_mask = (
        panel_sm["BM_rank"].notna()
        & panel_sm["signal_raw"].notna()
        & panel_sm["Market Cap"].notna()
        & (panel_sm["Market Cap"] > 0)
        & panel_sm["prc"].abs().gt(1)
    )
    value = panel_sm.loc[base_mask].copy()
    print(f"Univers investissable : {value['permno'].nunique():,} titres uniques")

    panel_sm = add_momentum(panel_sm)
    value = value.merge(
        panel_sm[["permno", "date", "mom_6m"]], on=["permno", "date"], how="left",
    )
    mom_med = value.groupby("date")["mom_6m"].transform("median")
    value["mom_ok"] = value["mom_6m"] >= mom_med

    value = add_quality_score(value)

    panel_filtered = value.loc[
        (value["long_eligible"] | value["short_eligible"])
        & (value["n_quality"] >= 2)
    ].copy()
    print(f"Univers filtré : {panel_filtered['permno'].nunique():,} titres")
    return panel_sm, panel_filtered
