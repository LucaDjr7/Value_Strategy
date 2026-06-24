"""1.8  Nettoyage des variables, illiquidité Amihud et coûts dynamiques.

Le nettoyage dépend de la variable. Pour xrd, capx, dp, txt, xint un manquant
vaut zéro (pas de dépense). Les dettes (dltt, dlc, lct) sont lissées puis
mises à zéro. Pour oibdp et oancf on exclut les firmes entièrement manquantes
avant de combler. xsga est reconstruit via revt - cogs - oibdp quand c'est
possible.

Vient ensuite l'illiquidité d'Amihud mensuelle, puis une grille de coûts de
transaction par seuils cross-sectionnels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les variables comptables du panel (cellule 1.8)."""
    panel = panel.copy()

    # Variables où l'absence signifie zéro
    for col in ["xrd", "capx", "dp", "txt", "xint"]:
        panel[col] = panel[col].fillna(0)

    # Dettes et passif courant : lissage puis zéro
    for col in ["dltt", "dlc", "lct"]:
        panel[col] = panel.groupby("gvkey")[col].transform(
            lambda x: x.ffill().bfill().fillna(0)
        )

    # OIBDP : exclure firmes totalement manquantes
    pct_oibdp = panel.groupby("gvkey")["oibdp"].apply(lambda x: x.isna().mean())
    gvkeys_oibdp_ok = pct_oibdp[pct_oibdp < 1.0].index
    panel = panel[panel["gvkey"].isin(gvkeys_oibdp_ok)].copy()
    panel["oibdp"] = panel.groupby("gvkey")["oibdp"].transform(
        lambda x: x.ffill().bfill()
    )

    # OANCF : exclure firmes totalement manquantes (majoritairement banques)
    pct_oancf = panel.groupby("gvkey")["oancf"].apply(lambda x: x.isna().mean())
    gvkeys_oancf_ok = pct_oancf[pct_oancf < 1.0].index
    panel = panel[panel["gvkey"].isin(gvkeys_oancf_ok)].copy()
    panel["oancf"] = panel.groupby("gvkey")["oancf"].transform(
        lambda x: x.ffill().bfill()
    )

    # XSGA : reconstruction si manquant
    mask_xsga = panel["xsga"] == 0
    panel.loc[mask_xsga, "xsga"] = (
        panel.loc[mask_xsga, "revt"]
        - panel.loc[mask_xsga, "cogs"]
        - panel.loc[mask_xsga, "oibdp"]
    ).clip(lower=0)

    print(f"Panel nettoyé : {panel.shape[0]:,} obs — {panel['permno'].nunique():,} titres")
    return panel


def add_illiquidity_costs(panel: pd.DataFrame) -> pd.DataFrame:
    """Illiquidité d'Amihud mensuelle et coût de transaction dynamique.

    Le coût ``tc_stock`` est attribué par seuils cross-sectionnels (30 %/70 %)
    de l'illiquidité : 5 bp (liquide), 15 bp (médian), 30 bp (illiquide).
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["permno", "date"])

    # Amihud mensuel = |ret| / volume dollar mensuel
    panel["amihud_monthly"] = (
        panel["ret"].astype(float).abs()
        / panel["dollar_vol_monthly_m"].astype(float)
    )
    panel["amihud_monthly"] = panel["amihud_monthly"].replace(
        [np.inf, -np.inf], np.nan
    )
    panel["amihud_used"] = panel["amihud_monthly"].astype(float)

    # Seuils cross-sectionnels par date
    panel["low_thresh"] = panel.groupby("date")["amihud_used"].transform(
        lambda x: x.quantile(0.30)
    ).astype(float)
    panel["high_thresh"] = panel.groupby("date")["amihud_used"].transform(
        lambda x: x.quantile(0.70)
    ).astype(float)

    mask_nan = (
        panel["amihud_used"].isna()
        | panel["low_thresh"].isna()
        | panel["high_thresh"].isna()
    )
    panel["tc_stock"] = np.where(
        mask_nan, np.nan,
        np.where(
            panel["amihud_used"] < panel["low_thresh"], 0.0005,
            np.where(panel["amihud_used"] > panel["high_thresh"], 0.0030, 0.0015),
        ),
    )

    tc_universe_monthly = panel.groupby("date")["tc_stock"].mean().sort_index()
    print(f"Coût TC moyen univers : {tc_universe_monthly.mean():.4f} = "
          f"{tc_universe_monthly.mean() * 100:.2f}%")
    print(f"Panel final : {panel.shape[0]:,} obs — {panel['permno'].nunique():,} titres")
    return panel
