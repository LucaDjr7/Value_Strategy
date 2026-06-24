"""1.7  Calcul KC / OC — Peters & Taylor (2017).

Knowledge Capital (KC) : R&D capitalisée, δ = 15 %/an
    KC_t = (1 − 0.15) × KC_{t−1} + XRD_t   ;  KC_0 = XRD_0 / (0.15 + g)

Organization Capital (OC) : SG&A capitalisée à 30 %, δ = 20 %/an
    OC_t = (1 − 0.20) × OC_{t−1} + 0.30 × XSGA_t  ;  OC_0 = 0.30·XSGA_0 / (0.20 + g)

Intuition : les dépenses R&D et marketing créent des actifs immatériels hors
bilan. En les capitalisant, on corrige le book-to-market pour les économies
modernes (tech, pharma) où ces actifs forment une large part de la valeur.
"""

from __future__ import annotations

import pandas as pd

from .. import config


def _compute_kc(group: pd.DataFrame, dep: float) -> pd.DataFrame:
    group = group.sort_values("datadate").copy()
    g = group["g_mean"].iloc[0]
    kc_prev, vals = None, []
    for _, row in group.iterrows():
        if kc_prev is None:
            kc = row["xrd"] / (dep + g) if (dep + g) > 0 else 0.0
        else:
            kc = (1 - dep) * kc_prev + row["xrd"]
        vals.append(kc)
        kc_prev = kc
    group["KC"] = vals
    return group


def _compute_oc(group: pd.DataFrame, dep: float, frac: float) -> pd.DataFrame:
    group = group.sort_values("datadate").copy()
    g = group["g_mean"].iloc[0]
    oc_prev, vals = None, []
    for _, row in group.iterrows():
        xsga_cap = frac * row["xsga"]
        if oc_prev is None:
            oc = xsga_cap / (dep + g) if (dep + g) > 0 else 0.0
        else:
            oc = (1 - dep) * oc_prev + xsga_cap
        vals.append(oc)
        oc_prev = oc
    group["OC"] = vals
    return group


def add_intangibles(panel: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes KC et OC au panel mensuel."""
    panel = panel.copy()
    panel["xrd"] = panel["xrd"].fillna(0)
    panel["xsga"] = panel["xsga"].fillna(0)

    panel = panel.sort_values(["gvkey", "datadate"])
    panel["sale_lag"] = panel.groupby("gvkey")["sale"].shift(1)
    panel["g_sale"] = (
        (panel["sale"] - panel["sale_lag"]) / panel["sale_lag"].abs()
    ).clip(-0.5, 0.5)

    g_by_firm = (
        panel.groupby("gvkey")["g_sale"]
        .mean().fillna(0.05).clip(0.0, 0.30).rename("g_mean")
    )
    panel = panel.merge(g_by_firm, on="gvkey", how="left")

    comp_annual = (
        panel[["gvkey", "datadate", "fyear", "xrd", "xsga", "g_mean"]]
        .drop_duplicates(subset=["gvkey", "fyear"])
        .sort_values(["gvkey", "datadate"])
        .copy()
    )

    print("Calcul KC...")
    comp_annual = comp_annual.groupby("gvkey", group_keys=False).apply(
        _compute_kc, dep=config.KC_DEPRECIATION,
    )
    print("Calcul OC...")
    comp_annual = comp_annual.groupby("gvkey", group_keys=False).apply(
        _compute_oc, dep=config.OC_DEPRECIATION, frac=config.OC_CAPITALIZED_FRACTION,
    )

    print(f"KC médian : {comp_annual['KC'].median():.1f}M")
    print(f"OC médian : {comp_annual['OC'].median():.1f}M")

    panel = panel.merge(
        comp_annual[["gvkey", "fyear", "KC", "OC"]],
        on=["gvkey", "fyear"], how="left",
    )
    return panel
