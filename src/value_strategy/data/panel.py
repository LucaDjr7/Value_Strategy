"""1.6  Merge CRSP et Compustat en panel mensuel (anti look-ahead bias).

Logique anti-look-ahead : ``avail_date = datadate + 6 mois`` (délai de
publication), puis ``merge_asof`` backward — chaque mois CRSP récupère le
dernier compte Compustat disponible, jamais un compte futur.
"""

from __future__ import annotations

import pandas as pd

COMP_COLS = [
    "gvkey", "avail_date", "datadate", "fyear", "conm",
    "ni", "ceq", "oancf", "sale", "revt", "ebit", "oibdp",
    "dp", "txt", "xint", "capx", "cogs", "at", "dltt", "dlc",
    "che", "lct", "csho", "prcc_f", "xrd", "xsga",
]


def merge_panel(
    df_crsp: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_link: pd.DataFrame,
) -> pd.DataFrame:
    """Fusionne CRSP (lié via CCM) et Compustat en panel mensuel.

    Parameters
    ----------
    df_crsp : CRSP mensuel avec delistings déjà intégrés.
    df_comp : Compustat annuel.
    df_link : table de lien CCM.
    """
    df_comp = df_comp.copy()
    df_crsp = df_crsp.copy()

    df_comp["avail_date"] = (
        df_comp["datadate"] + pd.DateOffset(months=6)
    ) + pd.offsets.MonthEnd(0)

    df_crsp["permno"] = df_crsp["permno"].astype(float)
    df_crsp["date"] = pd.to_datetime(df_crsp["date"])

    crsp_linked = df_crsp.merge(df_link, on="permno", how="inner")
    crsp_linked = crsp_linked[
        (crsp_linked["date"] >= crsp_linked["linkdt"])
        & (crsp_linked["date"] <= crsp_linked["linkenddt"])
    ].drop(columns=["linkdt", "linkenddt"])

    comp_for_merge = (
        df_comp[COMP_COLS].rename(columns={"avail_date": "date"}).copy()
    )

    crsp_linked["gvkey"] = crsp_linked["gvkey"].astype(str)
    comp_for_merge["gvkey"] = comp_for_merge["gvkey"].astype(str)
    crsp_linked["date"] = pd.to_datetime(crsp_linked["date"])
    comp_for_merge["date"] = pd.to_datetime(comp_for_merge["date"])

    crsp_linked = crsp_linked.sort_values("date").reset_index(drop=True)
    comp_for_merge = comp_for_merge.sort_values("date").reset_index(drop=True)

    panel = pd.merge_asof(
        crsp_linked, comp_for_merge,
        on="date", by="gvkey", direction="backward",
    )
    panel = panel.dropna(subset=["fyear"]).copy()

    print(f"Panel brut : {panel.shape[0]:,} obs — {panel['permno'].nunique():,} titres")
    return panel
