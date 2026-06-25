"""1.6  Merge CRSP and Compustat into a monthly panel (anti look-ahead bias).

Anti-look-ahead logic: ``avail_date = datadate + 6 months`` (publication lag),
then a backward ``merge_asof`` — each CRSP month picks up the latest available
Compustat record, never a future one.
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
    """Merge CRSP (linked via CCM) and Compustat into a monthly panel.

    Parameters
    ----------
    df_crsp : monthly CRSP with delistings already integrated.
    df_comp : annual Compustat.
    df_link : CCM link table.
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

    print(f"Raw panel: {panel.shape[0]:,} obs — {panel['permno'].nunique():,} stocks")
    return panel
