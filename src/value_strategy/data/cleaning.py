"""1.8  Variable cleaning, Amihud illiquidity and dynamic costs.

Cleaning depends on the variable. For xrd, capx, dp, txt, xint a missing value
means zero (no spending). Debts (dltt, dlc, lct) are smoothed then set to zero.
For oibdp and oancf, fully-missing firms are dropped before filling. xsga is
reconstructed via revt - cogs - oibdp when possible.

Then comes the monthly Amihud illiquidity, followed by a grid of transaction
costs based on cross-sectional thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """Clean the accounting variables of the panel (cell 1.8)."""
    panel = panel.copy()

    # Variables where a missing value means zero
    for col in ["xrd", "capx", "dp", "txt", "xint"]:
        panel[col] = panel[col].fillna(0)

    # Debt and current liabilities: smoothing then zero
    for col in ["dltt", "dlc", "lct"]:
        panel[col] = panel.groupby("gvkey")[col].transform(
            lambda x: x.ffill().bfill().fillna(0)
        )

    # OIBDP: drop fully-missing firms
    pct_oibdp = panel.groupby("gvkey")["oibdp"].apply(lambda x: x.isna().mean())
    gvkeys_oibdp_ok = pct_oibdp[pct_oibdp < 1.0].index
    panel = panel[panel["gvkey"].isin(gvkeys_oibdp_ok)].copy()
    panel["oibdp"] = panel.groupby("gvkey")["oibdp"].transform(
        lambda x: x.ffill().bfill()
    )

    # OANCF: drop fully-missing firms (mostly banks)
    pct_oancf = panel.groupby("gvkey")["oancf"].apply(lambda x: x.isna().mean())
    gvkeys_oancf_ok = pct_oancf[pct_oancf < 1.0].index
    panel = panel[panel["gvkey"].isin(gvkeys_oancf_ok)].copy()
    panel["oancf"] = panel.groupby("gvkey")["oancf"].transform(
        lambda x: x.ffill().bfill()
    )

    # XSGA: reconstruct if missing
    mask_xsga = panel["xsga"] == 0
    panel.loc[mask_xsga, "xsga"] = (
        panel.loc[mask_xsga, "revt"]
        - panel.loc[mask_xsga, "cogs"]
        - panel.loc[mask_xsga, "oibdp"]
    ).clip(lower=0)

    print(f"Cleaned panel: {panel.shape[0]:,} obs — {panel['permno'].nunique():,} stocks")
    return panel


def add_illiquidity_costs(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly Amihud illiquidity and dynamic transaction cost.

    The ``tc_stock`` cost is assigned via cross-sectional thresholds (30%/70%)
    of illiquidity: 5 bp (liquid), 15 bp (median), 30 bp (illiquid).
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["permno", "date"])

    # Monthly Amihud = |ret| / monthly dollar volume
    panel["amihud_monthly"] = (
        panel["ret"].astype(float).abs()
        / panel["dollar_vol_monthly_m"].astype(float)
    )
    panel["amihud_monthly"] = panel["amihud_monthly"].replace(
        [np.inf, -np.inf], np.nan
    )
    panel["amihud_used"] = panel["amihud_monthly"].astype(float)

    # Cross-sectional thresholds per date
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
    print(f"Avg universe TC cost: {tc_universe_monthly.mean():.4f} = "
          f"{tc_universe_monthly.mean() * 100:.2f}%")
    print(f"Final panel: {panel.shape[0]:,} obs — {panel['permno'].nunique():,} stocks")
    return panel
