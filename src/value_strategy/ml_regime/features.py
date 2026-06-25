"""4.1  Feature engineering V3 (CRSP + Compustat + FRED).

Builds a monthly market panel (~300 months) from the cross-sectional
aggregation of stocks, detrended into 24-month rolling z-scores:
  - CRSP: Amihud, turnover, zero_ret, rvol, dispersion, skewness...
  - Compustat: leverage, cash ratio (annual aggregate)
  - FRED: vix, credit_spread, term_spread, ted_spread...
  - Interactions: VIX x Amihud, Credit x Rvol
  - Momentum: changes (chg1, chg3) and moving averages (ma3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DETREND_WINDOW = 24


def compute_liquidity_features_v3(
    crsp: pd.DataFrame, comp: pd.DataFrame, macro: pd.DataFrame,
):
    """Build the market panel and the feature list. Returns (mkt, cols)."""
    print("\n[STEP 1] Feature engineering V3...")
    df = crsp.copy()
    for col in ["ret", "retx", "prc", "vol", "shrout"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    n_before = len(df)
    df = df.dropna(subset=["ret"]).copy()
    print(f"  Dropped {n_before - len(df):,} rows without ret")

    # -- Individual metrics --
    df["abs_ret"] = df["ret"].abs()
    df["dollar_vol"] = df["prc"].abs() * df["vol"]
    df["dollar_vol"] = df["dollar_vol"].replace(0, np.nan)
    df["amihud"] = df["abs_ret"] / df["dollar_vol"]
    df["amihud"] = df["amihud"].replace([np.inf, -np.inf], np.nan)
    df["turnover"] = df["vol"] / df["shrout"].replace(0, np.nan)
    df["turnover"] = df["turnover"].replace([np.inf, -np.inf], np.nan)
    df["zero_ret"] = (df["ret"] == 0).fillna(False).astype(int)
    df["neg_ret"] = (df["ret"] < 0).fillna(False).astype(int)

    # -- Cross-sectional aggregation -> market panel --
    mkt = df.groupby("date").agg(
        amihud_mkt=("amihud", "median"),
        turnover_mkt=("turnover", "median"),
        zero_ret_pct=("zero_ret", "mean"),
        rvol_mkt=("ret", "std"),
        dollar_vol_mkt=("dollar_vol", "sum"),
        ret_dispersion=("ret", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        pct_neg_ret=("neg_ret", "mean"),
        mkt_ret=("ret", "mean"),
        n_stocks=("permno", "nunique"),
        amihud_iqr=("amihud", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        ret_skewness=("ret", "skew"),
    ).reset_index()
    mkt = mkt.sort_values("date").reset_index(drop=True)
    print(f"  Raw market panel: {mkt.shape[0]} months "
          f"({mkt['date'].min().strftime('%Y-%m')} -> "
          f"{mkt['date'].max().strftime('%Y-%m')})")

    mkt["log_dollar_vol"] = np.log1p(mkt["dollar_vol_mkt"])

    # -- Advanced microstructure features --
    mkt["d_ret"] = mkt["mkt_ret"].diff()
    mkt["roll_cov"] = mkt["d_ret"].rolling(12).apply(
        lambda x: np.cov(x[1:], x[:-1])[0, 1] if len(x) > 2 else np.nan, raw=True,
    )
    mkt["roll_spread_mkt"] = mkt["roll_cov"].apply(
        lambda c: 2 * np.sqrt(-c) if (pd.notna(c) and c < 0) else 0
    )
    mkt["vol_of_amihud"] = mkt["amihud_mkt"].rolling(6, min_periods=3).std()
    mkt["amihud_autocorr"] = mkt["amihud_mkt"].rolling(12, min_periods=6).apply(
        lambda x: pd.Series(x).autocorr() if len(x) > 2 else 0, raw=True,
    ).fillna(0)

    # -- Annual Compustat aggregate --
    comp_clean = comp.dropna(subset=["dltt", "che", "at"]).copy()
    comp_clean = comp_clean[comp_clean["at"] > 0].copy()
    comp_agg = (
        comp_clean.groupby("fyear")[["dltt", "che", "at"]]
        .sum()
        .assign(
            leverage_mkt=lambda d: d["dltt"] / d["at"],
            cash_ratio_mkt=lambda d: d["che"] / d["at"],
        )[["leverage_mkt", "cash_ratio_mkt"]]
        .reset_index()
    )
    mkt["fyear"] = mkt["date"].dt.year
    mkt = mkt.merge(comp_agg, on="fyear", how="left")

    # -- Merge FRED macro --
    macro_cols_available = [c for c in macro.columns if c != "date"]
    mkt = mkt.merge(macro, on="date", how="left")
    mkt[macro_cols_available] = mkt[macro_cols_available].ffill().bfill()
    print(f"  Merged {len(macro_cols_available)} macro series")

    # -- Detrending (24-month rolling z-score) --
    raw_features = [
        "amihud_mkt", "turnover_mkt", "zero_ret_pct", "rvol_mkt",
        "log_dollar_vol", "ret_dispersion", "pct_neg_ret",
        "roll_spread_mkt", "vol_of_amihud", "amihud_autocorr",
        "amihud_iqr", "ret_skewness", "leverage_mkt", "cash_ratio_mkt",
    ] + macro_cols_available

    detrended_features = []
    for col in raw_features:
        if col not in mkt.columns:
            continue
        zcol = f"{col}_z"
        rm = mkt[col].rolling(DETREND_WINDOW, min_periods=12).mean()
        rs = mkt[col].rolling(DETREND_WINDOW, min_periods=12).std().replace(0, np.nan)
        mkt[zcol] = ((mkt[col] - rm) / rs).fillna(0)
        detrended_features.append(zcol)

    # -- Momentum features (changes & moving averages) --
    momentum_cols = ["amihud_mkt_z", "turnover_mkt_z", "rvol_mkt_z", "zero_ret_pct_z"]
    for mc in macro_cols_available:
        zmc = f"{mc}_z"
        if zmc in mkt.columns:
            momentum_cols.append(zmc)

    momentum_features = []
    for col in momentum_cols:
        if col not in mkt.columns:
            continue
        for lag, suffix in [(1, "_chg1"), (3, "_chg3")]:
            cname = f"{col}{suffix}"
            mkt[cname] = mkt[col].diff(lag)
            momentum_features.append(cname)
        ma3 = f"{col}_ma3"
        mkt[ma3] = mkt[col].rolling(3).mean()
        momentum_features.append(ma3)

    # -- Multi-horizon market returns --
    mkt["mkt_ret_3m"] = mkt["mkt_ret"].rolling(3).sum()
    mkt["mkt_ret_12m"] = mkt["mkt_ret"].rolling(12).sum()
    mkt["mkt_vol_3m"] = mkt["mkt_ret"].rolling(3).std()
    market_features = ["mkt_ret", "mkt_ret_3m", "mkt_ret_12m", "mkt_vol_3m"]

    # -- Interactions --
    interaction_features = []
    if "vix_z" in mkt.columns and "amihud_mkt_z" in mkt.columns:
        mkt["vix_x_amihud_z"] = mkt["vix_z"] * mkt["amihud_mkt_z"]
        interaction_features.append("vix_x_amihud_z")
    if "credit_spread_z" in mkt.columns and "rvol_mkt_z" in mkt.columns:
        mkt["credit_x_rvol_z"] = mkt["credit_spread_z"] * mkt["rvol_mkt_z"]
        interaction_features.append("credit_x_rvol_z")

    # -- Final assembly --
    feature_cols = (
        detrended_features + momentum_features + market_features + interaction_features
    )
    numeric_cols = mkt.select_dtypes(include=[np.number]).columns
    mkt[numeric_cols] = mkt[numeric_cols].replace([np.inf, -np.inf], np.nan)
    mkt = mkt.dropna(subset=["amihud_mkt"]).reset_index(drop=True)
    feature_cols = [c for c in feature_cols if c in mkt.columns]
    print(f"  Final market panel: {mkt.shape[0]} months | Features: {len(feature_cols)}")
    return mkt, feature_cols
