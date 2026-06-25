"""Part 5 — Dynamic short reduction during junk rallies.

When the ML (Part 4) detects a euphoria regime, the short weight goes from
100% to 50%: ``LS_adjusted = LONG - w * SHORT`` with w = 0.5 in euphoria,
1.0 otherwise. Costs: transition (repositioning 50% of the short book) +
borrow fee proportional to the short weight.

Timing: signal at month t -> applied at month t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .factors import compute_metrics


def build_dynamic_short(perf_oos: pd.DataFrame, signals_df: pd.DataFrame, costs_oos: dict):
    """Build the performance series with the ML-driven dynamic short.

    Parameters
    ----------
    perf_oos : OOS performance (index = date, columns LONG/SHORT/LS/LS_net).
    signals_df : ML signals (date, signal_gb, gb_prob).
    costs_oos : OOS cost dict from ``portfolio.compute_performance``.

    Returns
    -------
    perf_h : enriched monthly DataFrame (short_weight, LS_adjusted, LS_adj_net...).
    summary : dict {n_reduce, n_full, n_trans}.
    """
    print("=" * 60)
    print("  PART 5 — ML-DRIVEN DYNAMIC SHORT")
    print("=" * 60)

    # -- 5.1a  Prepare perf_oos
    perf_oos_df = perf_oos.copy()
    if "date" not in perf_oos_df.columns:
        perf_oos_df = perf_oos_df.reset_index()
        perf_oos_df.columns = perf_oos_df.columns.astype(str)
        date_col = [c for c in perf_oos_df.columns
                    if "date" in c.lower() or perf_oos_df[c].dtype == "datetime64[ns]"]
        if date_col:
            perf_oos_df = perf_oos_df.rename(columns={date_col[0]: "date"})
        else:
            perf_oos_df = perf_oos_df.rename(columns={perf_oos_df.columns[0]: "date"})
    perf_oos_df["date"] = pd.to_datetime(perf_oos_df["date"]) + pd.offsets.MonthEnd(0)

    # -- 5.1b  Align signals -> returns (t -> t+1)
    print("[5.1b] Aligning ML signals -> OOS returns...")
    sig = signals_df[["date", "signal_gb", "gb_prob"]].copy()
    sig["date"] = pd.to_datetime(sig["date"]) + pd.offsets.MonthEnd(0)
    sig["apply_date"] = sig["date"] + pd.offsets.MonthEnd(1)

    perf_h = perf_oos_df[["date", "LONG", "SHORT", "LS", "LS_net"]].copy()
    perf_h = perf_h.merge(
        sig[["apply_date", "signal_gb", "gb_prob"]].rename(columns={"apply_date": "date"}),
        on="date", how="left",
    )
    perf_h["signal_gb"] = perf_h["signal_gb"].fillna("FULL_SHORT")
    perf_h["gb_prob"] = perf_h["gb_prob"].fillna(0.0)
    perf_h = perf_h.sort_values("date").reset_index(drop=True)

    # -- 5.1c  Dynamic short weight
    perf_h["short_weight"] = np.where(
        perf_h["signal_gb"] == "SHORT_REDUCE",
        config.SHORT_WEIGHT_REDUCE,
        config.SHORT_WEIGHT_FULL,
    )
    perf_h["LS_adjusted"] = perf_h["LONG"] - perf_h["short_weight"] * perf_h["SHORT"]

    # -- 5.1d  Adjusted costs
    perf_h["weight_change"] = perf_h["short_weight"].diff().abs().fillna(0)
    perf_h["transition_cost"] = (
        (perf_h["weight_change"] > 0).astype(float) * config.TC_PER_TRANSITION
    )

    borrow_monthly_full = (costs_oos["borrow"] / 100) / 12
    perf_h["borrow_adj"] = perf_h["short_weight"] * borrow_monthly_full

    tc_monthly_base = (costs_oos["trans"] / 100) / 12
    perf_h["tc_adj"] = (
        tc_monthly_base * 0.5 + tc_monthly_base * 0.5 * perf_h["short_weight"]
    )

    perf_h["cost_adj"] = (
        perf_h["tc_adj"] + perf_h["borrow_adj"] + perf_h["transition_cost"]
    )
    perf_h["LS_adj_net"] = perf_h["LS_adjusted"] - perf_h["cost_adj"]

    # -- 5.1e  Summary
    n_reduce = int((perf_h["short_weight"] == config.SHORT_WEIGHT_REDUCE).sum())
    n_full = int((perf_h["short_weight"] == config.SHORT_WEIGHT_FULL).sum())
    n_trans = int((perf_h["weight_change"] > 0).sum())

    print(f"\n  Period: {perf_h['date'].min().strftime('%Y-%m')} -> "
          f"{perf_h['date'].max().strftime('%Y-%m')} ({len(perf_h)} months)")
    print(f"  SHORT_REDUCE months (50%): {n_reduce} ({n_reduce / len(perf_h):.0%})")
    print(f"  FULL_SHORT months (100%) : {n_full} ({n_full / len(perf_h):.0%})")
    print(f"  Transitions              : {n_trans}")
    print(f"  Total transition cost    : {perf_h['transition_cost'].sum() * 100:.2f}%")

    return perf_h, {"n_reduce": n_reduce, "n_full": n_full, "n_trans": n_trans}


def attach_ff_and_metrics(perf_h: pd.DataFrame, ff: pd.DataFrame):
    """Merge the FF factors and compute the compared metrics.

    Returns
    -------
    perf_h (with Mkt-RF, RF), metrics : dict {naked, adj, mkt}.
    """
    perf_h = perf_h.merge(ff[["date", "Mkt-RF", "RF"]], on="date", how="left")
    perf_h = perf_h.dropna(subset=["LS_net", "LS_adj_net"]).reset_index(drop=True)

    rf_oos = perf_h.set_index("date")["RF"]
    metrics = {
        "naked": compute_metrics(perf_h.set_index("date")["LS_net"], rf_series=rf_oos),
        "adj": compute_metrics(perf_h.set_index("date")["LS_adj_net"], rf_series=rf_oos),
        "mkt": compute_metrics(perf_h.set_index("date")["Mkt-RF"]),
    }
    return perf_h, metrics


def ff4_regression(perf_h: pd.DataFrame, ff: pd.DataFrame) -> dict:
    """FF4 regression (OOS) on both strategies. Returns a dict of results."""
    oos_ff = perf_h[["date", "LS_net", "LS_adj_net"]].merge(
        ff[["date", "Mkt-RF", "SMB", "HML", "Mom"]], on="date", how="inner",
    )
    X_oos = oos_ff[["Mkt-RF", "SMB", "HML", "Mom"]].values
    X_oos_c = np.column_stack([np.ones(len(X_oos)), X_oos])

    ff4_results = {}
    for name, col in [("Original", "LS_net"), ("Dynamic short", "LS_adj_net")]:
        y = oos_ff[col].values
        betas, _, _, _ = np.linalg.lstsq(X_oos_c, y, rcond=None)
        resid = y - X_oos_c @ betas
        n_obs, k = len(y), X_oos_c.shape[1]
        se = np.sqrt(
            np.sum(resid ** 2) / (n_obs - k)
            * np.diag(np.linalg.inv(X_oos_c.T @ X_oos_c))
        )
        t_stats = betas / se
        ff4_results[name] = {
            "alpha_ann": betas[0] * 12, "t_alpha": t_stats[0],
            "beta_mkt": betas[1], "beta_smb": betas[2],
            "beta_hml": betas[3], "beta_mom": betas[4],
        }
    return ff4_results
