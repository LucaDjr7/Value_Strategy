"""Part 2/3 — L/S portfolio construction and net performance.

Construction (semi-annual, June/December):
  LONG  : ENTRY     = value + quality + positive momentum
          HOLD      = value + quality (momentum ignored)
          EXIT      = no longer value, degraded quality, tenure > 3 years, distress
  SHORT : bottom 25% quality score among growth (bottom 20% B/M)

Performance: equal-weighted monthly return, net of dynamic transaction costs
(Amihud) and the borrow fee on the short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ----------------------------------------------------------------------------
# Portfolio construction
# ----------------------------------------------------------------------------
def construct_portfolios(panel_filtered: pd.DataFrame, start: str, end: str):
    """Build the long/short snapshots at each rebalancing date.

    Returns
    -------
    long_snapshots, short_snapshots : dict {date -> set(permno)}
    rebal_dates : sorted list of rebalancing dates.
    """
    rebal_dates = sorted([
        d for d in panel_filtered["date"].unique()
        if pd.Timestamp(start) <= d <= pd.Timestamp(end)
        and d.month in config.REBAL_MONTHS
    ])

    # Permnos with positive momentum per date (entry filter only)
    mom_by_date = {}
    for d, grp in panel_filtered.groupby("date"):
        mom_by_date[d] = set(grp.loc[grp["mom_6m"] > 0, "permno"])

    entry_date_tracker: dict = {}
    long_snapshots: dict = {}
    short_snapshots: dict = {}

    for idx, d in enumerate(rebal_dates):
        snap = panel_filtered.loc[panel_filtered["date"] == d].copy().set_index("permno")
        mom_ok = mom_by_date.get(d, set())
        d_prev = rebal_dates[idx - 1] if idx > 0 else None

        # -- New entrants: value + positive momentum + sufficient quality
        nouveaux = set(
            snap.loc[
                snap["long_eligible"]
                & snap.index.isin(mom_ok)
                & (snap["score"] >= config.SCORE_MIN_MAINTIEN)
                & snap["score"].notna()
            ].index
        )

        # -- Held names: momentum ignored, exit on fundamentals
        deja_en_port = long_snapshots.get(d_prev, set()) if d_prev else set()
        maintenus = set()
        for permno in deja_en_port:
            if permno not in snap.index:          # delisted -> exit
                continue
            row = snap.loc[permno]
            if not row["long_eligible"]:          # no longer value -> exit
                continue
            if pd.isna(row["score"]) or row["score"] < config.SCORE_MIN_MAINTIEN:
                continue                          # degraded quality -> exit
            entry = entry_date_tracker.get(permno, d)
            n_rebals = sum(1 for rd in rebal_dates if entry <= rd <= d)
            if n_rebals > config.MAX_SEJOUR_REBALS:  # tenure > 3 years -> exit
                continue
            if pd.notna(row.get("NetD/OIBDP")) and row["NetD/OIBDP"] > 20:
                continue                          # distress -> exit
            if pd.notna(row.get("ROCE")) and row["ROCE"] < -0.20:
                continue
            maintenus.add(permno)

        long_snapshots[d] = nouveaux | maintenus
        for permno in long_snapshots[d]:
            entry_date_tracker.setdefault(permno, d)

        # -- Short: bottom 25% quality among the growth bucket
        short_pool = (
            snap.loc[snap["short_eligible"]]["score"]
            .dropna().sort_values(ascending=True)
        )
        thresh = short_pool.quantile(config.SHORT_QUALITY_QUANTILE)
        short_snapshots[d] = set(short_pool[short_pool <= thresh].index)

    avg_long = np.mean([len(v) for v in long_snapshots.values()])
    avg_short = np.mean([len(v) for v in short_snapshots.values()])
    print(f"Number of rebalancing dates: {len(rebal_dates)}")
    print(f"Average LONG size:  {avg_long:.1f}")
    print(f"Average SHORT size: {avg_short:.1f}")
    return long_snapshots, short_snapshots, rebal_dates


# ----------------------------------------------------------------------------
# Costs
# ----------------------------------------------------------------------------
def estimate_borrow_fee(mcap: float) -> float:
    """Annual borrow fee (%) — decreasing grid by market cap."""
    if mcap >= 10_000:
        return 0.30
    elif mcap >= 5_000:
        return 0.50
    elif mcap >= 2_000:
        return 1.00
    elif mcap >= 1_000:
        return 2.00
    elif mcap >= 500:
        return 3.50
    else:
        return 6.00


def get_tc_dynamic(panel_sm: pd.DataFrame, snapshots_curr: dict) -> pd.Series:
    """Portfolio-level transaction cost at each rebalancing.

    Under equal weighting each stock weighs 1/N; if K stocks are traded, the
    total cost equals sum(tc_k) / N. Returns a Series (date -> decimal cost).
    """
    tc_by_date = {}
    dates = sorted(snapshots_curr.keys())
    for i, d in enumerate(dates):
        current_port = snapshots_curr[d]
        portfolio_size = len(current_port)
        if i == 0:
            permnos_traded = current_port
        else:
            permnos_traded = current_port ^ snapshots_curr[dates[i - 1]]
        snap_tc = panel_sm.loc[
            (panel_sm["date"] == d) & (panel_sm["permno"].isin(permnos_traded))
        ]["tc_stock"].dropna()
        if portfolio_size > 0 and len(snap_tc) > 0:
            tc_by_date[d] = snap_tc.sum() / portfolio_size
        else:
            tc_by_date[d] = 0.0
    return pd.Series(tc_by_date)


def monthly_ew_return(df: pd.DataFrame, pos_label: str) -> pd.Series:
    """Equal-weighted monthly return of the stocks in a given position."""
    return (
        df.loc[df["position"] == pos_label]
        .dropna(subset=["ret_next"])
        .query("ret_next >= -1.0 and ret_next <= 1.0")
        .groupby("date")["ret_next"]
        .mean()
        .rename(pos_label)
    )


# ----------------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------------
def compute_performance(
    panel_sm: pd.DataFrame,
    long_snapshots: dict,
    short_snapshots: dict,
    start: str,
    end: str,
):
    """Compute the monthly L/S performance net of costs.

    Returns
    -------
    perf : DataFrame indexed by date with columns LONG, SHORT, LS, LS_net.
    costs : dict of annualized costs (trans, borrow, total, monthly).
    """
    panel_sm = panel_sm.sort_values(["permno", "date"]).copy()

    # Delisting-adjusted return
    if "dlret" in panel_sm.columns:
        panel_sm["ret_adj"] = panel_sm["ret"].where(
            panel_sm["dlret"].isna(), panel_sm["dlret"]
        )
    else:
        panel_sm["ret_adj"] = panel_sm["ret"]

    panel_sm["ret_next"] = panel_sm.groupby("permno")["ret_adj"].shift(-1)
    panel_sm = panel_sm.drop(columns=["active_rebal"], errors="ignore")

    panel_sm = pd.merge_asof(
        panel_sm.sort_values("date"),
        pd.DataFrame({"active_rebal": sorted(long_snapshots.keys())}),
        left_on="date", right_on="active_rebal", direction="backward",
    )

    panel_sm["position"] = "NEUTRAL"
    for rd, longs in long_snapshots.items():
        panel_sm.loc[
            (panel_sm["active_rebal"] == rd) & (panel_sm["permno"].isin(longs)),
            "position",
        ] = "LONG"
    for rd, shorts in short_snapshots.items():
        panel_sm.loc[
            (panel_sm["active_rebal"] == rd) & (panel_sm["permno"].isin(shorts)),
            "position",
        ] = "SHORT"

    perf = pd.concat([
        monthly_ew_return(panel_sm, "LONG"),
        monthly_ew_return(panel_sm, "SHORT"),
    ], axis=1).dropna()
    perf["LS"] = perf["LONG"] - perf["SHORT"]

    # Borrow fee — median market cap of the shorts -> grid
    short_mcap = pd.concat([
        panel_sm.loc[
            (panel_sm["date"] == d) & (panel_sm["permno"].isin(shorts))
        ][["permno", "Market Cap"]]
        for d, shorts in short_snapshots.items()
    ], ignore_index=True)
    borrow_fee = (
        short_mcap.groupby("permno")["Market Cap"].median()
        .apply(estimate_borrow_fee).mean()
    )

    # Dynamic transaction costs
    tc_long = get_tc_dynamic(panel_sm, long_snapshots)
    tc_short = get_tc_dynamic(panel_sm, short_snapshots)
    n_years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25

    cout_trans = (tc_long.sum() + tc_short.sum()) / n_years * 100  # decimal -> %
    cout_borrow = borrow_fee * 0.50                                # already in %
    cout_total = cout_trans + cout_borrow
    cost_monthly = (cout_total / 100) / 12

    perf["LS_net"] = perf["LS"] - cost_monthly

    costs = {
        "trans": cout_trans,
        "borrow": cout_borrow,
        "total": cout_total,
        "monthly": cost_monthly,
    }
    print(f"\nPerformance computed: {len(perf)} months")
    print(f"Transaction cost: {cout_trans:.2f}%/yr")
    print(f"Borrow cost     : {cout_borrow:.2f}%/yr")
    print(f"Total cost      : {cout_total:.2f}%/yr")
    return perf, costs


# ----------------------------------------------------------------------------
# Diagnostic — average holding duration
# ----------------------------------------------------------------------------
def holding_durations(snapshots: dict) -> tuple[float, float, int]:
    """Average/median holding duration (months) for a set of snapshots."""
    dates = sorted(snapshots.keys())
    active: dict = {}
    durations = []
    for d in dates:
        current = snapshots[d]
        for p in list(active.keys()):
            if p not in current:
                entry = active.pop(p)
                durations.append((d - entry).days / 30.44)
        for p in current:
            active.setdefault(p, d)
    last = dates[-1]
    for p, entry in active.items():
        durations.append((last - entry).days / 30.44)
    return float(np.mean(durations)), float(np.median(durations)), len(durations)
