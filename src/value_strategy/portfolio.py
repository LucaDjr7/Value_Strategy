"""Partie 2/3 — Construction des portefeuilles L/S et performance nette.

Construction (semestrielle, juin/décembre) :
  LONG  : ENTRÉE   = value + qualité + momentum positif
          MAINTIEN = value + qualité (momentum ignoré)
          SORTIE   = plus value, qualité dégradée, séjour > 3 ans, détresse
  SHORT : bottom 25 % score qualité parmi growth (bottom 20 % B/M)

Performance : rendement mensuel equal-weighted, net des coûts de transaction
dynamiques (Amihud) et du borrow fee sur le short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ----------------------------------------------------------------------------
# Construction des portefeuilles
# ----------------------------------------------------------------------------
def construct_portfolios(panel_filtered: pd.DataFrame, start: str, end: str):
    """Construit les snapshots long/short à chaque date de rebalancement.

    Returns
    -------
    long_snapshots, short_snapshots : dict {date -> set(permno)}
    rebal_dates : liste triée des dates de rebalancement.
    """
    rebal_dates = sorted([
        d for d in panel_filtered["date"].unique()
        if pd.Timestamp(start) <= d <= pd.Timestamp(end)
        and d.month in config.REBAL_MONTHS
    ])

    # Permnos à momentum positif par date (filtre d'entrée uniquement)
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

        # ── Nouveaux entrants : value + momentum positif + qualité suffisante
        nouveaux = set(
            snap.loc[
                snap["long_eligible"]
                & snap.index.isin(mom_ok)
                & (snap["score"] >= config.SCORE_MIN_MAINTIEN)
                & snap["score"].notna()
            ].index
        )

        # ── Titres maintenus : momentum ignoré, sortie sur fondamentaux
        deja_en_port = long_snapshots.get(d_prev, set()) if d_prev else set()
        maintenus = set()
        for permno in deja_en_port:
            if permno not in snap.index:          # delisté -> sort
                continue
            row = snap.loc[permno]
            if not row["long_eligible"]:          # plus value -> sort
                continue
            if pd.isna(row["score"]) or row["score"] < config.SCORE_MIN_MAINTIEN:
                continue                          # qualité dégradée -> sort
            entry = entry_date_tracker.get(permno, d)
            n_rebals = sum(1 for rd in rebal_dates if entry <= rd <= d)
            if n_rebals > config.MAX_SEJOUR_REBALS:  # séjour > 3 ans -> sort
                continue
            if pd.notna(row.get("NetD/OIBDP")) and row["NetD/OIBDP"] > 20:
                continue                          # détresse -> sort
            if pd.notna(row.get("ROCE")) and row["ROCE"] < -0.20:
                continue
            maintenus.add(permno)

        long_snapshots[d] = nouveaux | maintenus
        for permno in long_snapshots[d]:
            entry_date_tracker.setdefault(permno, d)

        # ── Short : bottom 25 % qualité parmi le bucket growth
        short_pool = (
            snap.loc[snap["short_eligible"]]["score"]
            .dropna().sort_values(ascending=True)
        )
        thresh = short_pool.quantile(config.SHORT_QUALITY_QUANTILE)
        short_snapshots[d] = set(short_pool[short_pool <= thresh].index)

    avg_long = np.mean([len(v) for v in long_snapshots.values()])
    avg_short = np.mean([len(v) for v in short_snapshots.values()])
    print(f"Nb dates rebalancement : {len(rebal_dates)}")
    print(f"Taille moyenne LONG  : {avg_long:.1f}")
    print(f"Taille moyenne SHORT : {avg_short:.1f}")
    return long_snapshots, short_snapshots, rebal_dates


# ----------------------------------------------------------------------------
# Coûts
# ----------------------------------------------------------------------------
def estimate_borrow_fee(mcap: float) -> float:
    """Borrow fee annuel (%) — grille décroissante selon la market cap."""
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
    """Coût de transaction portfolio-level à chaque rebalancement.

    En equal-weight chaque titre pèse 1/N ; si on trade K titres, le coût total
    vaut sum(tc_k) / N. Renvoie une Series (date -> coût décimal).
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
    """Rendement mensuel equal-weighted des titres d'une position donnée."""
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
    """Calcule la performance mensuelle L/S nette de coûts.

    Returns
    -------
    perf : DataFrame indexé par date avec colonnes LONG, SHORT, LS, LS_net.
    costs : dict des coûts annualisés (trans, borrow, total, mensuel).
    """
    panel_sm = panel_sm.sort_values(["permno", "date"]).copy()

    # Rendement ajusté delisting
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

    # Borrow fee — médiane de market cap des shorts -> grille
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

    # Coûts de transaction dynamiques
    tc_long = get_tc_dynamic(panel_sm, long_snapshots)
    tc_short = get_tc_dynamic(panel_sm, short_snapshots)
    n_years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25

    cout_trans = (tc_long.sum() + tc_short.sum()) / n_years * 100  # décimal -> %
    cout_borrow = borrow_fee * 0.50                                # déjà en %
    cout_total = cout_trans + cout_borrow
    cost_monthly = (cout_total / 100) / 12

    perf["LS_net"] = perf["LS"] - cost_monthly

    costs = {
        "trans": cout_trans,
        "borrow": cout_borrow,
        "total": cout_total,
        "monthly": cost_monthly,
    }
    print(f"\nPerformance calculée : {len(perf)} mois")
    print(f"Coût transaction : {cout_trans:.2f}%/an")
    print(f"Coût borrow      : {cout_borrow:.2f}%/an")
    print(f"Coût total       : {cout_total:.2f}%/an")
    return perf, costs


# ----------------------------------------------------------------------------
# Diagnostic — durée de détention moyenne
# ----------------------------------------------------------------------------
def holding_durations(snapshots: dict) -> tuple[float, float, int]:
    """Durée de détention moyenne/médiane (mois) d'un jeu de snapshots."""
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
