"""Partie 5 — Réduction dynamique du short en junk rally.

Quand le ML (Partie 4) détecte un régime d'euphorie, le poids du short passe de
100 % à 50 % : ``LS_adjusted = LONG − w × SHORT`` avec w = 0.5 en euphoria,
1.0 sinon. Coûts : transition (repositionnement de 50 % du short book) +
borrow fee proportionnel au poids short.

Timing : signal au mois t -> appliqué au mois t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .factors import compute_metrics


def build_dynamic_short(perf_oos: pd.DataFrame, signals_df: pd.DataFrame, costs_oos: dict):
    """Construit la série de performance avec short dynamique piloté par le ML.

    Parameters
    ----------
    perf_oos : performance OOS (index = date, colonnes LONG/SHORT/LS/LS_net).
    signals_df : signaux ML (date, signal_gb, gb_prob).
    costs_oos : dict de coûts OOS issu de ``portfolio.compute_performance``.

    Returns
    -------
    perf_h : DataFrame mensuel enrichi (short_weight, LS_adjusted, LS_adj_net...).
    summary : dict {n_reduce, n_full, n_trans}.
    """
    print("=" * 60)
    print("  PARTIE 5 — SHORT DYNAMIQUE PILOTÉ PAR LE ML")
    print("=" * 60)

    # ── 5.1a  Préparer perf_oos ───────────────────────────────────────────
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

    # ── 5.1b  Alignement signaux -> rendements (t -> t+1) ─────────────────
    print("[5.1b] Alignement signaux ML → rendements OOS...")
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

    # ── 5.1c  Short weight dynamique ──────────────────────────────────────
    perf_h["short_weight"] = np.where(
        perf_h["signal_gb"] == "SHORT_REDUCE",
        config.SHORT_WEIGHT_REDUCE,
        config.SHORT_WEIGHT_FULL,
    )
    perf_h["LS_adjusted"] = perf_h["LONG"] - perf_h["short_weight"] * perf_h["SHORT"]

    # ── 5.1d  Coûts ajustés ───────────────────────────────────────────────
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

    # ── 5.1e  Résumé ──────────────────────────────────────────────────────
    n_reduce = int((perf_h["short_weight"] == config.SHORT_WEIGHT_REDUCE).sum())
    n_full = int((perf_h["short_weight"] == config.SHORT_WEIGHT_FULL).sum())
    n_trans = int((perf_h["weight_change"] > 0).sum())

    print(f"\n  Période : {perf_h['date'].min().strftime('%Y-%m')} → "
          f"{perf_h['date'].max().strftime('%Y-%m')} ({len(perf_h)} mois)")
    print(f"  Mois SHORT_REDUCE (50%) : {n_reduce} ({n_reduce / len(perf_h):.0%})")
    print(f"  Mois FULL_SHORT (100%)  : {n_full} ({n_full / len(perf_h):.0%})")
    print(f"  Transitions             : {n_trans}")
    print(f"  Coût transitions total  : {perf_h['transition_cost'].sum() * 100:.2f}%")

    return perf_h, {"n_reduce": n_reduce, "n_full": n_full, "n_trans": n_trans}


def attach_ff_and_metrics(perf_h: pd.DataFrame, ff: pd.DataFrame):
    """Merge les facteurs FF et calcule les métriques comparées.

    Returns
    -------
    perf_h (avec Mkt-RF, RF), metrics : dict {naked, adj, mkt}.
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
    """Régression FF4 (OOS) sur les deux stratégies. Renvoie un dict de résultats."""
    oos_ff = perf_h[["date", "LS_net", "LS_adj_net"]].merge(
        ff[["date", "Mkt-RF", "SMB", "HML", "Mom"]], on="date", how="inner",
    )
    X_oos = oos_ff[["Mkt-RF", "SMB", "HML", "Mom"]].values
    X_oos_c = np.column_stack([np.ones(len(X_oos)), X_oos])

    ff4_results = {}
    for name, col in [("Original", "LS_net"), ("Short dynamique", "LS_adj_net")]:
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
