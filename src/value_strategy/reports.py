"""Synthèses textuelles pour la console (tableaux IS vs OOS, verdict short dyn.)."""

from __future__ import annotations

import pandas as pd

from .factors import alpha_ff4, info_ratio_vs_hml, performance_stats

_METRICS = [
    ("Rendement annuel", "Ann. Return", "{:.2%}"),
    ("Volatilité", "Ann. Volatility", "{:.2%}"),
    ("Sharpe Ratio", "Sharpe", "{:.2f}"),
    ("Max Drawdown", "Max Drawdown", "{:.2%}"),
    ("Calmar Ratio", "Calmar", "{:.2f}"),
    ("Skewness", "Skewness", "{:.2f}"),
    ("VaR 5%", "VaR 5%", "{:.2%}"),
]


def print_is_oos_comparison(perf_is, perf_oos, ff5, mom_ff, costs_is, costs_oos):
    """3.5 — Tableau de synthèse IS vs OOS (stats nettes + alpha FF4)."""
    stats_is = performance_stats(perf_is["LS_net"], ff5)
    stats_oos = performance_stats(perf_oos["LS_net"], ff5)
    alpha_is, t_is, r2_is = alpha_ff4(perf_is["LS_net"], ff5, mom_ff)
    alpha_oos, t_oos, r2_oos = alpha_ff4(perf_oos["LS_net"], ff5, mom_ff)
    ir_is = info_ratio_vs_hml(perf_is["LS_net"], ff5)
    ir_oos = info_ratio_vs_hml(perf_oos["LS_net"], ff5)

    print("\n" + "=" * 65)
    print("COMPARAISON IS vs OOS — Stratégie nette (après tous coûts)")
    print("=" * 65)
    print(f"\n{'Métrique':<28} {'IS (2003-13)':>14} {'OOS (2014-24)':>15}")
    print("-" * 59)
    for label, key, fmt in _METRICS:
        vi = fmt.format(stats_is[key]) if pd.notna(stats_is[key]) else "—"
        vo = fmt.format(stats_oos[key]) if pd.notna(stats_oos[key]) else "—"
        print(f"{label:<28} {vi:>14} {vo:>15}")
    print("-" * 59)
    if pd.notna(alpha_is):
        print(f"{'Alpha FF4 annuel':<28} {alpha_is:>14.2%} {alpha_oos:>15.2%}")
        print(f"{'t-stat alpha FF4':<28} {t_is:>14.2f} {t_oos:>15.2f}")
        print(f"{'R² FF4':<28} {r2_is:>14.3f} {r2_oos:>15.3f}")
    if pd.notna(ir_is):
        print(f"{'Info. Ratio vs HML':<28} {ir_is:>14.2f} {ir_oos:>15.2f}")
    print("=" * 65)
    print(f"\n  → Variation Sharpe IS→OOS : {stats_oos['Sharpe'] - stats_is['Sharpe']:+.2f}")
    print(f"  → Coût IS : {costs_is['total']:.2f}%/an | "
          f"Coût OOS : {costs_oos['total']:.2f}%/an")
    return stats_is, stats_oos


def print_dynamic_short_verdict(perf_h, metrics, ff4_results, summary, costs_oos):
    """5.3 — Décision : faut-il garder le short dynamique ?"""
    m_naked, m_adj = metrics["naked"], metrics["adj"]
    s_no, s_yes = m_naked["Sharpe"], m_adj["Sharpe"]
    c_no, c_yes = m_naked["CAGR"], m_adj["CAGR"]
    d_no, d_yes = m_naked["MaxDD"], m_adj["MaxDD"]
    n_reduce, n_trans = summary["n_reduce"], summary["n_trans"]
    cout_orig = costs_oos["trans"] + costs_oos["borrow"]
    cout_adj = (perf_h["cost_adj"].sum() / len(perf_h)) * 12 * 100

    print("\n" + "=" * 70)
    print("  RÉSUMÉ FINAL — FAUT-IL GARDER LE SHORT DYNAMIQUE ?")
    print("=" * 70)
    print(f"  Sharpe  : {s_no:.3f} → {s_yes:.3f} ({s_yes - s_no:+.3f})")
    print(f"  CAGR    : {c_no:.1%} → {c_yes:.1%} ({c_yes - c_no:+.1%})")
    print(f"  MaxDD   : {d_no:.1%} → {d_yes:.1%} ({d_yes - d_no:+.1%})")
    print(f"  α FF4   : {ff4_results['Original']['alpha_ann']:.4f} → "
          f"{ff4_results['Short dynamique']['alpha_ann']:.4f}")
    print(f"  Mois short réduit (50%) : {n_reduce}/{len(perf_h)} "
          f"({n_reduce / len(perf_h):.0%}) | Transitions : {n_trans}")
    print(f"  Coût annualisé : {cout_orig:.2f}% → {cout_adj:.2f}%")

    if s_yes > s_no and d_yes > d_no:
        verdict = "✓ Sharpe AMÉLIORÉ ET MaxDD réduit → GARDER"
    elif s_yes > s_no:
        verdict = "✓ Sharpe amélioré → GARDER"
    elif d_yes > d_no and abs(s_yes - s_no) < 0.05:
        verdict = "~ MaxDD réduit, Sharpe stable → GARDER (protection tail risk)"
    elif c_yes > c_no:
        verdict = "~ CAGR augmente, Sharpe similaire → À CONSIDÉRER"
    elif d_yes > d_no:
        verdict = "~ MaxDD réduit mais Sharpe pénalisé → À CALIBRER"
    else:
        verdict = "✗ Pas d'amélioration claire → NE PAS GARDER"
    print(f"\n  VERDICT : {verdict}")
    print("=" * 70)
