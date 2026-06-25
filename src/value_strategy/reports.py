"""Console text summaries (IS vs OOS tables, dynamic short verdict)."""

from __future__ import annotations

import pandas as pd

from .factors import alpha_ff4, info_ratio_vs_hml, performance_stats

_METRICS = [
    ("Annual return", "Ann. Return", "{:.2%}"),
    ("Volatility", "Ann. Volatility", "{:.2%}"),
    ("Sharpe Ratio", "Sharpe", "{:.2f}"),
    ("Max Drawdown", "Max Drawdown", "{:.2%}"),
    ("Calmar Ratio", "Calmar", "{:.2f}"),
    ("Skewness", "Skewness", "{:.2f}"),
    ("VaR 5%", "VaR 5%", "{:.2%}"),
]


def print_is_oos_comparison(perf_is, perf_oos, ff5, mom_ff, costs_is, costs_oos):
    """3.5 — IS vs OOS summary table (net stats + FF4 alpha)."""
    stats_is = performance_stats(perf_is["LS_net"], ff5)
    stats_oos = performance_stats(perf_oos["LS_net"], ff5)
    alpha_is, t_is, r2_is = alpha_ff4(perf_is["LS_net"], ff5, mom_ff)
    alpha_oos, t_oos, r2_oos = alpha_ff4(perf_oos["LS_net"], ff5, mom_ff)
    ir_is = info_ratio_vs_hml(perf_is["LS_net"], ff5)
    ir_oos = info_ratio_vs_hml(perf_oos["LS_net"], ff5)

    print("\n" + "=" * 65)
    print("IS vs OOS COMPARISON — Net strategy (after all costs)")
    print("=" * 65)
    print(f"\n{'Metric':<28} {'IS (2003-13)':>14} {'OOS (2014-24)':>15}")
    print("-" * 59)
    for label, key, fmt in _METRICS:
        vi = fmt.format(stats_is[key]) if pd.notna(stats_is[key]) else "-"
        vo = fmt.format(stats_oos[key]) if pd.notna(stats_oos[key]) else "-"
        print(f"{label:<28} {vi:>14} {vo:>15}")
    print("-" * 59)
    if pd.notna(alpha_is):
        print(f"{'FF4 alpha (annual)':<28} {alpha_is:>14.2%} {alpha_oos:>15.2%}")
        print(f"{'FF4 alpha t-stat':<28} {t_is:>14.2f} {t_oos:>15.2f}")
        print(f"{'FF4 R-squared':<28} {r2_is:>14.3f} {r2_oos:>15.3f}")
    if pd.notna(ir_is):
        print(f"{'Info. Ratio vs HML':<28} {ir_is:>14.2f} {ir_oos:>15.2f}")
    print("=" * 65)
    print(f"\n  -> Sharpe change IS->OOS: {stats_oos['Sharpe'] - stats_is['Sharpe']:+.2f}")
    print(f"  -> IS cost: {costs_is['total']:.2f}%/yr | "
          f"OOS cost: {costs_oos['total']:.2f}%/yr")
    return stats_is, stats_oos


def print_dynamic_short_verdict(perf_h, metrics, ff4_results, summary, costs_oos):
    """5.3 — Decision: should we keep the dynamic short?"""
    m_naked, m_adj = metrics["naked"], metrics["adj"]
    s_no, s_yes = m_naked["Sharpe"], m_adj["Sharpe"]
    c_no, c_yes = m_naked["CAGR"], m_adj["CAGR"]
    d_no, d_yes = m_naked["MaxDD"], m_adj["MaxDD"]
    n_reduce, n_trans = summary["n_reduce"], summary["n_trans"]
    cout_orig = costs_oos["trans"] + costs_oos["borrow"]
    cout_adj = (perf_h["cost_adj"].sum() / len(perf_h)) * 12 * 100

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — SHOULD WE KEEP THE DYNAMIC SHORT?")
    print("=" * 70)
    print(f"  Sharpe : {s_no:.3f} -> {s_yes:.3f} ({s_yes - s_no:+.3f})")
    print(f"  CAGR   : {c_no:.1%} -> {c_yes:.1%} ({c_yes - c_no:+.1%})")
    print(f"  MaxDD  : {d_no:.1%} -> {d_yes:.1%} ({d_yes - d_no:+.1%})")
    print(f"  FF4 a  : {ff4_results['Original']['alpha_ann']:.4f} -> "
          f"{ff4_results['Dynamic short']['alpha_ann']:.4f}")
    print(f"  Short-reduced months (50%): {n_reduce}/{len(perf_h)} "
          f"({n_reduce / len(perf_h):.0%}) | Transitions: {n_trans}")
    print(f"  Annualized cost: {cout_orig:.2f}% -> {cout_adj:.2f}%")

    if s_yes > s_no and d_yes > d_no:
        verdict = "Sharpe IMPROVED AND MaxDD reduced -> KEEP"
    elif s_yes > s_no:
        verdict = "Sharpe improved -> KEEP"
    elif d_yes > d_no and abs(s_yes - s_no) < 0.05:
        verdict = "MaxDD reduced, Sharpe stable -> KEEP (tail-risk protection)"
    elif c_yes > c_no:
        verdict = "CAGR up, Sharpe similar -> CONSIDER"
    elif d_yes > d_no:
        verdict = "MaxDD reduced but Sharpe penalized -> CALIBRATE"
    else:
        verdict = "No clear improvement -> DO NOT KEEP"
    print(f"\n  VERDICT: {verdict}")
    print("=" * 70)
