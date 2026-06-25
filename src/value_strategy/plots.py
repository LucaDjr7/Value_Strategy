"""Presentation charts -> results/charts.

All notebook figures, faithfully ported with variables passed explicitly (no
more globals). Each function saves a PNG and returns the file path.

Performance figures (Parts 2-3):
  fig1 cumulative wealth, fig2 rolling Sharpe, fig3 annual returns,
  fig4 drawdown, fig5 calendar heatmap, fig6 robustness
Dashboards (Parts 4-5):
  ML regime dashboard, dynamic short, final cumulative wealth
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .factors import alpha_ff4, info_ratio_vs_hml, performance_stats

WINDOW = 36


def setup_style() -> None:
    """Apply the matplotlib style and create the output directory."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    })
    config.ensure_dirs()


def _ref_series(ff5):
    """Market and HML series from FF5 (or None)."""
    if ff5 is not None:
        return ff5["MKT"] + ff5["RF"], ff5["HML"]
    return None, None


def rolling_sharpe(series, ff5=None, window: int = WINDOW):
    """Annualized rolling Sharpe (dynamic RF if ff5 provided)."""
    if ff5 is None:
        excess = series
    else:
        rf_dynamic = ff5["RF"].reindex(series.index).ffill()
        excess = series - rf_dynamic
    return excess.rolling(window).mean() / series.rolling(window).std() * np.sqrt(12)


# ----------------------------------------------------------------------------
# Fig 1 — Cumulative wealth (base $1, log)
# ----------------------------------------------------------------------------
def plot_cumulative_wealth(perf_is, perf_oos, ff5):
    mkt_series, hml_series = _ref_series(ff5)
    fig, ax = plt.subplots(figsize=(14, 6))

    cum_is = (1 + perf_is["LS_net"].fillna(0)).cumprod()
    cum_oos = (1 + perf_oos["LS_net"].fillna(0)).cumprod()
    cum_oos_raccorde = cum_oos * cum_is.iloc[-1]

    ax.plot(cum_is.index, cum_is, color=config.C_IS, lw=2.5, ls="-",
            label="L/S net (IS 2003-2013)")
    ax.plot(cum_oos_raccorde.index, cum_oos_raccorde, color=config.C_OOS, lw=2.5,
            ls="--", label="L/S net (OOS 2014-2024)")

    if ff5 is not None:
        idx_full = cum_is.index.union(cum_oos_raccorde.index)
        for serie, color, label, ls in [
            (mkt_series, config.C_MKT, "US market (VW)", "-"),
            (hml_series, config.C_HML, "Passive HML (FF)", "--"),
        ]:
            c = (1 + serie.reindex(idx_full).fillna(0)).cumprod()
            ax.plot(c.index, c, color=color, lw=1.5, ls=ls, alpha=0.8, label=label)

    ax.axvline(pd.Timestamp(config.OOS_START), color=config.C_VLINE, lw=1.5, ls=":",
               label="OOS start (2014)")
    ax.axvspan(pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31"),
               alpha=0.06, color="red", label="Value Drawdown (2010-2020)")

    ax.set_yscale("log")
    ax.set_title("Cumulative Wealth — base $1, log scale\n"
                 "In-sample (2003-2013) | Out-of-sample (2014-2024)", fontsize=12)
    ax.set_ylabel("Value of a $1 invested portfolio")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, which="both")
    plt.tight_layout()
    return _save(fig, "fig1_cumulative_wealth.png")


# ----------------------------------------------------------------------------
# Fig 2 — Rolling Sharpe
# ----------------------------------------------------------------------------
def plot_rolling_sharpe(perf_is, perf_oos, ff5):
    mkt_series, hml_series = _ref_series(ff5)
    fig, ax = plt.subplots(figsize=(14, 5))

    rs_is = rolling_sharpe(perf_is["LS_net"], ff5)
    rs_oos = rolling_sharpe(perf_oos["LS_net"], ff5)
    ax.plot(rs_is.index, rs_is, color=config.C_IS, lw=2, ls="-", label="L/S net (IS)")
    ax.plot(rs_oos.index, rs_oos, color=config.C_OOS, lw=2, ls="--", label="L/S net (OOS)")

    if ff5 is not None:
        idx = rs_is.index.union(rs_oos.index)
        for serie, color, label, ls in [
            (mkt_series, config.C_MKT, "US market", "-"),
            (hml_series, config.C_HML, "Passive HML", "--"),
        ]:
            rs = rolling_sharpe(serie.reindex(idx).dropna(), ff5)
            ax.plot(rs.index, rs, color=color, lw=1.2, ls=ls, alpha=0.7, label=label)

    ax.axhline(0, linestyle="--", lw=0.8, color="black")
    ax.axhline(1, linestyle=":", lw=0.8, color=config.C_IS, alpha=0.5)
    ax.axvline(pd.Timestamp(config.OOS_START), color=config.C_VLINE, lw=1.5, ls=":",
               label="OOS start (2014)")
    ax.axvspan(pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31"),
               alpha=0.06, color="red", label="Value Drawdown (2010-2020)")
    ax.set_title(f"Rolling Sharpe Ratio ({WINDOW} months) — net series", fontsize=12)
    ax.set_ylabel("Annualized Sharpe")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return _save(fig, "fig2_rolling_sharpe.png")


# ----------------------------------------------------------------------------
# Fig 3 — Annual returns
# ----------------------------------------------------------------------------
def plot_annual_returns(perf_is, perf_oos, ff5):
    _, hml_series = _ref_series(ff5)
    fig, ax = plt.subplots(figsize=(14, 5))

    all_ls = pd.concat([perf_is["LS_net"], perf_oos["LS_net"]]).sort_index()
    ann_ls = all_ls.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    x_pos = np.arange(len(ann_ls))
    w = 0.30

    colors_bars = [config.C_IS if str(d.year) <= "2013" else config.C_OOS
                   for d in ann_ls.index]
    ax.bar(x_pos, ann_ls * 100, width=w * 2, color=colors_bars, alpha=0.85,
           label="_nolegend_")

    if ff5 is not None:
        ann_hml = hml_series.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        ann_hml = ann_hml.reindex(ann_ls.index)
        ax.bar(x_pos + w, ann_hml * 100, width=w, color=config.C_HML, alpha=0.6,
               label="Passive HML")

    ax.axhline(0, lw=0.8, color="black")
    ax.axvline(
        x_pos[list(ann_ls.index).index(
            next(d for d in ann_ls.index if d.year == 2013)
        )] + 0.5,
        color=config.C_VLINE, lw=1.5, ls=":", label="OOS start (2014)",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(d.year) for d in ann_ls.index], rotation=45, ha="right")
    ax.set_title("Annual L/S net return — IS (blue) vs OOS (green) vs HML (red)")
    ax.set_ylabel("Annual return (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    patch_is = mpatches.Patch(color=config.C_IS, alpha=0.85, label="L/S net (IS 2003-2013)")
    patch_oos = mpatches.Patch(color=config.C_OOS, alpha=0.85, label="L/S net (OOS 2014-2024)")
    patch_hml = mpatches.Patch(color=config.C_HML, alpha=0.6, label="Passive HML (FF)")
    ax.legend(handles=[patch_is, patch_oos, patch_hml], fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return _save(fig, "fig3_annual_returns.png")


# ----------------------------------------------------------------------------
# Fig 4 — Drawdown
# ----------------------------------------------------------------------------
def plot_drawdown(perf_is, perf_oos, ff5):
    _, hml_series = _ref_series(ff5)
    fig, ax = plt.subplots(figsize=(14, 5))

    for serie, color, label, lw, ls in [
        (perf_is["LS_net"], config.C_IS, "L/S net (IS)", 2.0, "-"),
        (perf_oos["LS_net"], config.C_OOS, "L/S net (OOS)", 2.0, "--"),
    ]:
        w_dd = (1 + serie.fillna(0)).cumprod()
        dd = (w_dd / w_dd.cummax() - 1) * 100
        ax.fill_between(dd.index, dd, 0, alpha=0.2, color=color)
        ax.plot(dd.index, dd, color=color, lw=lw, ls=ls, label=label)

    if ff5 is not None:
        idx_full = perf_is.index.union(perf_oos.index)
        s_hml = hml_series.reindex(idx_full).fillna(0)
        w_hml = (1 + s_hml).cumprod()
        dd_hml = (w_hml / w_hml.cummax() - 1) * 100
        ax.plot(dd_hml.index, dd_hml, color=config.C_HML, lw=1.5, ls=":", alpha=0.8,
                label="Passive HML")

    ax.axvline(pd.Timestamp(config.OOS_START), color=config.C_VLINE, lw=1.5, ls=":",
               label="OOS start")
    ax.set_title("Drawdown — IS + OOS vs passive HML")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return _save(fig, "fig4_drawdown.png")


# ----------------------------------------------------------------------------
# Fig 5 — Calendar heatmap (OOS)
# ----------------------------------------------------------------------------
def plot_calendar_heatmap(perf_oos):
    monthly_matrix = perf_oos["LS_net"].to_frame(name="ret")
    monthly_matrix["year"] = monthly_matrix.index.year
    monthly_matrix["month"] = monthly_matrix.index.month
    pivot = monthly_matrix.pivot(index="year", columns="month", values="ret") * 100
    pivot = pivot.astype(float)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")

    for i, year in enumerate(pivot.index):
        for j, month in enumerate(pivot.columns):
            val = pivot.loc[year, month]
            if pd.notna(val):
                color = "black" if abs(val) < vmax * 0.6 else "white"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")

    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for year in [2017, 2018, 2019, 2020]:
        if year in pivot.index.tolist():
            y_idx = pivot.index.tolist().index(year)
            ax.add_patch(plt.Rectangle((-0.5, y_idx - 0.5), 12, 1, linewidth=2.5,
                                       edgecolor="#212121", facecolor="none",
                                       linestyle=":"))
    if 2017 in pivot.index.tolist():
        y_start = pivot.index.tolist().index(2017)
        y_end = pivot.index.tolist().index(2020)
        ax.annotate("Value stress period", xy=(11.5, y_start - 0.5),
                    xytext=(12.2, (y_start + y_end) / 2), fontsize=8,
                    color="#212121", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#212121"))

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Net monthly return (%)", fontsize=9)
    ax.set_title("Calendar Heatmap — Monthly L/S net returns\n"
                 "Out-of-sample (2014-2024) | Outline: value stress period (2017-2020)",
                 fontsize=12)
    plt.tight_layout()
    return _save(fig, "fig5_calendar_heatmap.png")


# ----------------------------------------------------------------------------
# Fig 6 — Robustness by sub-period
# ----------------------------------------------------------------------------
def plot_robustness(perf_is, perf_oos, ff5):
    all_ls_full = pd.concat([perf_is["LS_net"], perf_oos["LS_net"]]).sort_index()
    subperiods = {
        "Crisis\n2007-09\n(IS)": ("2007-01-01", "2009-06-30"),
        "Recovery\n2009-13\n(IS)": ("2009-07-01", "2013-12-31"),
        "Post-crisis\n2014-16\n(OOS)": ("2014-01-01", "2016-12-31"),
        "Value drawdown\n2017-20\n(OOS)": ("2017-01-01", "2020-12-31"),
        "Rebound\n2021-24\n(OOS)": ("2021-01-01", "2024-12-31"),
    }
    hml_full = ff5["HML"].reindex(all_ls_full.index) if ff5 is not None else None

    sp_names, sp_sharpe_ls, sp_sharpe_hml = [], [], []
    for name, (start, end) in subperiods.items():
        sub_ls = all_ls_full.loc[start:end].dropna()
        if len(sub_ls) < 6:
            continue
        sp_names.append(name)
        sp_sharpe_ls.append(sub_ls.mean() / sub_ls.std() * np.sqrt(12))
        if ff5 is not None:
            sub_hml = hml_full.loc[start:end].dropna()
            sp_sharpe_hml.append(
                sub_hml.mean() / sub_hml.std() * np.sqrt(12) if len(sub_hml) > 5 else np.nan
            )
        else:
            sp_sharpe_hml.append(np.nan)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(sp_names))
    w = 0.35
    bars_ls = ax.bar(x - w / 2, sp_sharpe_ls, width=w, color=config.C_OOS, alpha=0.85,
                     label="L/S net (our strategy)")
    bars_hml = ax.bar(x + w / 2, sp_sharpe_hml, width=w, color=config.C_HML, alpha=0.65,
                      label="Passive HML (FF)")
    for bars, color in [(bars_ls, config.C_OOS), (bars_hml, config.C_HML)]:
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 0.04 if h >= 0 else -0.04
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:.2f}",
                    ha="center", va=va, fontsize=9, color=color, fontweight="bold")

    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="gray", lw=1.0, ls=":", alpha=0.6, label="Sharpe = 1.0")
    ax.axvline(1.5, color=config.C_VLINE, lw=1.5, ls=":", alpha=0.7)
    ax.text(0.5, ax.get_ylim()[0] if sp_sharpe_ls else -0.5, "IN-SAMPLE",
            ha="center", fontsize=8, color=config.C_IS, style="italic")
    ax.text(3.0, ax.get_ylim()[0] if sp_sharpe_ls else -0.5, "OUT-OF-SAMPLE",
            ha="center", fontsize=8, color=config.C_OOS, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(sp_names, fontsize=9)
    ax.set_ylabel("Annualized Sharpe", fontsize=10)
    ax.set_title("Robustness — Sharpe by sub-period\n"
                 "Our strategy (green) vs passive HML (red) across 5 market regimes",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return _save(fig, "fig6_robustness.png")


# ----------------------------------------------------------------------------
# ML dashboard (Part 4)
# ----------------------------------------------------------------------------
def plot_ml_dashboard(mkt_df, results_df, importance_df, metrics_df,
                      regime_method, avg_threshold):
    fig = plt.figure(figsize=(22, 24))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.3], hspace=0.45, wspace=0.32)
    colors = {"stress": "#e74c3c", "lr": "#3498db", "gb": "#e67e22"}

    ax1a = fig.add_subplot(gs[0, 0])
    valid = mkt_df.dropna(subset=["amihud_mkt"])
    ax1a.plot(valid["date"], valid["amihud_mkt"], color="#2c3e50", lw=0.8)
    ax1a.set_title("RAW Amihud (structural trend)", fontsize=11, fontweight="bold")
    ax1a.set_ylabel("Amihud")
    _year_axis(ax1a, 4)

    ax1b = fig.add_subplot(gs[0, 1])
    valid_z = mkt_df.dropna(subset=["amihud_mkt_z", "regime"])
    dates = valid_z["date"]
    regimes = valid_z["regime"].values
    ax1b.fill_between(dates, -3, 5, where=(regimes == 1), color=colors["stress"],
                      alpha=0.25, label="Stress (smoothed)")
    ax1b.plot(dates, valid_z["amihud_mkt_z"], color="#2c3e50", lw=0.8)
    ax1b.axhline(0, color="gray", ls="--", lw=0.5)
    ax1b.set_title("Detrended Amihud + smoothed regimes", fontsize=11, fontweight="bold")
    ax1b.set_ylabel("Z-score")
    ax1b.set_ylim(-3, 5)
    ax1b.legend(fontsize=9)
    _year_axis(ax1b, 4)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, -4, 6, where=(regimes == 1), color=colors["stress"],
                     alpha=0.12, label="Stress")
    for feat, fc, lbl in [
        ("vix_z", "#e74c3c", "VIX z-score"),
        ("credit_spread_z", "#3498db", "Credit spread z"),
        ("term_spread_z", "#27ae60", "Term spread z"),
    ]:
        if feat in valid_z.columns:
            ax2.plot(dates, valid_z[feat], color=fc, lw=0.9, alpha=0.85, label=lbl)
    ax2.axhline(0, color="gray", ls="--", lw=0.5)
    ax2.set_title("FRED macro features (z-scores) — stress regimes shaded",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylabel("Z-score")
    ax2.set_ylim(-4, 6)
    _year_axis(ax2, 2)

    ax3 = fig.add_subplot(gs[2, :])
    if len(results_df) > 0:
        oos = results_df.sort_values("date")
        ax3.fill_between(oos["date"], 0, 1, where=(oos["y_true"] == 1),
                         color=colors["stress"], alpha=0.20,
                         transform=ax3.get_xaxis_transform(),
                         label="Actual stress (HMM label)")
        ax3.plot(oos["date"], oos["lr_prob"], color=colors["lr"], lw=1.2, alpha=0.8,
                 label="Logistic Regression P(stress)")
        ax3.plot(oos["date"], oos["gb_prob"], color=colors["gb"], lw=1.2, alpha=0.8,
                 label="Gradient Boosting P(stress)")
        ax3.axhline(avg_threshold, color="#555555", ls="--", lw=1.2,
                    label=f"Mean optimal threshold = {avg_threshold:.2f}")
        ax3.set_ylabel("P(stress)")
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=9, loc="upper right")
        _year_axis(ax3, 2)
    else:
        ax3.text(0.5, 0.5, "No OOS data available\n(min_train too high?)",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=12, color="red")
    ax3.set_title("Walk-Forward OOS — P(stress t+1) | Gradient Boosting vs Logistic Regression",
                  fontsize=12, fontweight="bold")

    ax4 = fig.add_subplot(gs[3, 0])
    if importance_df is not None and len(importance_df) > 0:
        top12 = importance_df.head(12).iloc[::-1].copy()
        fc_colors = []
        for f in top12["feature"]:
            if any(m in f for m in ["vix", "credit", "term", "ted", "fed", "consumer"]):
                fc_colors.append("#e74c3c")
            elif "chg" in f or "ma3" in f:
                fc_colors.append("#3498db")
            else:
                fc_colors.append("#8e44ad")
        ax4.barh(top12["feature"], top12["importance"], xerr=top12["std"],
                 color=fc_colors, alpha=0.82, capsize=3)
        ax4.set_title("Feature Importance — Gradient Boosting\n"
                      "(red=FRED macro | blue=momentum | purple=level)",
                      fontsize=10, fontweight="bold")
        ax4.set_xlabel("Importance (permutation)")
        ax4.grid(axis="x", alpha=0.2)
    else:
        ax4.text(0.5, 0.5, "Feature importance unavailable", ha="center",
                 va="center", transform=ax4.transAxes)

    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis("off")
    if len(metrics_df) > 0:
        cell_text = []
        col_labels = ["Model"] + list(metrics_df.columns)
        for idx, row in metrics_df.iterrows():
            cell_text.append([idx] + [f"{v:.3f}" if isinstance(v, float) else str(v)
                                      for v in row.values])
        table = ax5.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                          loc="center", colWidths=[0.30] + [0.14] * len(metrics_df.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.0)
        _style_table_header(table, col_labels, cell_text)
    ax5.set_title("OOS metrics — Walk-Forward V3", fontsize=11, fontweight="bold",
                  pad=15, y=0.95)

    plt.suptitle(f"LIQUIDITY REGIME DETECTION V3\n"
                 f"{regime_method} + Smoothing (min_dur=3, bridge=2) + FRED features",
                 fontsize=14, fontweight="bold", y=1.01)
    return _save(fig, "liquidity_regime_dashboard_v3.png")


# ----------------------------------------------------------------------------
# Dynamic short dashboard (Part 5)
# ----------------------------------------------------------------------------
def plot_dynamic_short(perf_h, metrics, ff4_results, summary):
    m_naked, m_adj = metrics["naked"], metrics["adj"]
    n_reduce, n_trans = summary["n_reduce"], summary["n_trans"]

    fig = plt.figure(figsize=(20, 22))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1.2, 0.8, 1, 1.2], hspace=0.42, wspace=0.30)
    C_ORIG, C_ADJ, C_MKT = "#2196F3", "#FF9800", "#9E9E9E"
    C_REDUCE, C_FULL = "#e74c3c", "#4CAF50"

    dates_h = perf_h["date"]
    cum_orig = (1 + perf_h["LS_net"]).cumprod()
    cum_adj = (1 + perf_h["LS_adj_net"]).cumprod()
    cum_mkt = (1 + perf_h["Mkt-RF"]).cumprod()

    ax1 = fig.add_subplot(gs[0, :])
    reduce_mask = perf_h["signal_gb"] == "SHORT_REDUCE"
    ax1.fill_between(dates_h, 0, cum_adj.max() * 1.3, where=reduce_mask.values,
                     color=C_REDUCE, alpha=0.10, label="Short reduced to 50% (euphoria)")
    ax1.plot(dates_h, cum_orig, color=C_ORIG, lw=1.8,
             label=f"Original (Sharpe={m_naked['Sharpe']:.2f})")
    ax1.plot(dates_h, cum_adj, color=C_ADJ, lw=1.8,
             label=f"Dynamic short (Sharpe={m_adj['Sharpe']:.2f})")
    ax1.plot(dates_h, cum_mkt, color=C_MKT, lw=1.0, ls="--", label="Market (Mkt-RF)")
    ax1.set_title("OOS cumulative return — ML-driven dynamic short",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Value (base 1)")
    ax1.legend(fontsize=10, loc="upper left")
    _year_axis(ax1, 2)

    ax2 = fig.add_subplot(gs[1, :])
    dd_orig = cum_orig / cum_orig.cummax() - 1
    dd_adj = cum_adj / cum_adj.cummax() - 1
    ax2.fill_between(dates_h, dd_orig, 0, color=C_ORIG, alpha=0.30,
                     label=f"Original (MaxDD={m_naked['MaxDD']:.1%})")
    ax2.fill_between(dates_h, dd_adj, 0, color=C_ADJ, alpha=0.30,
                     label=f"Dyn. short (MaxDD={m_adj['MaxDD']:.1%})")
    ax2.plot(dates_h, dd_orig, color=C_ORIG, lw=0.8)
    ax2.plot(dates_h, dd_adj, color=C_ADJ, lw=0.8)
    ax2.set_title("Compared drawdowns", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=9, loc="lower left")
    _year_axis(ax2, 2)

    ax3 = fig.add_subplot(gs[2, 0])
    colors_w = [C_REDUCE if w < 1 else C_FULL for w in perf_h["short_weight"]]
    ax3.bar(dates_h, perf_h["short_weight"] * 100, width=25, color=colors_w, alpha=0.7)
    ax3.axhline(100, color="black", ls="--", lw=0.5, alpha=0.5)
    ax3.axhline(50, color=C_REDUCE, ls="--", lw=0.5, alpha=0.5)
    ax3.set_title("Short weight (%)\nred = reduced 50% | green = full 100%",
                  fontsize=10, fontweight="bold")
    ax3.set_ylabel("Short weight (%)")
    ax3.set_ylim(0, 120)
    _year_axis(ax3, 2)

    ax4 = fig.add_subplot(gs[2, 1])
    diff = (perf_h["LS_adj_net"] - perf_h["LS_net"]) * 100
    colors_d = [C_ADJ if v > 0 else C_ORIG for v in diff]
    ax4.bar(dates_h, diff, width=25, color=colors_d, alpha=0.7)
    ax4.axhline(0, color="black", lw=0.5)
    ax4.set_title("Monthly return delta (Dyn. short - Original, %)\n"
                  "orange = dyn. short wins | blue = original wins",
                  fontsize=10, fontweight="bold")
    ax4.set_ylabel("Return delta (%)")
    _year_axis(ax4, 2)

    ax5 = fig.add_subplot(gs[3, 0])
    ax5.axis("off")
    tbl = pd.DataFrame({"Original": m_naked, "Dyn. short": m_adj}).T
    cell_text, col_labels = [], ["Strategy"] + list(tbl.columns)
    for idx, row in tbl.iterrows():
        fmt = [idx]
        for col, val in row.items():
            if col in ["CAGR", "Vol", "MaxDD", "Hit Rate"]:
                fmt.append(f"{val:.1%}")
            elif col == "N months":
                fmt.append(f"{val:.0f}")
            else:
                fmt.append(f"{val:.2f}")
        cell_text.append(fmt)
    table = ax5.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                      loc="center", colWidths=[0.22] + [0.11] * len(tbl.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    _style_table_header(table, col_labels, cell_text)
    ax5.set_title("OOS metrics", fontsize=11, fontweight="bold", pad=15, y=0.92)

    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis("off")
    ff4_text = []
    ff4_labels = ["Strategy", "alpha (ann.)", "t(a)", "b_MKT", "b_SMB", "b_HML", "b_MOM"]
    for name, v in ff4_results.items():
        ff4_text.append([name, f"{v['alpha_ann']:.4f}", f"{v['t_alpha']:.2f}",
                         f"{v['beta_mkt']:.3f}", f"{v['beta_smb']:.3f}",
                         f"{v['beta_hml']:.3f}", f"{v['beta_mom']:.3f}"])
    table2 = ax6.table(cellText=ff4_text, colLabels=ff4_labels, cellLoc="center",
                       loc="center", colWidths=[0.26] + [0.13] * 6)
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2.2)
    _style_table_header(table2, ff4_labels, ff4_text)
    ax6.set_title("FF4 regression (OOS)", fontsize=11, fontweight="bold", pad=15, y=0.92)

    plt.suptitle(f"DYNAMIC SHORT — 50% reduction during Junk Rally (ML)\n"
                 f"Reduced months: {n_reduce}/{len(perf_h)} | Transitions: {n_trans}",
                 fontsize=14, fontweight="bold", y=1.01)
    return _save(fig, "dynamic_short_v3.png")


# ----------------------------------------------------------------------------
# Final figure — cumulative comparison (IS+OOS)
# ----------------------------------------------------------------------------
def plot_final_comparison(perf_is, perf_oos, perf_h, ff5, stats_is, stats_oos, m_adj):
    orig_full = pd.concat([perf_is["LS_net"], perf_oos["LS_net"]]).sort_index()
    orig_full = orig_full[~orig_full.index.duplicated(keep="first")]
    cum_orig = (1 + orig_full.fillna(0)).cumprod()

    adj_oos = perf_h.set_index("date")["LS_adj_net"].sort_index()
    adj_full = pd.concat([perf_is["LS_net"], adj_oos]).sort_index()
    adj_full = adj_full[~adj_full.index.duplicated(keep="first")]
    cum_adj = (1 + adj_full.fillna(0)).cumprod()

    idx_full = cum_orig.index
    hml_aligned = ff5["HML"].reindex(idx_full).fillna(0) if ff5 is not None else pd.Series(0, index=idx_full)
    cum_hml = (1 + hml_aligned).cumprod()

    fig, ax = plt.subplots(figsize=(16, 7))
    C_ORIG, C_ADJ, C_HML = "#2196F3", "#FF9800", "#F44336"
    ax.plot(cum_orig.index, cum_orig, color=C_ORIG, lw=2.5,
            label=f"L/S net original (Sharpe IS={stats_is['Sharpe']:.2f} | "
                  f"OOS={stats_oos['Sharpe']:.2f})")
    ax.plot(cum_adj.index, cum_adj, color=C_ADJ, lw=2.5, ls="--",
            label=f"L/S net dynamic short (Sharpe OOS={m_adj['Sharpe']:.2f})")
    ax.plot(cum_hml.index, cum_hml, color=C_HML, lw=1.8, ls=":",
            label="Passive HML Fama-French")
    ax.axvline(pd.Timestamp(config.OOS_START), color="#212121", lw=1.5, ls=":",
               label="OOS start (2014)")

    if "signal_gb" in perf_h.columns:
        for d in perf_h.loc[perf_h["signal_gb"] == "SHORT_REDUCE", "date"]:
            ax.axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15),
                       alpha=0.08, color=C_ADJ, zorder=0)
        ax.fill_between([], [], alpha=0.15, color=C_ADJ, label="Short-reduced months 50% (ML)")

    ax.set_yscale("log")
    ax.set_title("Cumulative Wealth — base $1, log scale\n"
                 "Original vs Dynamic Short vs passive HML\n"
                 "In-sample (2003-2013) | Out-of-sample (2014-2024)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Value of a $1 invested portfolio")
    _year_axis(ax, 2)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, which="both")
    for cum, color in [(cum_orig, C_ORIG), (cum_adj, C_ADJ), (cum_hml, C_HML)]:
        ax.annotate(f"${cum.iloc[-1]:.2f}", xy=(cum.index[-1], cum.iloc[-1]),
                    xytext=(10, 0), textcoords="offset points", fontsize=10,
                    fontweight="bold", color=color, va="center")
    plt.tight_layout()
    path = _save(fig, "fig_final_cumulative_comparison.png")
    print(f"  Final $1 value: original ${cum_orig.iloc[-1]:.2f} | "
          f"dyn. short ${cum_adj.iloc[-1]:.2f} | HML ${cum_hml.iloc[-1]:.2f}")
    return path


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------
def _year_axis(ax, every: int) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(every))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.2)


def _style_table_header(table, col_labels, cell_text) -> None:
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell_text) + 1):
        bg = "#ecf0f1" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)


def _save(fig, filename: str):
    config.ensure_dirs()
    path = config.CHARTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {path.relative_to(config.ROOT_DIR)}")
    return path
