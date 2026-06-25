"""Part 3 — Fama-French factors and performance statistics.

Direct download from the Kenneth French Data Library, then:
  - performance statistics (Sharpe, drawdown, VaR, t-stat...)
  - FF4 alpha (MKT + SMB + HML + MOM) with t-stat
  - information ratio vs passive HML

Two loaders coexist because they feed different steps:
``load_ff_factors`` (parts 3-4, ``ff5`` + ``mom_ff`` format) and
``download_ff_factors`` (part 5, flat ``ff`` format with ``Mkt-RF``).
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import requests
from numpy.linalg import lstsq
from scipy import stats

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_CSV.zip")


# ----------------------------------------------------------------------------
# 3.4  Load FF5 + momentum (ff5 / mom_ff format)
# ----------------------------------------------------------------------------
def _load_ff_factor(url: str):
    r = requests.get(url, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in raw.split("\n"):
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts[0]) == 6 and parts[0].isdigit():
            try:
                rows.append([int(parts[0])] + [float(x) for x in parts[1:] if x != ""])
            except ValueError:
                continue
    return rows


def load_ff_factors():
    """Load FF5 (MKT, SMB, HML, RMW, CMA, RF) and MOM. Returns (ff5, mom_ff).

    On a network failure, returns ``(None, None)`` (the rest of the pipeline
    handles missing factors).
    """
    try:
        ff5 = pd.DataFrame(_load_ff_factor(FF5_URL))
        ff5.columns = ["date", "MKT", "SMB", "HML", "RMW", "CMA", "RF"]
        ff5["date"] = pd.to_datetime(ff5["date"].astype(str), format="%Y%m") \
            + pd.offsets.MonthEnd(0)
        ff5 = ff5.set_index("date") / 100

        mom_ff = pd.DataFrame(_load_ff_factor(MOM_URL))
        mom_ff.columns = ["date", "MOM"]
        mom_ff["date"] = pd.to_datetime(mom_ff["date"].astype(str), format="%Y%m") \
            + pd.offsets.MonthEnd(0)
        mom_ff = mom_ff.set_index("date") / 100

        print("FF factors loaded successfully")
        return ff5, mom_ff
    except Exception as e:  # noqa: BLE001
        print(f"FF factors unavailable: {e}")
        return None, None


# ----------------------------------------------------------------------------
# 5.2a  Flat FF loader (ff format with Mkt-RF / Mom) — used in Part 5
# ----------------------------------------------------------------------------
def download_ff_factors() -> pd.DataFrame:
    """Load FF5 + momentum into a single flat DataFrame (``date`` column)."""
    def _fetch_ff(url):
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0]).decode("utf-8")
        lines = raw.split("\n")
        start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) >= 6 and stripped[:6].isdigit():
                start = i
                break
        end = start
        for i in range(start, len(lines)):
            stripped = lines[i].strip()
            if stripped == "" or (len(stripped) > 0 and not stripped[0].isdigit()):
                end = i
                break
        else:
            end = len(lines)
        return pd.read_csv(
            io.StringIO("\n".join(lines[start:end])),
            header=None, skipinitialspace=True,
        )

    df5 = _fetch_ff(FF5_URL)
    df5.columns = ["yyyymm", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    dfm = _fetch_ff(MOM_URL)
    dfm.columns = ["yyyymm", "Mom"]
    for d in (df5, dfm):
        d["yyyymm"] = d["yyyymm"].astype(str).str.strip()
    ff = df5.merge(dfm, on="yyyymm", how="inner")
    ff["date"] = pd.to_datetime(ff["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]:
        ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100
    return ff.dropna().sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3.5  Performance statistics and regressions
# ----------------------------------------------------------------------------
STAT_KEYS = [
    "Ann. Return", "Ann. Volatility", "Sharpe", "Max Drawdown", "Calmar",
    "Skewness", "Kurt.", "VaR 5%", "CVaR 5%", "t-stat", "p-value",
]


def performance_stats(series: pd.Series, ff5: pd.DataFrame | None = None) -> dict:
    """Performance statistics for a series of monthly returns.

    The Sharpe ratio uses a dynamic risk-free rate (``ff5['RF']``) if provided.
    """
    s = series.dropna()
    if len(s) < 12:
        return {k: np.nan for k in STAT_KEYS}

    mean_m = s.mean()
    ann_ret = (1 + mean_m) ** 12 - 1
    ann_vol = s.std() * np.sqrt(12)

    if ff5 is not None:
        rf_dynamic = ff5["RF"].reindex(s.index).ffill()
        ann_excess = (s - rf_dynamic).mean() * 12
    else:
        ann_excess = s.mean() * 12
    sharpe = ann_excess / ann_vol if ann_vol > 0 else np.nan

    wealth = (1 + s).cumprod()
    max_dd = (wealth / wealth.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    var_5 = s.quantile(0.05)
    cvar_5 = s[s <= var_5].mean()
    t_stat, p_val = stats.ttest_1samp(s, 0)

    return {
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Skewness": s.skew(),
        "Kurt.": s.kurtosis(),
        "VaR 5%": var_5,
        "CVaR 5%": cvar_5,
        "t-stat": t_stat,
        "p-value": p_val,
    }


def alpha_ff4(serie: pd.Series, ff5, mom_ff):
    """Annualized FF4 alpha (MKT+SMB+HML+MOM), t-stat, R-squared. Returns (alpha, t, r2)."""
    if ff5 is None:
        return np.nan, np.nan, np.nan
    ff = ff5[["MKT", "SMB", "HML"]].join(mom_ff[["MOM"]], how="inner")
    reg = serie.to_frame("ret").join(ff, how="inner").dropna()
    if len(reg) < 24:
        return np.nan, np.nan, np.nan
    y = reg["ret"].values
    X = np.column_stack([
        np.ones(len(y)), reg["MKT"].values, reg["SMB"].values,
        reg["HML"].values, reg["MOM"].values,
    ])
    c, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ c
    ss = np.sum(resid ** 2)
    r2 = 1 - ss / np.sum((y - y.mean()) ** 2)
    cov = ss / (len(y) - 5) * np.linalg.inv(X.T @ X)
    t = c[0] / np.sqrt(cov[0, 0])
    return c[0] * 12, t, r2


def info_ratio_vs_hml(serie: pd.Series, ff5) -> float:
    """Annualized information ratio vs passive HML (residual alpha / tracking)."""
    if ff5 is None:
        return np.nan
    hml = ff5["HML"].reindex(serie.index)
    mask = ~np.isnan(hml) & ~np.isnan(serie)
    X = np.column_stack([np.ones(mask.sum()), hml[mask].values])
    c, _, _, _ = lstsq(X, serie[mask].values, rcond=None)
    resid = serie[mask].values - X @ c
    return (c[0] / resid.std()) * np.sqrt(12)


def compute_metrics(monthly_ret: pd.Series, rf_series: pd.Series | None = None) -> dict:
    """Compact metrics (CAGR, Vol, Sharpe, MaxDD, Calmar, Sortino...).

    Used in Part 5 to compare static vs dynamic short.
    """
    r = monthly_ret.dropna()
    n = len(r)
    keys = ["CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "Sortino", "Skew",
            "Hit Rate", "N months"]
    if n < 12:
        return {k: np.nan for k in keys}
    cumul = (1 + r).cumprod()
    years = n / 12
    cagr = cumul.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(12)

    if rf_series is not None:
        rf_aligned = rf_series.reindex(r.index).ffill().bfill().fillna(0)
        ann_excess = (r - rf_aligned).mean() * 12
    else:
        ann_excess = r.mean() * 12
    sharpe = ann_excess / vol if vol > 0 else np.nan

    dd = cumul / cumul.cummax() - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd != 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(12)
    sortino = cagr / (downside + 1e-10)
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd,
            "Calmar": calmar, "Sortino": sortino, "Skew": r.skew(),
            "Hit Rate": (r > 0).mean(), "N months": n}
