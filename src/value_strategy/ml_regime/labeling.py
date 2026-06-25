"""4.2  Regime labeling (HMM/GMM) — euphoria / junk rally.

Target inversion relative to the stress model: the strategy suffers in low VIX
/ calm rally conditions (short squeeze), not in stress. So the active regime
(label 1) is identified as the cluster with the HIGHEST VIX (stress / high
volatility), then the regimes are smoothed (2-month bridge, 3-month minimum
duration).

Final convention: ``regime`` = 1 (active/euphoria), 0 (normal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..config import MIN_REGIME_SHARE
from . import HAS_HMMLEARN


def smooth_regimes(regimes, min_duration: int = 3, bridge_gap: int = 2):
    """Smooth a binary regime sequence: fill short gaps, then drop episodes
    shorter than ``min_duration``."""
    result = regimes.copy()
    n = len(result)
    i = 0
    while i < n:
        if result[i] == 1:
            j = i
            while j < n and result[j] == 1:
                j += 1
            k = j
            while k < n and k < j + bridge_gap + 1 and result[k] == 0:
                k += 1
            if k < n and result[k] == 1 and (k - j) <= bridge_gap:
                result[j:k] = 1
                continue
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if result[i] == 1:
            j = i
            while j < n and result[j] == 1:
                j += 1
            if (j - i) < min_duration:
                result[i:j] = 0
            i = j
        else:
            i += 1
    return result


def label_regimes_v3(mkt_df, n_regimes: int = 2, min_duration: int = 3, bridge_gap: int = 2):
    """Label the euphoria/normal regimes and smooth the result.

    Returns
    -------
    mkt_df, model, scaler_regime, trans_mat, trans_mat_raw, method
    """
    print(f"[STEP 2] Labeling EUPHORIA regimes V3 "
          f"(min_dur={min_duration}, bridge={bridge_gap})...")

    regime_features = ["vix_z", "rvol_mkt_z"]
    if "ret_dispersion_z" in mkt_df.columns:
        regime_features.append("ret_dispersion_z")
    if "credit_spread_z" in mkt_df.columns:
        regime_features.append("credit_spread_z")

    available = [c for c in regime_features if c in mkt_df.columns]
    valid_mask = mkt_df[available].notna().all(axis=1)
    X_regime = mkt_df.loc[valid_mask, available].values
    valid_idx = mkt_df.loc[valid_mask].index

    scaler_regime = StandardScaler()
    X_scaled = scaler_regime.fit_transform(X_regime)

    if HAS_HMMLEARN:
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                            n_iter=200, random_state=42)
        model.fit(X_scaled)
        hidden_states = model.predict(X_scaled)
        method = "GaussianHMM"
        trans_mat_raw = model.transmat_
    else:
        model = GaussianMixture(n_components=n_regimes, covariance_type="full",
                                n_init=10, random_state=42)
        model.fit(X_scaled)
        hidden_states = model.predict(X_scaled)
        method = "GaussianMixture"
        trans_mat_raw = np.zeros((n_regimes, n_regimes))
        for t in range(1, len(hidden_states)):
            trans_mat_raw[hidden_states[t - 1], hidden_states[t]] += 1
        trans_mat_raw = trans_mat_raw / trans_mat_raw.sum(axis=1, keepdims=True)

    mkt_df = mkt_df.copy()
    mkt_df.loc[valid_idx, "regime_raw"] = hidden_states

    # Active regime (label 1) = cluster with the HIGHEST VIX (stress / high
    # volatility). Convention aligned with the actual output of the original
    # notebook (regime 1 <=> high vix_z).
    vix_valid = mkt_df.loc[valid_idx, "vix_z"].values
    mean_vix_by_cluster = {
        k: vix_valid[hidden_states == k].mean() for k in np.unique(hidden_states)
    }
    stress_cluster = max(mean_vix_by_cluster, key=mean_vix_by_cluster.get)
    regimes_model = (hidden_states == stress_cluster).astype(int)
    if stress_cluster != 1:  # keep trans_mat_raw consistent with the remapping
        trans_mat_raw = trans_mat_raw[::-1, ::-1]

    # -- Guard against degeneracy --
    # On some data vintages the clustering collapses (one regime below
    # MIN_REGIME_SHARE). We then fall back to a robust, deterministic split:
    # a stress score = mean of the standardized features (all oriented so that
    # "higher = more stress"), with the median as the cutoff.
    minority_share = min(regimes_model.mean(), 1 - regimes_model.mean())
    if minority_share < MIN_REGIME_SHARE:
        stress_score = X_scaled.mean(axis=1)
        regimes_model = (stress_score > np.median(stress_score)).astype(int)
        method += " + stress-score fallback (degenerate clustering)"
        print(f"  ! Degenerate clustering (minority regime {minority_share:.1%} "
              f"< {MIN_REGIME_SHARE:.0%}) -> stress-score fallback (median)")

    mkt_df["regime_unsmoothed"] = np.nan
    mkt_df.loc[valid_idx, "regime_unsmoothed"] = regimes_model.astype(float)

    # After remapping: 1 = active regime (stress/euphoria), 0 = normal
    smoothed = smooth_regimes(regimes_model.copy(), min_duration=min_duration,
                              bridge_gap=bridge_gap)
    mkt_df.loc[valid_idx, "regime"] = smoothed.astype(float)

    trans_mat = np.zeros((2, 2))
    for t in range(1, len(smoothed)):
        trans_mat[smoothed[t - 1], smoothed[t]] += 1
    trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)

    mkt_df = mkt_df.drop(columns=["regime_raw"], errors="ignore")
    valid = mkt_df.dropna(subset=["regime"])
    rc = valid["regime"].value_counts().sort_index()
    total = rc.sum()
    print(f"  Method: {method}")
    print(f"  Normal   (0): {rc.get(0.0, 0):>4.0f} months ({rc.get(0.0, 0) / total:.1%})")
    print(f"  Euphoria (1): {rc.get(1.0, 0):>4.0f} months ({rc.get(1.0, 0) / total:.1%})")
    print(f"  Features: {available}")
    print(f"  Mean VIX in Normal  : "
          f"{valid.loc[valid['regime'] == 0, 'vix_z'].mean():+.2f}")
    print(f"  Mean VIX in Euphoria: "
          f"{valid.loc[valid['regime'] == 1, 'vix_z'].mean():+.2f}")
    return mkt_df, model, scaler_regime, trans_mat, trans_mat_raw, method
