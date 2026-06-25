"""Tests of the pure regime-detection functions (no WRDS)."""

import numpy as np
import pandas as pd

from value_strategy import portfolio
from value_strategy.ml_regime.labeling import label_regimes_v3, smooth_regimes
from value_strategy.ml_regime.model import find_optimal_threshold


def test_smooth_removes_short_episodes():
    """An episode shorter than min_duration is removed."""
    regimes = np.array([0, 0, 1, 1, 0, 0, 0])  # episode of 2 < min_duration=3
    out = smooth_regimes(regimes.copy(), min_duration=3, bridge_gap=0)
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0]


def test_smooth_bridges_short_gap():
    """A short gap between two episodes is filled."""
    regimes = np.array([1, 1, 1, 0, 1, 1, 1])  # gap of 1 <= bridge_gap=2
    out = smooth_regimes(regimes.copy(), min_duration=1, bridge_gap=2)
    assert out.tolist() == [1, 1, 1, 1, 1, 1, 1]


def test_find_optimal_threshold_in_unit_interval():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.15, 0.6])
    thresh = find_optimal_threshold(y_true, y_prob)
    assert 0.0 <= thresh <= 1.0


def test_label_regimes_guard_avoids_degenerate_split():
    """The guard ensures a non-degenerate split (both regimes present).

    Synthetic data: a contiguous minority block of stress (high VIX); whatever
    the clustering, the output must contain both regimes with a reasonable
    minority share.
    """
    rng = np.random.default_rng(0)
    n = 150
    stress = np.zeros(n)
    stress[60:100] = 1  # contiguous block of 40 months
    base = rng.normal(0, 0.3, n)
    mkt = pd.DataFrame({
        "vix_z": base + stress * 2.5,
        "rvol_mkt_z": base + stress * 2.0,
        "ret_dispersion_z": base + stress * 1.8,
        "credit_spread_z": base + stress * 1.5,
    })
    out, *_ = label_regimes_v3(mkt)
    regimes = out["regime"].dropna()
    assert set(regimes.unique()) == {0.0, 1.0}
    minority = min(regimes.mean(), 1 - regimes.mean())
    assert minority >= 0.10
    # The active regime (1) must correspond to the highest VIX
    assert out.loc[out["regime"] == 1, "vix_z"].mean() > \
        out.loc[out["regime"] == 0, "vix_z"].mean()


def test_estimate_borrow_fee_decreases_with_size():
    fees = [portfolio.estimate_borrow_fee(m)
            for m in [100, 600, 1500, 3000, 7000, 12000]]
    assert fees == sorted(fees, reverse=True)  # bigger = cheaper


def test_holding_durations_basic():
    import pandas as pd
    d1, d2, d3 = (pd.Timestamp("2010-06-30"), pd.Timestamp("2010-12-31"),
                  pd.Timestamp("2011-06-30"))
    snaps = {d1: {1.0, 2.0}, d2: {1.0}, d3: {1.0}}
    avg, med, n = portfolio.holding_durations(snaps)
    assert n == 2          # stock 2 (exits at d2) + stock 1 (active until d3)
    assert avg > 0 and med > 0
