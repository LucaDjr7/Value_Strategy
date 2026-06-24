"""Tests des statistiques de performance (sans réseau ni WRDS)."""

import numpy as np
import pandas as pd

from value_strategy import factors


def _series(mean=0.01, vol=0.04, n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    return pd.Series(rng.normal(mean, vol, n), index=idx)


def test_performance_stats_keys_and_sign():
    s = _series(mean=0.01)
    stats = factors.performance_stats(s)
    assert set(factors.STAT_KEYS) <= set(stats)
    assert stats["Ann. Return"] > 0
    assert stats["Ann. Volatility"] > 0


def test_performance_stats_short_series_returns_nan():
    s = _series(n=6)
    stats = factors.performance_stats(s)
    assert all(np.isnan(v) for v in stats.values())


def test_compute_metrics_sharpe_positive_for_positive_drift():
    s = _series(mean=0.02, vol=0.03, seed=1)
    m = factors.compute_metrics(s)
    assert m["N mois"] == 120
    assert m["Sharpe"] > 0
    assert 0 <= m["Hit Rate"] <= 1


def test_sharpe_uses_excess_return_when_rf_given():
    s = _series(mean=0.01, seed=2)
    rf = pd.Series(0.005, index=s.index)
    sharpe_no_rf = factors.compute_metrics(s)["Sharpe"]
    sharpe_rf = factors.compute_metrics(s, rf_series=rf)["Sharpe"]
    assert sharpe_rf < sharpe_no_rf  # retrancher le RF réduit le Sharpe
