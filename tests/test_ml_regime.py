"""Tests des fonctions pures de la détection de régime (sans WRDS)."""

import numpy as np
import pandas as pd

from value_strategy import portfolio
from value_strategy.ml_regime.labeling import label_regimes_v3, smooth_regimes
from value_strategy.ml_regime.model import find_optimal_threshold


def test_smooth_removes_short_episodes():
    """Un épisode plus court que min_duration est effacé."""
    regimes = np.array([0, 0, 1, 1, 0, 0, 0])  # épisode de 2 < min_duration=3
    out = smooth_regimes(regimes.copy(), min_duration=3, bridge_gap=0)
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0]


def test_smooth_bridges_short_gap():
    """Un trou court entre deux épisodes est comblé."""
    regimes = np.array([1, 1, 1, 0, 1, 1, 1])  # trou de 1 <= bridge_gap=2
    out = smooth_regimes(regimes.copy(), min_duration=1, bridge_gap=2)
    assert out.tolist() == [1, 1, 1, 1, 1, 1, 1]


def test_find_optimal_threshold_in_unit_interval():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.15, 0.6])
    thresh = find_optimal_threshold(y_true, y_prob)
    assert 0.0 <= thresh <= 1.0


def test_label_regimes_guard_avoids_degenerate_split():
    """Le garde-fou garantit un split non dégénéré (les deux régimes présents).

    Données synthétiques : un bloc contigu de stress (VIX élevé) minoritaire ;
    quel que soit le clustering, la sortie doit contenir les deux régimes avec
    une part minoritaire raisonnable.
    """
    rng = np.random.default_rng(0)
    n = 150
    stress = np.zeros(n)
    stress[60:100] = 1  # bloc contigu de 40 mois
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
    # Le régime actif (1) doit correspondre au VIX le plus élevé
    assert out.loc[out["regime"] == 1, "vix_z"].mean() > \
        out.loc[out["regime"] == 0, "vix_z"].mean()


def test_estimate_borrow_fee_decreases_with_size():
    fees = [portfolio.estimate_borrow_fee(m)
            for m in [100, 600, 1500, 3000, 7000, 12000]]
    assert fees == sorted(fees, reverse=True)  # plus gros = moins cher


def test_holding_durations_basic():
    import pandas as pd
    d1, d2, d3 = (pd.Timestamp("2010-06-30"), pd.Timestamp("2010-12-31"),
                  pd.Timestamp("2011-06-30"))
    snaps = {d1: {1.0, 2.0}, d2: {1.0}, d3: {1.0}}
    avg, med, n = portfolio.holding_durations(snaps)
    assert n == 2          # titre 2 (sort en d2) + titre 1 (actif jusqu'à d3)
    assert avg > 0 and med > 0
