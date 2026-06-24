"""Tests des fonctions pures de construction des signaux (sans WRDS)."""

import numpy as np
import pandas as pd

from value_strategy import config, signals


def test_momentum_skip2_formula():
    """mom_6m = produit des rendements t-6..t-2 (skip-2) − 1."""
    dates = pd.date_range("2010-01-31", periods=8, freq="ME")
    rets = [0.01, 0.02, -0.01, 0.03, 0.00, 0.04, 0.02, 0.05]
    df = pd.DataFrame({"permno": 1.0, "date": dates, "ret": rets})

    out = signals.add_momentum(df)
    # Au 8e mois (index 7) : cumul de ret décalés de 2, fenêtre 5 = indices 1..5
    expected = np.prod([1 + r for r in rets[1:6]]) - 1
    assert np.isclose(out["mom_6m"].iloc[7], expected)
    # Les 6 premiers mois n'ont pas assez d'historique
    assert out["mom_6m"].iloc[:6].isna().all()


def test_sector_rank_fallback_to_global():
    """Avec moins de SECTOR_MIN_COUNT titres, le rank retombe sur le global."""
    n = config.SECTOR_MIN_COUNT - 1  # secteur trop petit -> fallback
    df = pd.DataFrame({
        "permno": np.arange(n, dtype=float),
        "date": pd.Timestamp("2010-06-30"),
        "siccd": 1000,  # même secteur
        "signal_raw": np.linspace(-1, 1, n),
    })
    out = signals.add_sector_value_rank(df)
    assert np.allclose(
        out["BM_rank"].values, out["BM_rank_global"].values, equal_nan=True,
    )
    # Le titre au signal le plus élevé est long_eligible
    assert bool(out.loc[out["signal_raw"].idxmax(), "long_eligible"])


def test_quality_score_matches_mean_of_rank_columns():
    """Le score qualité = moyenne des rangs percentiels (réplication fidèle).

    NB convention du notebook : avec ``ascending=False`` pour ROCE/ROE/OM et
    ``ascending=True`` pour NetD/OIBDP, le composite est tel que les meilleurs
    fondamentaux obtiennent le score le plus BAS (voir README, "Point de
    vigilance sur le score qualité"). Ce test vérifie la mécanique exacte, pas
    une direction de qualité.
    """
    df = pd.DataFrame({
        "permno": [1.0, 2.0, 3.0],
        "date": pd.Timestamp("2010-06-30"),
        "ROCE": [0.1, 0.2, 0.3],
        "ROE": [0.1, 0.2, 0.3],
        "OM": [0.1, 0.2, 0.3],
        "NetD/OIBDP": [3.0, 2.0, 1.0],
    })
    out = signals.add_quality_score(df)

    rank_cols = [f"rank_{c.replace('/', '_')}" for c, _ in signals.QUALITY_SPECS]
    assert np.allclose(out["score"], out[rank_cols].mean(axis=1))
    assert (out["n_quality"] == 4).all()
    assert out["score"].between(0, 1).all()
    # Valeurs exactes attendues d'après la convention du notebook
    assert np.isclose(out.loc[0, "score"], 1.0)
    assert np.isclose(out.loc[2, "score"], 1 / 3)
