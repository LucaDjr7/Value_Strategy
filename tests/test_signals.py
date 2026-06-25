"""Tests of the pure signal-construction functions (no WRDS)."""

import numpy as np
import pandas as pd

from value_strategy import config, signals


def test_momentum_skip2_formula():
    """mom_6m = product of returns t-6..t-2 (skip-2) - 1."""
    dates = pd.date_range("2010-01-31", periods=8, freq="ME")
    rets = [0.01, 0.02, -0.01, 0.03, 0.00, 0.04, 0.02, 0.05]
    df = pd.DataFrame({"permno": 1.0, "date": dates, "ret": rets})

    out = signals.add_momentum(df)
    # At month 8 (index 7): cumul of returns shifted by 2, window 5 = indices 1..5
    expected = np.prod([1 + r for r in rets[1:6]]) - 1
    assert np.isclose(out["mom_6m"].iloc[7], expected)
    # The first 6 months lack enough history
    assert out["mom_6m"].iloc[:6].isna().all()


def test_sector_rank_fallback_to_global():
    """With fewer than SECTOR_MIN_COUNT stocks, the rank falls back to global."""
    n = config.SECTOR_MIN_COUNT - 1  # sector too small -> fallback
    df = pd.DataFrame({
        "permno": np.arange(n, dtype=float),
        "date": pd.Timestamp("2010-06-30"),
        "siccd": 1000,  # same sector
        "signal_raw": np.linspace(-1, 1, n),
    })
    out = signals.add_sector_value_rank(df)
    assert np.allclose(
        out["BM_rank"].values, out["BM_rank_global"].values, equal_nan=True,
    )
    # The stock with the highest signal is long_eligible
    assert bool(out.loc[out["signal_raw"].idxmax(), "long_eligible"])


def test_quality_score_matches_mean_of_rank_columns():
    """Quality score = mean of the percentile ranks (faithful replication).

    NB on the notebook convention: with ``ascending=False`` for ROCE/ROE/OM and
    ``ascending=True`` for NetD/OIBDP, the composite is such that the best
    fundamentals get the LOWEST score (see README, "Quality score, a caveat").
    This test verifies the exact mechanics, not a quality direction.
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
    # Exact values expected from the notebook convention
    assert np.isclose(out.loc[0, "score"], 1.0)
    assert np.isclose(out.loc[2, "score"], 1 / 3)
