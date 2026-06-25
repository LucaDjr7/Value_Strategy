"""4.6  Signal generation — SHORT_REDUCE / FULL_SHORT.

P(euphoria) > threshold -> SHORT_REDUCE (reduce short exposure).
Otherwise -> FULL_SHORT (keep 100% short).
"""

from __future__ import annotations

import numpy as np


def generate_signals_v3(results_df, threshold):
    """Convert the OOS probabilities into short-exposure signals."""
    print(f"[STEP 6] Euphoria signals (threshold={threshold:.3f})...")
    if len(results_df) == 0:
        print("  ! No OOS data — no signal (short stays full everywhere).")
        # Return an empty df but WITH the columns expected downstream (Part 5)
        empty = results_df.copy()
        for col in ("signal_gb", "signal_lr"):
            if col not in empty.columns:
                empty[col] = []
        return empty

    s = results_df.copy()
    s["signal_gb"] = np.where(s["gb_prob"] > threshold, "SHORT_REDUCE", "FULL_SHORT")
    s["signal_lr"] = np.where(s["lr_prob"] > threshold, "SHORT_REDUCE", "FULL_SHORT")
    n_reduce = (s["signal_gb"] == "SHORT_REDUCE").sum()
    n_full = (s["signal_gb"] == "FULL_SHORT").sum()
    print(f"  SHORT_REDUCE: {n_reduce} | FULL_SHORT: {n_full}")

    euphoria = s["y_true"] == 1
    if euphoria.sum() > 0:
        caught = ((s["signal_gb"] == "SHORT_REDUCE") & euphoria).sum()
        print(f"  Euphoria detected: {caught}/{euphoria.sum()} "
              f"({caught / euphoria.sum():.1%})")
    correct = (
        ((s["signal_gb"] == "SHORT_REDUCE") & (s["y_true"] == 1))
        | ((s["signal_gb"] == "FULL_SHORT") & (s["y_true"] == 0))
    )
    print(f"  Overall GB signal accuracy: {correct.mean():.1%}")
    return s
