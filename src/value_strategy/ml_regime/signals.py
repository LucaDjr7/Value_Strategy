"""4.6  Génération des signaux — SHORT_REDUCE / FULL_SHORT.

P(euphoria) > seuil -> SHORT_REDUCE (réduire l'exposition short).
Sinon -> FULL_SHORT (garder 100 % short).
"""

from __future__ import annotations

import numpy as np


def generate_signals_v3(results_df, threshold):
    """Convertit les probabilités OOS en signaux d'exposition short."""
    print(f"[STEP 6] Signaux euphoria (seuil={threshold:.3f})...")
    if len(results_df) == 0:
        print("  ⚠ Aucune donnée OOS — aucun signal (short restera plein partout).")
        # Renvoie un df vide mais AVEC les colonnes attendues en aval (Partie 5)
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
    print(f"  SHORT_REDUCE : {n_reduce} | FULL_SHORT : {n_full}")

    euphoria = s["y_true"] == 1
    if euphoria.sum() > 0:
        caught = ((s["signal_gb"] == "SHORT_REDUCE") & euphoria).sum()
        print(f"  Euphoria détectée : {caught}/{euphoria.sum()} "
              f"({caught / euphoria.sum():.1%})")
    correct = (
        ((s["signal_gb"] == "SHORT_REDUCE") & (s["y_true"] == 1))
        | ((s["signal_gb"] == "FULL_SHORT") & (s["y_true"] == 0))
    )
    print(f"  Précision globale signal GB : {correct.mean():.1%}")
    return s
