"""Orchestration de la Partie 4 (détection ML des régimes).

``run_ml_regime`` enchaîne features -> labeling -> walk-forward -> évaluation
-> signaux, et renvoie tous les artefacts nécessaires aux graphiques (Partie 4)
et au short dynamique (Partie 5).
"""

from __future__ import annotations

import pandas as pd

from .evaluation import compute_feature_importance, evaluate_models
from .features import compute_liquidity_features_v3
from .labeling import label_regimes_v3
from .model import prepare_supervised_data, walk_forward_validation_v3
from .signals import generate_signals_v3


def run_ml_regime(crsp_ml: pd.DataFrame, comp: pd.DataFrame, macro: pd.DataFrame) -> dict:
    """Exécute la détection de régime complète et renvoie un dict d'artefacts."""
    print("=" * 60)
    print("  PARTIE 4 — DÉTECTION DE RÉGIME (ML)")
    print("=" * 60)

    mkt_df, feature_cols = compute_liquidity_features_v3(crsp_ml, comp, macro)
    mkt_df, regime_model, scaler_regime, trans_mat, trans_mat_raw, regime_method = \
        label_regimes_v3(mkt_df)

    sup_df = prepare_supervised_data(mkt_df, feature_cols)
    results_df, last_lr, last_gb, last_scaler, avg_threshold = \
        walk_forward_validation_v3(sup_df, feature_cols)

    metrics_df = evaluate_models(results_df)
    if last_gb is not None:
        importance_df = compute_feature_importance(
            last_gb, last_scaler, feature_cols, sup_df,
        )
    else:
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": 0, "std": 0}
        )

    signals_df = generate_signals_v3(results_df, avg_threshold)

    print("\n" + "=" * 60)
    print("  RÉSUMÉ PARTIE 4 — DÉTECTION EUPHORIA")
    print("=" * 60)
    print(f"  Méthode régime   : {regime_method} + smoothing")
    print(f"  Features totales : {len(feature_cols)}")
    print(f"  Panel ML         : {mkt_df.shape[0]} mois")
    print(f"  Seuil moyen      : {avg_threshold:.3f}")

    return {
        "mkt_df": mkt_df,
        "feature_cols": feature_cols,
        "regime_method": regime_method,
        "trans_mat": trans_mat,
        "trans_mat_raw": trans_mat_raw,
        "results_df": results_df,
        "metrics_df": metrics_df,
        "importance_df": importance_df,
        "signals_df": signals_df,
        "avg_threshold": avg_threshold,
    }
