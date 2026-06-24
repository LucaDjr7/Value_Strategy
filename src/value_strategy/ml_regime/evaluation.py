"""4.4  Évaluation des modèles + feature importance (permutation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_models(results_df):
    """Métriques OOS : Accuracy, Precision, Recall, F1, AUC, Brier."""
    print("\n" + "=" * 60)
    print("  COMPARAISON MODÈLES — Out-of-Sample V3")
    print("=" * 60)

    if len(results_df) == 0:
        print("  ⚠ Aucune donnée OOS — évaluation impossible.")
        return pd.DataFrame(columns=[
            "Accuracy", "Precision (stress)", "Recall (stress)",
            "F1 (stress)", "AUC-ROC", "Brier Score",
        ])

    y = results_df["y_true"].values
    models_map = {
        "Logistic Regression": ("lr_pred", "lr_prob"),
        "Gradient Boosting": ("gb_pred", "gb_prob"),
    }
    metrics_list = []
    for name, (pred_col, prob_col) in models_map.items():
        preds = results_df[pred_col].values
        probs = results_df[prob_col].values
        try:
            auc = roc_auc_score(y, probs)
        except Exception:  # noqa: BLE001
            auc = np.nan
        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy_score(y, preds),
            "Precision (stress)": precision_score(y, preds, zero_division=0),
            "Recall (stress)": recall_score(y, preds, zero_division=0),
            "F1 (stress)": f1_score(y, preds, zero_division=0),
            "AUC-ROC": auc,
            "Brier Score": brier_score_loss(y, probs),
        })
        cm = confusion_matrix(y, preds)
        print(f"\n  {name}:")
        print(f"    Accuracy: {accuracy_score(y, preds):.3f} | "
              f"Precision: {precision_score(y, preds, zero_division=0):.3f} | "
              f"Recall: {recall_score(y, preds, zero_division=0):.3f}")
        print(f"    AUC: {auc:.3f} | Brier: {brier_score_loss(y, probs):.4f}")
        print(f"    Confusion matrix: {cm.tolist()}")

    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    best = metrics_df["AUC-ROC"].idxmax()
    print(f"\n  ★ Meilleur (AUC) : {best} ({metrics_df.loc[best, 'AUC-ROC']:.3f})")
    return metrics_df


def compute_feature_importance(model, scaler, feature_cols, df):
    """Permutation importance sur le dataset supervisé complet."""
    print("\n[STEP 4b] Feature importance (permutation)...")
    X = scaler.transform(df[feature_cols].values)
    y = df["target"].values
    result = permutation_importance(
        model, X, y, n_repeats=30, random_state=42, n_jobs=-1,
    )
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False)
    print("  Top 10 features :")
    for _, r in imp_df.head(10).iterrows():
        print(f"    {r['feature']:<35s} {r['importance']:.4f} ± {r['std']:.4f}")
    return imp_df
