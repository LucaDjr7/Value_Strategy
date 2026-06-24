"""4.3  Prédiction supervisée — walk-forward validation V3.

Expanding window : entraînement sur tout le passé, test sur les 12 mois
suivants (pas de look-ahead). Seuil de décision optimisé (F1) sur le train.
Modèles : Logistic Regression + Gradient Boosting (XGBoost si disponible).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

from . import HAS_XGB


def prepare_supervised_data(mkt_df, feature_cols):
    """Prépare le dataset supervisé : target = régime à t+1."""
    print("[STEP 3a] Préparation données supervisées...")
    df = mkt_df.copy().sort_values("date").reset_index(drop=True)
    df["target"] = df["regime"].shift(-1)
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    print(f"  Samples : {len(df)} | Stress : {df['target'].mean():.1%} | "
          f"Période : {df['date'].min().strftime('%Y-%m')} → "
          f"{df['date'].max().strftime('%Y-%m')}")
    return df


def find_optimal_threshold(y_true, y_prob):
    """Seuil qui maximise le F1 sur les données fournies."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1)
    return thresholds[min(best_idx, len(thresholds) - 1)]


def walk_forward_validation_v3(df, feature_cols, min_train=120, test_window=12):
    """Walk-forward expanding window avec seuil adaptatif.

    Returns
    -------
    results_df, last_lr, last_gb, scaler, avg_threshold
    """
    n = len(df)
    print(f"[STEP 3b] Walk-forward V3 "
          f"(min_train={min_train}, test={test_window}, n={n})...")

    if min_train >= n:
        print(f"  ✗ ERREUR : min_train ({min_train}) ≥ n ({n}).")
        empty = pd.DataFrame(columns=[
            "date", "y_true", "lr_pred", "lr_prob", "gb_pred", "gb_prob", "fold",
        ])
        return empty, None, None, StandardScaler(), 0.5

    results = {"date": [], "y_true": [], "lr_pred": [], "lr_prob": [],
               "gb_pred": [], "gb_prob": [], "fold": []}
    fold = 0
    scaler = StandardScaler()
    thresholds = []
    train_end = min_train
    lr = gb = None

    while train_end + test_window <= n:
        test_end = min(train_end + test_window, n)
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        X_train_s = scaler.fit_transform(train_df[feature_cols].values)
        X_test_s = scaler.transform(test_df[feature_cols].values)
        y_train = train_df["target"].values
        y_test = test_df["target"].values

        if len(np.unique(y_train)) < 2:        # une seule classe -> skip
            train_end += test_window
            continue

        w_ratio = max((y_train == 0).sum(), 1) / max((y_train == 1).sum(), 1)

        lr = LogisticRegression(penalty="l2", C=0.5, class_weight="balanced",
                                max_iter=2000, random_state=42)
        lr.fit(X_train_s, y_train)
        lr_prob = lr.predict_proba(X_test_s)[:, 1]

        if HAS_XGB:
            import xgboost as xgb
            gb = xgb.XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, scale_pos_weight=w_ratio,
                min_child_weight=5, reg_alpha=0.5, reg_lambda=1.0,
                random_state=42, eval_metric="logloss", verbosity=0)
        else:
            gb = GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.03,
                subsample=0.8, min_samples_leaf=10, random_state=42)
        gb.fit(X_train_s, y_train)
        gb_prob = gb.predict_proba(X_test_s)[:, 1]

        train_prob = gb.predict_proba(X_train_s)[:, 1]
        opt_thresh = np.clip(find_optimal_threshold(y_train, train_prob), 0.15, 0.70)
        thresholds.append(opt_thresh)

        results["date"].extend(test_df["date"].values)
        results["y_true"].extend(y_test)
        results["lr_pred"].extend((lr_prob >= opt_thresh).astype(int))
        results["lr_prob"].extend(lr_prob)
        results["gb_pred"].extend((gb_prob >= opt_thresh).astype(int))
        results["gb_prob"].extend(gb_prob)
        results["fold"].extend([fold] * len(y_test))
        fold += 1
        train_end += test_window

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df["date"] = pd.to_datetime(results_df["date"])
    avg_threshold = np.mean(thresholds) if thresholds else 0.5
    print(f"  Folds : {fold} | OOS : {len(results_df)} | "
          f"Seuil moyen : {avg_threshold:.3f}")
    return results_df, lr, gb, scaler, avg_threshold
