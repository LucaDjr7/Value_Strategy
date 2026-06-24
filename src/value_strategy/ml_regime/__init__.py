"""Partie 4 — Détection ML des régimes "junk rally" (euphoria).

Le pipeline enchaîne le feature engineering (CRSP, Compustat, FRED), le
labeling HMM/GMM des régimes, la prédiction supervisée en walk-forward,
l'évaluation, puis la génération des signaux SHORT_REDUCE / FULL_SHORT.

hmmlearn et xgboost sont optionnels : à défaut, on retombe respectivement sur
GaussianMixture et GradientBoostingClassifier de scikit-learn.
"""

try:
    from hmmlearn.hmm import GaussianHMM  # noqa: F401
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
