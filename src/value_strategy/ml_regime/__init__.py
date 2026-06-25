"""Part 4 — ML detection of "junk rally" (euphoria) regimes.

The pipeline chains feature engineering (CRSP, Compustat, FRED), HMM/GMM regime
labeling, supervised walk-forward prediction, evaluation, and then the
generation of the SHORT_REDUCE / FULL_SHORT signals.

hmmlearn and xgboost are optional: without them, the code falls back to
GaussianMixture and GradientBoostingClassifier from scikit-learn respectively.
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
