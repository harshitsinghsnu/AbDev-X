"""Single-task regression heads.

All heads are scikit-learn compatible (fit/predict) so they slot into the stacking ensemble
and share the same in-fold preprocessing pipeline (no leakage). Target standardization is
applied/inverted inside `fit`/`predict` when enabled.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_ridge(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)) -> Pipeline:
    """Standardize features then RidgeCV. This is the competition baseline head."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas)),
    ])


class LGBMHead(BaseEstimator, RegressorMixin):
    """LightGBM regressor with sensible small-data defaults; GBMs shine at n~250."""

    def __init__(self, random_state: int = 42, n_estimators: int = 600,
                 learning_rate: float = 0.02, num_leaves: int = 15,
                 subsample: float = 0.8, colsample_bytree: float = 0.5,
                 reg_lambda: float = 5.0, min_child_samples: int = 10):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples

    def fit(self, X, y):
        import lightgbm as lgb
        self.model_ = lgb.LGBMRegressor(
            random_state=self.random_state, n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, num_leaves=self.num_leaves,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda, min_child_samples=self.min_child_samples,
            subsample_freq=1, verbosity=-1)
        self.model_.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))


class TabPFNHead(BaseEstimator, RegressorMixin):
    """TabPFN v2 regressor head. Uses in-context Bayesian inference — no hyperparameter tuning.

    Requires one-time interactive authentication:
        python -c "from tabpfn import TabPFNRegressor; TabPFNRegressor().fit([[1,2],[3,4]], [1,2])"
    Run that in a terminal once; the token is cached in ~/.tabpfn.

    Best for small tabular datasets (n < 10k). TabPFN handles feature preprocessing internally.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as e:
            raise ImportError("tabpfn not installed. Run: pip install tabpfn") from e
        try:
            self.model_ = TabPFNRegressor(
                random_state=self.random_state,
                ignore_pretraining_limits=True,
                show_progress_bar=False,
            )
            self.model_.fit(np.asarray(X), np.asarray(y))
        except Exception as e:
            if "auth" in str(e).lower() or "token" in str(e).lower() or "api" in str(e).lower() or "socket" in str(e).lower():
                raise RuntimeError(
                    "TabPFN requires one-time browser authentication. Run this interactively:\n"
                    "  python -c \"from tabpfn import TabPFNRegressor; "
                    "TabPFNRegressor().fit([[1,2],[3,4]], [1,2])\"\n"
                    "The token is cached in ~/.tabpfn after that."
                ) from e
            raise
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))


def make_head(name: str, random_state: int = 42):
    if name == "ridge":
        return make_ridge()
    if name == "lgbm":
        return LGBMHead(random_state=random_state)
    if name == "tabpfn":
        return TabPFNHead(random_state=random_state)
    raise ValueError(f"Unknown head '{name}'")
