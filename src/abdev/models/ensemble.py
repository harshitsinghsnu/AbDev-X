"""Out-of-fold stacking ensemble for a single property.

Base learners produce out-of-fold (OOF) predictions on the training partition via an inner CV;
a ridge meta-learner combines them. Optionally a shared-trunk multi-task MLP contributes an
extra base column (trained leakage-free inside each inner fold on the full target matrix), so
correlated assays help the scored property. Everything is fit strictly within the outer
training fold. With a single base learner this degrades to that learner.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from .heads import make_head
from .multitask import MultiTaskMLP


class StackingRegressor:
    def __init__(self, base_names: List[str], random_state: int = 42, inner_folds: int = 5,
                 use_multitask: bool = False):
        self.base_names = base_names
        self.random_state = random_state
        self.inner_folds = inner_folds
        self.use_multitask = use_multitask and MultiTaskMLP.available()

    def fit(self, X, y, Y_multi: Optional[np.ndarray] = None, task_index: Optional[int] = None):
        """X:(n,d) y:(n,). Y_multi:(n,t) full target matrix (NaN allowed) for the MT base."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        self.fitted_, self.col_names_ = {}, []
        oof_cols: List[np.ndarray] = []
        kf = KFold(self.inner_folds, shuffle=True, random_state=self.random_state)

        for name in self.base_names:
            oof = np.full(n, np.nan)
            for tr, va in kf.split(X):
                m = make_head(name, self.random_state).fit(X[tr], y[tr])
                oof[va] = m.predict(X[va])
            oof_cols.append(oof)
            self.col_names_.append(name)
            self.fitted_[name] = make_head(name, self.random_state).fit(X, y)

        self.mt_full_ = None
        self.task_index_ = task_index
        if self.use_multitask and Y_multi is not None and task_index is not None:
            oof = np.full(n, np.nan)
            for tr, va in kf.split(X):
                mt = MultiTaskMLP([f"t{i}" for i in range(Y_multi.shape[1])],
                                  random_state=self.random_state).fit(X[tr], Y_multi[tr])
                oof[va] = mt.predict(X[va])[:, task_index]
            oof_cols.append(oof)
            self.col_names_.append("multitask")
            self.mt_full_ = MultiTaskMLP([f"t{i}" for i in range(Y_multi.shape[1])],
                                         random_state=self.random_state).fit(X, Y_multi)

        Z = np.column_stack(oof_cols)
        valid = ~np.isnan(Z).any(1)
        self.meta_ = (None if Z.shape[1] == 1
                      else RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0)).fit(Z[valid], y[valid]))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        cols = [self.fitted_[name].predict(X) for name in self.base_names]
        if self.mt_full_ is not None:
            cols.append(self.mt_full_.predict(X)[:, self.task_index_])
        Z = np.column_stack(cols)
        return Z[:, 0] if self.meta_ is None else self.meta_.predict(Z)
