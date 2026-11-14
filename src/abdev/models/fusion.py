"""AbDev-X: a multi-representation, biologically-grounded, explainable fusion model.

This is the primary contribution. Rather than a single high-capacity end-to-end network (which
overfits at n~250, see models/chain_aware.py), AbDev-X composes SIX complementary representations
-- each a weak/regularized base learner -- via a per-assay out-of-fold stacking meta-learner. The
learned meta-weights are directly interpretable: they reveal which biological signal drives each
developability property (e.g. germline/isotype -> Tm2; antibody-LM -> self-association).

Representations (chain-aware & biologically grounded):
  1. ESM-2 650M  (mean + CDR-pooled, PCA-denoised)      -- general folding/physicochemistry
  2. Interpretable physicochemical descriptors (165)     -- charge/pI/hydrophobic-moment/liabilities
  3. Germline/Isotype one-hot (hc/lc subtype)            -- constant-region identity (Fc/Tm driver)
  4. Structural/biophysical features (TAP,MOE,Aggrescan3D,AntiFold,DeepViscosity,Saprot) [ridge]
  5. ...the same structural block with a LightGBM head    -- nonlinear surface-property interactions
  6. AbLang2 paired antibody-LM embeddings               -- antibody-specific CDR grammar

Each base is trained strictly inside the training partition; inner-CV out-of-fold predictions feed
a ridge meta-learner (leakage-free). This module packages that design as a scikit-learn-style
estimator operating on a dict of named representation blocks.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _ridge_pca():
    return Pipeline([("s", StandardScaler()),
                     ("p", PCA(0.9, random_state=0, svd_solver="full")),
                     ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])


def _ridge():
    return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=7, subsample=.8,
                             colsample_bytree=.6, reg_lambda=5, min_child_samples=12,
                             subsample_freq=1, verbosity=-1, random_state=0)


# Default recipe: (block_name, head_factory, use_pca_ridge_flag)
DEFAULT_BASES = [
    ("ESM2", _ridge_pca),
    ("Descriptors", _ridge),
    ("Germline/Isotype", _ridge),
    ("BioFeatures", _ridge),
    ("BioFeatures_lgbm", _lgbm),
    ("AbLang2", _ridge),
]
# Which representation block each base consumes (name -> block key in the X dict).
BASE_BLOCK = {
    "ESM2": "esm2", "Descriptors": "descriptors", "Germline/Isotype": "isotype",
    "BioFeatures": "bio", "BioFeatures_lgbm": "bio", "AbLang2": "ablang2",
}


class AbDevFusion:
    """Per-assay out-of-fold stacking over named representation blocks.

    Parameters
    ----------
    bases : list of (name, head_factory) ; consumes blocks per BASE_BLOCK.
    inner_folds : inner CV used to generate leakage-free meta features.
    """

    def __init__(self, bases=DEFAULT_BASES, inner_folds: int = 5, random_state: int = 0):
        self.bases = bases
        self.inner_folds = inner_folds
        self.random_state = random_state

    def _stack(self, Xb):
        return [np.asarray(Xb[BASE_BLOCK[name]], float) for name, _ in self.bases]

    def fit(self, Xb: Dict[str, np.ndarray], y: np.ndarray):
        y = np.asarray(y, float)
        Xs = self._stack(Xb)
        n = len(y)
        kf = KFold(self.inner_folds, shuffle=True, random_state=self.random_state)
        oof = [np.full(n, np.nan) for _ in self.bases]
        for tr, va in kf.split(np.arange(n)):
            for bi, (name, mk) in enumerate(self.bases):
                oof[bi][va] = mk().fit(Xs[bi][tr], y[tr]).predict(Xs[bi][va])
        Z = np.column_stack(oof)
        self.meta_ = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(Z, y)
        self.fitted_ = [mk().fit(Xs[bi], y) for bi, (name, mk) in enumerate(self.bases)]
        self.weights_ = dict(zip([n for n, _ in self.bases], self.meta_.coef_))
        return self

    def predict(self, Xb: Dict[str, np.ndarray]) -> np.ndarray:
        Xs = self._stack(Xb)
        P = np.column_stack([m.predict(Xs[bi]) for bi, m in enumerate(self.fitted_)])
        return self.meta_.predict(P)

    def representation_importance(self) -> Dict[str, float]:
        """Meta-weights = biological interpretation (which signal drives this assay)."""
        return dict(self.weights_)
