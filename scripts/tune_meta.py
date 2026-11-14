"""Tune base learners + meta for the stacking ensemble (esp. to fix Titer)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from scipy.optimize import nnls
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.descriptors import build_descriptor_matrix
import experiment_bench as EB

ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
names = ds.df[ds.id_col]
DESC = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
CAT = np.nan_to_num(pd.get_dummies(ds.df[["hc_subtype", "lc_subtype"]].astype(str)).to_numpy(float))
BIO = np.nan_to_num(EB.load_bio_features(names).to_numpy(float))
ESM = np.nan_to_num(EB.load_esm())
ABL = np.nan_to_num(EB.load_ablang2())
ASSAYS = EB.ASSAYS; FOLD = "cluster_isotype"

def rpca(): return Pipeline([("s", StandardScaler()), ("p", PCA(0.9, random_state=0, svd_solver="full")), ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])
def rid(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])
def enet(): return Pipeline([("s", StandardScaler()), ("r", ElasticNetCV(l1_ratio=0.5, n_alphas=20, cv=3, max_iter=2000, random_state=0))])
def lgbm(): return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=7, subsample=.8, colsample_bytree=.6, reg_lambda=5, min_child_samples=12, subsample_freq=1, verbosity=-1, random_state=0)

def run(bases, meta="ridge"):
    res = {}
    for a in ASSAYS:
        y = ds.y(a); pooled = np.full(len(y), np.nan)
        for _, tr, te in fold_iter(ds, a, FOLD):
            oof = [np.full(len(tr), np.nan) for _ in bases]; kf = KFold(5, shuffle=True, random_state=0)
            for itr, iva in kf.split(np.arange(len(tr))):
                for bi, (X, mk) in enumerate(bases):
                    oof[bi][iva] = mk().fit(X[tr[itr]], y[tr[itr]]).predict(X[tr[iva]])
            Z = np.column_stack(oof)
            fit = [mk().fit(X[tr], y[tr]) for X, mk in bases]
            P = np.column_stack([f.predict(X[te]) for f, (X, mk) in zip(fit, bases)])
            if meta == "nnls":
                w, _ = nnls(np.column_stack([Z, np.ones(len(tr))]), y[tr])
                pooled[te] = np.column_stack([P, np.ones(len(te))]) @ w
            else:
                M = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(Z, y[tr])
                pooled[te] = M.predict(P)
        m = ~np.isnan(y) & ~np.isnan(pooled); res[a] = float(spearmanr(y[m], pooled[m]).statistic)
    res["MEAN"] = float(np.mean([res[a] for a in ASSAYS])); return res

def run_simplex(bases):
    """Meta = convex combination (weights>=0, sum=1) fit on OOF -> can't beat-worse-than-mean,
    and won't chase noise. Solved via NNLS on mean-centered columns + normalize."""
    res = {}
    for a in ASSAYS:
        y = ds.y(a); pooled = np.full(len(y), np.nan)
        for _, tr, te in fold_iter(ds, a, FOLD):
            oof = [np.full(len(tr), np.nan) for _ in bases]; kf = KFold(5, shuffle=True, random_state=0)
            for itr, iva in kf.split(np.arange(len(tr))):
                for bi, (X, mk) in enumerate(bases):
                    oof[bi][iva] = mk().fit(X[tr[itr]], y[tr[itr]]).predict(X[tr[iva]])
            Z = np.column_stack(oof)
            w, _ = nnls(Z, y[tr]); w = w / (w.sum() + 1e-9)
            fit = [mk().fit(X[tr], y[tr]) for X, mk in bases]
            P = np.column_stack([f.predict(X[te]) for f, (X, mk) in zip(fit, bases)])
            pooled[te] = P @ w
        m = ~np.isnan(y) & ~np.isnan(pooled); res[a] = float(spearmanr(y[m], pooled[m]).statistic)
    res["MEAN"] = float(np.mean([res[a] for a in ASSAYS])); return res

base = [(ESM, rpca), (DESC, rid), (CAT, rid), (BIO, rid), (BIO, lgbm), (ABL, rid)]
for name, r in [("base ridge-meta", run(base, "ridge")),
                ("base nnls-meta", run(base, "nnls")),
                ("base simplex-meta", run_simplex(base))]:
    print(f"{name:20s} " + " ".join(f"{a}={r[a]:.3f}" for a in ASSAYS) + f"  MEAN={r['MEAN']:.3f}", flush=True)
