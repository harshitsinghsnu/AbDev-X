"""Diagnose per-base standalone aggregated Spearman (esp. Titer)."""
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.descriptors import build_descriptor_matrix
import experiment_bench as EB

ds = load_gdpa1(str(ROOT / "configs/default.yaml")); names = ds.df[ds.id_col]
DESC = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
CAT = np.nan_to_num(pd.get_dummies(ds.df[["hc_subtype", "lc_subtype"]].astype(str)).to_numpy(float))
BIO = np.nan_to_num(EB.load_bio_features(names).to_numpy(float))
ESM = np.nan_to_num(EB.load_esm()); ABL = np.nan_to_num(EB.load_ablang2())
ASSAYS = EB.ASSAYS; FOLD = "cluster_isotype"

def rid(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])
def enet(): return Pipeline([("s", StandardScaler()), ("r", ElasticNetCV(l1_ratio=0.5, n_alphas=20, cv=3, max_iter=2000, random_state=0))])
def rpca(): return Pipeline([("s", StandardScaler()), ("p", PCA(0.9, random_state=0, svd_solver="full")), ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])

def single(X, mk):
    res = {}
    for a in ASSAYS:
        y = ds.y(a); pooled = np.full(len(y), np.nan)
        for _, tr, te in fold_iter(ds, a, FOLD):
            pooled[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        m = ~np.isnan(y) & ~np.isnan(pooled); res[a] = float(spearmanr(y[m], pooled[m]).statistic)
    return res

for nm, X, mk in [("ABL-ridge", ABL, rid), ("ABL-enet", ABL, enet), ("ESM-pca", ESM, rpca),
                   ("BIO-ridge", BIO, rid), ("CAT-ridge", CAT, rid), ("DESC-ridge", DESC, rid)]:
    r = single(X, mk)
    print(f"{nm:11s} " + " ".join(f"{a}={r[a]:.3f}" for a in ASSAYS), flush=True)
