"""Evaluate the AbDev-X stacking ensemble across all 3 generalization splits (aggregated Spearman)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.descriptors import build_descriptor_matrix
import experiment_bench as EB

ASSAYS = EB.ASSAYS
SPLITS = ["random", "cluster", "cluster_isotype"]

def rpca(): return Pipeline([("s", StandardScaler()), ("p", PCA(0.9, random_state=0, svd_solver="full")), ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])
def rid(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])
def lgbm(): return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=7, subsample=.8, colsample_bytree=.6, reg_lambda=5, min_child_samples=12, subsample_freq=1, verbosity=-1, random_state=0)

def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml")); names = ds.df[ds.id_col]
    DESC = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
    CAT = np.nan_to_num(pd.get_dummies(ds.df[["hc_subtype", "lc_subtype"]].astype(str)).to_numpy(float))
    BIO = np.nan_to_num(EB.load_bio_features(names).to_numpy(float))
    ESM = np.nan_to_num(EB.load_esm()); ABL = np.nan_to_num(EB.load_ablang2())
    bases = [(ESM, rpca), (DESC, rid), (CAT, rid), (BIO, rid), (BIO, lgbm), (ABL, rid)]

    rows = []
    for split in SPLITS:
        res = {}
        for a in ASSAYS:
            y = ds.y(a); pooled = np.full(len(y), np.nan)
            for _, tr, te in fold_iter(ds, a, split):
                oof = [np.full(len(tr), np.nan) for _ in bases]; kf = KFold(5, shuffle=True, random_state=0)
                for itr, iva in kf.split(np.arange(len(tr))):
                    for bi, (X, mk) in enumerate(bases):
                        oof[bi][iva] = mk().fit(X[tr[itr]], y[tr[itr]]).predict(X[tr[iva]])
                meta = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(np.column_stack(oof), y[tr])
                fit = [mk().fit(X[tr], y[tr]) for X, mk in bases]
                pooled[te] = meta.predict(np.column_stack([f.predict(X[te]) for f, (X, mk) in zip(fit, bases)]))
            m = ~np.isnan(y) & ~np.isnan(pooled); res[a] = float(spearmanr(y[m], pooled[m]).statistic)
        res["MEAN"] = float(np.mean([res[a] for a in ASSAYS]))
        rows.append(dict(model="AbDev-X (stacking)", split=split, **res))
        print(f"{split:16s} " + " ".join(f"{a}={res[a]:.3f}" for a in ASSAYS) + f"  MEAN={res['MEAN']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/abdevx_splits.csv", index=False)
    print("Saved results/abdevx_splits.csv")

if __name__ == "__main__":
    main()
