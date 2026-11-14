"""Re-evaluate AbDev-X in the OFFICIAL competition metric: AVERAGE Spearman across folds.

Also reports an isotype-ablation for Tm2: the held-out test set is all-IgG1, so any Tm2 gain that
comes from the germline/isotype block will NOT transfer. We quantify how much of Tm2 depends on it.
"""
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


def stack_metrics(ds, bases, split):
    """Return per-assay (avg-across-folds rho, aggregated rho)."""
    avg, agg = {}, {}
    for a in ASSAYS:
        y = ds.y(a); pooled = np.full(len(y), np.nan); per_fold = []
        for _, tr, te in fold_iter(ds, a, split):
            oof = [np.full(len(tr), np.nan) for _ in bases]; kf = KFold(5, shuffle=True, random_state=0)
            for itr, iva in kf.split(np.arange(len(tr))):
                for bi, (X, mk) in enumerate(bases):
                    oof[bi][iva] = mk().fit(X[tr[itr]], y[tr[itr]]).predict(X[tr[iva]])
            meta = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(np.column_stack(oof), y[tr])
            fit = [mk().fit(X[tr], y[tr]) for X, mk in bases]
            pte = meta.predict(np.column_stack([f.predict(X[te]) for f, (X, mk) in zip(fit, bases)]))
            pooled[te] = pte
            mm = ~np.isnan(y[te]) & ~np.isnan(pte)
            if mm.sum() >= 3:
                per_fold.append(spearmanr(y[te][mm], pte[mm]).statistic)
        avg[a] = float(np.nanmean(per_fold))
        m = ~np.isnan(y) & ~np.isnan(pooled); agg[a] = float(spearmanr(y[m], pooled[m]).statistic)
    return avg, agg


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml")); names = ds.df[ds.id_col]
    DESC = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
    CAT = np.nan_to_num(pd.get_dummies(ds.df[["hc_subtype", "lc_subtype"]].astype(str)).to_numpy(float))
    BIO = np.nan_to_num(EB.load_bio_features(names).to_numpy(float))
    ESM = np.nan_to_num(EB.load_esm()); ABL = np.nan_to_num(EB.load_ablang2())
    full = [(ESM, rpca), (DESC, rid), (CAT, rid), (BIO, rid), (BIO, lgbm), (ABL, rid)]
    noiso = [(ESM, rpca), (DESC, rid), (BIO, rid), (BIO, lgbm), (ABL, rid)]  # drop isotype block

    rows = []
    for split in SPLITS:
        avg, agg = stack_metrics(ds, full, split)
        for a in ASSAYS:
            rows.append(dict(model="AbDev-X", split=split, assay=a, avg_folds=round(avg[a], 3), aggregated=round(agg[a], 3)))
        print(f"[{split}] AVG-across-folds: " + " ".join(f"{a}={avg[a]:.3f}" for a in ASSAYS) + f"  MEAN={np.mean(list(avg.values())):.3f}", flush=True)
    # isotype ablation (Tm2) on cluster_isotype -- transfer risk to all-IgG1 heldout
    avg_ni, _ = stack_metrics(ds, noiso, "cluster_isotype")
    print(f"\n[cluster_isotype] Tm2 with isotype = {[r['avg_folds'] for r in rows if r['split']=='cluster_isotype' and r['assay']=='Tm2'][0]:.3f}"
          f" | Tm2 WITHOUT isotype = {avg_ni['Tm2']:.3f}  (heldout is all-IgG1 -> no-isotype is the transfer-relevant number)", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/leaderboard_metric.csv", index=False)


if __name__ == "__main__":
    main()
