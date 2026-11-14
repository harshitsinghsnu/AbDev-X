"""Per-fold MEAN Spearman (mean +/- std across folds) for AbDev-X and baselines, all 3 splits.

For each split and fold, we compute each assay's Spearman on that fold's held-out rows, average
across assays -> one value per fold, then report mean +/- std across the 5 folds. This is the
error bar for the model-comparison figure. Also saves per-assay per-fold std for all models.
"""
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
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import experiment_bench as EB
from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

ASSAYS = EB.ASSAYS
SPLITS = ["random", "cluster", "cluster_isotype"]


def rpca(): return Pipeline([("s", StandardScaler()), ("p", PCA(0.9, random_state=0, svd_solver="full")), ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])
def rid(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])
def enet(): return Pipeline([("s", StandardScaler()), ("r", ElasticNetCV(l1_ratio=0.5, n_alphas=20, cv=3, max_iter=2000, random_state=0))])
def lgbm(): return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=7, subsample=.8, colsample_bytree=.6, reg_lambda=5, min_child_samples=12, subsample_freq=1, verbosity=-1, random_state=0)


def stack_fold_preds(ds, bases, a, split):
    """Return dict fold_id -> (y_true, y_pred) for AbDev-X stacking."""
    y = ds.y(a); folds = ds.folds(split); out = {}
    for f in sorted(set(folds.tolist())):
        tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        oof = [np.full(len(tr), np.nan) for _ in bases]; kf = KFold(5, shuffle=True, random_state=0)
        for itr, iva in kf.split(np.arange(len(tr))):
            for bi, (X, mk) in enumerate(bases):
                oof[bi][iva] = mk().fit(X[tr[itr]], y[tr[itr]]).predict(X[tr[iva]])
        meta = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(np.column_stack(oof), y[tr])
        fit = [mk().fit(X[tr], y[tr]) for X, mk in bases]
        pred = meta.predict(np.column_stack([f_.predict(X[te]) for f_, (X, mk) in zip(fit, bases)]))
        out[f] = (y[te], pred)
    return out


def single_fold_preds(ds, X, mk, a, split):
    y = ds.y(a); folds = ds.folds(split); out = {}
    for f in sorted(set(folds.tolist())):
        tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        out[f] = (y[te], mk().fit(X[tr], y[tr]).predict(X[te]))
    return out


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml")); names = ds.df[ds.id_col]
    DESC = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
    CAT = np.nan_to_num(pd.get_dummies(ds.df[["hc_subtype", "lc_subtype"]].astype(str)).to_numpy(float))
    BIO = np.nan_to_num(EB.load_bio_features(names).to_numpy(float))
    ESM = np.nan_to_num(next(np.load(p)["emb"] for p in (ROOT / "data/processed").glob("esm2_*.npz")
                             if np.load(p)["emb"].shape == (len(ds.df), 2560)))
    ABL = np.nan_to_num(EB.load_ablang2())
    SAP = None
    bases = [(ESM, rpca), (DESC, rid), (CAT, rid), (BIO, rid), (BIO, lgbm), (ABL, rid)]

    models = {
        "AbDev-X (stacking)": ("stack", None, None),
        "ESM2-Ridge": ("single", ESM, rpca),
        "AbLang2-ElasticNet": ("single", ABL, enet),
        "SaProt-Ridge": ("single", None, rid),  # SaProt features loaded below
    }
    sap = pd.read_csv(ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1/Saprot_VH.csv")
    base = pd.DataFrame({"antibody_name": names.values}).merge(sap, on="antibody_name", how="left")
    SAPRO = np.nan_to_num(base.select_dtypes("number").to_numpy(float))

    rows = []
    for mname, (kind, X, mk) in models.items():
        for split in SPLITS:
            # collect per-fold per-assay preds
            fold_assay = {}
            for a in ASSAYS:
                if kind == "stack":
                    fp = stack_fold_preds(ds, bases, a, split)
                else:
                    Xuse = SAPRO if mname == "SaProt-Ridge" else X
                    fp = single_fold_preds(ds, Xuse, mk, a, split)
                for f, (yt, yp) in fp.items():
                    fold_assay.setdefault(f, {})[a] = spearmanr(yt, yp).statistic
            # per-fold MEAN across assays, then mean+/-std across folds
            fold_means = [np.nanmean(list(v.values())) for v in fold_assay.values()]
            rows.append(dict(model=mname, split=split, mean=round(float(np.mean(fold_means)), 3),
                             std=round(float(np.std(fold_means)), 3)))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/model_comparison_meanstd.csv", index=False)
    print(df.pivot(index="model", columns="split", values="mean").to_string())
    print("\nstd:")
    print(df.pivot(index="model", columns="split", values="std").to_string())


if __name__ == "__main__":
    main()
