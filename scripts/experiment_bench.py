"""Benchmark AbDev-X against the official 14 baselines using the EXACT competition protocol.

Protocol (from ginkgobioworks/abdev-benchmark):
  * fold column: hierarchical_cluster_IgG_isotype_stratified_fold
  * train on 4 folds, predict held-out 5th; repeat; pool all out-of-fold predictions
  * metric: 'aggregated' Spearman (one rho over pooled OOF predictions) per assay

Feature blocks (all biologically meaningful):
  * CAT   : germline/isotype one-hot (hc_subtype, lc_subtype)         -> thermostability driver
  * DESC  : our 165 interpretable physicochemical descriptors
  * BIO   : official precomputed TAP + MOE + Aggrescan3D + AntiFold + DeepViscosity + Saprot_VH
  * ESM   : ESM-2 650M mean+CDR embeddings (cached)
Per-representation base learners are stacked by a per-assay ridge meta-learner.
"""
from __future__ import annotations

import sys, warnings, glob, os
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.descriptors import build_descriptor_matrix
from scipy.stats import spearmanr

BENCH = ROOT / "external/abdev-benchmark"
FEAT = BENCH / "data/features/processed_features/GDPa1"
ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
FOLD = "cluster_isotype"


def load_bio_features(names: pd.Series) -> pd.DataFrame:
    """Merge official precomputed biological features on antibody_name, aligned to `names`."""
    base = pd.DataFrame({"antibody_name": names.values})
    files = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv",
             "DeepViscosity.csv", "Saprot_VH.csv"]
    for f in files:
        p = FEAT / f
        if not p.exists():
            continue
        df = pd.read_csv(p)
        num = df.select_dtypes("number").copy()
        num["antibody_name"] = df["antibody_name"].values
        num = num.rename(columns={c: f"{f[:-4]}::{c}" for c in num.columns if c != "antibody_name"})
        base = base.merge(num, on="antibody_name", how="left")
    feat = base.drop(columns=["antibody_name"])
    return feat


def load_esm():
    for f in (ROOT / "data/processed").glob("esm2_*.npz"):
        a = np.load(f)["emb"]
        if a.shape[1] == 2560:
            return a
    return None


def load_ablang2():
    p = ROOT / "data/processed/ablang2_gdpa1.npz"
    return np.load(p)["emb"] if p.exists() else None


def agg_spearman(y, pred, mask):
    m = mask & ~np.isnan(pred)
    return float(spearmanr(y[m], pred[m]).statistic)


def rpca():
    return Pipeline([("s", StandardScaler()), ("p", PCA(0.9, random_state=0, svd_solver="full")),
                     ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])
def rid():
    return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])
def lgbm():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=7, subsample=.8,
                             colsample_bytree=.6, reg_lambda=5, min_child_samples=12,
                             subsample_freq=1, verbosity=-1, random_state=0)


def stack_oof(ds, bases, assay):
    """Return pooled OOF predictions (aggregated protocol) + mean meta-weights for one assay."""
    y = ds.y(assay)
    n = len(y)
    pooled = np.full(n, np.nan)
    weights = []
    for _, tr, te in fold_iter(ds, assay, FOLD):
        # inner OOF for meta
        oof = [np.full(len(tr), np.nan) for _ in bases]
        kf = KFold(5, shuffle=True, random_state=0)
        for itr, iva in kf.split(np.arange(len(tr))):
            for bi, (X, mk) in enumerate(bases):
                oof[bi][iva] = mk().fit(X[tr[itr]], y[tr[itr]]).predict(X[tr[iva]])
        Z = np.column_stack(oof)
        meta = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(Z, y[tr])
        weights.append(meta.coef_)
        fit = [mk().fit(X[tr], y[tr]) for X, mk in bases]
        pooled[te] = meta.predict(np.column_stack([f.predict(X[te]) for f, (X, mk) in zip(fit, bases)]))
    return y, pooled, np.mean(weights, axis=0)


def baseline_bar():
    rows = []
    for f in glob.glob(str(BENCH / "outputs/evaluation/*_cv.csv")):
        m = os.path.basename(f).replace("_cv.csv", "")
        df = pd.read_csv(f)
        sub = df[(df["split"] == "test") & (df["fold"] == "aggregated")]
        for _, r in sub.iterrows():
            rows.append({"model": m, "assay": r["assay"], "spearman": r["spearman"]})
    p = pd.DataFrame(rows).pivot_table(index="model", columns="assay", values="spearman")
    return p


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    names = ds.df[ds.id_col]
    DESC = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
    CAT = np.nan_to_num(pd.get_dummies(ds.df[["hc_subtype", "lc_subtype"]].astype(str)).to_numpy(float))
    BIO = np.nan_to_num(load_bio_features(names).to_numpy(float))
    ESM = load_esm()
    ABL = load_ablang2()
    print(f"blocks: DESC{DESC.shape} CAT{CAT.shape} BIO{BIO.shape} "
          f"ESM{None if ESM is None else ESM.shape} ABL{None if ABL is None else ABL.shape}")

    block_names = ["ESM2", "Descriptors", "Germline/Isotype", "BioFeatures(ridge)", "BioFeatures(lgbm)"]
    bases = [(np.nan_to_num(ESM), rpca), (DESC, rid), (CAT, rid), (BIO, rid), (BIO, lgbm)]
    if ABL is not None:
        bases.append((np.nan_to_num(ABL), rid)); block_names.append("AbLang2")

    res, wrows = {}, []
    for a in ASSAYS:
        y, pooled, w = stack_oof(ds, bases, a)
        res[a] = agg_spearman(y, pooled, ~np.isnan(y))
        wrows.append(dict(assay=a, **{block_names[i]: round(float(w[i]), 3) for i in range(len(w))}))
    res["MEAN"] = float(np.mean([res[a] for a in ASSAYS]))

    bar = baseline_bar()[ASSAYS]
    best = {a: bar[a].max() for a in ASSAYS}
    bestmodel = {a: bar[a].idxmax() for a in ASSAYS}
    print("\n=== AbDev-X vs BEST official baseline (aggregated CV Spearman) ===")
    print(f"{'assay':16s} {'AbDev-X':>8s} {'best_base':>10s} {'baseline_model':>20s} {'win':>5s}")
    wins = 0
    rows = []
    for a in ASSAYS:
        w = res[a] > best[a]; wins += w
        print(f"{a:16s} {res[a]:8.3f} {best[a]:10.3f} {bestmodel[a]:>20s} {'YES' if w else 'no':>5s}")
        rows.append(dict(assay=a, abdevx=round(res[a], 3), best_baseline=round(best[a], 3),
                         best_baseline_model=bestmodel[a], win=bool(w)))
    print(f"{'MEAN':16s} {res['MEAN']:8.3f} {bar.mean(1).max():10.3f} {'(best-mean model)':>20s}")
    print(f"\nBeats best-per-assay baseline on {wins}/5 assays.")
    print(f"AbDev-X MEAN {res['MEAN']:.3f} vs best single-baseline MEAN {bar.mean(1).max():.3f}")

    outdir = ROOT / "results"; outdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "benchmark_comparison.csv", index=False)
    # full leaderboard: all baselines + AbDev-X
    full = bar.copy(); full["MEAN"] = full.mean(axis=1)
    ab = pd.DataFrame([{**{a: res[a] for a in ASSAYS}, "MEAN": res["MEAN"]}], index=["AbDevX"])
    pd.concat([ab, full]).sort_values("MEAN", ascending=False).round(3).to_csv(outdir / "full_leaderboard.csv")
    # biological interpretation: per-assay meta-weights over representations
    pd.DataFrame(wrows).to_csv(outdir / "representation_weights.csv", index=False)
    print("\n=== Biological interpretation: meta-weight per representation (per assay) ===")
    print(pd.DataFrame(wrows).to_string(index=False))
    print(f"\nSaved: benchmark_comparison.csv, full_leaderboard.csv, representation_weights.csv")


if __name__ == "__main__":
    main()
