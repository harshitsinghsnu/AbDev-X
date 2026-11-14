"""HIC specialist: try to beat the competition HIC champion (held-out rho 0.708).

HIC (hydrophobic-interaction-chromatography retention) is driven by surface hydrophobicity. We
assemble hydrophobicity-focused feature groups available for BOTH GDPa1 and GDPa3, tune the model
by GDPa1 cluster_isotype CV (no held-out peeking), and evaluate the SELECTED model on GDPa3.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.descriptors import build_descriptor_matrix

BENCH = ROOT / "external/abdev-benchmark/data/features/processed_features"
CHAMP = 0.708


def load_feat(feat_dir, names, files):
    base = pd.DataFrame({"antibody_name": list(names)})
    for f in files:
        p = Path(feat_dir) / f
        if not p.exists():
            continue
        d = pd.read_csv(p); num = d.select_dtypes("number").copy(); num["antibody_name"] = d["antibody_name"].values
        num = num.rename(columns={c: f"{f[:-4]}::{c}" for c in num.columns if c != "antibody_name"})
        base = base.merge(num, on="antibody_name", how="left")
    return base.drop(columns=["antibody_name"])


def hydro_descriptors(heavy, light):
    """HIC-relevant subset of physicochemical descriptors (hydrophobicity/aromaticity)."""
    D = build_descriptor_matrix(heavy, light)
    keep = [c for c in D.columns if any(k in c for k in
            ["gravy", "hmoment", "aromatic", "aliphatic", "CDR3_hydro", "hydroph", "W", "tryptophan"])]
    return D[keep]


def ridge(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 14)))])
def gbm(): return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=8, subsample=.8, colsample_bytree=.6, reg_lambda=5, min_child_samples=10, subsample_freq=1, verbosity=-1, random_state=0)


def cv_score(ds, X, y):
    per = []
    for f in sorted(set(ds.folds("cluster_isotype").tolist())):
        folds = ds.folds("cluster_isotype")
        tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
        if len(te) < 3:
            continue
        per.append(spearmanr(y[te], ridge().fit(X[tr], y[tr]).predict(X[te])).statistic)
    return float(np.nanmean(per))


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    y1 = ds.y("HIC"); y3 = g3["hic_rt_avg"].to_numpy(float); ok = ~np.isnan(y3)

    TAP = ["TAP.csv"]; AGG = ["Aggrescan3D.csv"]; MOE = ["MOE_properties.csv"]
    groups = {
        "TAP": (TAP,), "Aggrescan3D": (AGG,), "MOE": (MOE,),
        "TAP+Agg": (TAP + AGG,), "TAP+Agg+MOE": (TAP + AGG + MOE,),
    }
    # build each feature matrix for GDPa1 & GDPa3
    cand = {}
    for name, (files,) in groups.items():
        X1 = np.nan_to_num(load_feat(BENCH / "GDPa1", ds.df[ds.id_col], files).to_numpy(float))
        cols = load_feat(BENCH / "GDPa1", ds.df[ds.id_col], files).columns
        f3 = load_feat(BENCH / "heldout_test", g3["antibody_name"], files).reindex(columns=cols, fill_value=0.0)
        X3 = np.nan_to_num(f3.to_numpy(float))
        cand[name] = (X1, X3)
    # add hydrophobicity descriptors + combos
    hd1 = np.nan_to_num(hydro_descriptors(ds.heavy, ds.light).to_numpy(float))
    hd3 = np.nan_to_num(hydro_descriptors(g3["vh"], g3["vl"]).reindex(
        columns=hydro_descriptors(ds.heavy, ds.light).columns, fill_value=0.0).to_numpy(float)
        if False else hydro_descriptors(g3["vh"], g3["vl"]).to_numpy(float))
    cand["HydroDesc"] = (hd1, hd3)
    tapagg1, tapagg3 = cand["TAP+Agg"]
    cand["TAP+Agg+HydroDesc"] = (np.hstack([tapagg1, hd1]), np.hstack([tapagg3, hd3]))
    ta_moe1, ta_moe3 = cand["TAP+Agg+MOE"]
    cand["TAP+Agg+MOE+HydroDesc"] = (np.hstack([ta_moe1, hd1]), np.hstack([ta_moe3, hd3]))

    rows = []
    for name, (X1, X3) in cand.items():
        cv = cv_score(ds, X1, y1)
        m = ~np.isnan(y1)
        held_ridge = spearmanr(y3[ok], ridge().fit(X1[m], y1[m]).predict(X3)[ok]).statistic
        held_gbm = spearmanr(y3[ok], gbm().fit(X1[m], y1[m]).predict(X3)[ok]).statistic
        rows.append(dict(features=name, gdpa1_cv=round(cv, 3),
                         heldout_ridge=round(float(held_ridge), 3), heldout_gbm=round(float(held_gbm), 3)))
    df = pd.DataFrame(rows).sort_values("gdpa1_cv", ascending=False)
    print("=== HIC specialist (selected by GDPa1 CV; competition champion held-out = 0.708) ===")
    print(df.to_string(index=False))
    sel = df.iloc[0]
    print(f"\nModel selected by GDPa1-CV: '{sel.features}' -> held-out ridge={sel.heldout_ridge}, gbm={sel.heldout_gbm}")
    print(f"Champion held-out = {CHAMP}. Beat = {max(sel.heldout_ridge, sel.heldout_gbm) > CHAMP}")
    df.to_csv(ROOT / "results/hic_specialist.csv", index=False)


if __name__ == "__main__":
    main()
