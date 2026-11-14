"""HIC specialist with structure-based SAP features. Select by GDPa1 CV; evaluate on GDPa3."""
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
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from abdev.data.dataset import load_gdpa1
import hic_specialist as HS

SAP_COLS = ["sap_total", "sap_positive", "hydrophobic_sasa", "aromatic_sasa", "total_sasa", "hydrophobic_frac_sasa"]
CHAMP = 0.708


def ridge(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 14)))])
def gbm(): return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=8, subsample=.8, colsample_bytree=.6, reg_lambda=5, min_child_samples=10, subsample_freq=1, verbosity=-1, random_state=0)


def sap_matrix(csv, names):
    d = pd.read_csv(csv)
    base = pd.DataFrame({"antibody_name": list(names)}).merge(d, on="antibody_name", how="left")
    return np.nan_to_num(base[SAP_COLS].to_numpy(float))


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    y1 = ds.y("HIC"); y3 = g3["hic_rt_avg"].to_numpy(float); ok = ~np.isnan(y3)

    sap1 = sap_matrix(ROOT / "data/processed/sap_gdpa1.csv", ds.df[ds.id_col])
    sap3 = sap_matrix(ROOT / "data/processed/sap_heldout.csv", g3["antibody_name"])
    # reference block: TAP+Agg+MOE
    files = ["TAP.csv", "Aggrescan3D.csv", "MOE_properties.csv"]
    tam1 = np.nan_to_num(HS.load_feat(HS.BENCH / "GDPa1", ds.df[ds.id_col], files).to_numpy(float))
    cols = HS.load_feat(HS.BENCH / "GDPa1", ds.df[ds.id_col], files).columns
    tam3 = np.nan_to_num(HS.load_feat(HS.BENCH / "heldout_test", g3["antibody_name"], files).reindex(columns=cols, fill_value=0.0).to_numpy(float))

    from abdev.features.descriptors import build_descriptor_matrix
    hd1 = np.nan_to_num(HS.hydro_descriptors(ds.heavy, ds.light).to_numpy(float))
    hd3 = np.nan_to_num(HS.hydro_descriptors(g3["vh"], g3["vl"]).to_numpy(float))
    cands = {
        "SAP": (sap1, sap3),
        "SAP+HydroDesc": (np.hstack([sap1, hd1]), np.hstack([sap3, hd3])),
        "SAP+TAP+Agg+MOE": (np.hstack([sap1, tam1]), np.hstack([sap3, tam3])),
        "TAP+Agg+MOE (ref)": (tam1, tam3),
    }

    def cv_split(X, y, split):
        folds = ds.folds(split); per = []
        for f in sorted(set(folds.tolist())):
            tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
            if len(te) < 3:
                continue
            per.append(spearmanr(y[te], ridge().fit(X[tr], y[tr]).predict(X[te])).statistic)
        return float(np.nanmean(per))

    rows = []
    for name, (X1, X3) in cands.items():
        m = ~np.isnan(y1)
        hr = spearmanr(y3[ok], ridge().fit(X1[m], y1[m]).predict(X3)[ok]).statistic
        hg = spearmanr(y3[ok], gbm().fit(X1[m], y1[m]).predict(X3)[ok]).statistic
        rows.append(dict(features=name,
                         cv_isotype=round(cv_split(X1, y1, "cluster_isotype"), 3),
                         cv_cluster=round(cv_split(X1, y1, "cluster"), 3),
                         heldout_ridge=round(float(hr), 3), heldout_gbm=round(float(hg), 3)))
    df = pd.DataFrame(rows).sort_values("cv_cluster", ascending=False)
    print("=== HIC + structure SAP (champion held-out = 0.708) ===")
    print(df.to_string(index=False))
    best_held = df[["heldout_ridge", "heldout_gbm"]].to_numpy().max()
    print(f"\nBest held-out achieved: {best_held:.3f}  (champion {CHAMP}, beat={best_held>CHAMP})")
    df.to_csv(ROOT / "results/hic_sap.csv", index=False)


if __name__ == "__main__":
    main()
