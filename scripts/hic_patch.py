"""Test patch-SAP (spatial hydrophobic patches) on HIC held-out vs champion 0.708."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1

PATCH = ["patch_sap_max", "patch_sap_sum_pos", "patch_sap_top5", "patch_sap_n_hot", "patch_sap_mean"]
SAP = ["sap_total", "sap_positive", "hydrophobic_sasa", "aromatic_sasa", "total_sasa", "hydrophobic_frac_sasa"]


def ridge(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 14)))])


def mat(csv, names, cols):
    d = pd.read_csv(csv)
    b = pd.DataFrame({"antibody_name": list(names)}).merge(d, on="antibody_name", how="left")
    return np.nan_to_num(b[cols].to_numpy(float))


def cvs(ds, X, y, split):
    folds = ds.folds(split); per = []
    for f in sorted(set(folds.tolist())):
        tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
        if len(te) < 3:
            continue
        per.append(spearmanr(y[te], ridge().fit(X[tr], y[tr]).predict(X[te])).statistic)
    return float(np.nanmean(per))


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    y1 = ds.y("HIC"); y3 = g3["hic_rt_avg"].to_numpy(float); ok = ~np.isnan(y3); m = ~np.isnan(y1)
    P = ROOT / "data/processed"
    patch1 = mat(P / "sap_patch_gdpa1.csv", ds.df[ds.id_col], PATCH); patch3 = mat(P / "sap_patch_heldout.csv", g3["antibody_name"], PATCH)
    sap1 = mat(P / "sap_gdpa1.csv", ds.df[ds.id_col], SAP); sap3 = mat(P / "sap_heldout.csv", g3["antibody_name"], SAP)
    cands = {"PatchSAP": (patch1, patch3), "SAP(res)": (sap1, sap3),
             "PatchSAP+SAP": (np.hstack([patch1, sap1]), np.hstack([patch3, sap3]))}
    rows = []
    for nm, (X1, X3) in cands.items():
        hr = spearmanr(y3[ok], ridge().fit(X1[m], y1[m]).predict(X3)[ok]).statistic
        rows.append(dict(features=nm, cv_isotype=round(cvs(ds, X1, y1, "cluster_isotype"), 3),
                         cv_cluster=round(cvs(ds, X1, y1, "cluster"), 3), heldout=round(float(hr), 3)))
    df = pd.DataFrame(rows)
    print("=== HIC patch-SAP (champion held-out = 0.708) ===")
    print(df.to_string(index=False))
    print(f"\nBest held-out: {df['heldout'].max():.3f}")
    df.to_csv(ROOT / "results/hic_patch.csv", index=False)


if __name__ == "__main__":
    main()
