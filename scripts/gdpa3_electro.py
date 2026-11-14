"""Test structural electrostatic features on GDPa3 held-out (self-association / polyreactivity)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE
from abdev.data.dataset import load_gdpa1

COMP = GE.COMP_HELDOUT
EL = ["exp_pos", "exp_neg", "net_exp_charge", "charge_asym", "max_pos_patch", "max_neg_patch", "patch_charge_range"]
SAP = ["sap_total", "sap_positive", "hydrophobic_sasa", "aromatic_sasa", "total_sasa", "hydrophobic_frac_sasa"]


def head(pca=False):
    s = [("s", StandardScaler())]
    if pca:
        s.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    s.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 14))))
    return Pipeline(s)


def mat(csv, names, cols):
    d = pd.read_csv(csv); b = pd.DataFrame({"antibody_name": list(names)}).merge(d, on="antibody_name", how="left")
    return np.nan_to_num(b[cols].to_numpy(float))


def cv_iso(ds, X, y):
    folds = ds.folds("cluster_isotype"); per = []
    for f in sorted(set(folds.tolist())):
        tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
        if len(te) < 3:
            continue
        per.append(spearmanr(y[te], head().fit(X[tr], y[tr]).predict(X[te])).statistic)
    return float(np.nanmean(per))


def main():
    ds, g3, B1, B3 = GE.build_blocks()
    P = ROOT / "data/processed"
    el1 = mat(P / "electro_gdpa1.csv", ds.df[ds.id_col], EL); el3 = mat(P / "electro_heldout.csv", g3["antibody_name"], EL)
    sap1 = mat(P / "sap_gdpa1.csv", ds.df[ds.id_col], SAP); sap3 = mat(P / "sap_heldout.csv", g3["antibody_name"], SAP)
    struct1 = np.hstack([el1, sap1]); struct3 = np.hstack([el3, sap3])

    rows = []
    for a in GE.ASSAYS:
        y1 = ds.y(a); lab = ~np.isnan(y1)
        y3 = g3[GE.LABEL_MAP[a]].to_numpy(float); ok = ~np.isnan(y3)
        abl1, abl3 = np.nan_to_num(B1["ablang2"]), np.nan_to_num(B3["ablang2"])
        desc1, desc3 = np.nan_to_num(B1["descriptors"]), np.nan_to_num(B3["descriptors"])
        cands = {
            "electro": (el1, el3, False),
            "struct(el+sap)": (struct1, struct3, False),
            "ablang2+struct": (np.hstack([abl1, struct1]), np.hstack([abl3, struct3]), True),
            "desc+struct": (np.hstack([desc1, struct1]), np.hstack([desc3, struct3]), False),
        }
        rec = {"assay": a, "champion": COMP[a]}
        for nm, (X1, X3, pca) in cands.items():
            r = spearmanr(y3[ok], head(pca).fit(X1[lab], y1[lab]).predict(X3)[ok]).statistic
            rec[nm] = round(float(r), 3)
        rows.append(rec)
    df = pd.DataFrame(rows)
    print("=== Structural electrostatic/hydrophobic features on GDPa3 held-out ===")
    print(df.to_string(index=False))
    df.to_csv(ROOT / "results/gdpa3_electro.csv", index=False)


if __name__ == "__main__":
    main()
