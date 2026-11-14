"""Per-representation GDPa1->GDPa3 held-out diagnostic: which blocks transfer OOD?"""
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

ASSAYS = GE.ASSAYS


def rid(pca=False):
    steps = [("s", StandardScaler())]
    if pca:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def main():
    ds, g3, B1, B3 = GE.build_blocks()
    blocks = ["esm2", "descriptors", "isotype", "bio", "ablang2"]
    rows = []
    for a in ASSAYS:
        y1 = ds.y(a); m = ~np.isnan(y1)
        y3 = g3[GE.LABEL_MAP[a]].to_numpy(float); ok = ~np.isnan(y3)
        rec = {"assay": a}
        for blk in blocks:
            X1, X3 = np.nan_to_num(B1[blk])[m], np.nan_to_num(B3[blk])
            mdl = rid(pca=(blk in ("esm2", "ablang2"))).fit(X1, y1[m])
            p = mdl.predict(X3)
            rec[blk] = round(float(spearmanr(y3[ok], p[ok]).statistic), 3)
        rows.append(rec)
    df = pd.DataFrame(rows)
    print("=== Per-representation GDPa1->GDPa3 held-out Spearman ===")
    print(df.to_string(index=False))
    df.to_csv(ROOT / "results/gdpa3_perblock.csv", index=False)


if __name__ == "__main__":
    main()
