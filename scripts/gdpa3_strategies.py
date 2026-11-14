"""Generalization-aware combination: select/weight representations by GDPa1 OOD-CV (no held-out
peeking), then evaluate transfer to GDPa3. Tests whether OOD-CV-guided selection generalizes better
than random-CV stacking (the current fusion).
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
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, zscore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE
from abdev.data.splits import fold_iter

ASSAYS = GE.ASSAYS
BLOCKS = ["esm2", "descriptors", "isotype", "bio", "ablang2"]
COMP = GE.COMP_HELDOUT


def head(blk):
    steps = [("s", StandardScaler())]
    if blk in ("esm2", "ablang2"):
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def main():
    ds, g3, B1, B3 = GE.build_blocks()
    folds = ds.folds("cluster_isotype")

    rows = []
    for a in ASSAYS:
        y1 = ds.y(a); y3 = g3[GE.LABEL_MAP[a]].to_numpy(float); ok = ~np.isnan(y3)
        # per-block GDPa1 OOD-CV score + held-out prediction
        cv, heldpred = {}, {}
        for blk in BLOCKS:
            X1 = np.nan_to_num(B1[blk]); X3 = np.nan_to_num(B3[blk])
            per = []
            for f in sorted(set(folds.tolist())):
                tr = np.where((folds != f) & ~np.isnan(y1))[0]; te = np.where((folds == f) & ~np.isnan(y1))[0]
                if len(te) < 3 or len(tr) < 10:
                    continue
                p = head(blk).fit(X1[tr], y1[tr]).predict(X1[te])
                per.append(spearmanr(y1[te], p).statistic)
            cv[blk] = float(np.nanmean(per))
            m = ~np.isnan(y1)
            heldpred[blk] = head(blk).fit(X1[m], y1[m]).predict(X3)
        # strategy 1: best block by OOD-CV
        best = max(cv, key=cv.get)
        r_best = spearmanr(y3[ok], heldpred[best][ok]).statistic
        # strategy 2: OOD-CV-weighted average of z-scored block predictions (weights = relu(cv))
        w = np.array([max(cv[b], 0.0) for b in BLOCKS]); w = w / (w.sum() + 1e-9)
        Z = np.column_stack([zscore(heldpred[b]) for b in BLOCKS])
        r_wavg = spearmanr(y3[ok], (Z @ w)[ok]).statistic
        # strategy 3: top-2 blocks by OOD-CV, equal z-avg
        top2 = sorted(cv, key=cv.get, reverse=True)[:2]
        r_top2 = spearmanr(y3[ok], (zscore(heldpred[top2[0]]) + zscore(heldpred[top2[1]]))[ok]).statistic
        rows.append(dict(assay=a, best_block=best, best=round(float(r_best), 3),
                         wavg=round(float(r_wavg), 3), top2=round(float(r_top2), 3),
                         competition=COMP[a]))
    df = pd.DataFrame(rows)
    for c in ["best", "wavg", "top2"]:
        df.loc["MEAN", c] = round(df[c].mean(), 3)
    df.loc["MEAN", "assay"] = "MEAN"; df.loc["MEAN", "competition"] = round(np.mean(list(COMP.values())), 3)
    print("=== GDPa3 held-out via OOD-CV-guided selection (no held-out peeking) ===")
    print(df.to_string(index=False))
    df.to_csv(ROOT / "results/gdpa3_strategies.csv", index=False)


if __name__ == "__main__":
    main()
