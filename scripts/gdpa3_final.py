"""Final GDPa3 held-out comparison: naive fusion vs generalization-aware methods vs baselines vs
competition champions. All selection/weighting uses ONLY GDPa1 (no held-out peeking).
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE

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
    ufolds = [f for f in sorted(set(folds.tolist())) if f == f]

    res = {k: {} for k in ["fusion", "best_oodcv", "ood_stack", "ESM2", "AbLang2"]}
    for a in ASSAYS:
        y1 = ds.y(a); lab = ~np.isnan(y1)
        y3 = g3[GE.LABEL_MAP[a]].to_numpy(float); ok = ~np.isnan(y3)

        # per-block: OOD-CV score, OOF preds (cluster folds), full-data held-out preds
        cv, oof, held = {}, {}, {}
        for blk in BLOCKS:
            X1 = np.nan_to_num(B1[blk]); X3 = np.nan_to_num(B3[blk])
            o = np.full(len(y1), np.nan); per = []
            for f in ufolds:
                tr = np.where((folds != f) & lab)[0]; te = np.where((folds == f) & lab)[0]
                if len(te) < 3 or len(tr) < 10:
                    continue
                p = head(blk).fit(X1[tr], y1[tr]).predict(X1[te])
                o[te] = p; per.append(spearmanr(y1[te], p).statistic)
            cv[blk] = float(np.nanmean(per)); oof[blk] = o
            held[blk] = head(blk).fit(X1[lab], y1[lab]).predict(X3)

        # (a) naive fusion: ridge meta on RANDOM inner OOF -> approximate with full AbDevFusion held pred
        from abdev.models.fusion import AbDevFusion
        B1m = {k: v[lab] for k, v in B1.items()}
        res["fusion"][a] = spearmanr(y3[ok], AbDevFusion().fit(B1m, y1[lab]).predict(B3)[ok]).statistic
        # (b) best block by OOD-CV
        best = max(cv, key=cv.get)
        res["best_oodcv"][a] = spearmanr(y3[ok], held[best][ok]).statistic
        # (c) generalization-aware stack: ridge meta trained on cluster-fold OOF
        common = np.all([~np.isnan(oof[b]) for b in BLOCKS], axis=0) & lab
        Z = np.column_stack([oof[b][common] for b in BLOCKS])
        meta = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(Z, y1[common])
        Zh = np.column_stack([held[b] for b in BLOCKS])
        res["ood_stack"][a] = spearmanr(y3[ok], meta.predict(Zh)[ok]).statistic
        # baselines
        res["ESM2"][a] = spearmanr(y3[ok], held["esm2"][ok]).statistic
        res["AbLang2"][a] = spearmanr(y3[ok], held["ablang2"][ok]).statistic

    rows = []
    for model, d in res.items():
        r = {a: round(float(d[a]), 3) for a in ASSAYS}
        r["MEAN"] = round(float(np.mean(list(d.values()))), 3)
        r["model"] = model
        rows.append(r)
    rows.append({**{a: COMP[a] for a in ASSAYS}, "MEAN": round(np.mean(list(COMP.values())), 3), "model": "competition_top(champions)"})
    df = pd.DataFrame(rows)[["model"] + ASSAYS + ["MEAN"]]
    print("=== GDPa3 HELD-OUT (train GDPa1 -> predict GDPa3), Spearman ===")
    print(df.to_string(index=False))
    df.to_csv(ROOT / "results/gdpa3_final.csv", index=False)


if __name__ == "__main__":
    main()
