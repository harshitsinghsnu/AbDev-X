"""Improved held-out model: add AntiBERTy (OAS antibody LM) + structural SAP; per-assay OOD-CV
selection. Reports per-block held-out, OOD-CV-selected block, and oracle ceiling, vs champions.
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
from abdev.data.dataset import load_gdpa1

COMP = GE.COMP_HELDOUT
SAP_COLS = ["sap_total", "sap_positive", "hydrophobic_sasa", "aromatic_sasa", "total_sasa", "hydrophobic_frac_sasa"]


def head(pca=False):
    steps = [("s", StandardScaler())]
    if pca:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 14))))
    return Pipeline(steps)


def sap_mat(csv, names):
    d = pd.read_csv(csv); b = pd.DataFrame({"antibody_name": list(names)}).merge(d, on="antibody_name", how="left")
    return np.nan_to_num(b[SAP_COLS].to_numpy(float))


def main():
    ds, g3, B1, B3 = GE.build_blocks()
    # add AntiBERTy
    B1["antiberty"] = np.load(ROOT / "data/processed/antiberty_gdpa1.npz")["emb"]
    B3["antiberty"] = np.load(ROOT / "data/processed/antiberty_gdpa3.npz")["emb"]
    # add structural SAP (per-residue hydrophobic SASA)
    B1["sap"] = sap_mat(ROOT / "data/processed/sap_gdpa1.csv", ds.df[ds.id_col])
    B3["sap"] = sap_mat(ROOT / "data/processed/sap_heldout.csv", g3["antibody_name"])

    blocks = ["esm2", "descriptors", "isotype", "bio", "ablang2", "antiberty", "sap"]
    pca_blocks = {"esm2", "ablang2", "antiberty"}
    folds = ds.folds("cluster_isotype")

    rows, sel_rows = [], []
    for a in GE.ASSAYS:
        y1 = ds.y(a); lab = ~np.isnan(y1)
        y3 = g3[GE.LABEL_MAP[a]].to_numpy(float); ok = ~np.isnan(y3)
        cv, held = {}, {}
        for blk in blocks:
            X1 = np.nan_to_num(B1[blk]); X3 = np.nan_to_num(B3[blk])
            per = []
            for f in sorted(set(folds.tolist())):
                tr = np.where((folds != f) & lab)[0]; te = np.where((folds == f) & lab)[0]
                if len(te) < 3:
                    continue
                per.append(spearmanr(y1[te], head(blk in pca_blocks).fit(X1[tr], y1[tr]).predict(X1[te])).statistic)
            cv[blk] = float(np.nanmean(per))
            held[blk] = float(spearmanr(y3[ok], head(blk in pca_blocks).fit(X1[lab], y1[lab]).predict(X3)[ok]).statistic)
        rows.append({"assay": a, **{b: round(held[b], 3) for b in blocks}})
        best = max(cv, key=cv.get); oracle = max(held, key=held.get)
        sel_rows.append(dict(assay=a, selected_block=best, selected_heldout=round(held[best], 3),
                             oracle_block=oracle, oracle_heldout=round(held[oracle], 3), champion=COMP[a]))

    print("=== Per-block GDPa1->GDPa3 held-out Spearman (with AntiBERTy + SAP) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    sel = pd.DataFrame(sel_rows)
    sel.loc[len(sel)] = ["MEAN", "", round(sel["selected_heldout"].mean(), 3), "",
                         round(sel["oracle_heldout"].mean(), 3), round(np.mean(list(COMP.values())), 3)]
    print("\n=== OOD-CV-selected vs oracle vs champion (held-out) ===")
    print(sel.to_string(index=False))
    pd.DataFrame(rows).to_csv(ROOT / "results/gdpa3_perblock_improved.csv", index=False)
    sel.to_csv(ROOT / "results/gdpa3_selected_improved.csv", index=False)


if __name__ == "__main__":
    main()
