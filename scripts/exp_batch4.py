"""Experiments (7) PLM pooling ablation, (8) structure-source sensitivity (ABB3 vs MOE SAP)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import freesasa
from Bio.PDB import PDBParser
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE
from abdev.data.dataset import load_gdpa1
ASSAYS = GE.ASSAYS
KD = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5, "GLU": -3.5,
      "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8,
      "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2}
PARSER = PDBParser(QUIET=True)


def ridge(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])


def cv(ds, X, y):
    folds = ds.folds("cluster_isotype"); per = []
    for f in sorted(set(folds.tolist())):
        tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
        if len(te) < 3:
            continue
        per.append(spearmanr(y[te], ridge().fit(X[tr], y[tr]).predict(X[te])).statistic)
    return float(np.mean(per))


def pool(emb, mask, cdr, mode):
    # emb (N,L,D), mask (N,L), cdr (N,L)
    if mode == "mean":
        w = mask
    elif mode == "cdr":
        w = cdr
    else:  # framework
        w = mask * (1 - cdr)
    w = w / (w.sum(1, keepdims=True) + 1e-9)
    return np.einsum("nl,nld->nd", w, emb)


def sap_from(dir_path, name):
    p = Path(dir_path) / f"{name}.pdb"
    if not p.exists():
        return None
    try:
        ra = freesasa.calc(freesasa.Structure(str(p))).residueAreas()
    except Exception:
        return None
    hs = tot = 0.0
    for ch, residues in ra.items():
        for _, area in residues.items():
            sa = area.total; tot += sa
            hs += sa * max(KD.get(area.residueType, 0.0), 0.0)
    return [hs, tot, hs / tot if tot else 0.0]


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    Y3 = {a: g3[GE.LABEL_MAP[a]].to_numpy(float) for a in ASSAYS}

    # ---- (7) pooling ablation (ESM-2 per-residue) ----
    z = np.load(ROOT / "data/processed/esm_perres_150M.npz")
    rows = []
    for mode in ["mean", "cdr", "framework"]:
        vh = pool(z["vh_e"], z["vh_m"], z["vh_c"], mode); vl = pool(z["vl_e"], z["vl_m"], z["vl_c"], mode)
        X = np.nan_to_num(np.hstack([vh, vl]))
        rec = {"pooling": mode}
        for a in ASSAYS:
            rec[a] = round(cv(ds, X, ds.y(a)), 3)
        rec["MEAN"] = round(float(np.mean([rec[a] for a in ASSAYS])), 3)
        rows.append(rec)
    pdf = pd.DataFrame(rows); pdf.to_csv(ROOT / "results/exp_pooling.csv", index=False)
    print("=== (7) ESM-2 pooling ablation (cluster-isotype CV) ===")
    print(pdf.to_string(index=False))

    # ---- (8) structure-source sensitivity: ABB3 vs MOE SAP for HIC ----
    sources = {"AntiBodyBuilder3": "AntiBodyBuilder3", "MOE": "MOE_structures"}
    srow = []
    for sname, sub in sources.items():
        gdir = ROOT / f"external/abdev-benchmark/data/structures/{sub}/GDPa1"
        hdir = ROOT / f"external/abdev-benchmark/data/structures/{sub}/heldout_test"
        S1 = np.array([sap_from(gdir, str(n)) or [0, 0, 0] for n in ds.df[ds.id_col]])
        S3 = np.array([sap_from(hdir, str(n)) or [0, 0, 0] for n in g3["antibody_name"]])
        y = ds.y("HIC"); m = ~np.isnan(y); ok = ~np.isnan(Y3["HIC"])
        cvv = cv(ds, np.nan_to_num(S1), y)
        hov = spearmanr(Y3["HIC"][ok], ridge().fit(np.nan_to_num(S1[m]), y[m]).predict(np.nan_to_num(S3))[ok]).statistic
        found = int((S1.sum(1) != 0).sum())
        srow.append(dict(structure_source=sname, gdpa1_found=found, HIC_CV=round(cvv, 3), HIC_heldout=round(float(hov), 3)))
    sdf = pd.DataFrame(srow); sdf.to_csv(ROOT / "results/exp_structure_source.csv", index=False)
    print("\n=== (8) Structure-source sensitivity (SAP -> HIC) ===")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
