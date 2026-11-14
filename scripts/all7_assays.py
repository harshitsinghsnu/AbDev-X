"""Run AbDev-X OOD-CV + GDPa3 held-out evaluation on all 7 Jiang et al. assays.

The 7 assays used in Jiang et al. J. Chem. Inf. Model. 2026:
  tm1_nanodsf, tm2_nanodsf, tonset_nanodsf, hic_rt,
  sec_%monomer, acsins_dLmax_ph7.4, polyreactivity_prscore_cho

Uses identical block loading, head, and OOD-CV fold logic as bootstrap_ci.py.
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

# ---------------------------------------------------------------------------
# Assay definitions — GDPa1 column → GDPa3 column
# ---------------------------------------------------------------------------
ASSAY_MAP = {
    # name           GDPa1 column          GDPa3 column
    "HIC":        ("HIC",                 "hic_rt_avg"),
    "Tm2":        ("Tm2",                 "tm2_nanodsf_avg"),
    "Tm1":        ("Tm1",                 "tm1_nanodsf_avg"),
    "Tonset":     ("Tonset",              "tonset_nanodsf_avg"),
    "SEC":        ("SEC %Monomer",        "sec_%monomer_avg"),
    "AC-SINS":    ("AC-SINS_pH7.4",       "acsins_dLmax_ph7.4_avg"),
    "PR_CHO":     ("PR_CHO",              "polyreactivity_prscore_cho_avg"),
    "SMAC":       ("SMAC",                "smac_rt_avg"),
    "HAC":        ("HAC",                 "hac_rt_avg"),
}
# All assays: Jiang 7 + immunogenicity (SMAC, HAC)
JIANG_7 = ["HIC", "Tm2", "Tm1", "Tonset", "SEC", "AC-SINS", "PR_CHO", "SMAC", "HAC"]

# Jiang et al. best model per assay on GDPa3 (from paper text + Figure 5A)
# SMAC and HAC are not in Jiang et al. — marked None
JIANG_BEST = {
    "HIC":     0.558,   # hybrid ESM2+DP — exact from text
    "Tm2":     0.15,    # estimated from Figure 5A
    "Tm1":     0.10,    # estimated from Figure 5A
    "Tonset":  0.03,    # near zero, estimated
    "SEC":     0.00,    # near zero, stated in text
    "AC-SINS": 0.08,    # estimated
    "PR_CHO":  0.18,    # estimated
    "SMAC":    None,    # not in Jiang et al.
    "HAC":     None,    # not in Jiang et al.
}

BIO_FILES = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv"]
ISO_COLS  = ["hc_subtype_IgG1", "hc_subtype_IgG2", "hc_subtype_IgG4",
              "lc_subtype_Kappa", "lc_subtype_Lambda"]


def _isotype_block(df):
    d = pd.get_dummies(df[["hc_subtype", "lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS, fill_value=0).to_numpy(np.float32)


def _bio_block(feat_dir, names, cols=None):
    base = pd.DataFrame({"antibody_name": list(names)})
    for f in BIO_FILES:
        p = Path(feat_dir) / f
        if not p.exists(): continue
        d = pd.read_csv(p)
        num = d.select_dtypes("number").copy()
        num["antibody_name"] = d["antibody_name"].values
        num = num.rename(columns={c: f"{f[:-4]}::{c}" for c in num.columns
                                   if c != "antibody_name"})
        base = base.merge(num, on="antibody_name", how="left")
    X = base.drop(columns=["antibody_name"])
    if cols is None: cols = list(X.columns)
    return np.nan_to_num(X.reindex(columns=cols, fill_value=0.0).to_numpy(float)), cols


def _head(blk):
    steps = [("s", StandardScaler())]
    if blk in ("esm2", "ablang2"):
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def build_blocks(ds, g3):
    esm_files = list((ROOT / "data/processed").glob("esm2_*.npz"))
    esm1 = None
    for ef in esm_files:
        arr = np.load(ef)["emb"]
        if arr.shape == (len(ds.df), 2560):
            esm1 = arr; break

    abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    esm3 = np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"]
    abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]

    desc1 = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
    desc3 = np.nan_to_num(build_descriptor_matrix(g3["vh"], g3["vl"]).to_numpy(float))

    bio1, bio_cols = _bio_block(
        ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1",
        ds.df["antibody_name"])
    bio3, _ = _bio_block(
        ROOT / "external/abdev-benchmark/data/features/processed_features/heldout_test",
        g3["antibody_name"], cols=bio_cols)

    iso1 = _isotype_block(ds.df).astype(float)
    iso3 = _isotype_block(g3).astype(float)

    return {
        "esm2":    (np.nan_to_num(esm1), np.nan_to_num(esm3)),
        "ablang2": (np.nan_to_num(abl1), np.nan_to_num(abl3)),
        "isotype": (iso1, iso3),
        "desc":    (desc1, desc3),
        "bio":     (bio1, bio3),
    }


def run_assay(assay_name, y1, y3, blocks, folds, fold_ids):
    m1  = ~np.isnan(y1)
    ok3 = ~np.isnan(y3)

    cv_scores  = {}
    held_preds = {}

    for blk, (X1, X3) in blocks.items():
        per_fold = []
        for f in fold_ids:
            tr = np.where((folds != f) & m1)[0]
            te = np.where((folds == f) & m1)[0]
            if len(te) < 3 or len(tr) < 10: continue
            pipe = _head(blk)
            pipe.fit(X1[tr], y1[tr])
            p = pipe.predict(X1[te])
            per_fold.append(spearmanr(y1[te], p).statistic)
        cv_scores[blk]  = float(np.nanmean(per_fold)) if per_fold else -1.0
        held_preds[blk] = _head(blk).fit(X1[m1], y1[m1]).predict(X3)

    ranked   = sorted(cv_scores.items(), key=lambda x: -x[1])
    best_blk = ranked[0][0]
    rho      = float(spearmanr(y3[ok3], held_preds[best_blk][ok3]).statistic)
    cv_top   = "  ".join(f"{k}={v:.3f}" for k, v in ranked[:3])

    return best_blk, ranked[0][1], rho, cv_top


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds    = ds.folds("cluster_isotype")
    fold_ids = sorted(set(folds.tolist()))

    blocks = build_blocks(ds, g3)

    rows = []
    print(f"\n{'Assay':10s}  {'Best block':10s}  {'OOD-CV':7s}  {'AbDev-X':8s}  "
          f"{'Jiang best':10s}  {'Δ':7s}  CV top-3")
    print("-" * 90)

    for name in JIANG_7:
        g1_col, g3_col = ASSAY_MAP[name]

        # GDPa1 label — load directly from df column (works for all assays)
        if g1_col in ds.df.columns:
            y1 = ds.df[g1_col].to_numpy(float)
        else:
            try:
                y1 = ds.y(g1_col)
            except Exception:
                print(f"  WARNING: {g1_col} not in GDPa1"); continue

        # GDPa3 label
        if g3_col in g3.columns:
            y3 = g3[g3_col].to_numpy(float)
        else:
            print(f"  WARNING: {g3_col} not in gdpa3_table"); continue

        n1 = (~np.isnan(y1)).sum()
        n3 = (~np.isnan(y3)).sum()

        best_blk, cv_rho, rho, cv_top = run_assay(name, y1, y3, blocks, folds, fold_ids)
        jiang = JIANG_BEST.get(name)
        delta_str = f"{rho - jiang:+7.3f}" if jiang is not None else "   N/A "
        symbol    = ("✓" if rho > jiang else "✗") if jiang is not None else "-"
        jiang_str = f"{jiang:10.3f}" if jiang is not None else "       N/A"
        print(f"{name:10s}  {best_blk:10s}  {cv_rho:7.3f}  {rho:8.3f}  "
              f"{jiang_str}  {delta_str} {symbol}  {cv_top}  [n1={n1},n3={n3}]")

        rows.append({"assay": name, "g1_col": g1_col, "g3_col": g3_col,
                     "n_train": int(n1), "n_test": int(n3),
                     "best_blk": best_blk, "cv_rho": round(cv_rho, 3),
                     "abdevx": round(rho, 3),
                     "jiang_best": jiang if jiang is not None else np.nan,
                     "delta": round(rho - jiang, 3) if jiang is not None else np.nan})

    df = pd.DataFrame(rows)
    mean_abdevx     = df["abdevx"].mean()
    shared          = df[df["jiang_best"].notna()]
    mean_jiang      = shared["jiang_best"].mean()
    mean_abdevx_sh  = shared["abdevx"].mean()
    wins            = int((shared["abdevx"] > shared["jiang_best"]).sum())
    print("-" * 90)
    print(f"{'MEAN (all)':10s}  {'':10s}  {'':7s}  {mean_abdevx:8.3f}  {'':10s}")
    print(f"{'MEAN (7)':10s}  {'':10s}  {'':7s}  {mean_abdevx_sh:8.3f}  "
          f"{mean_jiang:10.3f}  {mean_abdevx_sh-mean_jiang:+7.3f}  wins={wins}/7")

    out = ROOT / "results/all7_assays.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
