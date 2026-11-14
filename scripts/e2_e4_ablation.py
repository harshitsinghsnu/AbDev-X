"""E2→E4 ablation: multi-view + biological residue masks.

E1: existing AbDev-X (single best block, per-assay OOD-CV)
E2: ESM-2 + AbLang2 concatenated (multi-view sequence)
E3: E2 + global structural/bio features
E4: biological descriptor masks per assay (hydrophobic / electrostatic / stability subsets)

All evaluated under same OOD-CV folds and GDPa3 held-out.
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

ASSAY_MAP = {
    "HIC":     ("HIC",          "hic_rt_avg"),
    "Tm2":     ("Tm2",          "tm2_nanodsf_avg"),
    "Tm1":     ("Tm1",          "tm1_nanodsf_avg"),
    "Tonset":  ("Tonset",       "tonset_nanodsf_avg"),
    "SEC":     ("SEC %Monomer", "sec_%monomer_avg"),
    "AC-SINS": ("AC-SINS_pH7.4","acsins_dLmax_ph7.4_avg"),
    "PR_CHO":  ("PR_CHO",       "polyreactivity_prscore_cho_avg"),
    "SMAC":    ("SMAC",         "smac_rt_avg"),
    "HAC":     ("HAC",          "hac_rt_avg"),
}

BIO_FILES = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv"]
ISO_COLS  = ["hc_subtype_IgG1", "hc_subtype_IgG2", "hc_subtype_IgG4",
             "lc_subtype_Kappa", "lc_subtype_Lambda"]


def _iso(df):
    d = pd.get_dummies(df[["hc_subtype", "lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS, fill_value=0).to_numpy(float)


def _bio(feat_dir, names, cols=None):
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


def _head(X):
    steps = [("s", StandardScaler())]
    if X.shape[1] > 100:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def ood_cv_rho(X1, y1, folds, fold_ids):
    m1 = ~np.isnan(y1)
    per_fold = []
    for f in fold_ids:
        tr = np.where((folds != f) & m1)[0]
        te = np.where((folds == f) & m1)[0]
        if len(te) < 3 or len(tr) < 10: continue
        pipe = _head(X1)
        pipe.fit(X1[tr], y1[tr])
        p = pipe.predict(X1[te])
        per_fold.append(spearmanr(y1[te], p).statistic)
    return float(np.nanmean(per_fold)) if per_fold else -1.0


def held_out_rho(X1, y1, X3, y3):
    m1 = ~np.isnan(y1); ok3 = ~np.isnan(y3)
    pipe = _head(X1)
    pipe.fit(X1[m1], y1[m1])
    pred = pipe.predict(X3)
    if ok3.sum() < 5: return np.nan
    return float(spearmanr(y3[ok3], pred[ok3]).statistic)


# ---- Biological mask definitions ----------------------------------------
# These are keyword-based filters on descriptor / bio column names.
# Each mask returns a boolean array over the feature columns.

HYDRO_KEYWORDS = ["hydro", "GRAVY", "aliphat", "aromat", "psh", "patch_hyd",
                  "Aggrescan", "hyd", "HydroScore"]
CHARGE_KEYWORDS = ["charge", "pI", "ppc", "pnc", "isoelectric", "zeta",
                   "dipole", "patch_pos", "patch_neg", "electr"]
STAB_KEYWORDS   = ["stab", "sasa", "gyrat", "pack", "mobil", "AntiFold",
                   "tm", "Tm", "cavity", "burial"]
LIAB_KEYWORDS   = ["deamid", "isomeriz", "glyc", "oxid", "cys", "liabil",
                   "nglyc", "asn", "asp"]
CDR_KEYWORDS    = ["cdr", "CDR", "loop", "h3", "l3", "h1", "h2", "l1", "l2"]
SURFACE_KEYWORDS = ["sasa", "SASA", "rSASA", "exposed", "surface", "patch"]

def _mask(cols, keywords):
    kw = [k.lower() for k in keywords]
    return np.array([any(k in c.lower() for k in kw) for c in cols])


def build_e4_block(desc_cols, bio_cols, desc1, bio1, desc3, bio3, assay):
    """Return (X_train, X_test) for E4 biological mask per assay."""
    all_cols = list(desc_cols) + list(bio_cols)
    X1 = np.hstack([desc1, bio1])
    X3 = np.hstack([desc3, bio3])

    # Assay-specific mask
    if assay == "HIC":
        mask = _mask(all_cols, HYDRO_KEYWORDS + SURFACE_KEYWORDS)
    elif assay in ("Tm1", "Tm2", "Tonset"):
        mask = _mask(all_cols, STAB_KEYWORDS + LIAB_KEYWORDS)
    elif assay == "SEC":
        mask = _mask(all_cols, HYDRO_KEYWORDS + STAB_KEYWORDS + LIAB_KEYWORDS)
    elif assay == "AC-SINS":
        mask = _mask(all_cols, CHARGE_KEYWORDS + SURFACE_KEYWORDS)
    elif assay in ("PR_CHO",):
        mask = _mask(all_cols, CHARGE_KEYWORDS + CDR_KEYWORDS + HYDRO_KEYWORDS)
    elif assay in ("SMAC", "HAC"):
        mask = _mask(all_cols, SURFACE_KEYWORDS + HYDRO_KEYWORDS + CHARGE_KEYWORDS)
    else:
        mask = np.ones(len(all_cols), dtype=bool)

    if mask.sum() < 5:
        mask = np.ones(len(all_cols), dtype=bool)

    return X1[:, mask], X3[:, mask], int(mask.sum())


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds    = ds.folds("cluster_isotype")
    fold_ids = sorted(set(folds.tolist()))

    # Feature loading
    esm_files = list((ROOT / "data/processed").glob("esm2_*.npz"))
    esm1 = None
    for ef in esm_files:
        arr = np.load(ef)["emb"]
        if arr.shape == (len(ds.df), 2560):
            esm1 = arr; break

    abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    esm3 = np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"]
    abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]

    desc_df1 = build_descriptor_matrix(ds.heavy, ds.light)
    desc_cols = list(desc_df1.columns)
    desc1 = np.nan_to_num(desc_df1.to_numpy(float))
    desc3 = np.nan_to_num(build_descriptor_matrix(g3["vh"], g3["vl"]).to_numpy(float))

    bio1, bio_cols = _bio(
        ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1",
        ds.df["antibody_name"])
    bio3, _ = _bio(
        ROOT / "external/abdev-benchmark/data/features/processed_features/heldout_test",
        g3["antibody_name"], cols=bio_cols)

    iso1 = _iso(ds.df); iso3 = _iso(g3)
    esm1n = np.nan_to_num(esm1); esm3n = np.nan_to_num(esm3)
    abl1n = np.nan_to_num(abl1); abl3n = np.nan_to_num(abl3)

    # E1 best block per assay (reproduce from all7_assays baseline)
    blocks_tr = {"esm2": esm1n, "ablang2": abl1n, "isotype": iso1,
                 "desc": desc1, "bio": bio1}
    blocks_te = {"esm2": esm3n, "ablang2": abl3n, "isotype": iso3,
                 "desc": desc3, "bio": bio3}

    print(f"\n{'Assay':10s}  {'E1-blk':8s}  {'E1':6s}  {'E2':6s}  {'E3':6s}  "
          f"{'E4-mask':8s}  {'E4':6s}  {'Best':6s}  {'E4-ndim':7s}")
    print("-"*80)

    rows = []
    for assay, (g1_col, g3_col) in ASSAY_MAP.items():
        y1 = ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df), np.nan)
        y3 = g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(len(g3), np.nan)

        # E1: best single block
        cv_scores = {b: ood_cv_rho(X, y1, folds, fold_ids) for b, X in blocks_tr.items()}
        best_blk  = max(cv_scores, key=lambda k: cv_scores[k])
        e1 = held_out_rho(blocks_tr[best_blk], y1, blocks_te[best_blk], y3)

        # E2: ESM-2 + AbLang2 concatenated, PCA 90%
        X2_tr = np.hstack([esm1n, abl1n])
        X2_te = np.hstack([esm3n, abl3n])
        e2 = held_out_rho(X2_tr, y1, X2_te, y3)

        # E3: E2 + bio + desc (all non-isotype features)
        X3_tr = np.hstack([esm1n, abl1n, bio1, desc1])
        X3_te = np.hstack([esm3n, abl3n, bio3, desc3])
        e3 = held_out_rho(X3_tr, y1, X3_te, y3)

        # E4: biological mask on desc+bio (no PLM — tests pure mechanistic signal)
        X4_tr, X4_te, ndim = build_e4_block(desc_cols, bio_cols, desc1, bio1,
                                             desc3, bio3, assay)
        e4 = held_out_rho(X4_tr, y1, X4_te, y3)

        best_e = max([(e1,"E1"),(e2,"E2"),(e3,"E3"),(e4,"E4")],
                     key=lambda x: x[0] if not np.isnan(x[0]) else -9)[1]

        def f(v): return f"{v:+6.3f}" if not np.isnan(v) else "   nan"
        print(f"{assay:10s}  {best_blk:8s}  {f(e1)}  {f(e2)}  {f(e3)}  "
              f"{'bio_mask':8s}  {f(e4)}  {best_e:6s}  {ndim:7d}")

        rows.append({"assay": assay, "best_blk": best_blk,
                     "E1": round(e1,3) if not np.isnan(e1) else None,
                     "E2": round(e2,3) if not np.isnan(e2) else None,
                     "E3": round(e3,3) if not np.isnan(e3) else None,
                     "E4": round(e4,3) if not np.isnan(e4) else None,
                     "best": best_e, "e4_ndim": ndim})

    print("-"*80)
    df = pd.DataFrame(rows)
    for col in ["E1","E2","E3","E4"]:
        print(f"  Mean {col}: {df[col].mean():.3f}")

    out = ROOT / "results/e2_e4_ablation.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
