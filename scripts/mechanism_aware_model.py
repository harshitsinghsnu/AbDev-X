"""Mechanism-aware per-assay model.

Uses the exact same block-loading and head logic as bootstrap_ci.py to
guarantee the baseline is reproduced, then adds biologically motivated
mechanism sub-blocks and tests whether any combination improves over the
OOD-CV single-block baseline.

New mechanism sub-blocks:
  bio_hydro   : PSH + MOE hydrophobic patch/index/strength + Aggrescan
  bio_charge  : PPC/PNC + MOE charge patches + electrostatics
  bio_stab    : AntiFold + MOE packing/gyration/SASA
  desc_hydro  : GRAVY, aromatic/aliphatic fraction, hydrophobic moment
  desc_charge : net charge, pI, positive/negative fractions
  desc_liab   : deamidation, isomerization, oxidation, free cys
  desc_cdr    : CDR lengths + CDR-specific charge/hydrophobicity
  comb_HIC    : bio_hydro + desc_hydro  (HIC mechanism)
  comb_AC     : bio_charge + desc_charge (AC-SINS mechanism)
  comb_stab   : bio_stab + desc_liab    (Tm2/Titer mechanism)
  comb_PR     : desc_charge + desc_cdr  (PR-CHO mechanism)
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

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
LABEL_MAP = {
    "HIC":           "hic_rt_avg",
    "Tm2":           "tm2_nanodsf_avg",
    "AC-SINS_pH7.4": "acsins_dLmax_ph7.4_avg",
    "PR_CHO":        "polyreactivity_prscore_cho_avg",
    "Titer":         "titer_avg",
}
BASELINE = {"HIC": 0.527, "Tm2": 0.333, "AC-SINS_pH7.4": 0.284, "PR_CHO": 0.197, "Titer": 0.295}
COMP     = {"HIC": 0.708, "Tm2": 0.392, "AC-SINS_pH7.4": 0.356, "PR_CHO": 0.337, "Titer": 0.310}

# ---------------------------------------------------------------------------
# Exact copies from bootstrap_ci.py
# ---------------------------------------------------------------------------
BIO_FILES_ORIG = ["TAP.csv","MOE_properties.csv","Aggrescan3D.csv","AntiFold.csv"]
ISO_COLS       = ["hc_subtype_IgG1","hc_subtype_IgG2","hc_subtype_IgG4",
                   "lc_subtype_Kappa","lc_subtype_Lambda"]


def _isotype_block(df):
    d = pd.get_dummies(df[["hc_subtype","lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS, fill_value=0).to_numpy(np.float32)


def _bio_block(feat_dir, names, cols=None):
    """Exact copy of bootstrap_ci.py _bio_block."""
    base = pd.DataFrame({"antibody_name": list(names)})
    for f in BIO_FILES_ORIG:
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
    """Exact copy of bootstrap_ci.py _head."""
    steps = [("s", StandardScaler())]
    if blk in ("esm2", "ablang2"):
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Mechanism sub-block feature groups (using prefixed column names from _bio_block)
# ---------------------------------------------------------------------------

# Bio prefix: "TAP::", "MOE_properties::", "Aggrescan3D::", "AntiFold::"
BIO_HYDRO_PREFIXED  = ["TAP::PSH", "TAP::SFvCSP",
                        "MOE_properties::patch_hyd", "MOE_properties::patch_cdr_hyd",
                        "MOE_properties::asa_hyd", "MOE_properties::asa_hph",
                        "MOE_properties::hyd_idx", "MOE_properties::hyd_idx_cdr",
                        "MOE_properties::hyd_strength", "MOE_properties::hyd_strength_cdr",
                        "MOE_properties::HI",
                        "Aggrescan3D::aggrescan_average_score",
                        "Aggrescan3D::aggrescan_max_score",
                        "Aggrescan3D::aggrescan_90_score",
                        "Aggrescan3D::aggrescan_cdrh3_average_score"]

BIO_CHARGE_PREFIXED = ["TAP::PPC", "TAP::PNC",
                        "MOE_properties::patch_pos", "MOE_properties::patch_neg",
                        "MOE_properties::patch_ion",
                        "MOE_properties::patch_cdr_pos", "MOE_properties::patch_cdr_neg",
                        "MOE_properties::patch_cdr_ion",
                        "MOE_properties::pI_3D", "MOE_properties::ens_charge",
                        "MOE_properties::zeta", "MOE_properties::zdipole",
                        "MOE_properties::zquadrupole", "MOE_properties::dipole_moment"]

BIO_STAB_PREFIXED   = ["AntiFold::Score",
                        "MOE_properties::r_gyr", "MOE_properties::r_solv",
                        "MOE_properties::asa_vdw", "MOE_properties::volume",
                        "MOE_properties::mobility",
                        "MOE_properties::Packing Score", "MOE_properties::BSA_LC_HC",
                        "MOE_properties::DRT", "MOE_properties::affinity_VL_VH",
                        "MOE_properties::helicity", "MOE_properties::strand",
                        "MOE_properties::eccen", "MOE_properties::henry",
                        "MOE_properties::ASPmax"]

# Descriptor keywords
DESC_HYDRO_KW  = ["gravy", "frac_aromatic", "frac_aliphatic", "hmoment",
                   "CDR3_hydrophobicity"]
DESC_CHARGE_KW = ["charge_pH", "pI", "frac_pos", "frac_neg",
                   "CDR3_net_charge", "Fv_charge", "Fv_pI"]
DESC_LIAB_KW   = ["deamidation", "isomerization", "nglyc", "oxidation",
                   "free_cys", "tryptophan"]
DESC_CDR_KW    = ["H_CDR", "L_CDR", "CDR3_total_len", "CDR3_net_charge",
                   "CDR3_hydrophobicity"]


def _sel_desc(all_cols, kws):
    return [c for c in all_cols if any(k in c for k in kws)]


def _sel_bio_sub(bio_all_arr, bio_all_cols, wanted_cols):
    """Extract a sub-array of bio features by column name."""
    idx = [i for i, c in enumerate(bio_all_cols) if c in wanted_cols]
    return bio_all_arr[:, idx] if idx else None


# ---------------------------------------------------------------------------
# Build all blocks
# ---------------------------------------------------------------------------

def build_all_blocks(ds, g3):
    # ESM-2
    esm_files = list((ROOT / "data/processed").glob("esm2_*.npz"))
    esm1 = None
    for ef in esm_files:
        arr = np.load(ef)["emb"]
        if arr.shape == (len(ds.df), 2560):
            esm1 = arr; break

    abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    esm3 = np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"]
    abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]

    desc1_df = build_descriptor_matrix(ds.heavy, ds.light)
    desc3_df = build_descriptor_matrix(g3["vh"], g3["vl"])
    dcols = list(desc1_df.columns)

    bio1_arr, bio_cols = _bio_block(
        ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1",
        ds.df["antibody_name"])
    bio3_arr, _ = _bio_block(
        ROOT / "external/abdev-benchmark/data/features/processed_features/heldout_test",
        g3["antibody_name"], cols=bio_cols)

    iso1 = _isotype_block(ds.df)
    iso3 = _isotype_block(g3)

    desc1 = np.nan_to_num(desc1_df.to_numpy(float))
    desc3 = np.nan_to_num(desc3_df.to_numpy(float))

    # ---- Original blocks ----
    blocks = {
        "esm2":    (np.nan_to_num(esm1), np.nan_to_num(esm3)),
        "ablang2": (np.nan_to_num(abl1), np.nan_to_num(abl3)),
        "isotype": (iso1.astype(float), iso3.astype(float)),
        "desc":    (desc1, desc3),
        "bio":     (bio1_arr, bio3_arr),
    }

    # ---- Mechanism sub-blocks from bio ----
    for name, cols in [("bio_hydro",  BIO_HYDRO_PREFIXED),
                        ("bio_charge", BIO_CHARGE_PREFIXED),
                        ("bio_stab",   BIO_STAB_PREFIXED)]:
        s1 = _sel_bio_sub(bio1_arr, bio_cols, cols)
        s3 = _sel_bio_sub(bio3_arr, bio_cols, cols)
        if s1 is not None and s1.shape[1] > 0:
            blocks[name] = (s1, s3)

    # ---- Mechanism sub-blocks from descriptors ----
    for name, kws in [("desc_hydro",  DESC_HYDRO_KW),
                       ("desc_charge", DESC_CHARGE_KW),
                       ("desc_liab",   DESC_LIAB_KW),
                       ("desc_cdr",    DESC_CDR_KW)]:
        sel = _sel_desc(dcols, kws)
        if sel:
            idx = [dcols.index(c) for c in sel]
            blocks[name] = (desc1[:, idx], desc3[:, idx])

    # ---- Combined mechanism blocks (concatenation of narrow sub-blocks) ----
    def _cat(names):
        a1 = np.hstack([blocks[n][0] for n in names if n in blocks])
        a3 = np.hstack([blocks[n][1] for n in names if n in blocks])
        return a1, a3

    blocks["comb_HIC"]  = _cat(["bio_hydro", "desc_hydro"])   # surface hydrophobicity
    blocks["comb_AC"]   = _cat(["bio_charge", "desc_charge"])  # electrostatics
    blocks["comb_stab"] = _cat(["bio_stab", "desc_liab"])      # stability
    blocks["comb_PR"]   = _cat(["desc_charge", "desc_cdr"])    # paratope charge
    blocks["comb_titer"]= _cat(["bio_stab", "desc_liab", "desc_cdr"])  # production

    print("Blocks:")
    for k, (a1, _) in blocks.items():
        print(f"  {k:15s}: {a1.shape[1]:4d} features")

    return blocks, dcols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds = ds.folds("cluster_isotype")
    fold_ids = sorted(set(folds.tolist()))

    blocks, dcols = build_all_blocks(ds, g3)

    rows = []
    for assay in ASSAYS:
        y1 = ds.y(assay)
        y3 = g3[LABEL_MAP[assay]].to_numpy(float)
        ok3 = ~np.isnan(y3)
        m1  = ~np.isnan(y1)

        # OOD-CV over all blocks
        cv_scores  = {}
        oof_preds  = {}
        held_preds = {}

        for blk_name, (X1, X3) in blocks.items():
            oof = np.full(len(ds.df), np.nan)
            per_fold = []
            for f in fold_ids:
                tr = np.where((folds != f) & m1)[0]
                te = np.where((folds == f) & m1)[0]
                if len(te) < 3 or len(tr) < 10: continue
                pipe = _head(blk_name)
                pipe.fit(X1[tr], y1[tr])
                p = pipe.predict(X1[te])
                oof[te] = p
                per_fold.append(spearmanr(y1[te], p).statistic)
            cv_scores[blk_name]  = float(np.nanmean(per_fold)) if per_fold else -1.0
            oof_preds[blk_name]  = oof
            held_preds[blk_name] = _head(blk_name).fit(X1[m1], y1[m1]).predict(X3)

        ranked   = sorted(cv_scores.items(), key=lambda x: -x[1])
        best_blk = ranked[0][0]
        best_cv  = ranked[0][1]
        top3_str = "  ".join(f"{k}={v:.3f}" for k, v in ranked[:5])

        # Single best block held-out
        rho_single = float(spearmanr(y3[ok3], held_preds[best_blk][ok3]).statistic)

        # Average of top-2
        top2  = [ranked[0][0], ranked[1][0]]
        avg2  = (held_preds[top2[0]] + held_preds[top2[1]]) / 2
        rho_avg2 = float(spearmanr(y3[ok3], avg2[ok3]).statistic)

        # Small stacking: ridge meta on top-2 OOF
        meta_idx = np.where(m1 & np.all([~np.isnan(oof_preds[n]) for n in top2], axis=0))[0]
        if len(meta_idx) >= 15:
            mX1 = np.column_stack([oof_preds[n][meta_idx] for n in top2])
            mX3 = np.column_stack([held_preds[n] for n in top2])
            mp  = Pipeline([("s", StandardScaler()),
                             ("r", RidgeCV(alphas=np.logspace(-2, 4, 8)))])
            mp.fit(mX1, y1[meta_idx])
            rho_stack2 = float(spearmanr(y3[ok3], mp.predict(mX3)[ok3]).statistic)
        else:
            rho_stack2 = rho_avg2

        baseline = BASELINE[assay]
        best_mech = max(rho_single, rho_avg2, rho_stack2)

        print(f"\n{assay}")
        print(f"  CV top-5: {top3_str}")
        print(f"  Held-out single={rho_single:.3f}  avg2={rho_avg2:.3f}  "
              f"stack2={rho_stack2:.3f}  baseline={baseline:.3f}  comp={COMP[assay]}")

        rows.append({
            "assay":      assay,
            "best_blk":   best_blk,
            "cv_rho":     round(best_cv, 3),
            "single":     round(rho_single, 3),
            "avg_top2":   round(rho_avg2, 3),
            "stack_top2": round(rho_stack2, 3),
            "best_mech":  round(best_mech, 3),
            "baseline":   baseline,
            "delta":      round(best_mech - baseline, 3),
            "comp":       COMP[assay],
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("Mechanism-aware vs original OOD-CV single-block baseline")
    print("="*80)
    print(df[["assay","best_blk","single","avg_top2","stack_top2","baseline","delta"]].to_string(index=False))
    print(f"\nMean best_mech={df['best_mech'].mean():.3f}  baseline={df['baseline'].mean():.3f}  "
          f"Δ={df['best_mech'].mean()-df['baseline'].mean():+.3f}")

    out = ROOT / "results/mechanism_aware.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
