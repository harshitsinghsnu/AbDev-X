"""Compute AW-MoE GDPa1 cross-validation ρ across all three split types.

Outer folds  : fold_iter(split_type) for random / cluster / cluster_isotype
Inner protocol: cluster_isotype folds within training set, same as aw_moe.py
Saves results to:
  results/awmoe_cv_splits.csv   – per-assay rho + mean  for each split
  results/awmoe_cv_errorbars.csv – per-assay mean ± std across outer folds

Companion to abdevx_splits.py (which runs the D-Stack stacking model).
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

# ── copy of helpers from aw_moe.py ───────────────────────────────────────────

GENUINE_ASSAYS = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "SMAC", "HAC"]
# all assays to evaluate (genuine 5 + 2 competition-only)
ALL_EVAL_ASSAYS = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "SMAC", "HAC", "Tm2", "Titer"]
ISO_INELIGIBLE  = {"Tm2", "Tm1", "Tonset", "SEC"}   # skip isotype block for these
# ds.df column names (same as assay names in GDPa1 dataset)
ASSAY_COL = {
    "HIC":          "HIC",
    "AC-SINS_pH7.4":"AC-SINS_pH7.4",
    "PR_CHO":       "PR_CHO",
    "SMAC":         "SMAC",
    "HAC":          "HAC",
    "Tm2":          "Tm2",
    "Titer":        "Titer",
}
# competition 5 assays for all_models_splits.csv compatibility
COMP_ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
COMP_COL = {
    "HIC":          "HIC",
    "Tm2":          "Tm2",
    "AC-SINS_pH7.4":"AC-SINS_pH7.4",
    "PR_CHO":       "PR_CHO",
    "Titer":        "Titer",
}

ISO_COLS  = ["hc_subtype_IgG1", "hc_subtype_IgG2", "hc_subtype_IgG4",
             "lc_subtype_Kappa", "lc_subtype_Lambda"]
BIO_FILES = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv"]

MGFS_MASKS = {
    "HIC":          {"inc": ["hydro","gravy","aliphat","aromat","patch","sasa","SASA"],
                     "exc": ["charge","pI","deamid","glyc"]},
    "AC-SINS_pH7.4":{"inc": ["charge","pI","ppc","pnc","patch_pos","patch_neg","dipole","electr"],
                     "exc": ["hydro","gravy"]},
    "PR_CHO":       {"inc": ["charge","cdr","CDR","hydro","gravy","patch"], "exc": []},
    "SMAC":         {"inc": ["sasa","SASA","patch","hydro","charge","exposed"], "exc": []},
    "HAC":          {"inc": ["sasa","SASA","patch","hydro","charge","exposed","stab"], "exc": []},
    "Tm2":          {"inc": ["stab","tm","Tm","pack","burial","sasa"], "exc": []},
    "Titer":        {"inc": ["charge","pI","cdr","CDR","hydro","sasa"], "exc": []},
}
MOE_EXPERTS = {
    "hydro":  ["hydro","gravy","aliphat","aromat","patch_hyd","hyd"],
    "charge": ["charge","pI","ppc","pnc","electr","dipole","patch_pos","patch_neg"],
    "stab":   ["stab","sasa","gyrat","pack","AntiFold","tm","Tm","cavity","burial"],
    "liab":   ["deamid","isomeriz","glyc","oxid","cys","nglyc","asn","asp"],
    "cdr":    ["cdr","CDR","loop","h3","l3","h1","h2","l1","l2"],
    "surface":["sasa","SASA","rSASA","exposed","surface","patch"],
    "global": ["len","mass","count","frac","total","net"],
}
MOE_ELIGIBILITY = {
    "HIC":          ["hydro","surface"],
    "AC-SINS_pH7.4":["charge","surface"],
    "PR_CHO":       ["charge","cdr","hydro"],
    "SMAC":         ["surface","hydro","charge"],
    "HAC":          ["surface","hydro","charge","stab"],
    "Tm2":          ["stab","global"],
    "Titer":        ["charge","cdr","hydro"],
}


def _head(X):
    steps = [("s", StandardScaler())]
    if X.shape[1] > 100:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def _mask_cols(all_cols, inc_kws, exc_kws=()):
    inc = [k.lower() for k in inc_kws]
    exc = [k.lower() for k in exc_kws]
    return np.array([(any(k in c.lower() for k in inc) and
                      not any(k in c.lower() for k in exc))
                     for c in all_cols])


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
    return np.nan_to_num(X.reindex(columns=cols, fill_value=0).to_numpy(float)), cols


def fit_aw_moe2(oof_matrix, y):
    m = ~np.isnan(y) & np.all(~np.isnan(oof_matrix), axis=1)
    if m.sum() < 10:
        return np.ones(oof_matrix.shape[1]) / oof_matrix.shape[1]
    P = oof_matrix[m]; yt = y[m]; K = P.shape[1]
    def loss(w): return -spearmanr(yt, P @ w).statistic + 0.1 * np.sum(w**2)
    w0 = np.ones(K) / K
    res = minimize(loss, w0, method="SLSQP",
                   bounds=[(0, None)] * K,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-9})
    w = np.maximum(res.x, 0); w /= w.sum()
    return w


# ── inner OOF for one outer-fold training set ─────────────────────────────────

def _inner_oof(tr_idx, te_idx, y_full,
               esm, abl, desc, bio, iso, db, all_db_cols, assay,
               inner_folds_full, iso_ok=True):
    """Return (oof_tr [len(tr_idx) x 4], te_preds [len(te_idx) x 4])."""
    n_tr = len(tr_idx)
    y_tr = y_full[tr_idx]
    m_tr = ~np.isnan(y_tr)

    inner_folds = inner_folds_full[tr_idx]
    inner_ids   = sorted(set(inner_folds.tolist()))

    oof = np.full((n_tr, 4), np.nan)
    te_preds = np.full((len(te_idx), 4), np.nan)

    # feature slices
    esm_tr = esm[tr_idx]; esm_te = esm[te_idx]
    abl_tr = abl[tr_idx]; abl_te = abl[te_idx]
    desc_tr = desc[tr_idx]; desc_te = desc[te_idx]
    bio_tr  = bio[tr_idx]; bio_te  = bio[te_idx]
    iso_tr  = iso[tr_idx]; iso_te  = iso[te_idx]
    db_tr   = db[tr_idx];  db_te   = db[te_idx]
    Xplm_tr = np.hstack([esm_tr, abl_tr]); Xplm_te = np.hstack([esm_te, abl_te])

    single_blocks = {"esm2": (esm_tr, esm_te), "ablang2": (abl_tr, abl_te),
                     "desc": (desc_tr, desc_te), "bio": (bio_tr, bio_te)}
    if iso_ok:
        single_blocks["isotype"] = (iso_tr, iso_te)

    # MGFS mask
    mgfs_cfg = MGFS_MASKS.get(assay, {"inc": [], "exc": []})
    mgfs_mask = _mask_cols(all_db_cols, mgfs_cfg["inc"], mgfs_cfg["exc"])
    if mgfs_mask.sum() < 5: mgfs_mask = np.ones(len(all_db_cols), bool)

    # F-MoE eligible experts
    eligible = MOE_ELIGIBILITY.get(assay, list(MOE_EXPERTS.keys()))
    fmoe_masks = []
    for exp in eligible:
        msk = _mask_cols(all_db_cols, MOE_EXPERTS.get(exp, []))
        if msk.sum() >= 3: fmoe_masks.append(msk)
    if not fmoe_masks: fmoe_masks = [np.ones(db.shape[1], bool)]

    # ── inner fold loop (compute OOF for each expert) ────────────────────────
    for f in inner_ids:
        ii_tr = np.where((inner_folds != f) & m_tr)[0]
        ii_te = np.where(inner_folds == f)[0]
        if len(ii_te) < 2 or len(ii_tr) < 8: continue

        # E1: best-block selection via inner-inner CV
        best_blk, best_cv = None, -9.0
        for blk, (Xb_tr, _) in single_blocks.items():
            scores = []
            ii2_folds = inner_folds[ii_tr]
            for f2 in sorted(set(ii2_folds.tolist())):
                a = np.where((ii2_folds != f2))[0]
                b = np.where((ii2_folds == f2))[0]
                if len(b) < 2 or len(a) < 6: continue
                p = _head(Xb_tr[ii_tr])
                p.fit(Xb_tr[ii_tr[a]], y_tr[ii_tr[a]])
                scores.append(spearmanr(y_tr[ii_tr[b]], p.predict(Xb_tr[ii_tr[b]])).statistic)
            if scores and np.nanmean(scores) > best_cv:
                best_cv = np.nanmean(scores); best_blk = blk
        if best_blk is None: best_blk = "desc"
        Xb_tr_e1, _ = single_blocks[best_blk]
        p_e1 = _head(Xb_tr_e1[ii_tr]); p_e1.fit(Xb_tr_e1[ii_tr], y_tr[ii_tr])
        oof[ii_te, 0] = p_e1.predict(Xb_tr_e1[ii_te])

        # E2: MGFS
        p_e2 = _head(db_tr[ii_tr][:, mgfs_mask])
        p_e2.fit(db_tr[ii_tr][:, mgfs_mask], y_tr[ii_tr])
        oof[ii_te, 1] = p_e2.predict(db_tr[ii_te][:, mgfs_mask])

        # E3: D-Stack (inner-nested ridge stacking)
        inner2_folds = inner_folds[ii_tr]
        in2_ids = sorted(set(inner2_folds.tolist()))
        ioof1 = np.zeros(len(ii_tr)); ioof2 = np.zeros(len(ii_tr))
        for f2 in in2_ids:
            a = np.where(inner2_folds != f2)[0]
            b = np.where(inner2_folds == f2)[0]
            if len(b) < 2 or len(a) < 6: continue
            pp1 = _head(Xplm_tr[ii_tr]); pp1.fit(Xplm_tr[ii_tr[a]], y_tr[ii_tr[a]]); ioof1[b] = pp1.predict(Xplm_tr[ii_tr[b]])
            pp2 = _head(db_tr[ii_tr]);   pp2.fit(db_tr[ii_tr[a]],   y_tr[ii_tr[a]]); ioof2[b] = pp2.predict(db_tr[ii_tr[b]])
        meta_in = np.column_stack([ioof1, ioof2])
        pp1f = _head(Xplm_tr[ii_tr]); pp1f.fit(Xplm_tr[ii_tr], y_tr[ii_tr])
        pp2f = _head(db_tr[ii_tr]);   pp2f.fit(db_tr[ii_tr],   y_tr[ii_tr])
        meta_te3 = np.column_stack([pp1f.predict(Xplm_tr[ii_te]),
                                    pp2f.predict(db_tr[ii_te])])
        m3 = Ridge(alpha=1.0); m3.fit(meta_in, y_tr[ii_tr])
        oof[ii_te, 2] = m3.predict(meta_te3)

        # E4: F-MoE
        exp_oofs = []; exp_cvs = []
        for msk in fmoe_masks:
            Xe_tr = db_tr[:, msk]
            inner2_folds2 = inner_folds[ii_tr]
            in2_ids2 = sorted(set(inner2_folds2.tolist()))
            e_oof = np.zeros(len(ii_tr)); e_scores = []
            for f2 in in2_ids2:
                a = np.where(inner2_folds2 != f2)[0]
                b = np.where(inner2_folds2 == f2)[0]
                if len(b) < 2 or len(a) < 6: continue
                pf = _head(Xe_tr[ii_tr]); pf.fit(Xe_tr[ii_tr[a]], y_tr[ii_tr[a]])
                e_oof[b] = pf.predict(Xe_tr[ii_tr[b]])
                e_scores.append(spearmanr(y_tr[ii_tr[b]], e_oof[b]).statistic)
            exp_oofs.append(e_oof)
            exp_cvs.append(float(np.nanmean(e_scores)) if e_scores else -1.0)
        cvs4 = np.maximum(np.array(exp_cvs), 0)
        ws4 = np.exp(cvs4 / 0.1); ws4 /= ws4.sum()
        stk_in = np.column_stack(exp_oofs)
        # fit test predictions per expert
        te_preds_fmoe = []
        for msk in fmoe_masks:
            Xe_tr = db_tr[:, msk]
            pf2 = _head(Xe_tr[ii_tr]); pf2.fit(Xe_tr[ii_tr], y_tr[ii_tr])
            te_preds_fmoe.append(pf2.predict(db_tr[ii_te][:, msk]))
        stk_te4 = np.column_stack(te_preds_fmoe)
        oof[ii_te, 3] = stk_te4 @ ws4

    # ── refit on full tr_idx → test predictions ───────────────────────────────
    if m_tr.sum() < 8:
        return oof, te_preds

    # E1: best block via full inner CV
    best_blk2, best_cv2 = None, -9.0
    for blk, (Xb_tr, _) in single_blocks.items():
        scores = []
        for f in inner_ids:
            ii_tr2 = np.where((inner_folds != f) & m_tr)[0]
            ii_te2 = np.where((inner_folds == f) & m_tr)[0]
            if len(ii_te2) < 2 or len(ii_tr2) < 6: continue
            p = _head(Xb_tr); p.fit(Xb_tr[ii_tr2], y_tr[ii_tr2])
            scores.append(spearmanr(y_tr[ii_te2], p.predict(Xb_tr[ii_te2])).statistic)
        if scores and np.nanmean(scores) > best_cv2:
            best_cv2 = np.nanmean(scores); best_blk2 = blk
    if best_blk2 is None: best_blk2 = "desc"
    Xb_tr2, Xb_te2 = single_blocks[best_blk2]
    p_e1f = _head(Xb_tr2[m_tr]); p_e1f.fit(Xb_tr2[m_tr], y_tr[m_tr])
    te_preds[:, 0] = p_e1f.predict(Xb_te2)

    # E2: MGFS
    p_e2f = _head(db_tr[m_tr][:, mgfs_mask]); p_e2f.fit(db_tr[m_tr][:, mgfs_mask], y_tr[m_tr])
    te_preds[:, 1] = p_e2f.predict(db_te[:, mgfs_mask])

    # E3: D-Stack
    oof1_full = np.zeros(n_tr); oof2_full = np.zeros(n_tr)
    for f in inner_ids:
        ii_tr2 = np.where((inner_folds != f) & m_tr)[0]
        ii_te2 = np.where(inner_folds == f)[0]
        if len(ii_te2) < 2 or len(ii_tr2) < 6: continue
        pp1 = _head(Xplm_tr); pp1.fit(Xplm_tr[ii_tr2], y_tr[ii_tr2]); oof1_full[ii_te2] = pp1.predict(Xplm_tr[ii_te2])
        pp2 = _head(db_tr);   pp2.fit(db_tr[ii_tr2],   y_tr[ii_tr2]); oof2_full[ii_te2] = pp2.predict(db_tr[ii_te2])
    meta_full = np.column_stack([oof1_full, oof2_full])
    pp1ff = _head(Xplm_tr[m_tr]); pp1ff.fit(Xplm_tr[m_tr], y_tr[m_tr])
    pp2ff = _head(db_tr[m_tr]);   pp2ff.fit(db_tr[m_tr],   y_tr[m_tr])
    m3f = Ridge(alpha=1.0); m3f.fit(meta_full[m_tr], y_tr[m_tr])
    meta_te3f = np.column_stack([pp1ff.predict(Xplm_te), pp2ff.predict(db_te)])
    te_preds[:, 2] = m3f.predict(meta_te3f)

    # E4: F-MoE
    te_preds_fmoe2 = []; cvs_full = []
    for msk in fmoe_masks:
        Xe_tr2 = db_tr[:, msk]; scores = []
        for f in inner_ids:
            ii_tr2 = np.where((inner_folds != f) & m_tr)[0]
            ii_te2 = np.where((inner_folds == f) & m_tr)[0]
            if len(ii_te2) < 2 or len(ii_tr2) < 6: continue
            pf = _head(Xe_tr2); pf.fit(Xe_tr2[ii_tr2], y_tr[ii_tr2])
            scores.append(spearmanr(y_tr[ii_te2], pf.predict(Xe_tr2[ii_te2])).statistic)
        pf2 = _head(Xe_tr2[m_tr]); pf2.fit(Xe_tr2[m_tr], y_tr[m_tr])
        te_preds_fmoe2.append(pf2.predict(db_te[:, msk]))
        cvs_full.append(float(np.nanmean(scores)) if scores else -1.0)
    cvs4f = np.maximum(np.array(cvs_full), 0)
    ws4f = np.exp(cvs4f / 0.1); ws4f /= ws4f.sum()
    te_preds[:, 3] = np.column_stack(te_preds_fmoe2) @ ws4f

    return oof, te_preds


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    names = ds.df[ds.id_col]

    # load features
    esm_files = list((ROOT / "data/processed").glob("esm2_*.npz"))
    esm = None
    for ef in esm_files:
        arr = np.load(ef)["emb"]
        if arr.shape == (len(ds.df), 2560): esm = arr; break
    abl = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]

    desc_df = build_descriptor_matrix(ds.heavy, ds.light)
    desc_cols = list(desc_df.columns)
    desc = np.nan_to_num(desc_df.to_numpy(float))
    bio, bio_cols = _bio(ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1", names)
    iso = _iso(ds.df)

    esm  = np.nan_to_num(esm)
    abl  = np.nan_to_num(abl)
    db   = np.hstack([desc, bio])
    all_db_cols = desc_cols + list(bio_cols)

    # inner folds always cluster_isotype (same as aw_moe.py training protocol)
    inner_folds_full = ds.folds("cluster_isotype")

    SPLITS = ["random", "cluster", "cluster_isotype"]
    ALL_ASSAYS = ALL_EVAL_ASSAYS   # 7 assays (5 genuine + Tm2 + Titer)
    COMP5 = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]

    rows_split = []     # for awmoe_cv_splits.csv (mean per split, pooled-rho style)
    rows_errbar = []    # for awmoe_cv_errorbars.csv (per-fold mean ± std)

    for split in SPLITS:
        print(f"\n=== Split: {split} ===", flush=True)
        outer_folds = ds.folds(split)
        outer_fold_ids = sorted(set(outer_folds.tolist()))

        # pool predictions across outer folds (same as abdevx_splits.py)
        assay_rho_pooled = {}
        # also collect per-fold rho for error bars
        assay_fold_rhos = {a: [] for a in ALL_ASSAYS}

        for assay in ALL_ASSAYS:
            iso_ok = assay not in ISO_INELIGIBLE
            col = ASSAY_COL[assay]
            y = ds.df[col].to_numpy(float) if col in ds.df.columns else np.full(len(ds.df), np.nan)
            m_labeled = ~np.isnan(y)
            pooled = np.full(len(y), np.nan)

            for f in outer_fold_ids:
                # outer fold split — use only labeled samples for training
                tr_idx = np.where((outer_folds != f) & m_labeled)[0]
                te_idx = np.where(outer_folds == f)[0]
                y_te = y[te_idx]
                if (~np.isnan(y_te)).sum() < 3 or len(tr_idx) < 10: continue

                oof, te_preds = _inner_oof(
                    tr_idx, te_idx, y,
                    esm, abl, desc, bio, iso, db, all_db_cols, assay,
                    inner_folds_full, iso_ok=iso_ok
                )
                # fit SLSQP on inner OOF
                w = fit_aw_moe2(oof, y[tr_idx])
                pred_awmoe = te_preds @ w
                pooled[te_idx] = pred_awmoe

                # per-fold rho (only labeled test samples)
                ok = ~np.isnan(y_te) & ~np.isnan(pred_awmoe)
                if ok.sum() >= 3:
                    assay_fold_rhos[assay].append(
                        float(spearmanr(y_te[ok], pred_awmoe[ok]).statistic))

            m = ~np.isnan(y) & ~np.isnan(pooled)
            rho = float(spearmanr(y[m], pooled[m]).statistic) if m.sum() >= 3 else float("nan")
            assay_rho_pooled[assay] = rho
            print(f"  {assay:14s} pooled rho = {rho:.3f}  (folds: {len(assay_fold_rhos[assay])})", flush=True)

        mean_5 = float(np.nanmean([assay_rho_pooled.get(a, float("nan")) for a in GENUINE_ASSAYS]))
        print(f"  {'MEAN (5hs)':14s}            = {mean_5:.3f}", flush=True)
        rows_split.append(dict(model="AbDev-X (AW-MoE)", split=split,
                               **{a: round(assay_rho_pooled.get(a, float("nan")), 4) for a in ALL_ASSAYS},
                               MEAN=round(mean_5, 4)))

        for assay in ALL_ASSAYS:
            folds_rho = assay_fold_rhos[assay]
            rows_errbar.append(dict(
                model="AbDev-X",
                split=split,
                assay=assay,
                mean=round(float(np.nanmean(folds_rho)), 4) if folds_rho else float("nan"),
                std=round(float(np.nanstd(folds_rho)), 4) if folds_rho else float("nan"),
                n_folds=len(folds_rho),
            ))

    df_split = pd.DataFrame(rows_split)
    df_errbar = pd.DataFrame(rows_errbar)
    df_split.to_csv(ROOT / "results/awmoe_cv_splits.csv", index=False)
    df_errbar.to_csv(ROOT / "results/awmoe_cv_errorbars.csv", index=False)
    print("\nSaved results/awmoe_cv_splits.csv")
    print("Saved results/awmoe_cv_errorbars.csv")

    # also append to all_models_splits.csv so figures pick it up
    out_all = ROOT / "results/all_models_splits.csv"
    if out_all.exists():
        existing = pd.read_csv(out_all)
        # remove stale AW-MoE rows if any
        existing = existing[~existing["model"].str.contains("AW-MoE", na=False)]
        # build rows in the same column format (COMP5 for compatibility + SMAC/HAC as extra)
        split_rows = []
        for s in SPLITS:
            row = df_split[df_split.split == s].iloc[0]
            d = {"model": "AbDev-X (AW-MoE)", "split": s}
            for a in COMP5:
                d[a] = row.get(a, float("nan"))
            split_rows.append(d)
        updated = pd.concat([existing, pd.DataFrame(split_rows)], ignore_index=True)
        updated.to_csv(out_all, index=False)
        print(f"Updated {out_all}")

    print("\n--- Summary ---")
    print(df_split.to_string(index=False))


if __name__ == "__main__":
    main()
