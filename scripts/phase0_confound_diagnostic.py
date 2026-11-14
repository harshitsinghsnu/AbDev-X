"""Phase 0: Confound diagnostic for all 9 assays.

For every assay × every block, computes:
  1. Overall Spearman rho
  2. Within-isotype rho (IgG1, IgG2, IgG4 separately)
  3. Isotype-only baseline (just one-hot isotype)
  4. Performance after removing isotype from the block (desc/bio only)
  5. IgG1-only evaluation

Central question: does a selected block retain signal after controlling for isotype?
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


def _isotype_block(df):
    d = pd.get_dummies(df[["hc_subtype", "lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS, fill_value=0).to_numpy(float)


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


def _head(dim):
    steps = [("s", StandardScaler())]
    if dim > 100:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def _rho(y_true, y_pred, mask=None):
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    ok = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if ok.sum() < 5:
        return np.nan, ok.sum()
    r = spearmanr(y_true[ok], y_pred[ok]).statistic
    return float(r), int(ok.sum())


def loocv_predict(X, y, mask):
    """Leave-one-out predictions within mask."""
    idx = np.where(mask)[0]
    preds = np.full(len(y), np.nan)
    for i in idx:
        tr = [j for j in idx if j != i]
        if len(tr) < 5: continue
        pipe = _head(X.shape[1])
        pipe.fit(X[tr], y[tr])
        preds[i] = pipe.predict(X[[i]])[0]
    return preds


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds = ds.folds("cluster_isotype")
    fold_ids = sorted(set(folds.tolist()))

    # Build feature blocks once
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
    bio1, bio_cols = _bio_block(
        ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1",
        ds.df["antibody_name"])
    bio3, _ = _bio_block(
        ROOT / "external/abdev-benchmark/data/features/processed_features/heldout_test",
        g3["antibody_name"], cols=bio_cols)
    desc3 = np.nan_to_num(build_descriptor_matrix(g3["vh"], g3["vl"]).to_numpy(float))

    iso1 = _isotype_block(ds.df)
    iso3 = _isotype_block(g3)

    blocks_tr = {
        "esm2": np.nan_to_num(esm1),
        "ablang2": np.nan_to_num(abl1),
        "isotype": iso1,
        "desc": desc1,
        "bio": bio1,
    }
    blocks_te = {
        "esm2": np.nan_to_num(esm3),
        "ablang2": np.nan_to_num(abl3),
        "isotype": iso3,
        "desc": desc3,
        "bio": bio3,
    }

    # Isotype labels for stratification
    iso_labels = ds.df["hc_subtype"].values  # IgG1, IgG2, IgG4

    rows = []
    print("\n" + "="*100)
    print(f"{'Assay':10s}  {'Block':8s}  {'Overall':8s}  "
          f"{'IgG1':7s}(n)  {'IgG2':7s}(n)  {'IgG4':7s}(n)  "
          f"{'IgG1-only CV':12s}  {'ISO-baseline':12s}  Flag")
    print("="*100)

    for assay, (g1_col, g3_col) in ASSAY_MAP.items():
        y1 = ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df), np.nan)
        y3 = g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(len(g3), np.nan)
        m1 = ~np.isnan(y1)

        # OOD-CV to select best block
        cv_scores = {}
        for blk, X1 in blocks_tr.items():
            per_fold = []
            for f in fold_ids:
                tr = np.where((folds != f) & m1)[0]
                te = np.where((folds == f) & m1)[0]
                if len(te) < 3 or len(tr) < 10: continue
                pipe = _head(X1.shape[1])
                pipe.fit(X1[tr], y1[tr])
                p = pipe.predict(X1[te])
                per_fold.append(spearmanr(y1[te], p).statistic)
            cv_scores[blk] = float(np.nanmean(per_fold)) if per_fold else -1.0

        best_blk = max(cv_scores, key=lambda k: cv_scores[k])
        X_best = blocks_tr[best_blk]
        X_best_te = blocks_te[best_blk]

        # Overall held-out rho
        pipe_all = _head(X_best.shape[1])
        pipe_all.fit(X_best[m1], y1[m1])
        pred3 = pipe_all.predict(X_best_te)
        ok3 = ~np.isnan(y3)
        rho_overall, n_te = _rho(y3, pred3, ok3)

        # Isotype-only baseline on held-out
        pipe_iso = _head(iso1.shape[1])
        pipe_iso.fit(iso1[m1], y1[m1])
        pred_iso3 = pipe_iso.predict(iso3)
        rho_iso_base, _ = _rho(y3, pred_iso3, ok3)

        # Within-isotype LOO-CV rho on training set
        iso_results = {}
        for ig in ["IgG1", "IgG2", "IgG4"]:
            iso_mask = (iso_labels == ig) & m1
            if iso_mask.sum() < 8:
                iso_results[ig] = (np.nan, iso_mask.sum())
                continue
            preds_loo = loocv_predict(X_best, y1, iso_mask)
            r, n = _rho(y1, preds_loo, iso_mask)
            iso_results[ig] = (r, iso_mask.sum())

        # IgG1-only evaluation: train on IgG1, predict IgG1 portion of GDPa3
        ig1_tr_mask = (iso_labels == "IgG1") & m1
        ig1_te_mask = (g3["hc_subtype"] == "IgG1").to_numpy() & ok3
        rho_ig1_only = np.nan
        if ig1_tr_mask.sum() >= 10 and ig1_te_mask.sum() >= 5:
            pipe_ig1 = _head(X_best.shape[1])
            pipe_ig1.fit(X_best[ig1_tr_mask], y1[ig1_tr_mask])
            pred_ig1 = pipe_ig1.predict(X_best_te)
            rho_ig1_only, _ = _rho(y3, pred_ig1, ig1_te_mask)

        # Confound flag: within-isotype rho is much lower than overall OOD-CV
        ig1_r = iso_results["IgG1"][0]
        ig4_r = iso_results["IgG4"][0]
        flag = ""
        if best_blk == "isotype":
            if (not np.isnan(ig1_r) and ig1_r < 0.1) or (not np.isnan(ig4_r) and ig4_r < 0.1):
                flag = "⚠ CONFOUND"
            elif abs(rho_iso_base - rho_overall) < 0.05:
                flag = "⚠ ISO-DRIVEN"

        def fmt(r, n): return f"{r:+.3f}({n:d})" if not np.isnan(r) else f"  nan ({n:d})"

        print(f"{assay:10s}  {best_blk:8s}  {rho_overall:+8.3f}  "
              f"{fmt(*iso_results['IgG1']):12s}  "
              f"{fmt(*iso_results['IgG2']):12s}  "
              f"{fmt(*iso_results['IgG4']):12s}  "
              f"{rho_ig1_only:+12.3f}  "
              f"{rho_iso_base:+12.3f}  {flag}")

        rows.append({
            "assay": assay, "best_blk": best_blk,
            "cv_rho": round(cv_scores[best_blk], 3),
            "held_out_rho": round(rho_overall, 3),
            "iso_baseline_rho": round(rho_iso_base, 3) if not np.isnan(rho_iso_base) else None,
            "within_IgG1_loo": round(ig1_r, 3) if not np.isnan(ig1_r) else None,
            "within_IgG2_loo": round(iso_results["IgG2"][0], 3) if not np.isnan(iso_results["IgG2"][0]) else None,
            "within_IgG4_loo": round(ig4_r, 3) if not np.isnan(ig4_r) else None,
            "igG1_only_eval": round(rho_ig1_only, 3) if not np.isnan(rho_ig1_only) else None,
            "confound_flag": flag.strip(),
        })

    print("="*100)
    df = pd.DataFrame(rows)
    out = ROOT / "results/phase0_confound.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    print("\n--- Confound summary ---")
    flagged = df[df["confound_flag"] != ""]
    if len(flagged):
        for _, r in flagged.iterrows():
            print(f"  {r['assay']:10s}  {r['confound_flag']}  "
                  f"block={r['best_blk']}  iso_baseline={r['iso_baseline_rho']}  "
                  f"held_out={r['held_out_rho']}")
    else:
        print("  No confounds flagged.")


if __name__ == "__main__":
    main()
