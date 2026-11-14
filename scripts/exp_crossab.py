"""Exp 1: CrossAbSense-style frozen-encoder + cross-attention decoder.

Per-residue ESM2-150M embeddings (frozen, cached in esm_perres_150M.npz) are used to build:
  (a) mean-pooled VH / VL baselines
  (b) CDR-only pooled features
  (c) cross-attention context: for each VH-CDR3 residue, attend over VL-CDR residues (no learned weights)
  (d) concatenated [VH_mean, VL_mean, VH_CDR_mean, cross_ctx] -> PCA -> RidgeCV

Evaluated on random / cluster / cluster_isotype CV + GDPa3 heldout.
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE

ASSAYS = GE.ASSAYS
SPLITS = ["random", "cluster", "cluster_isotype"]


def _pca_ridge():
    return Pipeline([
        ("s", StandardScaler()),
        ("p", PCA(0.9, random_state=0, svd_solver="full")),
        ("r", RidgeCV(alphas=np.logspace(-2, 4, 12))),
    ])


def build_crossab_features(pr_npz_path: Path, n1: int = 246):
    """Build cross-attention features from per-residue ESM2-150M embeddings.

    Returns dict with keys for each feature variant, shape (n1, d).
    """
    d = np.load(pr_npz_path, allow_pickle=True)
    vh_e = d["vh_e"].astype(np.float32)   # (246, 140, 640)
    vh_m = d["vh_m"].astype(bool)          # (246, 140) - real residue mask
    vh_c = d["vh_c"].astype(bool)          # (246, 140) - CDR indicator
    vl_e = d["vl_e"].astype(np.float32)
    vl_m = d["vl_m"].astype(bool)
    vl_c = d["vl_c"].astype(bool)

    def safe_mean(emb, mask):
        mask = mask.astype(np.float32)
        s = (emb * mask[:, :, None]).sum(1)
        c = mask.sum(1, keepdims=True).clip(min=1)
        return s / c   # (n, 640)

    vh_mean = safe_mean(vh_e, vh_m)                        # (n, 640)
    vl_mean = safe_mean(vl_e, vl_m)                        # (n, 640)
    vh_cdr_mean = safe_mean(vh_e, vh_m & vh_c)             # (n, 640)
    vl_cdr_mean = safe_mean(vl_e, vl_m & vl_c)             # (n, 640)

    # Cross-attention: VH-CDR3 residues attend over VL-CDR residues (frozen, no learned weights)
    # CDR-H3 = positions 97-108 roughly - already in vh_c; use same mask
    # For each antibody: A = softmax(VH_CDR_residues @ VL_CDR_residues.T / sqrt(d)) -> mean over VH dim
    n = vh_e.shape[0]
    d_model = vh_e.shape[2]
    cross_ctx = np.zeros((n, d_model), dtype=np.float32)
    for i in range(n):
        vh_cdr_res = vh_e[i, vh_m[i] & vh_c[i]]   # (nVH_cdr, 640)
        vl_cdr_res = vl_e[i, vl_m[i] & vl_c[i]]   # (nVL_cdr, 640)
        if len(vh_cdr_res) == 0 or len(vl_cdr_res) == 0:
            continue
        # Attention scores (nVH_cdr, nVL_cdr)
        scores = vh_cdr_res @ vl_cdr_res.T / np.sqrt(d_model)
        scores -= scores.max(1, keepdims=True)
        attn = np.exp(scores) / np.exp(scores).sum(1, keepdims=True)  # (nVH_cdr, nVL_cdr)
        ctx_per_vh = attn @ vl_cdr_res                                  # (nVH_cdr, 640)
        cross_ctx[i] = ctx_per_vh.mean(0)

    # Feature variants
    feat_vh = vh_mean                                       # 640
    feat_vl = vl_mean                                       # 640
    feat_concat = np.concatenate([vh_mean, vl_mean], 1)    # 1280
    feat_cdr = np.concatenate([vh_cdr_mean, vl_cdr_mean], 1)  # 1280
    feat_cross = cross_ctx                                  # 640
    feat_full = np.concatenate([vh_mean, vl_mean, vh_cdr_mean, cross_ctx], 1)  # 2560

    return {
        "VH_mean": feat_vh,
        "VL_mean": feat_vl,
        "VH+VL_mean": feat_concat,
        "CDR_mean": feat_cdr,
        "CrossAttn_ctx": feat_cross,
        "CrossAbSense": feat_full,
    }


def eval_cv(feats, ds, split_name, assay):
    """Return mean Spearman over folds for a feature matrix."""
    folds = ds.folds(split_name)
    ufolds = sorted(set(f for f in folds.tolist() if f == f))
    y = ds.y(assay)
    lab = ~np.isnan(y)
    scores = []
    for f in ufolds:
        tr = np.where((folds != f) & lab)[0]
        te = np.where((folds == f) & lab)[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        preds = _pca_ridge().fit(feats[tr], y[tr]).predict(feats[te])
        scores.append(spearmanr(y[te], preds).statistic)
    return float(np.nanmean(scores)) if scores else np.nan


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading data ...")
    ds, g3, B1, B3 = GE.build_blocks()

    pr_path = ROOT / "data/processed/esm_perres_150M.npz"
    print("Building cross-attention features ...")
    feats_train = build_crossab_features(pr_path, n1=246)

    # Build held-out versions for GDPa3: use ESM2-150M mean pools for g3
    # GDPa3 doesn't have per-residue embeddings cached; use ablang2 as proxy for heldout
    # For cross-attention, we can only evaluate on GDPa1 (we have per-res for train only)

    rows = []
    for feat_name, X1 in feats_train.items():
        X1 = np.nan_to_num(X1)
        for assay in ASSAYS:
            y1 = ds.y(assay)
            lab = ~np.isnan(y1)
            row = {"model": feat_name, "assay": assay}
            for sp in SPLITS:
                row[f"cv_{sp}"] = round(eval_cv(X1, ds, sp, assay), 3)
            # Heldout: can only train on GDPa1 and predict using B3["ablang2"] for non-per-res variants
            # For the CrossAbSense feature itself, report "N/A" for GDPa3 (no per-res for g3)
            row["gdpa3"] = None
            rows.append(row)

    # Ablang2 baseline (equivalent single block from gdpa3_final)
    for assay in ASSAYS:
        y1 = ds.y(assay)
        lab = ~np.isnan(y1)
        X1 = np.nan_to_num(B1["ablang2"])
        X3 = np.nan_to_num(B3["ablang2"])
        y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        ok = ~np.isnan(y3)
        row = {"model": "AbLang2_baseline", "assay": assay}
        for sp in SPLITS:
            row[f"cv_{sp}"] = round(eval_cv(X1, ds, sp, assay), 3)
        pred3 = _pca_ridge().fit(X1[lab], y1[lab]).predict(X3)
        row["gdpa3"] = round(float(spearmanr(y3[ok], pred3[ok]).statistic), 3)
        rows.append(row)

    # ESM2-650M baseline
    for assay in ASSAYS:
        y1 = ds.y(assay)
        lab = ~np.isnan(y1)
        X1 = np.nan_to_num(B1["esm2"])
        X3 = np.nan_to_num(B3["esm2"])
        y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        ok = ~np.isnan(y3)
        row = {"model": "ESM2_650M_baseline", "assay": assay}
        for sp in SPLITS:
            row[f"cv_{sp}"] = round(eval_cv(X1, ds, sp, assay), 3)
        pred3 = _pca_ridge().fit(X1[lab], y1[lab]).predict(X3)
        row["gdpa3"] = round(float(spearmanr(y3[ok], pred3[ok]).statistic), 3)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Pivot to wide: mean over assays
    summary_cols = [f"cv_{sp}" for sp in SPLITS] + ["gdpa3"]
    summary = (df.groupby("model")[summary_cols]
                 .mean().round(3)
                 .sort_values("cv_cluster_isotype", ascending=False))
    print("\n=== Exp 1: CrossAbSense-style encoder comparison (mean over assays) ===")
    print(summary.to_string())
    out_path = ROOT / "results/exp_crossab.csv"
    df.to_csv(out_path, index=False)
    summary.to_csv(ROOT / "results/exp_crossab_summary.csv")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
