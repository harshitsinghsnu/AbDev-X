"""Exp 2: Systematic frozen encoder screen.

Tests all cached encoders:
  - ESM2-650M (2560-dim)
  - ESM2-150M-pooled (640-dim, mean-pool from per-residue)
  - AbLang2 (480-dim)
  - AntiBERTy (1024-dim)
  - ProGen2 (2048-dim)
  - ProtBert (2048-dim)
  - IgBERT (2048-dim)
  - Descriptor-based (baseline)

Reports per-assay cluster_isotype CV + GDPa3 heldout Spearman.
Also reports whether block selection changes vs existing AbDev-X selection.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE

ASSAYS = GE.ASSAYS
SPLITS = ["random", "cluster", "cluster_isotype"]


def _head(n_features, high_dim=False):
    steps = [("s", StandardScaler())]
    if high_dim:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def pool_perres(npz_path: Path) -> np.ndarray:
    """Mean-pool per-residue ESM2-150M embeddings using real-residue mask."""
    d = np.load(npz_path, allow_pickle=True)
    vh_e = d["vh_e"].astype(np.float32)
    vh_m = d["vh_m"].astype(np.float32)
    vl_e = d["vl_e"].astype(np.float32)
    vl_m = d["vl_m"].astype(np.float32)
    def mpool(e, m):
        return (e * m[:, :, None]).sum(1) / m.sum(1, keepdims=True).clip(min=1)
    vh_pooled = mpool(vh_e, vh_m)   # (n, 640)
    vl_pooled = mpool(vl_e, vl_m)   # (n, 640)
    return np.concatenate([vh_pooled, vl_pooled], axis=1)   # (n, 1280)


def eval_cv(X, ds, split_name, assay, high_dim=True):
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
        pipe = _head(X.shape[1], high_dim=high_dim)
        preds = pipe.fit(X[tr], y[tr]).predict(X[te])
        scores.append(spearmanr(y[te], preds).statistic)
    return float(np.nanmean(scores)) if scores else np.nan


def load_encoders(ds):
    """Load all available encoder embeddings for GDPa1."""
    enc = {}

    # ESM2-650M (pre-existing in B1)
    esm_files = list((ROOT / "data/processed").glob("esm2_*.npz"))
    for ep in esm_files:
        emb = np.load(ep)["emb"]
        if emb.shape == (246, 2560):
            enc["ESM2_650M"] = np.nan_to_num(emb)
            break

    # ESM2-150M pooled from per-residue
    pr_path = ROOT / "data/processed/esm_perres_150M.npz"
    if pr_path.exists():
        enc["ESM2_150M_pooled"] = np.nan_to_num(pool_perres(pr_path))

    # AbLang2
    p = ROOT / "data/processed/ablang2_gdpa1.npz"
    if p.exists():
        enc["AbLang2"] = np.nan_to_num(np.load(p)["emb"])

    # AntiBERTy
    p = ROOT / "data/processed/antiberty_gdpa1.npz"
    if p.exists():
        enc["AntiBERTy"] = np.nan_to_num(np.load(p)["emb"])

    # ProGen2
    p = ROOT / "data/processed/progen2_gdpa1.npz"
    if p.exists():
        enc["ProGen2"] = np.nan_to_num(np.load(p)["emb"])

    # ProtBert
    p = ROOT / "data/processed/protbert_gdpa1.npz"
    if p.exists():
        enc["ProtBert"] = np.nan_to_num(np.load(p)["emb"])

    # IgBERT
    p = ROOT / "data/processed/igbert_gdpa1.npz"
    if p.exists():
        enc["IgBERT"] = np.nan_to_num(np.load(p)["emb"])

    return enc


def load_encoders_g3():
    """Load all available encoder embeddings for GDPa3 heldout."""
    enc3 = {}

    # ESM2-650M
    p = ROOT / "data/processed/gdpa3_esm.npz"
    if p.exists():
        enc3["ESM2_650M"] = np.nan_to_num(np.load(p)["emb"])

    # AbLang2
    p = ROOT / "data/processed/gdpa3_ablang2.npz"
    if p.exists():
        enc3["AbLang2"] = np.nan_to_num(np.load(p)["emb"])

    # AntiBERTy
    p = ROOT / "data/processed/antiberty_gdpa3.npz"
    if p.exists():
        enc3["AntiBERTy"] = np.nan_to_num(np.load(p)["emb"])

    # IgBERT
    p = ROOT / "data/processed/igbert_gdpa3.npz"
    if p.exists():
        enc3["IgBERT"] = np.nan_to_num(np.load(p)["emb"])

    # No GDPa3 versions available for: ESM2_150M_pooled, ProGen2, ProtBert
    return enc3


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading data ...")
    ds, g3, B1, B3 = GE.build_blocks()

    print("Loading encoders ...")
    enc1 = load_encoders(ds)
    enc3 = load_encoders_g3()
    # Also add descriptor block
    enc1["Descriptors"] = np.nan_to_num(B1["descriptors"])
    enc1["Isotype"] = np.nan_to_num(B1["isotype"])

    print(f"Encoders loaded: {list(enc1.keys())}")
    for k, v in enc1.items():
        print(f"  {k}: {v.shape}")

    rows = []
    best_per_assay = {a: {"cv_score": -9, "encoder": None} for a in ASSAYS}

    for enc_name, X1 in enc1.items():
        high_dim = X1.shape[1] >= 480
        print(f"  Evaluating {enc_name} ...")
        X3 = enc3.get(enc_name)

        for assay in ASSAYS:
            y1 = ds.y(assay)
            lab = ~np.isnan(y1)
            row = {"encoder": enc_name, "assay": assay, "dim": X1.shape[1]}

            for sp in SPLITS:
                cv_s = eval_cv(X1, ds, sp, assay, high_dim=high_dim)
                row[f"cv_{sp}"] = round(cv_s, 3)

            # GDPa3 heldout
            if X3 is not None:
                y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
                ok = ~np.isnan(y3)
                pipe = _head(X1.shape[1], high_dim=high_dim)
                pred3 = pipe.fit(X1[lab], y1[lab]).predict(X3)
                row["gdpa3"] = round(float(spearmanr(y3[ok], pred3[ok]).statistic), 3)
            else:
                row["gdpa3"] = None

            rows.append(row)

            # Track best by cluster_isotype CV
            if row["cv_cluster_isotype"] > best_per_assay[assay]["cv_score"]:
                best_per_assay[assay] = {"cv_score": row["cv_cluster_isotype"], "encoder": enc_name}

    df = pd.DataFrame(rows)

    print("\n=== Exp 2: Encoder Screen - cluster_isotype CV (per assay) ===")
    pivot = df.pivot_table(index="encoder", columns="assay", values="cv_cluster_isotype", aggfunc="mean")
    pivot["MEAN"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("MEAN", ascending=False)
    print(pivot.round(3).to_string())

    print("\n=== GDPa3 Heldout (where available) ===")
    pivot3 = df.pivot_table(index="encoder", columns="assay", values="gdpa3", aggfunc="mean")
    if not pivot3.empty:
        pivot3["MEAN"] = pivot3.mean(axis=1)
        print(pivot3.round(3).to_string())

    print("\n=== Best encoder per assay (cluster_isotype CV) ===")
    for assay, info in best_per_assay.items():
        print(f"  {assay}: {info['encoder']} ({info['cv_score']:.3f})")

    out_path = ROOT / "results/exp_encoder_screen.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
