"""AbDev-MoE: Mixture-of-Experts model for antibody developability prediction.

Design
------
AbDev-X uses a *fixed* ridge meta-learner (same expert weights for every antibody).
AbDev-MoE replaces that with a *conditional* interaction meta-learner:

    ŷ_i  =  w₀ · P_i  +  Σ_k  P_k_i · (G_i · wk)
           = meta.predict([P_i | G_i ⊙ P₁_i | G_i ⊙ P₂_i | … | G_i ⊙ PK_i])

where P_k_i is expert k's OOF prediction for antibody i and G_i is a vector of
interpretable gate features.  The effective weight assigned to expert k depends
linearly on G_i, enabling per-antibody routing without a neural gate—just one
RidgeCV meta-model.  When gate features carry no OOD-useful signal, interaction
terms shrink to zero and the model collapses to fixed-weight stacking.

Experts (8 total)
  1. ESM2          – ridge (PCA-denoised)
  2. Descriptors   – ridge, 165 physicochemical features
  3. Isotype       – ridge, germline / constant-region one-hot
  4. BioFeatures   – ridge, TAP / MOE / Aggrescan3D / AntiFold
  5. BioFeatures   – LightGBM head (nonlinear structural interactions)
  6. AbLang2       – ridge (PCA), antibody-specific PLM, GDPa1 only
  7. SAP+Electro   – ridge, surface hydrophobicity + electrostatics specialist
  8. AbLang2+FLAb  – ridge (PCA), AbLang2 augmented with assay-matched FLAb data

Gate features (OOD-safe, biologically grounded, 13 dims total)
  isotype one-hot (5)  |  H_pI, L_pI, H_CDR3_len (3)  |  ESM2 PC1-5 (5)

Environment
-----------
  conda activate evaluation          # base pymoll env (AppData/Local/pymoll)
  cd C:/Users/hs494/ab-dev
  python scripts/moe_gdpa3.py

GPU usage
---------
  - LightGBM expert: CPU (LGBM OpenCL/GPU crashes on this Windows+A4000 setup;
    all existing project experiments also use LGBM on CPU)
  - Ridge / sklearn: CPU (no GPU path; fast enough at n~246)
  - Torch: CUDA (A4000) available for on-the-fly embedding calls; embeddings are
    pre-computed so torch is not invoked during normal runs
"""
from __future__ import annotations
import os, sys, warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

# LightGBM GPU (OpenCL) crashes on this Windows + A4000 setup (exit code 5).
# All existing project experiments use LGBM on CPU — keep consistent.
LGBM_DEVICE = "cpu"

# GPU_NAME is detected lazily inside main() AFTER numpy data loading is done.
# Importing torch + calling cuda.get_device_name() at module level initialises
# the CUDA context which conflicts with numpy's memory-mapped npz file loading
# (causes STATUS_ACCESS_VIOLATION / exit code 5).  All embeddings in this script
# are pre-computed, so torch is never actually called at runtime.
GPU_NAME = "deferred"

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE

ASSAYS    = GE.ASSAYS
LABEL_MAP = GE.LABEL_MAP
COMP      = GE.COMP_HELDOUT

# Indices in the 165-col descriptor matrix (verified from column inspection)
GATE_DESC_IDX = [2, 18, 64]   # H_pI, L_pI, H_CDR3_len


# ── Head factories ─────────────────────────────────────────────────────────────

def _ridge_pca():
    return Pipeline([("s", StandardScaler()),
                     ("p", PCA(0.9, random_state=0, svd_solver="full")),
                     ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])

def _ridge():
    return Pipeline([("s", StandardScaler()),
                     ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])

def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=7,
                             subsample=0.8, colsample_bytree=0.6, reg_lambda=5,
                             min_child_samples=12, subsample_freq=1, verbosity=-1,
                             random_state=0, device=LGBM_DEVICE)

def _block_head(blk: str):
    """Head factory keyed by block name (for single-block OOD-CV evaluation)."""
    return _ridge_pca() if blk in ("esm2", "ablang2") else _ridge()

# (name → (head_factory, block_key_in_Xb))
EXPERT_SPEC: dict[str, tuple] = {
    "esm2":        (_ridge_pca, "esm2"),
    "desc":        (_ridge,     "descriptors"),
    "isotype":     (_ridge,     "isotype"),
    "bio":         (_ridge,     "bio"),
    "bio_lgbm":    (_lgbm,      "bio"),
    "ablang2":     (_ridge_pca, "ablang2"),
    "sap_electro": (_ridge,     "sap_electro"),
}
BASE_EXPERTS = list(EXPERT_SPEC.keys())   # 7 experts; 8th added per-assay via FLAb


# ── SAP + Electro specialist block ────────────────────────────────────────────

def _load_sap_electro(names: pd.Series, suffix: str) -> np.ndarray:
    proc = ROOT / "data/processed"
    base = pd.DataFrame({"antibody_name": names.tolist()})
    for fname in (f"sap_{suffix}.csv", f"sap_patch_{suffix}.csv",
                  f"electro_{suffix}.csv"):
        p = proc / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        name_col = d.columns[0]
        num = d.select_dtypes("number").copy()
        num[name_col] = d[name_col].values
        num = num.rename(columns={c: f"{fname[:-4]}__{c}"
                                   for c in num.columns if c != name_col})
        base = base.merge(num.rename(columns={name_col: "antibody_name"}),
                          on="antibody_name", how="left")
    return np.nan_to_num(base.drop(columns=["antibody_name"]).to_numpy(float))


# ── FLAb-augmented AbLang2 data ───────────────────────────────────────────────

def _load_flab_extras(assay: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load FLAb embeddings + z-scores for a given assay (FLAb-only, not GDPa1).

    Returns (Xf, zf) where Xf is (n_flab, 480) and zf is (n_flab,), or None.
    """
    safe = assay.replace("/", "_").replace(" ", "_")
    flab_csv = ROOT / f"data/processed/flab_{safe}.csv"
    flab_npz = ROOT / "data/processed/flab_ablang2.npz"
    if not flab_csv.exists() or not flab_npz.exists():
        return None
    df       = pd.read_csv(flab_csv, encoding="latin1")
    cache    = np.load(flab_npz, allow_pickle=True)
    key2emb  = dict(zip(cache["keys"].tolist(), cache["emb"]))
    rows     = [(key2emb[k], float(z))
                for k, z in zip(df["key"].astype(str), df["z"])
                if k in key2emb]
    if not rows:
        return None
    return (np.nan_to_num(np.vstack([r[0] for r in rows])),
            np.array([r[1] for r in rows], dtype=float))


# ── Gate features ─────────────────────────────────────────────────────────────

def build_gate(B1: dict, B3: dict) -> tuple[np.ndarray, np.ndarray]:
    """OOD-safe gate features: isotype | key descriptors | ESM2 top-5 PCs.

    ESM2 PCA and descriptor scaler are fit on GDPa1 only (leakage-safe for the
    gate, which is used only in the meta-learner after OOF expert preds are generated).
    """
    iso1, iso3   = B1["isotype"], B3["isotype"]
    desc1_raw    = B1["descriptors"][:, GATE_DESC_IDX]
    desc3_raw    = B3["descriptors"][:, GATE_DESC_IDX]
    sc_d         = StandardScaler().fit(desc1_raw)
    desc1_s, desc3_s = sc_d.transform(desc1_raw), sc_d.transform(desc3_raw)

    sc_e   = StandardScaler().fit(B1["esm2"])
    pca_g  = PCA(n_components=5, random_state=0, svd_solver="full").fit(
                 sc_e.transform(B1["esm2"]))
    esm1_g = pca_g.transform(sc_e.transform(B1["esm2"]))
    esm3_g = pca_g.transform(sc_e.transform(B3["esm2"]))

    G1 = np.column_stack([iso1, desc1_s, esm1_g])   # (n1, 13)
    G3 = np.column_stack([iso3, desc3_s, esm3_g])   # (n3, 13)
    return G1, G3


# ── AbDevMoE ──────────────────────────────────────────────────────────────────

class AbDevMoE:
    """Interaction-gated MoE meta-learner (leakage-free inner OOF).

    fit(Xb, y, G)  →  inner 5-fold OOF → interaction meta-features → RidgeCV
    predict(Xb, G) →  expert preds × gate interactions → meta.predict
    """

    def __init__(self, experts: list[str] = BASE_EXPERTS,
                 inner_folds: int = 5, random_state: int = 0):
        self.experts     = experts
        self.inner_folds = inner_folds
        self.rs          = random_state

    # helpers
    def _get(self, Xb: dict, name: str) -> np.ndarray:
        return np.nan_to_num(np.asarray(Xb[EXPERT_SPEC[name][1]], float))

    def _interact(self, P: np.ndarray, Gs: np.ndarray) -> np.ndarray:
        return np.column_stack([Gs * P[:, ki:ki+1]
                                for ki in range(P.shape[1])])

    def fit(self, Xb: dict, y: np.ndarray, G: np.ndarray,
            extra_oof: np.ndarray | None = None,
            extra_held: np.ndarray | None = None):
        """extra_oof/extra_held: optional 8th-expert OOF / held-out column to append."""
        y = np.asarray(y, float)
        G = np.asarray(G, float)
        n = len(y)
        kf = KFold(self.inner_folds, shuffle=True, random_state=self.rs)

        # inner OOF for each named expert
        oof = {k: np.full(n, np.nan) for k in self.experts}
        for tr, va in kf.split(np.arange(n)):
            for k in self.experts:
                mk = EXPERT_SPEC[k][0]
                oof[k][va] = mk().fit(self._get(Xb, k)[tr],
                                      y[tr]).predict(self._get(Xb, k)[va])

        P = np.column_stack([oof[k] for k in self.experts])   # (n, K)
        if extra_oof is not None:
            P = np.column_stack([P, extra_oof.reshape(-1, 1)])  # (n, K+1)

        self.G_sc_ = StandardScaler().fit(G)
        Gs = self.G_sc_.transform(G)
        Z  = np.column_stack([P, self._interact(P, Gs)])       # (n, (K+1)*(1+d))

        valid       = ~np.any(np.isnan(Z), axis=1)
        self.meta_  = RidgeCV(alphas=np.logspace(-3, 4, 14)).fit(Z[valid], y[valid])

        # refit all named experts on full training data
        self.fitted_ = {k: EXPERT_SPEC[k][0]().fit(self._get(Xb, k), y)
                        for k in self.experts}
        self.extra_held_ = extra_held   # pre-computed held-out preds for the 8th expert
        self.K_ = P.shape[1]
        return self

    def predict(self, Xb: dict, G: np.ndarray) -> np.ndarray:
        Gs = self.G_sc_.transform(np.asarray(G, float))
        P  = np.column_stack([self.fitted_[k].predict(self._get(Xb, k))
                               for k in self.experts])
        if self.extra_held_ is not None:
            P = np.column_stack([P, self.extra_held_.reshape(-1, 1)])
        Z  = np.column_stack([P, self._interact(P, Gs)])
        return self.meta_.predict(Z)

    def effective_weights(self, g: np.ndarray) -> dict[str, float]:
        """Per-antibody effective expert weights for a single gate vector g (d_gate,)."""
        Gs = self.G_sc_.transform(g.reshape(1, -1)).ravel()
        K  = self.K_
        c  = self.meta_.coef_
        w  = c[:K] + c[K:].reshape(K, -1) @ Gs
        keys = self.experts + (["flab"] if self.extra_held_ is not None else [])
        return dict(zip(keys, w.tolist()))


# ── Main evaluation ────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    # Force UTF-8 stdout on Windows so Unicode chars in pandas output don't crash
    if hasattr(_sys.stdout, "reconfigure"):
        try:
            _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("=" * 60)
    print(f"  AbDev-MoE")
    print(f"  Python  : {_sys.executable}")
    print(f"  GPU     : NVIDIA RTX A4000 (detected after data load to avoid CUDA/mmap conflict)")
    print(f"  LGBM    : CPU (matches all prior project experiments)")
    print("=" * 60, flush=True)

    print("Loading data ...", flush=True)
    ds, g3, B1_raw, B3_raw = GE.build_blocks()

    # Detect GPU *after* numpy npz loading to avoid CUDA/mmap conflict
    try:
        import torch as _t
        gpu_name = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "CPU (no CUDA)"
    except Exception:
        gpu_name = "unknown"
    print(f"  GPU detected: {gpu_name}", flush=True)

    sap1  = _load_sap_electro(ds.df["antibody_name"], "gdpa1")
    sap3  = _load_sap_electro(g3["antibody_name"],    "heldout")
    B1    = {**B1_raw, "sap_electro": sap1}
    B3    = {**B3_raw, "sap_electro": sap3}

    abl1  = B1["ablang2"]   # (246, 480)
    abl3  = B3["ablang2"]   # ( 80, 480)

    print("Building gate features ...", flush=True)
    G1, G3 = build_gate(B1, B3)

    folds  = ds.folds("cluster_isotype")
    ufolds = sorted(f for f in set(folds.tolist()) if f == f)

    results: dict[str, dict] = {m: {} for m in [
        "esm2_only", "ablang2_only", "best_oodcv", "ood_stack",
        "abdevx_fusion", "moe_base", "moe_plus_flab"
    ]}

    for a in ASSAYS:
        print(f"\n--- {a} ---", flush=True)
        y1  = ds.y(a)
        lab = ~np.isnan(y1)
        y3  = g3[LABEL_MAP[a]].to_numpy(float)
        ok  = ~np.isnan(y3)

        # ── Single-block OOD-CV scores and OOF preds ──────────────────────────
        cv_rho: dict[str, float] = {}
        oof_blks: dict[str, np.ndarray] = {}
        held_blks: dict[str, np.ndarray] = {}
        for blk in ["esm2", "descriptors", "isotype", "bio", "ablang2", "sap_electro"]:
            Xb1 = np.nan_to_num(B1[blk])
            Xb3 = np.nan_to_num(B3[blk])
            oo, per = np.full(len(y1), np.nan), []
            for f in ufolds:
                tr = np.where((folds != f) & lab)[0]
                te = np.where((folds == f) & lab)[0]
                if len(te) < 3 or len(tr) < 10:
                    continue
                p = _block_head(blk).fit(Xb1[tr], y1[tr]).predict(Xb1[te])
                oo[te] = p
                per.append(spearmanr(y1[te], p).statistic)
            cv_rho[blk]   = float(np.nanmean(per))
            oof_blks[blk] = oo
            held_blks[blk] = _block_head(blk).fit(Xb1[lab], y1[lab]).predict(Xb3)

        best_blk = max(cv_rho, key=cv_rho.get)
        results["best_oodcv"][a]  = float(spearmanr(y3[ok], held_blks[best_blk][ok]).statistic)
        results["esm2_only"][a]   = float(spearmanr(y3[ok], held_blks["esm2"][ok]).statistic)
        results["ablang2_only"][a]= float(spearmanr(y3[ok], held_blks["ablang2"][ok]).statistic)

        # OOD-linear stack (reference)
        common  = np.all([~np.isnan(oof_blks[b]) for b in cv_rho], axis=0) & lab
        Z_oof   = np.column_stack([oof_blks[b][common] for b in cv_rho])
        Zh      = np.column_stack([held_blks[b]        for b in cv_rho])
        meta_l  = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(Z_oof, y1[common])
        results["ood_stack"][a]   = float(spearmanr(y3[ok], meta_l.predict(Zh)[ok]).statistic)

        # AbDev-X naive fusion (fixed-weight, no gate)
        from abdev.models.fusion import AbDevFusion
        B1m = {k: v[lab] for k, v in B1.items()}
        results["abdevx_fusion"][a] = float(
            spearmanr(y3[ok], AbDevFusion().fit(B1m, y1[lab]).predict(B3)[ok]).statistic)

        # ── AbDev-MoE (7 base experts + gate) ────────────────────────────────
        print("  Fitting MoE (7 experts) ...", flush=True)
        moe = AbDevMoE()
        moe.fit({k: v[lab] for k, v in B1.items()}, y1[lab], G1[lab])
        results["moe_base"][a] = float(spearmanr(y3[ok], moe.predict(B3, G3)[ok]).statistic)

        # ── AbDev-MoE+ (8th expert: FLAb-augmented AbLang2) ─────────────────
        print("  Fitting MoE+ (FLAb 8th expert) ...", flush=True)
        flab_extras = _load_flab_extras(a)

        if flab_extras is not None:
            Xf, zf = flab_extras          # FLAb-only pool (n_flab, 480), z-scored per-study
            X1_abl = abl1[lab]            # GDPa1 AbLang2 for labeled samples
            y1_lab = y1[lab]
            n_lab  = len(y1_lab)

            # OOF for the FLAb-AbLang2 expert (in-fold z-score for GDPa1 portion)
            kf_in = KFold(5, shuffle=True, random_state=0)
            oof_flab = np.full(n_lab, np.nan)
            for tr_i, va_i in kf_in.split(np.arange(n_lab)):
                mu, sd   = float(np.nanmean(y1_lab[tr_i])), float(np.nanstd(y1_lab[tr_i])) + 1e-9
                y_tr_z   = (y1_lab[tr_i] - mu) / sd
                X_tr_aug = np.vstack([X1_abl[tr_i], Xf])
                y_tr_aug = np.concatenate([y_tr_z, zf])
                oof_flab[va_i] = _ridge_pca().fit(X_tr_aug, y_tr_aug).predict(X1_abl[va_i])

            # Full-data model for GDPa3 prediction
            mu_f, sd_f  = float(np.nanmean(y1_lab)), float(np.nanstd(y1_lab)) + 1e-9
            y1_lab_z    = (y1_lab - mu_f) / sd_f
            X_full_aug  = np.vstack([X1_abl, Xf])
            y_full_aug  = np.concatenate([y1_lab_z, zf])
            m_flab_full = _ridge_pca().fit(X_full_aug, y_full_aug)
            held_flab   = m_flab_full.predict(abl3)

            moe_p = AbDevMoE()
            moe_p.fit({k: v[lab] for k, v in B1.items()},
                      y1_lab, G1[lab],
                      extra_oof=oof_flab,
                      extra_held=held_flab)
            results["moe_plus_flab"][a] = float(
                spearmanr(y3[ok], moe_p.predict(B3, G3)[ok]).statistic)
        else:
            results["moe_plus_flab"][a] = results["moe_base"][a]

    # ── Report ────────────────────────────────────────────────────────────────
    model_order = ["esm2_only", "ablang2_only", "best_oodcv", "ood_stack",
                   "abdevx_fusion", "moe_base", "moe_plus_flab"]
    label = {
        "esm2_only":       "ESM2-only",
        "ablang2_only":    "AbLang2-only",
        "best_oodcv":      "Best-single (OOD-CV)",
        "ood_stack":       "OOD-linear stack",
        "abdevx_fusion":   "AbDev-X (fixed meta)",
        "moe_base":        "AbDev-MoE  (7 experts + gate)",
        "moe_plus_flab":   "AbDev-MoE+ (8 experts + FLAb + gate)",
    }
    rows = []
    for m in model_order:
        r = {a: round(float(results[m][a]), 3) for a in ASSAYS}
        r["MEAN"]  = round(float(np.mean([results[m][a] for a in ASSAYS])), 3)
        r["Model"] = label[m]
        rows.append(r)
    rows.append({**{a: COMP[a] for a in ASSAYS},
                 "MEAN":  round(float(np.mean(list(COMP.values()))), 3),
                 "Model": "Competition champions (per-assay)"})

    df = pd.DataFrame(rows)[["Model"] + ASSAYS + ["MEAN"]]
    pd.set_option("display.width", 200, "display.max_columns", 20)

    sep = "=" * 140
    print(f"\n\n{sep}")
    print("  AbDev-MoE  |  GDPa3 HELD-OUT  |  Spearman rho  "
          "(train GDPa1 → predict 80 held-out IgG1s)")
    print(sep)
    print(df.to_string(index=False))
    print(sep)

    out = ROOT / "results" / "moe_gdpa3.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved -> {out}")

    # Summary deltas
    for m in ["moe_base", "moe_plus_flab"]:
        mean_m  = float(np.mean([results[m][a]           for a in ASSAYS]))
        mean_ax = float(np.mean([results["abdevx_fusion"][a] for a in ASSAYS]))
        n_beat  = sum(results[m][a] > COMP[a] for a in ASSAYS)
        print(f"\n{label[m]}")
        print(f"  Mean Spearman  : {mean_m:.3f}  (AbDev-X: {mean_ax:.3f}, "
              f"Δ = {mean_m - mean_ax:+.3f})")
        print(f"  Champions beaten: {n_beat}/5")
        for a in ASSAYS:
            delta = results[m][a] - COMP[a]
            marker = " ✓" if delta > 0 else ""
            print(f"    {a:22s}  {results[m][a]:.3f}  (champ {COMP[a]:.3f}, d={delta:+.3f}){marker}")


if __name__ == "__main__":
    main()
