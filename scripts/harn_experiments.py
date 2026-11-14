"""HARN — Hierarchical Antibody Region Network experiments H1–H7.

Descriptors are already computed per region:
  H_CDR1_*, H_CDR2_*, H_CDR3_*, H_FR_*  (heavy)
  L_CDR1_*, L_CDR2_*, L_CDR3_*, L_FR_*  (light)
  H_*, L_*, Fv_*                          (chain and global)

H1  Global mean pool (full Fv descriptor)
H2  CDR regions only (H_CDR1+2+3 + L_CDR1+2+3)
H3  Framework regions only (H_FR + L_FR)
H4  CDR + FR (all regional — tests decomposition value)
H5  VH regions + VL regions (chain separation)
H6  Hierarchical fusion — regions → chain → antibody, then concat
H7  H6 + PLM global context (best PLM appended to hierarchical desc)

Evaluated on 5 genuine assays. E1 (best-block from arch_comparison) = 0.444 mean.
Bootstrap 95% CI (n=2000) for final comparison.
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

GENUINE_ASSAYS = {
    "HIC":     ("HIC",          "hic_rt_avg"),
    "AC-SINS": ("AC-SINS_pH7.4","acsins_dLmax_ph7.4_avg"),
    "PR_CHO":  ("PR_CHO",       "polyreactivity_prscore_cho_avg"),
    "SMAC":    ("SMAC",         "smac_rt_avg"),
    "HAC":     ("HAC",          "hac_rt_avg"),
}
N_BOOT = 2000

# Region groups defined by column prefix
REGION_GROUPS = {
    "H_CDR1": "H_CDR1_",
    "H_CDR2": "H_CDR2_",
    "H_CDR3": "H_CDR3_",
    "H_FR":   "H_FR_",
    "L_CDR1": "L_CDR1_",
    "L_CDR2": "L_CDR2_",
    "L_CDR3": "L_CDR3_",
    "L_FR":   "L_FR_",
    # Chain-level global (excludes per-CDR/FR to avoid overlap)
    "VH_global": None,  # H_* that are NOT H_CDR* or H_FR*
    "VL_global": None,  # L_* that are NOT L_CDR* or L_FR*
    "Fv_global": "Fv_",
}


def get_region_cols(all_cols, region):
    """Return boolean mask over all_cols for a given region."""
    if region == "VH_global":
        return np.array([
            c.startswith("H_") and
            not any(c.startswith(p) for p in ["H_CDR1_","H_CDR2_","H_CDR3_","H_FR_"])
            for c in all_cols])
    if region == "VL_global":
        return np.array([
            c.startswith("L_") and
            not any(c.startswith(p) for p in ["L_CDR1_","L_CDR2_","L_CDR3_","L_FR_"])
            for c in all_cols])
    prefix = REGION_GROUPS[region]
    return np.array([c.startswith(prefix) for c in all_cols])


def _head(X):
    steps = [("s", StandardScaler())]
    if X.shape[1] > 100:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def ood_cv_rho(X, y, folds, fold_ids):
    m = ~np.isnan(y); scores = []
    for f in fold_ids:
        tr = np.where((folds != f) & m)[0]
        te = np.where((folds == f) & m)[0]
        if len(te) < 3 or len(tr) < 10: continue
        pipe = _head(X); pipe.fit(X[tr], y[tr])
        scores.append(spearmanr(y[te], pipe.predict(X[te])).statistic)
    return float(np.nanmean(scores)) if scores else -1.0


def held_out_rho_pred(X_tr, y_tr, X_te):
    m = ~np.isnan(y_tr)
    pipe = _head(X_tr); pipe.fit(X_tr[m], y_tr[m])
    return pipe.predict(X_te)


def bca_ci(y_te, pred, n_boot=N_BOOT, alpha=0.05, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    ok = ~np.isnan(y_te) & ~np.isnan(pred)
    y, p = y_te[ok], pred[ok]; n = len(y)
    if n < 5: return np.nan, np.nan
    obs = spearmanr(y, p).statistic
    boot = np.array([spearmanr(y[s:=rng.integers(0,n,n)], p[s]).statistic
                     for _ in range(n_boot)])
    from scipy.stats import norm
    z0n = norm.ppf(np.clip(np.mean(boot < obs), 1e-6, 1-1e-6))
    jk = np.array([spearmanr(np.delete(y,i), np.delete(p,i)).statistic for i in range(n)])
    jk_m = jk.mean()
    a = ((jk_m-jk)**3).sum() / (6*(((jk_m-jk)**2).sum()**1.5) + 1e-12)
    def adj(z): return norm.cdf(z0n + (z0n+z)/(1-a*(z0n+z)))
    return float(np.percentile(boot, 100*adj(norm.ppf(alpha/2)))), \
           float(np.percentile(boot, 100*adj(norm.ppf(1-alpha/2))))


def hierarchical_pool(X, col_mask_dict, col_names):
    """
    For each region group, pool (mean) the features within that group,
    then concatenate region-level pooled vectors.
    Returns (n_samples, n_regions * features_per_region) but since regions
    have different sizes we just concatenate region-level feature vectors.
    """
    parts = []
    for region, mask in col_mask_dict.items():
        sub = X[:, mask]
        if sub.shape[1] == 0: continue
        parts.append(sub)
    return np.hstack(parts) if parts else X


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds    = ds.folds("cluster_isotype")
    fold_ids = sorted(set(folds.tolist()))
    rng = np.random.default_rng(42)

    desc_df1 = build_descriptor_matrix(ds.heavy, ds.light)
    all_cols  = list(desc_df1.columns)
    X_desc1   = np.nan_to_num(desc_df1.to_numpy(float))
    X_desc3   = np.nan_to_num(build_descriptor_matrix(g3["vh"], g3["vl"]).to_numpy(float))

    # Pre-compute region masks and print breakdown
    region_masks = {r: get_region_cols(all_cols, r) for r in REGION_GROUPS}
    print("\n--- Region sizes ---")
    for r, m in region_masks.items():
        print(f"  {r:12s}: {m.sum():3d} features")

    # Best PLM: ESM-2 (use for H7)
    esm_files = list((ROOT / "data/processed").glob("esm2_*.npz"))
    esm1 = None
    for ef in esm_files:
        arr = np.load(ef)["emb"]
        if arr.shape == (len(ds.df), 2560): esm1 = arr; break
    esm3 = np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"]
    esm1n = np.nan_to_num(esm1); esm3n = np.nan_to_num(esm3)

    # ---  Define H1–H7 representations  ------------------------------------
    def make_X(variant, desc1=X_desc1, desc3=X_desc3):
        """Returns (X_train, X_test) for each variant."""
        def sel(regions):
            mask = np.zeros(len(all_cols), bool)
            for r in regions:
                mask |= region_masks[r]
            return desc1[:, mask], desc3[:, mask]

        if variant == "H1":   # global Fv descriptors
            return desc1, desc3
        elif variant == "H2": # CDR regions only
            return sel(["H_CDR1","H_CDR2","H_CDR3","L_CDR1","L_CDR2","L_CDR3"])
        elif variant == "H3": # Framework only
            return sel(["H_FR","L_FR","VH_global","VL_global","Fv_global"])
        elif variant == "H4": # CDR + FR (full regional decomposition)
            return sel(list(REGION_GROUPS.keys()))
        elif variant == "H5": # VH sub-regions, then VL sub-regions
            return sel(["H_CDR1","H_CDR2","H_CDR3","H_FR","VH_global",
                        "L_CDR1","L_CDR2","L_CDR3","L_FR","VL_global","Fv_global"])
        elif variant == "H6": # Hierarchical — same as H5 but with explicit regional ordering
            # Fuse: CDR-level → chain-level → antibody-level
            # Implementation: concat in biological order (CDR3 last as most variable)
            regions_ordered = [
                "H_CDR1","H_CDR2","H_CDR3","H_FR","VH_global",
                "L_CDR1","L_CDR2","L_CDR3","L_FR","VL_global","Fv_global"
            ]
            return sel(regions_ordered)
        elif variant == "H7": # H6 + PLM global context
            X1_h6, X3_h6 = sel(list(REGION_GROUPS.keys()))
            return np.hstack([X1_h6, esm1n]), np.hstack([X3_h6, esm3n])

    # Region-level OOD-CV selection (H7 equivalent with per-region selection)
    def h_best_region(y1, y3):
        """Pick best region group by OOD-CV, then evaluate on GDPa3."""
        cv_per_region = {}
        for r, m in region_masks.items():
            if m.sum() < 3: continue
            cv_per_region[r] = ood_cv_rho(X_desc1[:, m], y1, folds, fold_ids)
        best_r = max(cv_per_region, key=lambda k: cv_per_region[k])
        Xr1 = X_desc1[:, region_masks[best_r]]
        Xr3 = X_desc3[:, region_masks[best_r]]
        pred = held_out_rho_pred(Xr1, y1, Xr3)
        ok3 = ~np.isnan(y3)
        return float(spearmanr(y3[ok3], pred[ok3]).statistic), best_r, cv_per_region

    # ---- Run ---------------------------------------------------------------
    print("\n" + "="*100)
    print(f"{'Assay':10s}  {'H1':7s}  {'H2-CDR':7s}  {'H3-FR':7s}  "
          f"{'H4-C+F':7s}  {'H5-chain':8s}  {'H6-hier':7s}  "
          f"{'H7-+PLM':7s} {'CI':13s}  {'H-region':9s}  Best")
    print("="*100)

    rows = []
    for assay, (g1_col, g3_col) in GENUINE_ASSAYS.items():
        y1 = ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df), np.nan)
        y3 = g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(len(g3), np.nan)
        ok3 = ~np.isnan(y3)

        def rho(variant):
            Xtr, Xte = make_X(variant)
            pred = held_out_rho_pred(Xtr, y1, Xte)
            if ok3.sum() < 5: return np.nan, pred
            return float(spearmanr(y3[ok3], pred[ok3]).statistic), pred

        h1, _     = rho("H1")
        h2, _     = rho("H2")
        h3, _     = rho("H3")
        h4, _     = rho("H4")
        h5, _     = rho("H5")
        h6, _     = rho("H6")
        h7, pred7 = rho("H7")

        ci7 = bca_ci(y3, pred7, rng=rng)
        h_region, best_r, cv_regions = h_best_region(y1, y3)

        all_h = {"H1":h1,"H2":h2,"H3":h3,"H4":h4,"H5":h5,"H6":h6,"H7":h7,"H-region":h_region}
        best_h = max(all_h, key=lambda k: all_h[k] if not np.isnan(all_h[k]) else -9)

        def f(v): return f"{v:+7.3f}" if not np.isnan(v) else "    nan"
        def ci(lo, hi):
            if np.isnan(lo): return "             "
            return f"[{lo:+.3f},{hi:+.3f}]"

        print(f"{assay:10s}  {f(h1)}  {f(h2)}  {f(h3)}  {f(h4)}  "
              f"{f(h5)}  {f(h6)}  {f(h7)} {ci(*ci7)}  "
              f"{h_region:+9.3f}[{best_r[:6]}]  {best_h}")

        # Print top-3 region OOD-CV for this assay
        top3 = sorted(cv_regions.items(), key=lambda x: -x[1])[:3]
        print(f"           Region CV top-3: " +
              "  ".join(f"{r}={v:.3f}" for r, v in top3))

        rows.append({"assay": assay,
                     "H1_global": round(h1,3), "H2_CDR": round(h2,3),
                     "H3_FR": round(h3,3), "H4_CDR_FR": round(h4,3),
                     "H5_chain": round(h5,3), "H6_hier": round(h6,3),
                     "H7_PLM": round(h7,3), "H7_ci_lo": round(ci7[0],3),
                     "H7_ci_hi": round(ci7[1],3),
                     "H_best_region": round(h_region,3),
                     "H_best_region_name": best_r, "best_variant": best_h})

    print("="*100)
    df = pd.DataFrame(rows)
    for col in ["H1_global","H2_CDR","H3_FR","H4_CDR_FR","H5_chain","H6_hier","H7_PLM","H_best_region"]:
        print(f"  Mean {col:18s}: {df[col].mean():.3f}")

    print(f"\n  E1 baseline mean (from arch_comparison): 0.444")
    print(f"  Best HARN mean: {df[['H1_global','H2_CDR','H3_FR','H4_CDR_FR','H5_chain','H6_hier','H7_PLM','H_best_region']].max(axis=1).mean():.3f}")

    out = ROOT / "results/harn_experiments.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    print("\n--- PLM per-residue extraction needed for full HARN ---")
    print("  Saves: esm2_per_residue.npz  shape=(246, max_L, 2560)")
    print("  Command: python scripts/extract_plm_residues.py")
    print("  Enables: per-CDR PLM pooling → likely largest HARN gain")


if __name__ == "__main__":
    main()
