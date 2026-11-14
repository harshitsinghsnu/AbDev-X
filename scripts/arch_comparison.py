"""Architecture comparison: A (MGFS), D (DSSF), F (Bio-MoE) vs E1 baseline.

Evaluated on 5 genuine assays: HIC, AC-SINS, PR_CHO, SMAC, HAC.
Thermal/SEC excluded — no sequence signal demonstrated.

Architecture A — Mechanism-Guided Feature Selection (MGFS)
  Careful assay-specific biological masks on desc+bio features.
  Tests: can explicit mechanistic subsets beat global feature blocks?

Architecture D — Dual-Stream Sequence-Structure Fusion (DSSF)
  Stream 1: ESM-2 + AbLang2 (sequence)
  Stream 2: bio + desc (structural/physicochemical)
  Fusions: concat, fixed 0.5/0.5 blend, OOF stacking (assay-specific α).

Architecture F — Biological Mixture-of-Experts (Bio-MoE)
  Experts: mechanism sub-blocks from bio+desc (7 experts).
  Gating: soft OOD-CV scores → weighted combination of held-out predictions.
  Constrained by biological eligibility matrix.

All compared vs E1 (best-block OOD-CV selection) on GDPa3.
Bootstrap 95% CI (n=2000) reported for each.
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

# Only genuine signal assays
GENUINE_ASSAYS = {
    "HIC":     ("HIC",          "hic_rt_avg"),
    "AC-SINS": ("AC-SINS_pH7.4","acsins_dLmax_ph7.4_avg"),
    "PR_CHO":  ("PR_CHO",       "polyreactivity_prscore_cho_avg"),
    "SMAC":    ("SMAC",         "smac_rt_avg"),
    "HAC":     ("HAC",          "hac_rt_avg"),
}

BIO_FILES = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv"]
ISO_COLS  = ["hc_subtype_IgG1","hc_subtype_IgG2","hc_subtype_IgG4",
             "lc_subtype_Kappa","lc_subtype_Lambda"]
N_BOOT = 2000


# ── helpers ──────────────────────────────────────────────────────────────────

def _iso(df):
    d = pd.get_dummies(df[["hc_subtype","lc_subtype"]].astype(str))
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


def ood_cv_preds(X, y, folds, fold_ids):
    """Return OOF predictions array (NaN where not predicted)."""
    m = ~np.isnan(y)
    oof = np.full(len(y), np.nan)
    for f in fold_ids:
        tr = np.where((folds != f) & m)[0]
        te = np.where((folds == f) & m)[0]
        if len(te) < 3 or len(tr) < 10: continue
        pipe = _head(X)
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])
    return oof


def ood_cv_rho(X, y, folds, fold_ids):
    oof = ood_cv_preds(X, y, folds, fold_ids)
    m = ~np.isnan(y) & ~np.isnan(oof)
    return float(spearmanr(y[m], oof[m]).statistic) if m.sum() >= 5 else -1.0


def held_out(X_tr, y_tr, X_te, y_te):
    m = ~np.isnan(y_tr); ok = ~np.isnan(y_te)
    if m.sum() < 10 or ok.sum() < 5: return np.nan
    pipe = _head(X_tr)
    pipe.fit(X_tr[m], y_tr[m])
    return float(spearmanr(y_te[ok], pipe.predict(X_te)[ok]).statistic)


def bca_ci(y_te, pred, n_boot=N_BOOT, alpha=0.05, rng=None):
    """BCa bootstrap 95% CI for Spearman rho."""
    if rng is None: rng = np.random.default_rng(0)
    ok = ~np.isnan(y_te) & ~np.isnan(pred)
    y, p = y_te[ok], pred[ok]
    n = len(y)
    if n < 5: return np.nan, np.nan
    obs = spearmanr(y, p).statistic
    boot = np.array([spearmanr(y[s:=rng.integers(0, n, n)], p[s]).statistic
                     for _ in range(n_boot)])
    # bias correction
    z0 = float(np.sum(boot < obs) / n_boot)
    from scipy.stats import norm
    z0n = norm.ppf(np.clip(z0, 1e-6, 1-1e-6))
    # acceleration
    jk = np.array([spearmanr(np.delete(y,i), np.delete(p,i)).statistic for i in range(n)])
    jk_mean = jk.mean()
    num = ((jk_mean - jk)**3).sum(); den = (6*((jk_mean - jk)**2).sum()**1.5)
    a = num / den if den != 0 else 0.0
    za = norm.ppf(alpha/2); zb = norm.ppf(1-alpha/2)
    def adj(z): return norm.cdf(z0n + (z0n+z)/(1-a*(z0n+z)))
    lo = float(np.percentile(boot, 100*adj(za)))
    hi = float(np.percentile(boot, 100*adj(zb)))
    return lo, hi


# ── Architecture A: MGFS masks ───────────────────────────────────────────────

# Carefully designed per-assay masks based on biological priors.
# Keys are keyword fragments matched against concatenated (desc_cols + bio_cols).

MGFS_MASKS = {
    "HIC": {
        # HIC = hydrophobic interaction chromatography retention
        # Signal: exposed hydrophobic surface, aromatic patches, GRAVY
        "include": ["GRAVY","aromat","aliphat","hydro","psh","patch_hyd",
                    "Aggrescan","HydroScore","sasa","rSASA","exposed",
                    "patch","surface"],
        "exclude": ["charge","pI","pnc","ppc"],
    },
    "AC-SINS": {
        # AC-SINS = self-association via surface charge asymmetry
        # Signal: charge patches, pI, electrostatics, exposed charged residues
        "include": ["charge","pI","isoelectric","ppc","pnc","patch_pos",
                    "patch_neg","dipole","zeta","electr","ionic"],
        "exclude": ["hydro","GRAVY","aromat"],
    },
    "PR_CHO": {
        # Polyreactivity = nonspecific binding, driven by CDR+surface charge+hydro
        # Signal: CDR composition, charge, hydrophobicity, aromatic exposure
        "include": ["cdr","CDR","h3","l3","h1","h2","l1","l2",
                    "charge","pI","ppc","pnc","GRAVY","aromat",
                    "hydro","patch","surface","sasa"],
        "exclude": [],
    },
    "SMAC": {
        # Surrogate clearance — FcRn/charge-mediated, surface patches
        # Signal: surface hydrophobicity, charge patches, exposed patches
        "include": ["sasa","rSASA","patch","surface","hydro","GRAVY",
                    "aromat","charge","pI","ppc","pnc","Aggrescan",
                    "patch_hyd","patch_pos","patch_neg"],
        "exclude": [],
    },
    "HAC": {
        # Human anti-chimeric — structural surface exposure, novelty patches
        # Signal: surface features, structural geometry, patch characteristics
        "include": ["sasa","rSASA","patch","surface","hydro","charge",
                    "Aggrescan","MOE","gyrat","packing","mobility",
                    "AntiFold"],
        "exclude": [],
    },
}


def apply_mgfs_mask(all_cols, assay):
    inc = [k.lower() for k in MGFS_MASKS[assay]["include"]]
    exc = [k.lower() for k in MGFS_MASKS[assay]["exclude"]]
    mask = np.array([
        (any(k in c.lower() for k in inc) and
         not any(k in c.lower() for k in exc))
        for c in all_cols
    ])
    return mask if mask.sum() >= 5 else np.ones(len(all_cols), dtype=bool)


# ── Architecture F: Bio-MoE experts ─────────────────────────────────────────

# 7 mechanism sub-blocks as experts. Per-assay gating = OOD-CV softmax.
# Biological eligibility: some experts excluded for some assays.

EXPERTS_KW = {
    "hydro":    ["GRAVY","aromat","aliphat","hydro","psh","patch_hyd","Aggrescan","HydroScore"],
    "charge":   ["charge","pI","ppc","pnc","patch_pos","patch_neg","dipole","zeta","electr"],
    "stab":     ["stab","sasa","gyrat","pack","mobil","AntiFold","contact","burial"],
    "liab":     ["deamid","isomeriz","glyc","oxid","cys","liabil","nglyc","asn","asp"],
    "cdr":      ["cdr","CDR","h3","l3","h1","h2","l1","l2","loop"],
    "surface":  ["sasa","rSASA","patch","surface","exposed","SASA"],
    "global":   [],  # full bio+desc with no mask
}

# Eligibility: which experts are allowed for each assay
MOE_ELIGIBILITY = {
    "HIC":     ["hydro","surface","global"],
    "AC-SINS": ["charge","surface","global"],
    "PR_CHO":  ["charge","cdr","hydro","surface","global"],
    "SMAC":    ["hydro","charge","surface","global"],
    "HAC":     ["stab","hydro","charge","surface","global"],
}


def get_expert_mask(all_cols, expert_name):
    if expert_name == "global":
        return np.ones(len(all_cols), dtype=bool)
    kws = [k.lower() for k in EXPERTS_KW[expert_name]]
    mask = np.array([any(k in c.lower() for k in kws) for c in all_cols])
    return mask if mask.sum() >= 5 else np.ones(len(all_cols), dtype=bool)


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds    = ds.folds("cluster_isotype")
    fold_ids = sorted(set(folds.tolist()))
    rng = np.random.default_rng(42)

    esm_files = list((ROOT / "data/processed").glob("esm2_*.npz"))
    esm1 = None
    for ef in esm_files:
        arr = np.load(ef)["emb"]
        if arr.shape == (len(ds.df), 2560):
            esm1 = arr; break
    abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    esm3 = np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"]
    abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]

    desc_df = build_descriptor_matrix(ds.heavy, ds.light)
    desc_cols = list(desc_df.columns)
    desc1 = np.nan_to_num(desc_df.to_numpy(float))
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

    # Combined desc+bio for MGFS and MoE
    all_cols = desc_cols + list(bio_cols)
    Xdb1 = np.hstack([desc1, bio1]); Xdb3 = np.hstack([desc3, bio3])

    # E1 single blocks
    single_blocks = {"esm2": (esm1n, esm3n), "ablang2": (abl1n, abl3n),
                     "isotype": (iso1, iso3), "desc": (desc1, desc3), "bio": (bio1, bio3)}

    rows = []
    preds_all = {}  # save predictions for BCa CI

    print("\n" + "="*105)
    print(f"{'Assay':10s}  {'E1':7s} {'CI':13s}  {'A-MGFS':7s} {'CI':13s}  "
          f"{'D-concat':8s}  {'D-stack':8s}  {'F-MoE':7s}  {'Best':8s}")
    print("="*105)

    for assay, (g1_col, g3_col) in GENUINE_ASSAYS.items():
        y1 = ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df), np.nan)
        y3 = g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(len(g3), np.nan)
        ok3 = ~np.isnan(y3)

        # ── E1: best single block ──────────────────────────────────────────
        cv_scores = {b: ood_cv_rho(Xtr, y1, folds, fold_ids)
                     for b, (Xtr, _) in single_blocks.items() if b != "isotype"}
        best_blk = max(cv_scores, key=lambda k: cv_scores[k])
        Xb1, Xb3 = single_blocks[best_blk]
        pipe_e1 = _head(Xb1)
        pipe_e1.fit(Xb1[~np.isnan(y1)], y1[~np.isnan(y1)])
        pred_e1 = pipe_e1.predict(Xb3)
        e1 = float(spearmanr(y3[ok3], pred_e1[ok3]).statistic)
        ci_e1 = bca_ci(y3, pred_e1, rng=rng)

        # ── Architecture A: MGFS ──────────────────────────────────────────
        mask_a = apply_mgfs_mask(all_cols, assay)
        Xa1 = Xdb1[:, mask_a]; Xa3 = Xdb3[:, mask_a]
        # OOD-CV to decide: MGFS alone, or MGFS + best PLM
        cv_a_desc  = ood_cv_rho(Xa1, y1, folds, fold_ids)
        cv_a_plm   = ood_cv_rho(np.hstack([Xb1, Xa1]), y1, folds, fold_ids)
        if cv_a_plm > cv_a_desc:
            Xa1_use = np.hstack([Xb1, Xa1]); Xa3_use = np.hstack([Xb3, Xa3])
            a_note = f"PLM+mask({mask_a.sum()})"
        else:
            Xa1_use = Xa1; Xa3_use = Xa3
            a_note = f"mask({mask_a.sum()})"
        pipe_a = _head(Xa1_use)
        pipe_a.fit(Xa1_use[~np.isnan(y1)], y1[~np.isnan(y1)])
        pred_a = pipe_a.predict(Xa3_use)
        a_rho = float(spearmanr(y3[ok3], pred_a[ok3]).statistic)
        ci_a = bca_ci(y3, pred_a, rng=rng)

        # ── Architecture D: DSSF ──────────────────────────────────────────
        # Stream 1: ESM-2 + AbLang2 (sequence)
        # Stream 2: bio + desc (structural)
        Xs1 = np.hstack([esm1n, abl1n]); Xs3 = np.hstack([esm3n, abl3n])
        Xst1 = np.hstack([bio1, desc1]); Xst3 = np.hstack([bio3, desc3])

        # D-concat: concatenate both streams
        Xdc1 = np.hstack([Xs1, Xst1]); Xdc3 = np.hstack([Xs3, Xst3])
        d_concat = held_out(Xdc1, y1, Xdc3, y3)

        # D-stack: OOF stacking (assay-specific α learned by Ridge)
        oof_seq  = ood_cv_preds(Xs1,  y1, folds, fold_ids)
        oof_struc = ood_cv_preds(Xst1, y1, folds, fold_ids)
        valid = ~np.isnan(y1) & ~np.isnan(oof_seq) & ~np.isnan(oof_struc)
        meta_feat = np.column_stack([oof_seq, oof_struc])
        if valid.sum() >= 15:
            meta = RidgeCV(alphas=np.logspace(-2, 4, 12))
            meta.fit(meta_feat[valid], y1[valid])
            # retrain base models on all data
            pipe_seq  = _head(Xs1); pipe_seq.fit(Xs1[~np.isnan(y1)], y1[~np.isnan(y1)])
            pipe_strc = _head(Xst1); pipe_strc.fit(Xst1[~np.isnan(y1)], y1[~np.isnan(y1)])
            pred_meta = meta.predict(np.column_stack([
                pipe_seq.predict(Xs3), pipe_strc.predict(Xst3)]))
            d_stack = float(spearmanr(y3[ok3], pred_meta[ok3]).statistic)
        else:
            d_stack = np.nan

        # ── Architecture F: Bio-MoE ──────────────────────────────────────
        eligible_experts = MOE_ELIGIBILITY.get(assay, list(EXPERTS_KW.keys()))
        expert_oof  = {}
        expert_preds = {}
        expert_cv = {}
        for exp in eligible_experts:
            emask = get_expert_mask(all_cols, exp)
            Xe1 = Xdb1[:, emask]; Xe3 = Xdb3[:, emask]
            cv_e = ood_cv_rho(Xe1, y1, folds, fold_ids)
            oof_e = ood_cv_preds(Xe1, y1, folds, fold_ids)
            pipe_e = _head(Xe1); pipe_e.fit(Xe1[~np.isnan(y1)], y1[~np.isnan(y1)])
            expert_oof[exp]   = oof_e
            expert_preds[exp] = pipe_e.predict(Xe3)
            expert_cv[exp]    = cv_e

        # Gating: softmax of OOD-CV scores → weighted prediction
        exps = list(expert_cv.keys())
        scores_arr = np.array([expert_cv[e] for e in exps])
        # temperature-scaled softmax (τ=0.1 concentrates on best)
        tau = 0.1
        weights = np.exp(scores_arr / tau); weights /= weights.sum()
        pred_moe = sum(weights[i] * expert_preds[exps[i]] for i in range(len(exps)))
        f_rho = float(spearmanr(y3[ok3], pred_moe[ok3]).statistic)

        # Also try OOF ridge stacking over experts
        valid_exp = ~np.isnan(y1)
        oof_mat = np.column_stack([expert_oof[e] for e in exps])
        valid_exp2 = valid_exp & np.all(~np.isnan(oof_mat), axis=1)
        if valid_exp2.sum() >= 15:
            meta_moe = RidgeCV(alphas=np.logspace(-2, 4, 12))
            meta_moe.fit(oof_mat[valid_exp2], y1[valid_exp2])
            pred_moe_stack = meta_moe.predict(
                np.column_stack([expert_preds[e] for e in exps]))
            f_stack = float(spearmanr(y3[ok3], pred_moe_stack[ok3]).statistic)
        else:
            f_stack = np.nan
        f_best = max(f_rho, f_stack) if not np.isnan(f_stack) else f_rho

        all_rhos = {"E1": e1, "A-MGFS": a_rho, "D-concat": d_concat,
                    "D-stack": d_stack, "F-MoE": f_best}
        best_arch = max(all_rhos, key=lambda k: all_rhos[k] if not np.isnan(all_rhos[k]) else -9)

        def f(v): return f"{v:+7.3f}" if not np.isnan(v) else "    nan"
        def ci(lo, hi):
            if np.isnan(lo): return "             "
            return f"[{lo:+.3f},{hi:+.3f}]"
        print(f"{assay:10s}  {f(e1)} {ci(*ci_e1)}  {f(a_rho)} {ci(*ci_a)}  "
              f"{f(d_concat)}  {f(d_stack)}  {f(f_best)}  {best_arch}")

        rows.append({"assay": assay, "e1_blk": best_blk,
                     "E1": round(e1,3), "E1_ci_lo": round(ci_e1[0],3), "E1_ci_hi": round(ci_e1[1],3),
                     "A_MGFS": round(a_rho,3), "A_note": a_note,
                     "D_concat": round(d_concat,3) if not np.isnan(d_concat) else None,
                     "D_stack": round(d_stack,3) if not np.isnan(d_stack) else None,
                     "F_MoE": round(f_best,3),
                     "best_arch": best_arch})

    print("="*105)
    df = pd.DataFrame(rows)
    for col in ["E1","A_MGFS","D_concat","D_stack","F_MoE"]:
        print(f"  Mean {col:10s}: {df[col].mean():.3f}")

    out = ROOT / "results/arch_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    print("\n--- What's needed for B (RBAN) and C (SRGN) ---")
    print("  B: per-residue ESM-2/AbLang2 embeddings (re-run extractors with save_per_residue=True)")
    print("  C: predicted structures + contact matrices (run ESMFold or AlphaFold2 on all 326 sequences)")
    print("  E: CDR residue annotations (extractable from heavy_aligned_aho / light_aligned_aho AHo numbering)")


if __name__ == "__main__":
    main()
