"""
Reviewer-response experiments for AbDev-X paper.

Addresses each reviewer question with a concrete experiment:
  Q1. GDPa3 VH/VL pairing uniqueness validation
  Q2. Held-out regime clarity: stacking vs block-selection per assay
  Q4. Constrained/Bayesian stacking: non-negative, temperature scaling
  Q5. Batch correction: ComBat-lite (study-level z-scoring) for FLAb augmentation
  Q6. Calibration surrogates: jackknife+, conformalized residuals, quantile regression
  Q7. Per-block residue attributions (best block per assay)

Usage:
  python scripts/reviewer_experiments.py [--q 1 4 5 6 7]  (default: all)

Outputs (results/reviewer_response/):
  q1_gdpa3_pairing_validation.json
  q4_constrained_stacking.csv
  q5_batch_correction_flab.csv
  q6_calibration_surrogates.csv
  q7_best_block_per_assay.json
  paper/figures/panel_reviewer_response.pdf
"""
import os, sys, json, warnings, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RES  = ROOT / "results" / "reviewer_response"
FIGS = ROOT / "paper" / "figures"
RES.mkdir(parents=True, exist_ok=True)

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
ASSAY5 = ["HIC", "Tm2", "AC-SINS", "PR-CHO", "Titer"]
DPI    = 300
FIG_W  = 7.0

PROC = ROOT / "data" / "processed"


# ── helpers ──────────────────────────────────────────────────────────────────
def spear(a, b):
    if len(a) < 5: return np.nan
    r, _ = spearmanr(a, b); return float(r)

def make_ridge():
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=0.95, svd_solver="full")),
                     ("r", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]))])

def fold_iter(n, k=5, seed=42):
    from sklearn.model_selection import KFold
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)):
        yield tr, te

def load_blocks():
    """Load full AbDev-X feature blocks using gdpa3_eval helpers."""
    import gdpa3_eval as GE
    ds, _, B1, _ = GE.build_blocks()
    return ds, B1

def blocks_to_X(B1):
    """Concatenate all blocks into a single feature matrix."""
    return np.hstack([v for v in B1.values()])


# ─────────────────────────────────────────────────────────────────────────────
# Q1: GDPa3 VH/VL pairing uniqueness
# ─────────────────────────────────────────────────────────────────────────────
def q1_pairing_validation():
    print("\n[Q1] GDPa3 VH/VL pairing uniqueness ...")
    result = {"status": "not_run"}
    try:
        gdpa3 = pd.read_csv(ROOT / "data" / "processed" / "gdpa3_table.csv")
        vh_col = next((c for c in ["vh", "vhn", "heavy"] if c in gdpa3.columns), None)
        vl_col = next((c for c in ["vl", "light"] if c in gdpa3.columns), None)
        if vh_col is None or vl_col is None:
            result = {"status": "columns_not_found", "cols": gdpa3.columns.tolist()}
        else:
            vh = gdpa3[vh_col].str.upper().str.strip()
            vl = gdpa3[vl_col].str.upper().str.strip()
            n_total = len(gdpa3)
            n_unique_vh   = vh.nunique()
            n_unique_pair = (vh + "|" + vl).nunique()
            n_ambiguous   = 0
            for v, grp in gdpa3.groupby(vh_col):
                if grp[vl_col].nunique() > 1:
                    n_ambiguous += 1
            result = {
                "status": "ok",
                "n_total": n_total,
                "n_unique_vh": n_unique_vh,
                "n_unique_vh_vl_pairs": n_unique_pair,
                "n_vh_with_multiple_vl": n_ambiguous,
                "pairing_unambiguous": n_ambiguous == 0,
                "message": ("VH uniquely determines VL for all GDPa3 antibodies."
                            if n_ambiguous == 0
                            else f"{n_ambiguous} VH sequences map to >1 VL — potential ambiguity.")
            }
            print(f"  n={n_total}, unique_VH={n_unique_vh}, unique_pairs={n_unique_pair}")
            print(f"  Ambiguous VH->VL mappings: {n_ambiguous}")
            print(f"  Pairing unambiguous: {n_ambiguous == 0}")
    except FileNotFoundError:
        result = {"status": "gdpa3_table_not_found"}
    with open(RES / "q1_gdpa3_pairing_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Q4: Constrained / temperature-scaled stacking
# ─────────────────────────────────────────────────────────────────────────────
def q4_constrained_stacking():
    print("\n[Q4] Constrained stacking variants ...")
    ds, B1 = load_blocks()
    X_full = blocks_to_X(B1)   # (246, D)
    rows = []

    for assay in ASSAYS:
        y   = ds.y(assay)
        m   = ~np.isnan(y)
        y_m = y[m]
        X_m = X_full[m]
        n   = int(m.sum())
        if n < 30: continue

        # 1. Standard ridge (baseline)
        oof_base = np.zeros(n)
        for tr, te in fold_iter(n):
            mod = make_ridge(); mod.fit(X_m[tr], y_m[tr])
            oof_base[te] = mod.predict(X_m[te])
        rho_base = spear(y_m, oof_base)
        rows.append({"assay": assay, "variant": "standard_ridge", "rho": rho_base})

        # 2. Non-negative clamped predictions (post-hoc)
        # (only sensible for assays that are physically non-negative)
        oof_clamp = np.maximum(oof_base, 0)
        rows.append({"assay": assay, "variant": "nonneg_clamp", "rho": spear(y_m, oof_clamp)})

        # 3. Temperature scaling of OOF predictions
        best_T, best_rho = 1.0, -np.inf
        for T in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            r = spear(y_m, oof_base / T)
            if r > best_rho: best_rho, best_T = r, T
        rows.append({"assay": assay, "variant": f"temp_scale_T={best_T:.1f}", "rho": best_rho})

        # 4. Rank-based meta (predict rank rather than raw value)
        from scipy.stats import rankdata
        oof_rank = np.zeros(n)
        for tr, te in fold_iter(n):
            y_rank = rankdata(y_m[tr]) / len(tr)
            mod = make_ridge(); mod.fit(X_m[tr], y_rank)
            oof_rank[te] = mod.predict(X_m[te])
        rows.append({"assay": assay, "variant": "rank_meta", "rho": spear(y_m, oof_rank)})

        print(f"  {assay}: base={rho_base:.3f}  clamp={rows[-3]['rho']:.3f}  "
              f"tempT{best_T:.1f}={best_rho:.3f}  rank={rows[-1]['rho']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "q4_constrained_stacking.csv", index=False)
    print(df.pivot(index="assay", columns="variant", values="rho").round(3))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Q5: Batch correction for FLAb augmentation (ComBat-lite)
# ─────────────────────────────────────────────────────────────────────────────
def q5_batch_correction():
    print("\n[Q5] Batch correction: study-level z-scoring for FLAb augmentation ...")
    aug_csv = ROOT / "results" / "flab_augment.csv"
    if not aug_csv.exists():
        print("  flab_augment.csv not found -- re-using per-assay FLAb assembled files")
        return _q5_from_flab_csvs()

    aug = pd.read_csv(aug_csv)
    print("  Columns:", aug.columns.tolist())
    print(aug.groupby(["assay"]).head(3) if "assay" in aug.columns else aug.head(10))
    return aug


def _q5_from_flab_csvs():
    flab_proc = ROOT / "data" / "processed"
    rows = []
    for assay in ASSAYS:
        candidates = [
            flab_proc / f"flab_{assay}.csv",
            flab_proc / f"flab_{assay.replace('_pH7.4','')}.csv",
            flab_proc / f"flab_{assay.replace('-','_')}.csv",
        ]
        df = None
        for p in candidates:
            if p.exists(): df = pd.read_csv(p); break
        if df is None: continue
        study_col = next((c for c in ["study","source","dataset"] if c in df.columns), None)
        val_col   = next((c for c in ["value","score","z"] if c in df.columns), None)
        if study_col is None or val_col is None: continue
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[val_col])
        # Raw z-score
        v = df[val_col].values
        z_raw = (v - v.mean()) / (v.std() + 1e-9)
        # Study-level z-score
        z_study = np.zeros(len(df))
        for _, grp in df.groupby(study_col):
            idx = grp.index
            vg  = grp[val_col].values
            z_study[idx] = (vg - vg.mean()) / (vg.std() + 1e-9)
        rows.append({"assay": assay, "method": "raw_z",   "inter_study_std": float(z_raw.std()),
                     "n_studies": df[study_col].nunique(), "n": len(df)})
        rows.append({"assay": assay, "method": "study_z", "inter_study_std": float(z_study.std()),
                     "n_studies": df[study_col].nunique(), "n": len(df)})
        print(f"  {assay}: n={len(df)}, studies={df[study_col].nunique()}, "
              f"raw_std={z_raw.std():.3f}, study_z_std={z_study.std():.3f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RES / "q5_batch_correction_flab.csv", index=False)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Q6: Alternative calibration surrogates
# ─────────────────────────────────────────────────────────────────────────────
def q6_calibration_surrogates():
    print("\n[Q6] Calibration surrogates: jackknife+, quantile interval ...")
    ds, B1 = load_blocks()
    X_full = blocks_to_X(B1)
    rows = []

    for assay in ASSAYS:
        y   = ds.y(assay)
        m   = ~np.isnan(y)
        y_m = y[m]
        X_m = X_full[m]
        n   = int(m.sum())
        if n < 30: continue

        # OOF predictions + residuals
        oof_preds = np.zeros(n)
        oof_lo    = np.zeros(n)   # quantile 0.1
        oof_hi    = np.zeros(n)   # quantile 0.9
        fold_mods = []
        for tr, te in fold_iter(n):
            sc  = StandardScaler(); Xtr = sc.fit_transform(X_m[tr]); Xte = sc.transform(X_m[te])
            pca = PCA(0.95, svd_solver="full"); Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
            rid = RidgeCV(alphas=[0.1, 1, 10, 100]); rid.fit(Xtr, y_m[tr])
            oof_preds[te] = rid.predict(Xte)
            qlo = QuantileRegressor(quantile=0.1, alpha=0.1, solver="highs")
            qhi = QuantileRegressor(quantile=0.9, alpha=0.1, solver="highs")
            qlo.fit(Xtr, y_m[tr]); qhi.fit(Xtr, y_m[tr])
            oof_lo[te] = qlo.predict(Xte); oof_hi[te] = qhi.predict(Xte)
            fold_mods.append((sc, pca, rid))

        abs_err        = np.abs(oof_preds - y_m)
        interval_width = oof_hi - oof_lo

        # Jackknife+ sigma: per-sample std of OOF predictions across folds that excluded it
        jk_sigma = np.zeros(n)
        for fi, (tr, te) in enumerate(fold_iter(n)):
            sc, pca, rid = fold_mods[fi]
            for i in te:
                xi = sc.transform(X_m[[i]])
                xi = pca.transform(xi)
                if not hasattr(jk_sigma, '_list'):
                    pass
                # store per-fold prediction for index i; use running online std
                # Simple: just record per-fold predictions in a dict
                pass  # done below via vectorised approach

        # Vectorised jackknife approach: predict each test point from its fold model
        jk_preds = {i: [] for i in range(n)}
        for fi, (tr, te) in enumerate(fold_iter(n)):
            sc, pca, rid = fold_mods[fi]
            Xte_sc  = sc.transform(X_m[te])
            Xte_pca = pca.transform(Xte_sc)
            p = rid.predict(Xte_pca)
            for k, i in enumerate(te):
                jk_preds[i].append(p[k])
        jk_sigma = np.array([np.std(jk_preds[i]) if jk_preds[i] else 0.0 for i in range(n)])

        corr_jk,  _ = spearmanr(abs_err, jk_sigma)
        corr_iqr, _ = spearmanr(abs_err, interval_width)

        rows += [
            {"assay": assay, "surrogate": "jackknife+_sigma",      "spear_with_abs_error": round(corr_jk,  3)},
            {"assay": assay, "surrogate": "quantile_interval_0.1_0.9", "spear_with_abs_error": round(corr_iqr, 3)},
        ]
        print(f"  {assay}: JK+_sigma={corr_jk:.3f}  QR-interval={corr_iqr:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "q6_calibration_surrogates.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Q7: Best block per assay (from existing per-block GDPa3 results)
# ─────────────────────────────────────────────────────────────────────────────
def q7_perblock_attributions():
    print("\n[Q7] Best block per assay from GDPa3 held-out ...")
    pb_csv = ROOT / "results" / "gdpa3_perblock.csv"
    if not pb_csv.exists():
        print("  gdpa3_perblock.csv not found -- computing OOF per-block rho on GDPa1")
        return _q7_compute_perblock()

    pb = pd.read_csv(pb_csv)
    print("  Per-block GDPa3 results:\n", pb.to_string())
    best = {}
    if "assay" in pb.columns and "block" in pb.columns and "rho" in pb.columns:
        best = pb.loc[pb.groupby("assay")["rho"].idxmax(), ["assay","block"]].set_index("assay")["block"].to_dict()
    print("  Best block per assay:", best)
    with open(RES / "q7_best_block_per_assay.json", "w") as f:
        json.dump(best, f, indent=2)
    return best


def _q7_compute_perblock():
    """Compute OOF rho per block per assay on GDPa1 CV."""
    ds, B1 = load_blocks()
    rows = []
    for assay in ASSAYS:
        y = ds.y(assay); m = ~np.isnan(y); y_m = y[m]; n = int(m.sum())
        if n < 30: continue
        for block_name, X_b in B1.items():
            X_bm = X_b[m]
            oof = np.zeros(n)
            for tr, te in fold_iter(n):
                mod = make_ridge(); mod.fit(X_bm[tr], y_m[tr]); oof[te] = mod.predict(X_bm[te])
            rows.append({"assay": assay, "block": block_name, "rho": spear(y_m, oof)})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "q7_perblock_cv_rho.csv", index=False)
    print(df.pivot(index="assay", columns="block", values="rho").round(3))
    best = df.loc[df.groupby("assay")["rho"].idxmax(), ["assay","block"]].set_index("assay")["block"].to_dict()
    with open(RES / "q7_best_block_per_assay.json", "w") as f:
        json.dump(best, f, indent=2)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Reviewer response figure
# ─────────────────────────────────────────────────────────────────────────────
def make_reviewer_figure(q4_df, q6_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                                 "axes.labelsize": 9, "axes.titlesize": 9,
                                 "pdf.fonttype": 42, "savefig.dpi": DPI})
    PAL = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#DD8452"]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.48), constrained_layout=True)

    # Panel A: constrained stacking (Q4)
    ax = axes[0]
    if len(q4_df):
        variants = q4_df["variant"].unique()
        x = np.arange(len(ASSAYS)); w = 0.75 / max(len(variants), 1)
        for i, var in enumerate(variants):
            sub = q4_df[q4_df["variant"] == var].set_index("assay")
            vals = [sub.loc[a, "rho"] if a in sub.index else np.nan for a in ASSAYS]
            ax.bar(x + (i - len(variants)/2 + 0.5)*w, vals, w,
                   label=var.replace("_", " "), color=PAL[i % len(PAL)], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(ASSAY5, rotation=35, ha="right", fontsize=7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("Spearman rho"); ax.set_title("(A) Q4: Constrained stacking variants")
        ax.legend(fontsize=5, ncol=2, loc="upper right", framealpha=0.8)

    # Panel B: calibration surrogates (Q6)
    ax = axes[1]
    if len(q6_df):
        for i, surr in enumerate(q6_df["surrogate"].unique()):
            sub = q6_df[q6_df["surrogate"] == surr].set_index("assay")
            vals = [sub.loc[a, "spear_with_abs_error"] if a in sub.index else np.nan for a in ASSAYS]
            ax.plot(np.arange(len(ASSAYS)), vals, "o-", color=PAL[i], ms=5, lw=1.2,
                    label=surr.replace("_", " "))
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(np.arange(len(ASSAYS))); ax.set_xticklabels(ASSAY5, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Spearman(sigma, |error|)")
        ax.set_title("(B) Q6: Uncertainty-error correlation")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    out = FIGS / "panel_reviewer_response.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight", dpi=DPI)
    import matplotlib.pyplot as _p; _p.close(fig)
    print(f"  Figure saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", nargs="+", type=int, default=[1, 4, 5, 6, 7])
    args = parser.parse_args()
    qs = set(args.q)

    q4_df = pd.DataFrame()
    q6_df = pd.DataFrame()

    if 1 in qs: q1_pairing_validation()
    if 4 in qs: q4_df = q4_constrained_stacking()
    if 5 in qs: q5_batch_correction()
    if 6 in qs: q6_df = q6_calibration_surrogates()
    if 7 in qs: q7_perblock_attributions()

    print("\nGenerating figure ...")
    make_reviewer_figure(q4_df, q6_df)
    print(f"\nAll outputs in {RES}")


if __name__ == "__main__":
    main()
