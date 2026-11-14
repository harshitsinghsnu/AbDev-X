"""
Physics-Informed Constrained Heads for AbDev-X.

Three experiments:
  1. Isotonic-constrained HIC: project predictions to be monotone with SAP descriptor
  2. Rank-loss regularized AC-SINS: align prediction ordering with charge descriptor
  3. Multi-task shared latent: single PCA encoder, assay-specific ridge heads

All use the same AbDev-X feature blocks (descriptors + ESM2 + AbLang2 + isotype + bio).

Usage:
  python scripts/physics_constrained.py

Outputs:
  results/physics_constrained/physics_hic.csv
  results/physics_constrained/physics_rank_reg.csv
  results/physics_constrained/multitask_cv.csv
  paper/figures/panel_physics_constrained.pdf
"""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata
from sklearn.linear_model import RidgeCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RES  = ROOT / "results" / "physics_constrained"
FIGS = ROOT / "paper" / "figures"
RES.mkdir(parents=True, exist_ok=True)

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
ASSAY5 = ["HIC", "Tm2", "AC-SINS", "PR-CHO", "Titer"]
DPI    = 300
FIG_W  = 7.0

AA_HYDRO   = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
               "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
               "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
AA_CHARGE  = {"R":1.0,"K":1.0,"H":0.1,"D":-1.0,"E":-1.0}


# ── helpers ──────────────────────────────────────────────────────────────────
def spear(a, b):
    if len(a) < 5: return np.nan
    r, _ = spearmanr(a, b); return float(r)

def fold_iter(n, k=5, seed=42):
    from sklearn.model_selection import KFold
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)):
        yield tr, te

def make_ridge():
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=0.95, svd_solver="full")),
                     ("r", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]))])

def sap_proxy(seqs):
    return np.array([np.mean([AA_HYDRO.get(aa, 0.0) for aa in str(s).upper() if aa in AA_HYDRO] or [0.0])
                     for s in seqs])

def charge_proxy(seqs):
    return np.array([sum(AA_CHARGE.get(aa, 0.0) for aa in str(s).upper()) for s in seqs])


# ── Experiment 1: Isotonic-constrained HIC ───────────────────────────────────
def exp_isotonic_hic(ds, B1):
    print("\n[Physics] Exp 1: Isotonic-constrained HIC prediction ...")
    y = ds.y("HIC"); m = ~np.isnan(y); y_m = y[m]; n = int(m.sum())
    if n < 30: return []

    X_m  = np.hstack([v[m] for v in B1.values()])
    seqs = (ds.heavy + ds.light).values[m]
    sap  = sap_proxy(seqs)

    oof_base = np.zeros(n); oof_iso = np.zeros(n)
    for tr, te in fold_iter(n):
        mod = make_ridge(); mod.fit(X_m[tr], y_m[tr])
        pred_raw     = mod.predict(X_m[te])
        oof_base[te] = pred_raw
        # Isotonic: force prediction to be monotone-increasing with SAP
        order    = np.argsort(sap[te])
        inv_ord  = np.argsort(order)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(sap[te][order], pred_raw[order])
        oof_iso[te] = iso.predict(sap[te])[inv_ord]

    rho_base = spear(y_m, oof_base)
    rho_iso  = spear(y_m, oof_iso)
    kt_base, _ = kendalltau(sap, oof_base)
    kt_iso,  _ = kendalltau(sap, oof_iso)
    print(f"  HIC: baseline={rho_base:.3f}  isotonic={rho_iso:.3f}")
    print(f"  KendallTau(pred, SAP): base={kt_base:.3f}  iso={kt_iso:.3f}")
    return [
        {"assay": "HIC", "method": "baseline_ridge", "rho": rho_base, "kt_sap": kt_base},
        {"assay": "HIC", "method": "isotonic_sap",   "rho": rho_iso,  "kt_sap": kt_iso},
    ]


# ── Experiment 2: Rank-loss regularized AC-SINS / PR_CHO ─────────────────────
def exp_rank_regularized(ds, B1):
    print("\n[Physics] Exp 2: Rank-loss regularized prediction ...")
    rows = []
    for assay, ref_fn in [("AC-SINS_pH7.4", charge_proxy), ("PR_CHO", charge_proxy)]:
        y = ds.y(assay); m = ~np.isnan(y); y_m = y[m]; n = int(m.sum())
        if n < 30: continue
        X_m  = np.hstack([v[m] for v in B1.values()])
        seqs = (ds.heavy + ds.light).values[m]
        ref  = ref_fn(seqs)

        for lam in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]:
            oof = np.zeros(n)
            for tr, te in fold_iter(n):
                if lam > 0:
                    rank_ref = rankdata(ref[tr]) / len(tr)
                    rank_y   = rankdata(y_m[tr]) / len(tr)
                    y_aug    = (1 - lam) * rank_y + lam * rank_ref
                else:
                    y_aug = y_m[tr]
                mod = make_ridge(); mod.fit(X_m[tr], y_aug)
                oof[te] = mod.predict(X_m[te])
            rho = spear(y_m, oof)
            rows.append({"assay": assay, "lambda": lam, "rho": rho})
        bdf = pd.DataFrame([r for r in rows if r["assay"] == assay])
        print(f"  {assay}: best_rho={bdf['rho'].max():.3f} at lam={bdf.loc[bdf['rho'].idxmax(),'lambda']}")
    return rows


# ── Experiment 3: Multi-task shared latent space ─────────────────────────────
def exp_multitask_shared(ds, B1):
    print("\n[Physics] Exp 3: Multi-task shared latent space ...")
    X_all = np.hstack([v for v in B1.values()])   # (246, D)
    sc  = StandardScaler(); X_sc = sc.fit_transform(X_all)
    pca = PCA(n_components=0.95, svd_solver="full"); Z = pca.fit_transform(X_sc)
    print(f"  Shared PCA: {X_all.shape[1]} -> {Z.shape[1]} dims "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    rows = []
    for assay in ASSAYS:
        y = ds.y(assay); m = ~np.isnan(y); y_m = y[m]; n = int(m.sum())
        if n < 30: continue
        Z_m = Z[m]; X_m = X_all[m]

        # Shared-PCA ridge
        oof_shared = np.zeros(n)
        for tr, te in fold_iter(n):
            rid = RidgeCV(alphas=[0.01,0.1,1,10,100,1000])
            rid.fit(Z_m[tr], y_m[tr]); oof_shared[te] = rid.predict(Z_m[te])
        rho_shared = spear(y_m, oof_shared)

        # Per-assay ridge (separate PCA per assay -- standard AbDev-X style)
        oof_indiv = np.zeros(n)
        for tr, te in fold_iter(n):
            mod = make_ridge(); mod.fit(X_m[tr], y_m[tr]); oof_indiv[te] = mod.predict(X_m[te])
        rho_indiv = spear(y_m, oof_indiv)

        rows.append({"assay": assay, "rho_individual": rho_indiv,
                     "rho_shared": rho_shared, "delta": rho_shared - rho_indiv})
        print(f"  {assay}: individual={rho_indiv:.3f}  shared_latent={rho_shared:.3f}  "
              f"delta={rho_shared-rho_indiv:+.3f}")
    return rows


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(hic_rows, rank_rows, mt_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                                 "axes.labelsize": 9, "axes.titlesize": 9, "pdf.fonttype": 42})
    PAL = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_W * 0.52), constrained_layout=True)

    ax = axes[0]
    if hic_rows:
        df = pd.DataFrame(hic_rows)
        ax.bar(np.arange(len(df)), df["rho"], color=[PAL[0], PAL[1]], width=0.5)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df["method"].str.replace("_", "\n"), fontsize=7, rotation=0)
        ax.set_ylabel("Spearman rho"); ax.set_title("(A) Isotonic HIC vs SAP")
        ax.axhline(0, color="grey", lw=0.5)
        for i, r in enumerate(df.itertuples()):
            ax.text(i, r.rho + 0.01, f"{r.rho:.3f}", ha="center", fontsize=7)

    ax = axes[1]
    if rank_rows:
        df = pd.DataFrame(rank_rows)
        for i, assay in enumerate(df["assay"].unique()):
            sub = df[df["assay"] == assay]
            lbl = assay.replace("_pH7.4", "").replace("_", "-")
            ax.plot(sub["lambda"], sub["rho"], "o-", color=PAL[i], ms=5, lw=1.2, label=lbl)
        ax.set_xlabel("Rank-loss lambda"); ax.set_ylabel("Spearman rho")
        ax.set_title("(B) Rank-loss regularization")
        ax.axhline(0, color="grey", lw=0.5); ax.legend(fontsize=6, loc="best", framealpha=0.8)

    ax = axes[2]
    if mt_rows:
        df  = pd.DataFrame(mt_rows)
        x   = np.arange(len(df))
        short_lbl = df["assay"].str.replace("_pH7.4","").str.replace("AC-SINS","ACSINS").str.replace("_","-")
        ax.bar(x - 0.2, df["rho_individual"], 0.38, color=PAL[0], label="Per-assay")
        ax.bar(x + 0.2, df["rho_shared"],     0.38, color=PAL[1], label="Shared")
        ax.set_xticks(x)
        ax.set_xticklabels(short_lbl, fontsize=6, rotation=35, ha="right")
        ax.set_ylabel("Spearman rho"); ax.set_title("(C) Multi-task shared latent")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8); ax.axhline(0, color="grey", lw=0.5)
    out = FIGS / "panel_physics_constrained.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight", dpi=DPI)
    import matplotlib.pyplot as _p; _p.close(fig)
    print(f"  Figure: {out.name}")


def main():
    import gdpa3_eval as GE
    ds, _, B1, _ = GE.build_blocks()

    hic_rows  = exp_isotonic_hic(ds, B1)
    rank_rows = exp_rank_regularized(ds, B1)
    mt_rows   = exp_multitask_shared(ds, B1)

    pd.DataFrame(hic_rows).to_csv(RES / "physics_hic.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(RES / "physics_rank_reg.csv", index=False)
    pd.DataFrame(mt_rows).to_csv(RES / "multitask_cv.csv", index=False)

    make_figure(hic_rows, rank_rows, mt_rows)
    print(f"\nResults in {RES}")


if __name__ == "__main__":
    main()
