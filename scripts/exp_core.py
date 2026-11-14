"""Core experiments: (1) leave-one-representation-out ablation, (2) learning curves,
(3) paired-bootstrap significance, (5) CDR-vs-framework predictivity.

All on the AbDev-X block set built by gdpa3_eval.build_blocks (esm2, descriptors, isotype, bio,
ablang2). CV = per-fold-average Spearman on cluster-isotype; held-out = GDPa1->GDPa3.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE
from abdev.models.fusion import AbDevFusion, DEFAULT_BASES, BASE_BLOCK
from abdev.features.descriptors import build_descriptor_matrix

ASSAYS = GE.ASSAYS
REPS = ["esm2", "descriptors", "isotype", "bio", "ablang2"]
FIG = ROOT / "results/figures"


def cv_perfold(ds, blocks, y, bases, split="cluster_isotype"):
    folds = ds.folds(split); per = []
    for f in sorted(set(folds.tolist())):
        tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        Btr = {k: v[tr] for k, v in blocks.items()}; Bte = {k: v[te] for k, v in blocks.items()}
        m = AbDevFusion(bases=bases).fit(Btr, y[tr])
        per.append(spearmanr(y[te], m.predict(Bte)).statistic)
    return float(np.mean(per))


def heldout(B1, B3, y1, y3, bases):
    m = ~np.isnan(y1); ok = ~np.isnan(y3)
    B1m = {k: v[m] for k, v in B1.items()}
    pred = AbDevFusion(bases=bases).fit(B1m, y1[m]).predict(B3)
    return float(spearmanr(y3[ok], pred[ok]).statistic), pred, ok


def bases_without(rep):
    return [b for b in DEFAULT_BASES if BASE_BLOCK[b[0]] != rep]


def main():
    ds, g3, B1, B3 = GE.build_blocks()
    B1 = {k: np.nan_to_num(v) for k, v in B1.items()}; B3 = {k: np.nan_to_num(v) for k, v in B3.items()}
    Y3 = {a: g3[GE.LABEL_MAP[a]].to_numpy(float) for a in ASSAYS}

    # ---------------- (1) leave-one-representation-out ----------------
    rows = []
    full_cv, full_ho = {}, {}
    for a in ASSAYS:
        y = ds.y(a)
        full_cv[a] = cv_perfold(ds, B1, y, DEFAULT_BASES)
        full_ho[a] = heldout(B1, B3, y, Y3[a], DEFAULT_BASES)[0]
    for rep in REPS:
        bs = bases_without(rep)
        for a in ASSAYS:
            y = ds.y(a)
            cv = cv_perfold(ds, B1, y, bs); ho = heldout(B1, B3, y, Y3[a], bs)[0]
            rows.append(dict(dropped=rep, assay=a, dCV=round(full_cv[a] - cv, 3), dHeldout=round(full_ho[a] - ho, 3)))
    abl = pd.DataFrame(rows); abl.to_csv(ROOT / "results/exp_ablation.csv", index=False)
    print("=== (1) Leave-one-out: mean drop when each representation is removed ===")
    print(abl.groupby("dropped")[["dCV", "dHeldout"]].mean().round(3).to_string())

    # ablation heatmap (dHeldout)
    piv = abl.pivot(index="dropped", columns="assay", values="dHeldout").reindex(REPS)[ASSAYS]
    fig, ax = plt.subplots(figsize=(8, 4)); im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ASSAYS))); ax.set_xticklabels(ASSAYS, rotation=20, ha="right")
    ax.set_yticks(range(len(REPS))); ax.set_yticklabels(REPS)
    for i in range(len(REPS)):
        for j in range(len(ASSAYS)):
            ax.text(j, i, f"{piv.to_numpy()[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Held-out Spearman drop when representation removed"); fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(FIG / "figS2_ablation.png", dpi=160); plt.close(fig)

    # ---------------- (2) learning curves ----------------
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]; lc_rows = []
    folds = ds.folds("cluster_isotype")
    for a in ASSAYS:
        y = ds.y(a)
        for fr in fracs:
            per = []
            for f in sorted(set(folds.tolist())):
                tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
                if len(te) < 3:
                    continue
                rng = np.random.RandomState(0); sub = rng.choice(tr, max(10, int(fr * len(tr))), replace=False)
                Btr = {k: v[sub] for k, v in B1.items()}; Bte = {k: v[te] for k, v in B1.items()}
                per.append(spearmanr(y[te], AbDevFusion().fit(Btr, y[sub]).predict(Bte)).statistic)
            lc_rows.append(dict(assay=a, frac=fr, rho=round(float(np.mean(per)), 3)))
    lc = pd.DataFrame(lc_rows); lc.to_csv(ROOT / "results/exp_learning_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); cols = [plt.cm.viridis(v) for v in np.linspace(.12, .88, len(ASSAYS))]
    for a, c in zip(ASSAYS, cols):
        sub = lc[lc.assay == a]; ax.plot(sub.frac, sub.rho, "-o", color=c, label=a)
    ax.set_xlabel("Training-set fraction"); ax.set_ylabel("CV Spearman ρ"); ax.set_title("Learning curves (AbDev-X, cluster-isotype CV)")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(FIG / "figS3_learning_curves.png", dpi=160); plt.close(fig)
    print("\n=== (2) Learning curves (rho at 20% vs 100% train) ===")
    print(lc.pivot(index="assay", columns="frac", values="rho")[[0.2, 1.0]].to_string())

    # ---------------- (3) paired bootstrap significance (held-out) ----------------
    def block_pred(y1, block, B1b, B3b):
        from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA; from sklearn.linear_model import RidgeCV
        m = ~np.isnan(y1)
        steps = [("s", StandardScaler())]
        if block in ("esm2", "ablang2"):
            steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
        steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
        return Pipeline(steps).fit(B1b[block][m], y1[m]).predict(B3b[block])

    sig_rows = []
    for a in ASSAYS:
        y1 = ds.y(a); y3 = Y3[a]; ok = ~np.isnan(y3)
        ab = heldout(B1, B3, y1, y3, DEFAULT_BASES)[1][ok]
        esm = block_pred(y1, "esm2", B1, B3)[ok]; abl = block_pred(y1, "ablang2", B1, B3)[ok]
        yv = y3[ok]; rng = np.random.RandomState(0); n = len(yv)
        for name, base in [("ESM2", esm), ("AbLang2", abl)]:
            d = []
            for _ in range(2000):
                bi = rng.choice(n, n, replace=True)
                if len(np.unique(yv[bi])) < 3:
                    continue
                d.append(spearmanr(yv[bi], ab[bi]).statistic - spearmanr(yv[bi], base[bi]).statistic)
            d = np.array(d); p = float(np.mean(d <= 0))  # one-sided: AbDev-X <= baseline
            sig_rows.append(dict(assay=a, vs=name, delta_rho=round(float(np.mean(d)), 3),
                                 ci_low=round(float(np.percentile(d, 2.5)), 3),
                                 ci_high=round(float(np.percentile(d, 97.5)), 3), p_one_sided=round(p, 3)))
    sig = pd.DataFrame(sig_rows); sig.to_csv(ROOT / "results/exp_significance.csv", index=False)
    print("\n=== (3) Held-out paired bootstrap: AbDev-X - baseline (Delta rho [95% CI], p) ===")
    print(sig.to_string(index=False))

    # ---------------- (5) CDR vs framework predictivity ----------------
    D = build_descriptor_matrix(ds.heavy, ds.light)
    cdr_cols = [c for c in D.columns if "CDR" in c and any(k in c for k in ["gravy", "hmoment", "aromatic", "charge", "frac_pos", "frac_neg", "pI"])]
    fw_cols = [c for c in D.columns if "_FR_" in c and any(k in c for k in ["gravy", "hmoment", "aromatic", "charge", "frac_pos", "frac_neg", "pI"])]
    cf_rows = []
    for a in ASSAYS:
        y = ds.y(a); m = ~np.isnan(y)
        cdr_r = np.nanmean([abs(spearmanr(D[c].to_numpy()[m], y[m]).statistic) for c in cdr_cols])
        fw_r = np.nanmean([abs(spearmanr(D[c].to_numpy()[m], y[m]).statistic) for c in fw_cols])
        cf_rows.append(dict(assay=a, CDR_meanabs_rho=round(float(cdr_r), 3), FR_meanabs_rho=round(float(fw_r), 3),
                            CDR_wins=bool(cdr_r > fw_r)))
    cf = pd.DataFrame(cf_rows); cf.to_csv(ROOT / "results/exp_cdr_vs_fw.csv", index=False)
    print("\n=== (5) CDR vs framework mean|rho| with each assay ===")
    print(cf.to_string(index=False))


if __name__ == "__main__":
    main()
