"""Generate publication-ready figures from the committed result CSVs."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = R / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.labelsize": 12, "figure.dpi": 150, "savefig.dpi": 200})
ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
LBL = {"HIC": "HIC", "Tm2": "Tm2", "AC-SINS_pH7.4": "AC-SINS", "PR_CHO": "PR-CHO", "Titer": "Titer"}
COMP = {"HIC": 0.708, "Tm2": 0.392, "AC-SINS_pH7.4": 0.356, "PR_CHO": 0.337, "Titer": 0.310}
RED = "#C44E52"


def SEQ(n):
    """Sequential (viridis) colors for grouped bar graphs."""
    return [plt.cm.viridis(v) for v in np.linspace(0.12, 0.88, n)]


EBAR = dict(capsize=3, ecolor="#333333", error_kw={"lw": 1.1})


def fig_generalization():
    cv = pd.read_csv(R / "cv_errorbars.csv")
    cvx = cv[cv.model == "AbDev-X"].set_index("assay").reindex(ASSAYS)
    hb = pd.read_csv(R / "heldout_errorbars.csv").set_index("assay").reindex(ASSAYS)
    x = np.arange(len(ASSAYS)); w = 0.38
    c_cv, c_ho = SEQ(2)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, cvx["mean"], w, yerr=cvx["std"], color=c_cv, label="AbDev-X: GDPa1 CV (mean±std)", **EBAR)
    err = np.vstack([hb["heldout_mean"] - hb["ci_low"], hb["ci_high"] - hb["heldout_mean"]])
    ax.bar(x + w / 2, hb["heldout_mean"], w, yerr=err, color=c_ho, label="AbDev-X: GDPa3 held-out (95% CI)", **EBAR)
    ax.scatter(x + w / 2, [COMP[a] for a in ASSAYS], color=RED, marker="D", zorder=6, s=55, label="Competition champion (held-out)")
    ax.set_xticks(x); ax.set_xticklabels([LBL[a] for a in ASSAYS]); ax.set_ylabel("Spearman ρ")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("Cross-validation vs. real held-out generalization (GDPa1 → GDPa3)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=True, fontsize=9, borderpad=0.4,
               bbox_transform=fig.transFigure)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(FIG / "fig2_generalization.png", bbox_inches="tight"); plt.close(fig)


def fig_model_comparison():
    df = pd.read_csv(R / "model_comparison_meanstd.csv")
    order = ["AbDev-X (stacking)", "ESM2-Ridge", "AbLang2-ElasticNet", "SaProt-Ridge"]
    splits = ["random", "cluster", "cluster_isotype"]; x = np.arange(len(splits)); w = 0.2
    colors = SEQ(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, m in enumerate(order):
        sub = df[df.model == m].set_index("split").reindex(splits)
        ax.bar(x + (i - 1.5) * w, sub["mean"], w, yerr=sub["std"], color=colors[i], label=m, **EBAR)
    ax.set_xticks(x); ax.set_xticklabels(["random\n(IID)", "cluster\n(OOD)", "cluster-isotype\n(OOD)"])
    ax.set_ylabel("Mean Spearman ρ (CV, mean±std)")
    ax.set_title("GDPa1 CV across generalization splits")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=True, fontsize=9, borderpad=0.4,
               bbox_transform=fig.transFigure)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(FIG / "fig1_model_comparison.png", bbox_inches="tight"); plt.close(fig)


def fig_plm_ablation():
    df = pd.read_csv(R / "plm_ablation.csv").sort_values("MEAN", ascending=False)
    x = np.arange(len(ASSAYS)); n = len(df); w = 0.8 / n; colors = SEQ(n)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, (_, r) in enumerate(df.iterrows()):
        ax.bar(x + (i - n / 2 + 0.5) * w, [r[a] for a in ASSAYS], w,
               yerr=[r[a + "_std"] for a in ASSAYS], color=colors[i],
               label=f"{r['PLM']} ({r['MEAN']})", **EBAR)
    ax.set_xticks(x); ax.set_xticklabels([LBL[a] for a in ASSAYS])
    ax.set_ylabel("Spearman ρ (CV, mean±std)"); ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("PLM ablation (GDPa1 cluster-isotype CV, ridge head)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, frameon=True, fontsize=8, borderpad=0.4,
               bbox_transform=fig.transFigure)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(FIG / "fig3_plm_ablation.png", bbox_inches="tight"); plt.close(fig)


def fig_repweights():
    w = pd.read_csv(R / "representation_weights.csv").set_index("assay").reindex(ASSAYS)
    cols = [c for c in w.columns]
    M = w[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(ASSAYS))); ax.set_yticklabels([LBL[a] for a in ASSAYS])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Per-assay stacking meta-weights (biological interpretation)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="meta-weight"); fig.tight_layout(); fig.savefig(FIG / "fig4_repweights.png"); plt.close(fig)


def fig_explainability_panel():
    imgs = [(R / "explain/shap_HIC.png", "SHAP feature attribution (HIC)"),
            (R / "explain/structures/brazikumab_hic.png", "Hydrophobic surface hotspots (HIC)"),
            (R / "explain/structures/zolbetuximab_charge.png", "Surface charge patches (AC-SINS)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (p, t) in zip(axes, imgs):
        if p.exists():
            ax.imshow(mpimg.imread(p)); ax.set_title(t, fontsize=12)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "fig5_explainability.png"); plt.close(fig)


def main():
    fig_model_comparison(); fig_generalization(); fig_plm_ablation(); fig_repweights(); fig_explainability_panel()
    print("Figures written to", FIG)
    for f in sorted(FIG.glob("*.png")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
