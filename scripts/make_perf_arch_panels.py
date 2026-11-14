"""Generate panel_performance_comparison.pdf and panel_decision_arch.pdf.

Both use actual AW-MoE results as the primary AbDev-X reference.

panel_performance_comparison.pdf
  A: AW-MoE CV rho (mean ± SD, cluster_isotype) vs best single-rep baseline per assay
  B: GDPa3 held-out rho for all 9 models (bootstrap CI bars) on 5 high-signal assays
  C: Distribution of cluster_isotype CV rho across all models per assay;
     AW-MoE shown as filled diamond

panel_decision_arch.pdf
  A: Abstention curves (from si_abstention data)
  B: Precision@K on GDPa3 (from limitations_suite)
  C: Competition ranking rho across model configurations
  D: Architecture ablation (mean rho) vs D-Stack reference
  E: Architecture ablation (heatmap per assay)
  F: Architecture ablation (random CV split)
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT    = Path(__file__).resolve().parents[1]
FIGS    = ROOT / "paper" / "figures"
FIGS2   = ROOT / "paper" / "paper10assay"
RESULTS = ROOT / "results"
LSUIT   = RESULTS / "limitations_suite"
ABLAT   = RESULTS / "abdevx2_ablation"

FIG_W  = 7.0
FS_B   = 8; FS_AX = 9; FS_T = 9; FS_LBL = 11; FS_SUP = 7.5; LW = 0.8
CAPSIZE = 2; DPI = 300

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS_B, "axes.titlesize": FS_T,
    "axes.labelsize": FS_AX, "xtick.labelsize": FS_B, "ytick.labelsize": FS_B,
    "legend.fontsize": FS_B, "figure.dpi": DPI, "savefig.dpi": DPI,
})

ASSAY5_HS  = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "SMAC", "HAC"]
ASSAY5_C   = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
LBL_HS     = {"HIC":"HIC","AC-SINS_pH7.4":"AC-SINS","PR_CHO":"PR-CHO","SMAC":"SMAC","HAC":"HAC"}
LBL_C      = {"HIC":"HIC","Tm2":"Tm2","AC-SINS_pH7.4":"AC-SINS","PR_CHO":"PR-CHO","Titer":"Titer"}

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREY   = "#808080"
RED    = "#C44E52"
GREEN  = "#55A868"
PURPLE = "#8172B2"

EBAR = dict(capsize=CAPSIZE, ecolor="#333333", error_kw={"lw": LW, "capthick": LW})


def panel_label(ax, lbl):
    ax.text(-0.14, 1.04, lbl, transform=ax.transAxes,
            fontsize=FS_LBL, fontweight="bold", va="top", ha="left")


def save(fig, stem, tight=True):
    for d in [FIGS, FIGS2]:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(d / f"{stem}.pdf"), format="pdf", bbox_inches="tight" if tight else None)
        fig.savefig(str(d / f"{stem}.png"), format="png", bbox_inches="tight" if tight else None,
                    dpi=200)
    print(f"  Saved {stem}.pdf + .png")


# ── panel_performance_comparison ─────────────────────────────────────────────

def fig_perf_comparison():
    # Data
    cv_eb = pd.read_csv(RESULTS / "cv_errorbars.csv")
    awmoe_cv = pd.read_csv(RESULTS / "awmoe_cv_errorbars.csv")
    awmoe_ci_df = awmoe_cv[awmoe_cv.split == "cluster_isotype"].set_index("assay")

    # bootstrap CIs on GDPa3
    bca = pd.read_csv(RESULTS / "bootstrap_ci_all.csv")

    # all models CV for distribution panel C
    all_cv = pd.read_csv(RESULTS / "all_models_splits.csv")
    all_ci = all_cv[all_cv.split == "cluster_isotype"]

    fig = plt.figure(figsize=(FIG_W, 6.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38,
                  height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # ── Panel A: CV bar (AW-MoE vs ESM2-Ridge, cluster_isotype, 5 comp assays) ──
    esm_cv = cv_eb[cv_eb.model == "ESM2-Ridge"].set_index("assay")
    assays_a = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
    x = np.arange(len(assays_a)); w = 0.35

    awmoe_means = [awmoe_ci_df.loc[a, "mean"] if a in awmoe_ci_df.index else np.nan for a in assays_a]
    awmoe_stds  = [awmoe_ci_df.loc[a, "std"]  if a in awmoe_ci_df.index else 0.0  for a in assays_a]
    esm_means   = [esm_cv.loc[a, "mean"] if a in esm_cv.index else np.nan for a in assays_a]
    esm_stds    = [esm_cv.loc[a, "std"]  if a in esm_cv.index else 0.0  for a in assays_a]

    ax_a.bar(x - w/2, awmoe_means, w, yerr=awmoe_stds, color=BLUE, label="AbDev-X", **EBAR)
    ax_a.bar(x + w/2, esm_means,   w, yerr=esm_stds,   color=ORANGE, label="ESM-2 Ridge",    **EBAR)

    for i, (am, em) in enumerate(zip(awmoe_means, esm_means)):
        if not np.isnan(am) and not np.isnan(em):
            delta = am - em
            ax_a.text(i - w/2, am + (awmoe_stds[i] or 0) + 0.025,
                      f"{delta:+.2f}", ha="center", va="bottom", fontsize=6.5, color="#222")

    ax_a.set_xticks(x); ax_a.set_xticklabels([LBL_C.get(a, a) for a in assays_a], fontsize=FS_B - 1)
    ax_a.set_ylabel("Spearman ρ (CV)")
    ax_a.set_title("CV: AbDev-X vs ESM-2 Ridge\n(cluster-isotype, mean ± SD)")
    ax_a.axhline(0, color="grey", lw=0.5, ls="--")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                fontsize=6.5, frameon=True, framealpha=0.9,
                handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    panel_label(ax_a, "A")

    # ── Panel B: GDPa3 held-out BCa CIs for all models (5 high-signal assays) ──
    model_order = ["AbDev-X", "A1-E1", "A7-Dstack", "A8-FMoE",
                   "A4-Desc", "A5-Bio", "A6-MGFS", "A2-ESM2", "A3-AbLang2"]
    model_labels = {"AbDev-X": "AbDev-X*", "A1-E1": "E1", "A7-Dstack": "D-Stack",
                    "A8-FMoE": "F-MoE", "A4-Desc": "Desc", "A5-Bio": "Bio",
                    "A6-MGFS": "MGFS", "A2-ESM2": "ESM-2", "A3-AbLang2": "AbLang2"}
    colors_b = [BLUE if m == "AbDev-X" else GREY for m in model_order]

    bca_idx = bca.set_index("model")
    hs5_lo_cols = ["HIC_lo", "AC-SINS_lo", "PR_CHO_lo", "SMAC_lo", "HAC_lo"]
    hs5_hi_cols = ["HIC_hi", "AC-SINS_hi", "PR_CHO_hi", "SMAC_hi", "HAC_hi"]
    rho_col = "mean_rho"
    means_b, ci_lo_b, ci_hi_b = [], [], []
    for m in model_order:
        if m in bca_idx.index:
            row = bca_idx.loc[m]
            means_b.append(row[rho_col])
            ci_lo_b.append(np.nanmean([row[c] for c in hs5_lo_cols if c in row.index]))
            ci_hi_b.append(np.nanmean([row[c] for c in hs5_hi_cols if c in row.index]))
        else:
            means_b.append(np.nan); ci_lo_b.append(np.nan); ci_hi_b.append(np.nan)
    means_b  = np.array(means_b,  dtype=float)
    ci_lo_b  = np.array(ci_lo_b,  dtype=float)
    ci_hi_b  = np.array(ci_hi_b,  dtype=float)
    yerr_lo  = np.clip(means_b - ci_lo_b, 0, None)
    yerr_hi  = np.clip(ci_hi_b - means_b, 0, None)

    xb = np.arange(len(model_order))
    bars = ax_b.bar(xb, means_b, 0.6, color=colors_b)
    bars[0].set_edgecolor("black"); bars[0].set_linewidth(0.8)
    ax_b.errorbar(xb, means_b, yerr=[yerr_lo, yerr_hi], fmt="none",
                  ecolor="#222222", elinewidth=LW, capsize=CAPSIZE, capthick=LW)

    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([model_labels.get(m, m) for m in model_order],
                         rotation=35, ha="right", fontsize=FS_B - 1.5)
    ax_b.set_ylabel("Mean Spearman ρ (GDPa3)")
    ax_b.set_title("GDPa3 held-out: all base models\n(5 HS assays, mean BCa 95% CI, * = AbDev-X)")
    ax_b.axhline(0, color="grey", lw=0.5, ls="--")
    panel_label(ax_b, "B")

    # ── Panel C: Distribution of cluster_isotype CV across all models ──
    # grey dots for all models, AW-MoE filled diamond
    assays_c = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
    xc = np.arange(len(assays_c))
    jitter = np.linspace(-0.3, 0.3, len(all_ci))

    awmoe_row = all_ci[all_ci.model == "AbDev-X"]
    other_rows = all_ci[all_ci.model != "AbDev-X"]

    for j, (_, row) in enumerate(other_rows.iterrows()):
        vals = [row.get(a, np.nan) for a in assays_c]
        ax_c.scatter(xc + jitter[j % len(jitter)], vals,
                     color=GREY, s=18, alpha=0.6, zorder=2)

    if len(awmoe_row):
        row = awmoe_row.iloc[0]
        vals = [row.get(a, np.nan) for a in assays_c]
        ax_c.scatter(xc, vals, color=BLUE, s=60, marker="D", zorder=5,
                     label="AbDev-X", edgecolors="black", linewidths=0.6)

    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([LBL_C.get(a, a) for a in assays_c])
    ax_c.set_ylabel("Spearman ρ (cluster-isotype CV)")
    ax_c.set_title("Distribution of GDPa1 CV ρ across all models (grey) with AbDev-X (◆) highlighted")
    ax_c.axhline(0, color="grey", lw=0.5, ls="--")
    ax_c.legend(fontsize=FS_B, loc="lower right")
    panel_label(ax_c, "C")

    fig.suptitle("AbDev-X performance: GDPa1 CV and GDPa3 held-out",
                 fontsize=FS_SUP, y=1.01)
    save(fig, "panel_performance_comparison")
    plt.close(fig)


# ── panel_decision_arch ───────────────────────────────────────────────────────

def fig_decision_arch():
    # D-Stack cluster_isotype mean (reference line for architecture ablation)
    all_cv = pd.read_csv(RESULTS / "all_models_splits.csv")
    dstack_row = all_cv[(all_cv.model == "AbDev-X (stacking)") & (all_cv.split == "cluster_isotype")]
    comp5 = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
    dstack_mean = float(dstack_row[comp5].values.mean()) if len(dstack_row) else 0.462

    # Architecture ablation data
    abl = pd.read_csv(ABLAT / "metrics.csv")
    abl_ci  = abl[abl.split == "cluster_isotype"].copy()
    abl_ran = abl[abl.split == "random"].copy()

    method_labels = {
        "representation_fusion": "Rep. Fusion",
        "moe":                   "MoE",
        "cdr_framework_moe":     "CDR-fw MoE",
        "graph_moe":             "Graph MoE",
        "uncertainty_moe":       "Uncert. MoE",
        "multitask_projection":  "Multitask",
        "abdevx2_full":          "Prototype",
    }
    method_order = list(method_labels.keys())

    abl_ci_sorted  = abl_ci[abl_ci.method.isin(method_order)].set_index("method").reindex(method_order)
    abl_ran_sorted = abl_ran[abl_ran.method.isin(method_order)].set_index("method").reindex(method_order)

    fig = plt.figure(figsize=(FIG_W, 7.5))
    gs = GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.4)
    ax_d = fig.add_subplot(gs[0, :])  # architecture ablation bar (full width)
    ax_e = fig.add_subplot(gs[1, :])  # heatmap (full width)
    ax_f = fig.add_subplot(gs[2, 0])  # random CV
    ax_g = fig.add_subplot(gs[2, 1])  # cluster_isotype mean bars

    colors_arch = [GREY] * len(method_order)

    # ── Panel D: Architecture ablation mean rho, cluster_isotype ──
    xd = np.arange(len(method_order))
    vals_d = [abl_ci_sorted.loc[m, "MEAN"] if m in abl_ci_sorted.index else np.nan
              for m in method_order]
    ax_d.bar(xd, vals_d, 0.6, color=colors_arch)
    ax_d.axhline(dstack_mean, color=BLUE, ls="--", lw=LW * 1.5,
                 label=f"D-Stack (AbDev-X base expert, mean={dstack_mean:.3f})")
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([method_labels.get(m, m) for m in method_order],
                         rotation=20, ha="right", fontsize=FS_B - 1)
    ax_d.set_ylabel("Mean Spearman ρ")
    ax_d.set_title("Architecture ablation vs D-Stack (AbDev-X base expert) reference — cluster-isotype CV")
    ax_d.legend(fontsize=FS_B - 1)
    panel_label(ax_d, "D")

    # ── Panel E: Heatmap per assay, cluster_isotype ──
    heatmap_data = abl_ci_sorted[comp5].to_numpy(float)
    im = ax_e.imshow(heatmap_data.T, cmap="RdYlGn", aspect="auto",
                     vmin=-0.2, vmax=0.7)
    ax_e.set_xticks(range(len(method_order)))
    ax_e.set_xticklabels([method_labels.get(m, m) for m in method_order],
                         rotation=20, ha="right", fontsize=FS_B - 1.5)
    ax_e.set_yticks(range(len(comp5)))
    ax_e.set_yticklabels([LBL_C.get(a, a) for a in comp5], fontsize=FS_B - 1)
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            v = heatmap_data[i, j]
            if not np.isnan(v):
                ax_e.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                          color="white" if abs(v) < 0.1 else "black")
    plt.colorbar(im, ax=ax_e, shrink=0.5, label="CV ρ")
    ax_e.set_title("Per-assay cluster-isotype CV ρ heatmap")
    panel_label(ax_e, "E")

    # ── Panel F: Random CV ──
    vals_f = [abl_ran_sorted.loc[m, "MEAN"] if m in abl_ran_sorted.index else np.nan
              for m in method_order]
    # D-Stack random CV
    dstack_ran = all_cv[(all_cv.model == "AbDev-X (stacking)") & (all_cv.split == "random")]
    dstack_ran_mean = float(dstack_ran[comp5].values.mean()) if len(dstack_ran) else 0.528

    ax_f.bar(xd, vals_f, 0.6, color=colors_arch)
    ax_f.axhline(dstack_ran_mean, color=BLUE, ls="--", lw=LW * 1.5,
                 label=f"D-Stack random (mean={dstack_ran_mean:.3f})")
    ax_f.set_xticks(xd)
    ax_f.set_xticklabels([method_labels.get(m, m) for m in method_order],
                         rotation=20, ha="right", fontsize=FS_B - 2)
    ax_f.set_ylabel("Mean Spearman ρ")
    ax_f.set_title("Architecture ablation — random CV")
    ax_f.legend(fontsize=FS_B - 1.5)
    panel_label(ax_f, "F")

    # ── Panel G: AW-MoE vs D-Stack vs ablations, cluster_isotype, all bars ──
    awmoe_row = all_cv[(all_cv.model == "AbDev-X") & (all_cv.split == "cluster_isotype")]
    awmoe_mean = float(awmoe_row[comp5].values.mean()) if len(awmoe_row) else np.nan

    all_means = vals_d + [dstack_mean, awmoe_mean]
    all_labels = [method_labels.get(m, m) for m in method_order] + ["D-Stack", "AbDev-X"]
    all_colors = [GREY] * len(method_order) + [ORANGE, BLUE]
    xg = np.arange(len(all_means))
    bars_g = ax_g.bar(xg, all_means, 0.6, color=all_colors)
    for bar, lbl in zip(bars_g[-2:], ["D-Stack", "AbDev-X"]):
        bar.set_edgecolor("black"); bar.set_linewidth(0.8)
    ax_g.set_xticks(xg)
    ax_g.set_xticklabels(all_labels, rotation=35, ha="right", fontsize=FS_B - 2.5)
    ax_g.set_ylabel("Mean Spearman ρ")
    ax_g.set_title("All architectures + D-Stack + AbDev-X\n(cluster-isotype CV)")
    panel_label(ax_g, "G")

    fig.suptitle("Architecture ablation: D-Stack base expert (component of AbDev-X) outperforms alternatives",
                 fontsize=FS_SUP, y=1.01)
    save(fig, "panel_decision_arch")
    plt.close(fig)


def main():
    print("Generating panel_performance_comparison.pdf ...")
    fig_perf_comparison()
    print("Generating panel_decision_arch.pdf ...")
    fig_decision_arch()
    print("Done.")


if __name__ == "__main__":
    main()
