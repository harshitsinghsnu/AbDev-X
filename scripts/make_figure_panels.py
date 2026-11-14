"""
Generate publication-quality figure panels from existing single figures.

Panels produced:
  panel_shap_structures.png  – SHAP beeswarm (HIC) + hydrophobic surface + charge surface  (missing Fig 6)
  panel_results_overview.png – CV comparison + CV-vs-held-out  (Figs 2+4 combined)
  panel_plm_interp.png       – PLM ablation + meta-weights heatmap  (Figs 3+5 combined)
  panel_si_abdev.png         – ablation + learning-curves + significance  (already in text; regenerate larger)
  panel_si_calibration.png   – calibration + leakage  (already in text; regenerate larger)
  panel_si_transfer.png      – cross-dataset transfer + augmentation  (existing figS9 standalone → panel)
  panel_si_pooling_ig.png    – pooling ablation + IG CDR enrichment  (Figs S7+S6 combined)
  panel_si_explainability.png – explain per-residue ESM + per-antibody heatmap + category stacked
"""

import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT  = Path(__file__).resolve().parents[1]
FIGS  = ROOT / "paper" / "figures"
RFIGS = ROOT / "results" / "figures"
EXPL  = ROOT / "results" / "explain"
OUT   = FIGS   # write final panels into paper/figures/

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})

def load(path):
    return mpimg.imread(str(path))

def imshow(ax, img, title="", label="", fontsize=12):
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    if label:
        ax.text(-0.04, 1.04, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="top", ha="right")

# ──────────────────────────────────────────────────────────────────────
# Panel 1: SHAP beeswarm + structure maps  (the missing "Figure 6")
# ──────────────────────────────────────────────────────────────────────
def panel_shap_structures():
    shap_hic   = load(FIGS / "shap_HIC.png")
    hic_surf   = load(FIGS / "brazikumab_hic.png")
    charge_surf= load(FIGS / "zolbetuximab_charge.png")

    fig = plt.figure(figsize=(18, 6))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    imshow(ax0, shap_hic,
           title="(A) SHAP feature attributions — HIC\n(LightGBM on named physicochemical features)",
           label="A")

    ax1 = fig.add_subplot(gs[1])
    imshow(ax1, hic_surf,
           title="(B) Exposed hydrophobicity → HIC\n(orange/red = hydrophobic hotspots; brazikumab)",
           label="B")

    ax2 = fig.add_subplot(gs[2])
    imshow(ax2, charge_surf,
           title="(C) Surface charge → AC-SINS\n(red/blue = positive/negative patches; zolbetuximab)",
           label="C")

    fig.suptitle(
        "Figure: Explainability — SHAP attributions and structure-mapped residue signals\n"
        "Left: top-12 LightGBM SHAP features for HIC coloured by value. "
        "Middle: per-residue exposed hydrophobicity (SASA × Kyte–Doolittle) mapped onto the antibody surface. "
        "Right: signed exposed charge mapped onto the surface; patches drive self-association (AC-SINS).",
        fontsize=8.5, y=0.02, ha="center"
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT / "panel_shap_structures.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Panel 2: Results overview  (CV bars + CV-vs-held-out)
# ──────────────────────────────────────────────────────────────────────
def panel_results_overview():
    cv_img  = load(FIGS / "fig1_model_comparison.png")
    gen_img = load(FIGS / "fig2_generalization.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    imshow(axes[0], cv_img,
           title="(A) GDPa1 CV mean Spearman ρ across generalization splits",
           label="A")
    imshow(axes[1], gen_img,
           title="(B) GDPa1 cross-validation vs GDPa3 real held-out",
           label="B")

    fig.suptitle(
        "AbDev-X generalization results. "
        "(A) AbDev-X leads all baselines; margin widens under harder OOD splits. "
        "(B) CV (purple) vs. held-out (green, 95% CI); competition champions shown as red diamonds.",
        fontsize=9, y=0.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT / "panel_results_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Panel 3: PLM ablation + meta-weights  (model interpretation)
# ──────────────────────────────────────────────────────────────────────
def panel_plm_interp():
    plm_img    = load(FIGS / "fig3_plm_ablation.png")
    weight_img = load(FIGS / "fig4_repweights.png")

    fig = plt.figure(figsize=(16, 5.5))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1], wspace=0.06)

    ax0 = fig.add_subplot(gs[0])
    imshow(ax0, plm_img,
           title="(A) Five-PLM ablation — GDPa1 cluster-isotype CV ρ per assay",
           label="A")

    ax1 = fig.add_subplot(gs[1])
    imshow(ax1, weight_img,
           title="(B) Per-assay stacking meta-weights across six representations",
           label="B")

    fig.suptitle(
        "Representation analysis. "
        "(A) AbLang2 (antibody-specific) leads overall; ESM-2 leads on HIC — motivating fusion. "
        "(B) Learned meta-weights: Tm2 → isotype; HIC → bio+ESM-2; AC-SINS → AbLang2.",
        fontsize=9, y=0.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT / "panel_plm_interp.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Panel 4: SI — ablation + learning curves + significance (larger)
# ──────────────────────────────────────────────────────────────────────
def panel_si_abdev():
    abl = load(FIGS / "figS2_ablation.png")
    lc  = load(FIGS / "figS3_learning_curves.png")
    sig = load(FIGS / "figS8_significance.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    imshow(axes[0], abl,
           title="(A) Held-out ρ drop — leave-one-representation-out", label="A")
    imshow(axes[1], lc,
           title="(B) Learning curves (CV ρ vs. training fraction)", label="B")
    imshow(axes[2], sig,
           title="(C) Paired-bootstrap Δρ vs. baselines (held-out)", label="C")

    fig.suptitle(
        "SI: Representation importance, data efficiency, and statistical significance. "
        "(A) Germline/isotype and AbLang2 contribute most. "
        "(B) All assays show non-saturating learning curves — data is the bottleneck. "
        "(C) AbDev-X significantly beats ESM-2 and AbLang2 on HIC and Tm2 (* p < 0.05).",
        fontsize=9, y=0.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT / "panel_si_abdev.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Panel 5: SI — calibration + leakage
# ──────────────────────────────────────────────────────────────────────
def panel_si_calibration():
    cal = load(FIGS / "figS5_calibration.png")
    lkg = load(FIGS / "figS4_leakage.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    imshow(axes[0], cal,
           title="(A) Held-out calibration: ensemble variance vs. absolute error", label="A")
    imshow(axes[1], lkg,
           title="(B) Leakage audit: GDPa3→GDPa1 max VH+VL sequence identity", label="B")

    fig.suptitle(
        "SI: Calibration and leakage. "
        "(A) Ensemble variance is weakly correlated with error (ρ = 0.13–0.17 for HIC/AC-SINS; near-zero elsewhere). "
        "(B) No GDPa3 antibody exceeds 0.95 identity to any GDPa1 antibody — genuine OOD test confirmed.",
        fontsize=9, y=0.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT / "panel_si_calibration.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Panel 6: SI — cross-dataset transfer + augmentation (figS9)
# ──────────────────────────────────────────────────────────────────────
def panel_si_transfer():
    # figS9 already combines both; just re-export at higher quality with label
    trn = load(FIGS / "figS9_transfer_scaling.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    imshow(ax, trn, title="Cross-dataset transfer and per-study FLAb augmentation contribution")
    fig.suptitle(
        "SI: External augmentation. "
        "Left: GDPa1↔FLAb transfer (AC-SINS ρ=0.30/0.41; titer ≈ 0). "
        "Right: per-study augmentation delta (red=harmful); Hutchinson Tm2 panel adds +0.099; Garbinski titer +0.067.",
        fontsize=9, y=0.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT / "panel_si_transfer.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Panel 7: SI — pooling ablation + IG CDR enrichment
# ──────────────────────────────────────────────────────────────────────
def panel_si_pooling_ig():
    pool = load(FIGS / "figS7_pooling.png")
    ig   = load(FIGS / "figS6_ig_cdr.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    imshow(axes[0], pool,
           title="(A) ESM-2 pooling ablation: mean vs. CDR vs. framework", label="A")
    imshow(axes[1], ig,
           title="(B) Integrated Gradients: CDR hotspot enrichment per assay", label="B")

    fig.suptitle(
        "SI: Pooling and residue-level attribution. "
        "(A) CDR-focused pooling improves HIC (0.55 vs 0.44) and AC-SINS; framework is best for PR-CHO. "
        "(B) Chain-aware IG attributions are 2.5–3.5× CDR-enriched across all five assays.",
        fontsize=9, y=0.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT / "panel_si_pooling_ig.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Panel 8: SI — per-residue ESM + per-antibody block-SHAP heatmap + category stacked
# ──────────────────────────────────────────────────────────────────────
def panel_si_explainability():
    perres  = RFIGS / "explain_4_perres_esm150m.png"
    heatmap = RFIGS / "explain_2_perantibody_heatmap.png"
    stacked = EXPL  / "shap_category_stacked.png"

    imgs = []
    lbls = []
    titles = []
    for path, lbl, ttl in [
        (perres,  "A", "Per-residue ESM-2 150M gradient attribution (5 assays)"),
        (heatmap, "B", "Per-antibody block SHAP heatmap (HIC, sorted by label)"),
        (stacked, "C", "Feature-category SHAP stacked per antibody (all assays)"),
    ]:
        if path.exists():
            imgs.append(load(path))
            lbls.append(lbl)
            titles.append(ttl)

    n = len(imgs)
    if n == 0:
        print("  Skipped panel_si_explainability — no source images found")
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5))
    if n == 1:
        axes = [axes]
    for ax, img, lbl, ttl in zip(axes, imgs, lbls, titles):
        imshow(ax, img, title=f"({lbl}) {ttl}", label=lbl)

    fig.suptitle(
        "SI: Multi-level explainability details. "
        "(A) Modest CDR enrichment (1.1–1.2×) from ESM-2 proxy gradients. "
        "(B) Per-antibody block dominance: most HIC antibodies are bio-driven; outliers show complementary ESM-2/AbLang2. "
        "(C) Hydrophobic features dominate HIC; ~15% antibodies are charge-driven (dual-mechanism HIC).",
        fontsize=9, y=0.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT / "panel_si_explainability.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Panel 9: SI — block-SHAP waterfall + per-antibody SHAP heatmap (wider view)
# ──────────────────────────────────────────────────────────────────────
def panel_si_waterfall():
    waterfall = EXPL / "shap_waterfall_lgbm.png"
    heatmap   = EXPL / "shap_perantibody_heatmap.png"

    if not waterfall.exists() or not heatmap.exists():
        print("  Skipped panel_si_waterfall — source images missing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    imshow(axes[0], load(waterfall),
           title="(A) Per-antibody SHAP waterfall — lowest/median/highest antibody per assay",
           label="A")
    imshow(axes[1], load(heatmap),
           title="(B) SHAP heatmap — top-20 features × antibodies (sorted by assay label)",
           label="B")

    fig.suptitle(
        "SI: Per-antibody LightGBM SHAP decomposition. "
        "(A) Waterfall plots show feature-level attribution for three representative antibodies per assay. "
        "(B) Heatmap (RdBu) reveals which features push each prediction up or down.",
        fontsize=9, y=0.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT / "panel_si_waterfall.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ──────────────────────────────────────────────────────────────────────
# Panel 10: Antibody ranking comparison  (composite rho + P@K + scatter)
# ──────────────────────────────────────────────────────────────────────
def panel_ranking():
    """Ranking comparison: composite Spearman rho, Precision@K, true vs pred scatter."""
    ranking_fig = OUT / "fig_ranking.png"
    if not ranking_fig.exists():
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "fig_ranking", Path(__file__).parent / "fig_ranking.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    if not ranking_fig.exists():
        print("  Skipped panel_ranking — fig_ranking.png not found")
        return

    img = load(ranking_fig)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    imshow(ax, img,
           title="Antibody developability ranking on GDPa3 (n=80): "
                 "composite Spearman rho, Precision@K, and true vs predicted scatter")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("png", "pdf"):
        out = OUT / f"panel_ranking.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved {out.name}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figure panels ...")
    panel_shap_structures()
    panel_results_overview()
    panel_plm_interp()
    panel_si_abdev()
    panel_si_calibration()
    panel_si_transfer()
    panel_si_pooling_ig()
    panel_si_explainability()
    panel_si_waterfall()
    panel_ranking()
    print(f"\nAll panels → {OUT}")
