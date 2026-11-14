"""
Combined figure panels (v2) -- journal PDF format.
All outputs: PDF, 7.0-inch page width, 8 pt base font.

Outputs (all .pdf in paper/figures/):
  panel_fig23          - Fig 2+3: results overview (2x2 grid, 4 panels A-D)
  panel_fig45          - Fig 4+5: SHAP+structures + 6-panel explainability (stacked)
  panel_fig67          - Fig 6+7: ablation+learning+sig + pooling+IG (5 panels 3+2)
  panel_si_explainability - 3 rows: ESM perres | block heatmap | cat stacked
  panel_si_waterfall_only - waterfall plots only
  panel_si_shap_heatmap   - per-antibody SHAP heatmap (standalone)
  panel_si_crossabsense   - CrossAbSense vs AbDev-X (SI-17)
  panel_si_encoder_ext    - Extended encoder screen (SI-18)
  panel_si_struct_sources - Structural feature groups (SI-19)
  panel_si_abstention     - Abstention tradeoff curves (SI-20)
  panel_si_precisionk     - Precision@K enrichment (SI-26)
  panel_si_arch_ablation  - Architecture ablation (SI-27)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

ROOT    = Path(__file__).resolve().parents[1]
FIGS    = ROOT / "paper" / "figures"
RFIGS   = ROOT / "results" / "figures"
EXPL    = ROOT / "results" / "explain"
LSUIT   = ROOT / "results" / "limitations_suite"
ABLAT   = ROOT / "results" / "abdevx2_ablation"
RESULTS = ROOT / "results"
OUT     = FIGS

# ── Journal style constants ──────────────────────────────────────────────────
FIG_W   = 7.0    # inches (full text width)
FS_B    = 8      # base / tick labels
FS_AX   = 9      # axis labels
FS_T    = 9      # panel titles (slightly smaller to save space)
FS_LBL  = 11     # (A) (B) panel letters
FS_SUP  = 7.5    # suptitle / legend text
LW      = 0.8
CAPSIZE = 2
DPI     = 300

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        FS_B,
    "axes.titlesize":   FS_T,
    "axes.labelsize":   FS_AX,
    "xtick.labelsize":  FS_B,
    "ytick.labelsize":  FS_B,
    "legend.fontsize":  FS_B,
    "figure.dpi":       DPI,
    "savefig.dpi":      DPI,
    "lines.linewidth":  LW,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# Colour palette (Seaborn default)
PAL = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#DD8452",
       "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

ASSAYS  = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
ASSAY5  = ["HIC", "Tm2", "AC-SINS", "PR-CHO", "Titer"]
ALABEL  = {"HIC": "HIC", "Tm2": "Tm2", "AC-SINS_pH7.4": "AC-SINS",
           "AC-SINS": "AC-SINS", "PR_CHO": "PR-CHO", "PR-CHO": "PR-CHO", "Titer": "Titer"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def load(p):
    return mpimg.imread(str(p))

def imshow(ax, img, title="", label=""):
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=FS_T, pad=2)
    if label:
        ax.text(-0.03, 1.04, label, transform=ax.transAxes,
                fontsize=FS_LBL, fontweight="bold", va="top", ha="right")

def imshow_fill(ax, img, title="", label=""):
    """Display image filling the full axes (no letterboxing)."""
    ax.imshow(img, aspect="auto")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=FS_T, pad=2)
    if label:
        ax.text(-0.03, 1.04, label, transform=ax.transAxes,
                fontsize=FS_LBL, fontweight="bold", va="top", ha="right")

def save(fig, stem):
    name = stem.replace(".png", ".pdf")
    if not name.endswith(".pdf"):
        name += ".pdf"
    path = OUT / name
    fig.savefig(str(path), format="pdf", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ════════════════════════════════════════════════════════════════════════════
# 1. panel_gen_rep: GDPa1 CV bars + CV-vs-held-out scatter + PLM ablation
#    + meta-weights heatmap. Drawn directly from CSVs; legends sit horizontally
#    above each subplot (no overlap with plot area).
# ════════════════════════════════════════════════════════════════════════════
def fig23():
    _A5   = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
    _A5HS = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "SMAC", "HAC"]  # 5 high-signal for Panel A
    _LBL  = {"HIC": "HIC", "Tm2": "Tm2", "AC-SINS_pH7.4": "AC-SINS",
             "PR_CHO": "PR-CHO", "Titer": "Titer",
             "SMAC": "SMAC", "HAC": "HAC"}
    _COMP = {"HIC": 0.708, "Tm2": 0.392, "AC-SINS_pH7.4": 0.356,
             "PR_CHO": 0.337, "Titer": 0.310}
    EBAR  = dict(capsize=CAPSIZE, error_kw={"lw": 0.6})

    cv  = pd.read_csv(RESULTS / "cv_errorbars.csv")
    awmoe_eb = pd.read_csv(RESULTS / "awmoe_cv_errorbars.csv")  # per-fold mean±std per assay×split

    # AW-MoE held-out values (corrected from stale heldout_errorbars.csv)
    # awmoe_all9.csv has point estimates; bootstrap_ci_all.csv has BCa CIs for 5 high-signal assays
    _awmoe = pd.read_csv(RESULTS / "awmoe_all9.csv").set_index("assay")
    _bca   = pd.read_csv(RESULTS / "bootstrap_ci_all.csv")
    _bca_row = _bca[_bca["model"] == "AbDev-X"].iloc[0]
    # CIs for Tm2 and Titer from SI Table (BCa 95%, B=2000)
    _AWMOE_CI = {
        "HIC":          (_bca_row["HIC_lo"],     _bca_row["HIC_hi"]),
        "AC-SINS_pH7.4":(_bca_row["AC-SINS_lo"], _bca_row["AC-SINS_hi"]),
        "PR_CHO":       (_bca_row["PR_CHO_lo"],  _bca_row["PR_CHO_hi"]),
        "Tm2":          (-0.063, 0.371),
        "Titer":        (-0.147, 0.293),
    }
    _awmoe_assay_map = {"HIC": "HIC", "Tm2": "Tm2", "AC-SINS_pH7.4": "AC-SINS",
                        "PR_CHO": "PR_CHO", "Titer": "Titer"}
    hb = pd.DataFrame({
        "assay":        _A5,
        "heldout_mean": [float(_awmoe.loc[_awmoe_assay_map[a], "rho"]) for a in _A5],
        "ci_low":       [_AWMOE_CI[a][0] for a in _A5],
        "ci_high":      [_AWMOE_CI[a][1] for a in _A5],
    }).set_index("assay").reindex(_A5)
    plm = pd.read_csv(RESULTS / "plm_ablation.csv").sort_values("MEAN", ascending=False)
    wt  = pd.read_csv(RESULTS / "representation_weights.csv").set_index("assay").reindex(_A5)

    # legend style: centered above each subplot, one horizontal row
    # original model colors: dark navy → blue → teal → gray
    MODEL_COLS  = ["#1A3A6E", "#3A7CC1", "#17BECF", "#7F7F7F"]
    # assay colors: match original seaborn deep palette
    ASSAY_COLS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#bcbd22"]

    # legend style shared across A, B, C
    def _leg(ax, ncol, y_anchor=1.01):
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, y_anchor), ncol=ncol,
                  frameon=True, framealpha=0.94, borderpad=0.3,
                  fontsize=FS_B - 1, handlelength=1.1, handletextpad=0.35,
                  columnspacing=0.65)

    fig = plt.figure(figsize=(FIG_W, FIG_W * 1.12))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.72, wspace=0.44)

    # ── A: AbDev-X GDPa1 CV ρ per assay across split strategies ─────────
    ax_a = fig.add_subplot(gs[0, 0])
    splits  = ["random", "cluster", "cluster_isotype"]
    slabels = ["Random\n(IID)", "Cluster\n(OOD)", "Clust-Iso\n(OOD)"]
    SPLIT_COLS = ["#1A3A6E", "#3A7CC1", "#7EC8E3"]
    assay_short = [_LBL[a] for a in _A5HS]
    x = np.arange(len(_A5HS)); w = 0.25
    for i, (spl, scol, slbl) in enumerate(zip(splits, SPLIT_COLS, slabels)):
        sub = awmoe_eb[(awmoe_eb.model == "AbDev-X") & (awmoe_eb.split == spl)]
        sub = sub.set_index("assay").reindex(_A5HS)
        ax_a.bar(x + (i - 1) * w, sub["mean"], w, yerr=sub["std"],
                 color=scol, label=slbl.replace("\n", " "), **EBAR)
    ax_a.axhline(0, color="grey", lw=0.5)
    ax_a.set_xticks(x); ax_a.set_xticklabels(assay_short, fontsize=FS_B)
    ax_a.set_ylabel("Spearman ρ (mean ± std across folds)", fontsize=FS_AX)
    ax_a.set_title("AbDev-X GDPa1 CV across splits", fontsize=FS_T)
    _leg(ax_a, ncol=3)
    ax_a.text(-0.03, 1.38, "(A)", transform=ax_a.transAxes,
              fontsize=FS_LBL, fontweight="bold", va="top", ha="right")

    # ── B: CV ρ vs held-out ρ scatter ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    cvx = cv[cv.model == "AbDev-X"].set_index("assay").reindex(_A5)
    for i, (a, col) in enumerate(zip(_A5, ASSAY_COLS)):
        cv_m  = cvx.loc[a, "mean"]; cv_s = cvx.loc[a, "std"]
        ho_m  = hb.loc[a, "heldout_mean"]
        ci_lo = hb.loc[a, "ci_low"];  ci_hi = hb.loc[a, "ci_high"]
        ax_b.errorbar(cv_m, ho_m,
                      xerr=cv_s, yerr=[[ho_m - ci_lo], [ci_hi - ho_m]],
                      fmt="o", color=col, markersize=6,
                      capsize=3, elinewidth=0.9, label=_LBL[a])
        ax_b.plot(cv_m, _COMP[a], marker="D", color=col,
                  fillstyle="none", markersize=7, markeredgewidth=1.0)
    all_vals = ([cvx.loc[a, "mean"] for a in _A5] +
                [hb.loc[a, "heldout_mean"] for a in _A5] +
                list(_COMP.values()))
    lo = min(0.0, min(all_vals) - 0.05)
    hi = max(all_vals) * 1.15
    # clamp so Tm2 upper CI doesn't blow axes out
    hi = min(hi, 1.0)
    ax_b.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax_b.set_xlim(lo, hi); ax_b.set_ylim(lo, hi)
    ax_b.set_xlabel("GDPa1 CV ρ (mean ± std)", fontsize=FS_AX)
    ax_b.set_ylabel("GDPa3 held-out ρ (BCa 95% CI)", fontsize=FS_AX)
    ax_b.set_title("CV vs held-out  (◇ = champion)", fontsize=FS_T)
    _leg(ax_b, ncol=5)                          # 5 short names → 1 row
    ax_b.text(-0.03, 1.20, "(B)", transform=ax_b.transAxes,
              fontsize=FS_LBL, fontweight="bold", va="top", ha="right")

    # ── C: PLM ablation — PLMs on x, assays as bar groups ─────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    plm_names = plm["PLM"].tolist()
    plm_short = [p.replace(" (Ab)", "").replace(" (small)", "")
                   .replace(" (650M)", "") for p in plm_names]
    n_a = len(_A5); w = 0.80 / n_a
    x   = np.arange(len(plm_names))
    for j, (a, col) in enumerate(zip(_A5, ASSAY_COLS)):
        vals = [plm.loc[plm["PLM"] == p, a].values[0] for p in plm_names]
        errs = [plm.loc[plm["PLM"] == p, a + "_std"].values[0] for p in plm_names]
        ax_c.bar(x + (j - n_a / 2 + 0.5) * w, vals, w,
                 yerr=errs, color=col, label=_LBL[a], **EBAR)
    ax_c.axhline(0, color="grey", lw=0.5)
    ax_c.set_xticks(x); ax_c.set_xticklabels(plm_short, fontsize=FS_B)
    ax_c.set_ylabel("Cluster-isotype CV ρ", fontsize=FS_AX)
    ax_c.set_title("PLM ablation (per assay)", fontsize=FS_T)
    _leg(ax_c, ncol=5)                          # 5 short names → 1 row
    ax_c.text(-0.03, 1.20, "(C)", transform=ax_c.transAxes,
              fontsize=FS_LBL, fontweight="bold", va="top", ha="right")

    # ── D: per-assay stacking meta-weights heatmap ────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    cols = list(wt.columns)
    M    = wt[cols].to_numpy(float)
    vmax = np.nanmax(np.abs(M))
    im   = ax_d.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    short_cols = ["ESM2", "Desc.", "Iso.", "Bio-R", "Bio-L", "AbL2"]
    ax_d.set_xticks(range(len(cols)))
    ax_d.set_xticklabels(short_cols[:len(cols)], rotation=30, ha="right",
                         fontsize=FS_B - 1)
    ax_d.set_yticks(range(len(_A5)))
    ax_d.set_yticklabels([_LBL[a] for a in _A5], fontsize=FS_B)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax_d.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                      fontsize=FS_B - 2)
    ax_d.set_title("Stacking meta-weights per block", fontsize=FS_T)
    fig.colorbar(im, ax=ax_d, shrink=0.72, label="Meta-weight")
    ax_d.text(-0.03, 1.04, "(D)", transform=ax_d.transAxes,
              fontsize=FS_LBL, fontweight="bold", va="top", ha="right")

    out_dir = ROOT / "paper" / "paper10assay"
    fig.savefig(str(out_dir / "panel_gen_rep.pdf"), format="pdf",
                bbox_inches="tight", dpi=DPI)
    fig.savefig(str(out_dir / "panel_gen_rep.png"), format="png",
                bbox_inches="tight", dpi=DPI)
    # keep paper/figures copy in sync
    fig.savefig(str(OUT / "panel_gen_rep.pdf"), format="pdf",
                bbox_inches="tight", dpi=DPI)
    fig.savefig(str(OUT / "panel_fig23.pdf"), format="pdf",
                bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("  Saved panel_gen_rep.pdf + .png (paper10assay and paper/figures)")


# ════════════════════════════════════════════════════════════════════════════
# 2. Combined Fig 4+5: SHAP+structures (row 1) + 6-panel explainability (row 2)
# ════════════════════════════════════════════════════════════════════════════
def fig45():
    shap_img  = load(FIGS / "shap_HIC.png")
    hic_surf  = load(FIGS / "brazikumab_hic.png")
    chg_surf  = load(FIGS / "zolbetuximab_charge.png")
    explain6  = load(FIGS / "panel_explainability.png")

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.92))
    gs  = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.55],
                   hspace=0.10, wspace=0.05)

    imshow(fig.add_subplot(gs[0, 0]), shap_img,
           label="A", title="SHAP (HIC): top-12 physicochemical features")
    imshow(fig.add_subplot(gs[0, 1]), hic_surf,
           label="B", title="Hydrophobic surface: Brazikumab / HIC")
    imshow(fig.add_subplot(gs[0, 2]), chg_surf,
           label="C", title="Charge surface: Zolbetuximab / AC-SINS")

    ax_bot = fig.add_subplot(gs[1, :])
    ax_bot.imshow(explain6)
    ax_bot.axis("off")
    ax_bot.text(-0.01, 1.02, "D-I", transform=ax_bot.transAxes,
                fontsize=FS_LBL, fontweight="bold", va="bottom")
    ax_bot.set_title(
        "(D) Block SHAP fractions  |  (E) Per-antibody heatmap  |  (F) Named features  |"
        "  (G) Category stacked  |  (H) ESM gradient  |  (I) 3D B-factors",
        fontsize=FS_B)

    fig.tight_layout(rect=[0, 0, 1, 1])
    save(fig, "panel_fig45")


# ════════════════════════════════════════════════════════════════════════════
# 3. Combined Fig 6+7: ablation+learning+sig (row 1) + pooling+IG (row 2)
# ════════════════════════════════════════════════════════════════════════════
def fig67():
    abl  = load(FIGS / "figS2_ablation.png")
    lc   = load(FIGS / "figS3_learning_curves.png")
    sig  = load(FIGS / "figS8_significance.png")
    pool = load(FIGS / "figS7_pooling.png")
    ig   = load(FIGS / "figS6_ig_cdr.png")

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.68))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.12, wspace=0.05)

    imshow(fig.add_subplot(gs[0, 0]), abl,  label="A",
           title="Leave-one-rep-out: held-out rho drop")
    imshow(fig.add_subplot(gs[0, 1]), lc,   label="B",
           title="Learning curves (CV rho vs. training fraction)")
    imshow(fig.add_subplot(gs[0, 2]), sig,  label="C",
           title="Paired-bootstrap Delta-rho (AbDev-X - baseline)")
    imshow(fig.add_subplot(gs[1, 0]), pool, label="D",
           title="ESM-2 pooling ablation: mean vs CDR vs framework")
    imshow(fig.add_subplot(gs[1, 1]), ig,   label="E",
           title="Integrated Gradients: CDR enrichment (chain-aware net)")
    fig.add_subplot(gs[1, 2]).axis("off")

    fig.tight_layout(rect=[0, 0, 1, 1])
    save(fig, "panel_fig67")


# ════════════════════════════════════════════════════════════════════════════
# 4. SI Fig 11 -- 3 rows vertical (A top, B middle, C bottom)
# ════════════════════════════════════════════════════════════════════════════
def fix_fig11():
    paths = [RFIGS / "explain_4_perres_esm150m.png",
             RFIGS / "explain_2_perantibody_heatmap.png",
             EXPL  / "shap_category_stacked.png"]
    imgs = [load(p) for p in paths if p.exists()]
    if len(imgs) < 3:
        print("  Skipped panel_si_explainability -- source image missing"); return

    # Panel A: full page width (top row spans both columns)
    # Panels B + C: side-by-side below
    fig = plt.figure(figsize=(FIG_W, FIG_W * 1.30))
    gs = GridSpec(2, 2, figure=fig, hspace=0.03, wspace=0.03,
                  height_ratios=[1.0, 1.1])
    ax_a = fig.add_subplot(gs[0, :])  # spans full width
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    imshow_fill(ax_a, imgs[0], label="A",
                title="Per-residue ESM-2 150M gradient attribution (CDR shaded; 1.1-1.2x CDR enrichment)")
    imshow_fill(ax_b, imgs[1], label="B",
                title="Per-antibody block SHAP heatmap (HIC, sorted by label)")
    imshow_fill(ax_c, imgs[2], label="C",
                title="Feature-category SHAP stacked per antibody (all 5 assays, sorted by label)")
    fig.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3)
    save(fig, "panel_si_explainability")


# ════════════════════════════════════════════════════════════════════════════
# 5. SI Fig 13 split: waterfall only (A) and per-antibody heatmap (B)
# ════════════════════════════════════════════════════════════════════════════
def split_fig13():
    wf_p = EXPL / "shap_waterfall_lgbm.png"
    hm_p = EXPL / "shap_perantibody_heatmap.png"

    if wf_p.exists():
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.82))
        imshow(ax, load(wf_p),
               title="LightGBM SHAP waterfall -- lowest / median / highest antibody per assay")
        fig.tight_layout(rect=[0, 0, 1, 1])
        save(fig, "panel_si_waterfall_only")

    if hm_p.exists():
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.45))
        imshow(ax, load(hm_p),
               title="SHAP heatmap: top-20 features x antibodies (sorted by label, RdBu)")
        fig.tight_layout(rect=[0, 0, 1, 1])
        save(fig, "panel_si_shap_heatmap")


# ════════════════════════════════════════════════════════════════════════════
# 6. CrossAbSense comparison (SI-17)
# ════════════════════════════════════════════════════════════════════════════
def si_crossabsense():
    df = pd.read_csv(RESULTS / "exp_crossab_errorbars.csv")
    models = ["AbLang2_baseline", "ESM2_650M_baseline", "CrossAbSense", "CDR_mean", "VH+VL_mean"]
    mlabels = {"AbLang2_baseline": "AbLang2-Ridge", "ESM2_650M_baseline": "ESM-2-Ridge",
               "CrossAbSense": "CrossAbSense", "CDR_mean": "CDR mean-pool",
               "VH+VL_mean": "VH+VL mean-pool"}
    colors = [plt.cm.viridis(v) for v in np.linspace(0.1, 0.88, len(models))]
    n = len(models)
    w = 0.13
    half = n / 2 * w + 0.05

    fig, (ax_cv, ax_ho) = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.50),
                                        constrained_layout=False)
    fig.subplots_adjust(bottom=0.26, wspace=0.32)
    x = np.arange(len(ASSAY5))

    abdevx_cv = [0.617, -0.011, 0.545, 0.457, 0.317]
    abdevx_ho = [0.527, 0.333, 0.300, 0.197, 0.294]

    for i, (m, col) in enumerate(zip(models, colors)):
        sub = df[df["model"] == m].set_index("assay")
        cv = [sub.loc[a, "cv_mean"] if a in sub.index else np.nan for a in ASSAYS]
        ce = [sub.loc[a, "cv_std"]  if a in sub.index else 0      for a in ASSAYS]
        ax_cv.bar(x + (i - n/2 + 0.5)*w, cv, w*0.88, color=col, label=mlabels[m],
                  yerr=ce, capsize=CAPSIZE, error_kw={"lw": 0.6})

    for j, (xj, v) in enumerate(zip(x, abdevx_cv)):
        ax_cv.plot([xj - half, xj + half], [v, v], "k--", lw=LW * 1.5)
    ax_cv.plot([], [], "k--", lw=LW * 1.5, label="AbDev-X")
    ax_cv.set_xticks(x); ax_cv.set_xticklabels(ASSAY5, rotation=30, ha="right", fontsize=FS_B)
    ax_cv.set_ylabel("CV Spearman rho")
    ax_cv.set_title("(A) GDPa1 cluster-isotype CV rho")
    ax_cv.axhline(0, color="grey", lw=0.5)

    for i, (m, col) in enumerate(zip(models, colors)):
        sub = df[df["model"] == m].set_index("assay")
        ho, lo, hi = [], [], []
        for a in ASSAYS:
            if a in sub.index and not pd.isna(sub.loc[a, "gdpa3"]):
                v = sub.loc[a, "gdpa3"]
                ho.append(v)
                lo.append(v - sub.loc[a, "gdpa3_ci_lo"])
                hi.append(sub.loc[a, "gdpa3_ci_hi"] - v)
            else:
                ho.append(np.nan); lo.append(0); hi.append(0)
        ax_ho.bar(x + (i - n/2 + 0.5)*w, ho, w*0.88, color=col, label=mlabels[m],
                  yerr=[lo, hi], capsize=CAPSIZE, error_kw={"lw": 0.6})

    for j, (xj, v) in enumerate(zip(x, abdevx_ho)):
        ax_ho.plot([xj - half, xj + half], [v, v], "k--", lw=LW * 1.5)
    ax_ho.plot([], [], "k--", lw=LW * 1.5, label="AbDev-X")
    ax_ho.set_xticks(x); ax_ho.set_xticklabels(ASSAY5, rotation=30, ha="right", fontsize=FS_B)
    ax_ho.set_ylabel("Held-out Spearman rho (GDPa3)")
    ax_ho.set_title("(B) GDPa3 held-out rho (bootstrap +/- 95% CI)")
    ax_ho.axhline(0, color="grey", lw=0.5)

    handles, labels = ax_cv.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="lower center",
               bbox_to_anchor=(0.5, 0.01), fontsize=FS_B - 1, framealpha=0.9)
    save(fig, "panel_si_crossabsense")


# ════════════════════════════════════════════════════════════════════════════
# 7. Extended encoder screen with IgBERT (SI-18)
# ════════════════════════════════════════════════════════════════════════════
def si_encoder_ext():
    df = pd.read_csv(RESULTS / "exp_encoder_screen_errorbars.csv")
    encs = ["ESM2_650M", "AbLang2", "IgBERT", "AntiBERTy", "ProtBert", "ProGen2", "Descriptors", "Isotype"]
    elabs = {"ESM2_650M": "ESM-2\n650M", "AbLang2": "AbLang2", "IgBERT": "IgBERT",
             "AntiBERTy": "AntiBERTy", "ProtBert": "ProtBert", "ProGen2": "ProGen2",
             "Descriptors": "Descriptors", "Isotype": "Isotype"}
    ecols = PAL[:len(encs)]

    cv_m = {e: df[df["encoder"] == e]["cv_mean"].mean() for e in encs}
    cv_s = {e: df[df["encoder"] == e]["cv_mean"].std()  for e in encs}
    ho_m = {e: df[df["encoder"] == e]["gdpa3"].dropna().mean() for e in encs}
    ho_s = {e: df[df["encoder"] == e]["gdpa3"].dropna().std()  for e in encs}
    ecols = [plt.cm.viridis(v) for v in np.linspace(0.1, 0.88, len(encs))]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.48), constrained_layout=True)

    xe = np.arange(len(encs))
    ax0.bar(xe, [cv_m[e] for e in encs], color=ecols, edgecolor="white", lw=0.4,
            yerr=[cv_s[e] for e in encs], capsize=CAPSIZE, error_kw={"lw": 0.6})
    ax0.axhline(0.462, color="k", ls="--", lw=LW * 1.5, label="AbDev-X (fusion)")
    ax0.set_xticks(xe); ax0.set_xticklabels([elabs[e] for e in encs], rotation=30, ha="right", fontsize=FS_B)
    ax0.set_ylabel("Mean CV Spearman rho (+/- std across assays)")
    ax0.set_title("(A) Single-encoder CV performance")
    ax0.axhline(0, color="grey", lw=0.5); ax0.legend(fontsize=FS_B - 1)

    enc_ho = [e for e in encs if not np.isnan(ho_m.get(e, np.nan))]
    ho_cols = [ecols[encs.index(e)] for e in enc_ho]
    ax1.bar(range(len(enc_ho)), [ho_m[e] for e in enc_ho], color=ho_cols, edgecolor="white", lw=0.4,
            yerr=[ho_s.get(e, 0) for e in enc_ho], capsize=CAPSIZE, error_kw={"lw": 0.6})
    ax1.axhline(0.327, color="k", ls="--", lw=LW * 1.5, label="AbDev-X held-out mean")
    ax1.set_xticks(range(len(enc_ho)))
    ax1.set_xticklabels([elabs[e] for e in enc_ho], rotation=30, ha="right", fontsize=FS_B)
    ax1.set_ylabel("Mean held-out Spearman rho (+/- std)")
    ax1.set_title("(B) GDPa3 held-out transfer"); ax1.axhline(0, color="grey", lw=0.5)
    ax1.legend(fontsize=FS_B - 1)

    save(fig, "panel_si_encoder_ext")


# ════════════════════════════════════════════════════════════════════════════
# 8. Structural feature source comparison (SI-19)
# ════════════════════════════════════════════════════════════════════════════
def si_struct_sources():
    df = pd.read_csv(RESULTS / "exp_struct_errorbars.csv")
    groups = ["MOE", "SAP", "Electro", "Aggrescan3D", "Combined_struct", "AbLang2", "ESM2_650M"]
    glabs  = {"MOE": "MOE", "SAP": "SAP", "Electro": "Electro",
               "Aggrescan3D": "Aggrescan3D", "Combined_struct": "Combined",
               "AbLang2": "AbLang2", "ESM2_650M": "ESM-2"}
    gcols  = [plt.cm.viridis(v) for v in np.linspace(0.1, 0.88, len(groups))]
    x = np.arange(len(ASSAY5))
    w = 0.11

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.52),
                                    constrained_layout=False)
    fig.subplots_adjust(bottom=0.24, wspace=0.35)

    for i, (g, col) in enumerate(zip(groups, gcols)):
        sub = df[df["block"] == g].set_index("assay")
        vals = [sub.loc[a, "cv_mean"] if a in sub.index else np.nan for a in ASSAYS]
        errs = [sub.loc[a, "cv_std"]  if a in sub.index else 0      for a in ASSAYS]
        ax0.bar(x + (i - 3)*w, vals, w, color=col, label=glabs[g],
                yerr=errs, capsize=CAPSIZE, error_kw={"lw": 0.6})
    ax0.set_xticks(x); ax0.set_xticklabels(ASSAY5, rotation=30, ha="right", fontsize=FS_B)
    ax0.set_ylabel("CV Spearman rho"); ax0.set_title("(A) Structural feature group CV")
    ax0.axhline(0, color="grey", lw=0.5)

    means = [df[df["block"] == g]["cv_mean"].mean() for g in groups]
    ax1.bar(range(len(groups)), means, color=gcols, edgecolor="white", lw=0.4)
    ax1.set_xticks(range(len(groups)))
    ax1.set_xticklabels([glabs[g] for g in groups], rotation=30, ha="right", fontsize=FS_B)
    ax1.set_ylabel("Mean CV Spearman rho"); ax1.set_title("(B) Mean CV rho (5 assays)")
    ax1.axhline(0, color="grey", lw=0.5)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="lower center",
               bbox_to_anchor=(0.5, 0.01), fontsize=FS_B - 1, framealpha=0.9)
    save(fig, "panel_si_struct_sources")


# ════════════════════════════════════════════════════════════════════════════
# 9. Abstention tradeoff curves (SI-20)
# ════════════════════════════════════════════════════════════════════════════
def si_abstention():
    df_cv = pd.read_csv(RESULTS / "exp_abstention_errorbars.csv")
    df_ho = pd.read_csv(LSUIT / "abstention_tradeoff.csv")
    cols5 = PAL[:5]
    ls5   = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.42))

    for i, a in enumerate(ASSAYS):
        sub = df_cv[df_cv["assay"] == a]
        if not sub.empty:
            ax0.plot(sub["abstain_frac"], sub["rho"],
                     color=cols5[i], ls=ls5[i], marker="o", ms=3, lw=LW * 1.5,
                     label=ALABEL.get(a, a))
    ax0.set_xlabel("Abstain fraction"); ax0.set_ylabel("Spearman rho")
    ax0.set_title("(A) CV abstention (GDPa1)")
    ax0.axhline(0, color="grey", lw=0.5); ax0.legend()

    for i, a in enumerate(ASSAYS):
        sub = df_ho[df_ho["assay"] == a]
        if not sub.empty:
            ax1.plot(sub["abstain_fraction"], sub["retained_spearman"],
                     color=cols5[i], ls=ls5[i], marker="o", ms=3, lw=LW * 1.5,
                     label=ALABEL.get(a, a))
    ax1.set_xlabel("Abstain fraction"); ax1.set_ylabel("Spearman rho")
    ax1.set_title("(B) Held-out abstention (GDPa3)")
    ax1.axhline(0, color="grey", lw=0.5); ax1.legend()

    fig.tight_layout()
    save(fig, "panel_si_abstention")


# ════════════════════════════════════════════════════════════════════════════
# 10. Precision@K enrichment (SI-26)
# ════════════════════════════════════════════════════════════════════════════
def si_precisionk():
    df = pd.read_csv(LSUIT / "decision_metrics.csv")
    aord  = ["HIC", "Titer", "AC-SINS_pH7.4", "Tm2", "PR_CHO"]
    alab  = {"HIC": "HIC", "Titer": "Titer", "AC-SINS_pH7.4": "AC-SINS",
              "Tm2": "Tm2", "PR_CHO": "PR-CHO"}
    ks    = [5, 10, 20]
    kcols = [PAL[0], PAL[1], PAL[2]]
    x = np.arange(len(aord))
    w = 0.25

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.42))

    for i, (k, col) in enumerate(zip(ks, kcols)):
        sub = df[df["k"] == k].set_index("assay")
        vals = [sub.loc[a, "precision_at_k"] if a in sub.index else 0 for a in aord]
        ax0.bar(x + (i-1)*w, vals, w, color=col, label=f"P@{k}")
    ax0.set_xticks(x); ax0.set_xticklabels([alab[a] for a in aord])
    ax0.set_ylabel("Precision@K"); ax0.set_title("(A) Precision@K")
    ax0.axhline(0, color="grey", lw=0.5); ax0.legend(); ax0.set_ylim(0, 0.75)

    for i, (k, col) in enumerate(zip(ks, kcols)):
        sub = df[df["k"] == k].set_index("assay")
        vals = [sub.loc[a, "enrichment"] if a in sub.index else 0 for a in aord]
        ax1.bar(x + (i-1)*w, vals, w, color=col, label=f"P@{k}")
    ax1.axhline(1.0, color="k", ls="--", lw=LW * 1.5, label="Random (1x)")
    ax1.set_xticks(x); ax1.set_xticklabels([alab[a] for a in aord])
    ax1.set_ylabel("Enrichment (P@K / random)"); ax1.set_title("(B) Enrichment vs. random")
    ax1.legend()

    fig.tight_layout()
    save(fig, "panel_si_precisionk")


# ════════════════════════════════════════════════════════════════════════════
# 11. Architecture ablation (SI-27)
# ════════════════════════════════════════════════════════════════════════════
def si_arch_ablation():
    df = pd.read_csv(ABLAT / "metrics.csv")
    mord  = ["multitask_projection", "cdr_framework_moe", "representation_fusion",
             "graph_moe", "uncertainty_moe", "abdevx2_full"]
    mlab  = {"multitask_projection": "Multitask\nproj.", "cdr_framework_moe": "CDR-fwk\nMoE",
              "representation_fusion": "Rep.\nfusion", "graph_moe": "Graph\nMoE",
              "uncertainty_moe": "Uncert.\nMoE", "abdevx2_full": "Full\n2.0"}
    splits = ["random", "cluster", "cluster_isotype"]
    slab   = {"random": "Random", "cluster": "Cluster", "cluster_isotype": "Cluster-isotype"}
    scols  = [PAL[0], PAL[1], PAL[2]]
    x = np.arange(len(mord))
    w = 0.25

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.44))

    for i, (sp, col) in enumerate(zip(splits, scols)):
        sub = df[df["split"] == sp].set_index("method")
        means = [sub.loc[m, "MEAN"] if m in sub.index else np.nan for m in mord]
        ax0.bar(x + (i-1)*w, means, w, color=col, label=slab[sp])
    ax0.axhline(0.462, color="k", ls="--", lw=LW * 1.5, label="AbDev-X")
    ax0.set_xticks(x); ax0.set_xticklabels([mlab[m] for m in mord])
    ax0.set_ylabel("Mean Spearman rho"); ax0.set_title("(A) Architecture ablation: mean across assays")
    ax0.axhline(0, color="grey", lw=0.5); ax0.legend(fontsize=FS_B - 1)

    sub_ci = df[df["split"] == "cluster_isotype"].set_index("method")
    acols5 = PAL[:5]
    for j, (a, acol) in enumerate(zip(["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"], acols5)):
        vals = [sub_ci.loc[m, a] if m in sub_ci.index else np.nan for m in mord]
        ax1.bar(x + (j-2)*0.15, vals, 0.14, color=acol, label=ALABEL.get(a, a))
    ax1.set_xticks(x); ax1.set_xticklabels([mlab[m] for m in mord])
    ax1.axhline(0, color="grey", lw=0.5)
    ax1.set_ylabel("CV Spearman rho"); ax1.set_title("(B) Per-assay (cluster-isotype CV)")
    ax1.legend(fontsize=FS_B - 1, ncol=3)

    fig.tight_layout()
    save(fig, "panel_si_arch_ablation")


# ════════════════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Building journal-format PDF panels ...")
    fig23()
    fig45()
    fig67()
    fix_fig11()
    split_fig13()
    si_crossabsense()
    si_encoder_ext()
    si_struct_sources()
    si_abstention()
    si_precisionk()
    si_arch_ablation()
    print(f"Done. Figures written to {OUT}")
