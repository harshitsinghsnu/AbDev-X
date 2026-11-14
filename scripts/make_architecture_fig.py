"""Generate updated AbDev-X architecture figure (fig0_architecture_v3.pdf).

Layout mirrors slide 4:
  (a) Input  |  (b) Representations  |  (c) Four Experts + Fusion  |  (d) Outputs

New vs old:
  - Fusion column now shows 4 expert boxes (E1, A-MGFS, D-Stack, F-MoE)
    with assay-specific weighted combination
  - Outputs extended to all 10 assays
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT  = Path(__file__).resolve().parents[1]
OUT   = ROOT / "paper" / "paper10assay"
OUT2  = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
OUT2.mkdir(parents=True, exist_ok=True)

# ── Colour palette matching the slide ─────────────────────────────────────────
C_ESM    = "#B5313C"   # red      – ESM-2
C_ABLANG = "#9B59B6"   # purple   – AbLang2
C_GERM   = "#5DADE2"   # blue     – Germline/isotype
C_DESC   = "#58B85C"   # green    – Physicochemical
C_STRUCT = "#E5A020"   # amber    – Structural features
C_E1     = "#2E86AB"   # teal     – E1 expert
C_AMGFS  = "#58B85C"   # green    – A-MGFS
C_DSTACK = "#9B59B6"   # purple   – D-Stack
C_FMOE   = "#E5A020"   # amber    – F-MoE
C_FUSE   = "#1A3A6E"   # navy     – fusion/weights
C_OUT    = "#F0C27F"   # peach    – output assays

# ── Canvas ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18); ax.set_ylim(0, 10)
ax.axis("off")

# helper: rounded rectangle
def rbox(ax, x, y, w, h, color, text, fontsize=8.5, text_color="white",
         alpha=1.0, bold=False, lw=0):
    fc = mpatches.FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.08",
                                 facecolor=color, edgecolor="white",
                                 linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(fc)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold" if bold else "normal",
            wrap=True, zorder=4)

# helper: arrow
def arrow(ax, x0, y0, x1, y1, color="#444444", lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=5)

# ── Column labels ──────────────────────────────────────────────────────────────
for xc, lbl in [(1.3, "(a) Input"), (5.2, "(b) Representations"),
                (10.5, "(c) Four Experts + Fusion"), (16.0, "(d) Outputs")]:
    ax.text(xc, 9.6, lbl, ha="center", va="center", fontsize=10,
            color="#333333", style="italic")

# dividers
for xd in [2.9, 7.6, 14.1]:
    ax.axvline(xd, color="#cccccc", lw=0.8, ls="--", zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# (a) INPUT
# ══════════════════════════════════════════════════════════════════════════════
ax.text(1.3, 7.8, "Antibody VH+VL\nsequence", ha="center", va="center",
        fontsize=8.5, color="#333")
ax.text(1.3, 6.3, "+ predicted 3D\nstructure\n(ABodyBuilder3)", ha="center",
        va="center", fontsize=8, color="#555")

# stylised antibody shape (Y)
for ang, (dx, dy) in zip([0], [(0, 0)]):
    ax.plot([1.3, 1.1, 1.0], [5.6, 5.0, 4.4], color="#4472C4", lw=2.5, solid_capstyle="round")
    ax.plot([1.3, 1.5, 1.6], [5.6, 5.0, 4.4], color="#70AD47", lw=2.5, solid_capstyle="round")
    ax.plot([1.3, 1.3],      [5.6, 4.0],       color="#7030A0", lw=2.5, solid_capstyle="round")
ax.text(1.3, 3.7, "Fab", ha="center", va="center", fontsize=7.5, color="#555")

# ══════════════════════════════════════════════════════════════════════════════
# (b) REPRESENTATIONS  (5 blocks)
# ══════════════════════════════════════════════════════════════════════════════
reps = [
    (C_ESM,    "ESM-2 650M\n(general PLM)"),
    (C_ABLANG, "AbLang2\n(antibody PLM)"),
    (C_GERM,   "Germline / Isotype\n(HC/LC subtype: IgG1/2/4, K/λ)"),
    (C_DESC,   "Physicochemical\ndescriptors (165)"),
    (C_STRUCT, "Structural features\n(SAP · TAP · Aggrescan3D · AntiFold)"),
]
x_rep = 3.1; w_rep = 4.2; h_rep = 0.80
y_starts = [8.6, 7.5, 6.35, 5.2, 4.05]

for (col, lbl), y0 in zip(reps, y_starts):
    rbox(ax, x_rep, y0, w_rep, h_rep, col, lbl, fontsize=8)

# arrows: input → each rep block
for y0, h in zip(y_starts, [h_rep]*5):
    arrow(ax, 2.9, y0 + h/2, x_rep, y0 + h/2)

# ══════════════════════════════════════════════════════════════════════════════
# (c) FOUR EXPERTS + FUSION
# ══════════════════════════════════════════════════════════════════════════════
x_exp = 7.8; w_exp = 2.4; h_exp = 0.90

experts = [
    (C_E1,     "E1\nBest-Block\nSelector",          8.1),
    (C_AMGFS,  "A-MGFS\nAssay-Masked\nDescriptors", 6.8),
    (C_DSTACK, "D-Stack\nPLM+Desc\nRidge Stacker",  5.5),
    (C_FMOE,   "F-MoE\nFeature-Level\nMixture",     4.2),
]

for col, lbl, y0 in experts:
    rbox(ax, x_exp, y0, w_exp, h_exp, col, lbl, fontsize=7.8)

# arrows: rep blocks → each expert (simplified fan)
rep_right = x_rep + w_rep   # 7.3
for _, _, y_exp in experts:
    for y_rep, h in zip(y_starts, [h_rep]*5):
        ax.annotate("", xy=(x_exp, y_exp + h_exp/2),
                    xytext=(rep_right, y_rep + h/2),
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                    lw=0.6, connectionstyle="arc3,rad=0.0"),
                    zorder=2)

# expert → fusion arrow heads
x_exp_right = x_exp + w_exp   # 10.2
for _, _, y0 in experts:
    arrow(ax, x_exp_right, y0 + h_exp/2, 11.0, y0 + h_exp/2, lw=1.0)

# ── Fusion / weight combination box ───────────────────────────────────────────
x_fuse = 11.0; w_fuse = 2.7; h_fuse = 5.0; y_fuse = 3.6
rbox(ax, x_fuse, y_fuse, w_fuse, h_fuse, C_FUSE,
     "Assay-specific\nlearned weights\n(sum-to-1, SLSQP)\n\nŷ⁽ᵃ⁾ = Σₖ wₖ⁽ᵃ⁾ ŷₖ⁽ᵃ⁾",
     fontsize=8, bold=False)

# mini weight-heatmap inside fusion box
hm_x = x_fuse + 0.25; hm_y = y_fuse + 0.35
hm_w = 0.5; hm_h = 0.4
hm_data = np.array([[0.6, 0.1, 0.2, 0.1],
                     [0.1, 0.5, 0.3, 0.1],
                     [0.3, 0.2, 0.4, 0.1],
                     [0.1, 0.1, 0.5, 0.3],
                     [0.2, 0.3, 0.1, 0.4]])
for ri in range(5):
    for ci in range(4):
        v = hm_data[ri, ci]
        c_hm = plt.cm.Blues(0.3 + 0.7*v)
        fc = mpatches.FancyBboxPatch(
            (hm_x + ci*hm_w, hm_y + ri*hm_h), hm_w*0.9, hm_h*0.85,
            boxstyle="square,pad=0", facecolor=c_hm, edgecolor="white",
            linewidth=0.3, zorder=4)
        ax.add_patch(fc)
ax.text(x_fuse + w_fuse/2, y_fuse + 0.22,
        "Expert weights per assay",
        ha="center", va="center", fontsize=6.5, color="#aaccff", zorder=5)

# arrow: fusion → outputs
arrow(ax, x_fuse + w_fuse, y_fuse + h_fuse/2, 13.9, y_fuse + h_fuse/2, lw=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# (d) OUTPUTS – all 10 assays
# ══════════════════════════════════════════════════════════════════════════════
assays = [
    ("HAC",      "Aggregation"),
    ("SMAC",     "Aggregation"),
    ("HIC",      "Hydrophobicity"),
    ("AC-SINS",  "Self-association"),
    ("PR-CHO",   "Expression"),
    ("Tm2",      "Thermostability"),
    ("Tm1",      "Thermostability"),
    ("Tonset",   "Thermostability"),
    ("SEC",      "Colloidal"),
    ("Titer",    "Expression"),
]
x_out = 14.1; w_out = 3.4; h_out = 0.60; gap = 0.14
total_h = len(assays) * (h_out + gap)
y_top = y_fuse + h_fuse  # top of fusion box

for i, (name, cat) in enumerate(assays):
    y0 = y_top - (i+1)*(h_out + gap) + gap
    rbox(ax, x_out, y0, w_out, h_out, C_OUT, name,
         fontsize=8.5, text_color="#333333", bold=True)
    ax.text(x_out + w_out + 0.08, y0 + h_out/2,
            cat, ha="left", va="center", fontsize=6.8, color="#888")
    # connector from fusion
    ax.annotate("", xy=(x_out, y0 + h_out/2),
                xytext=(x_fuse + w_fuse, y_fuse + h_fuse/2),
                arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                lw=0.5, connectionstyle="arc3,rad=0.0"),
                zorder=2)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(9.0, 10.0, "AbDev-X", ha="center", va="center",
        fontsize=16, fontweight="bold", color="#1A3A6E")

# ── Legend for experts ─────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_E1,     label="E1 — per-assay best-block selector"),
    mpatches.Patch(facecolor=C_AMGFS,  label="A-MGFS — assay-specific masked descriptors"),
    mpatches.Patch(facecolor=C_DSTACK, label="D-Stack — PLM+descriptor ridge meta-learner"),
    mpatches.Patch(facecolor=C_FMOE,   label="F-MoE — feature-level mixture of experts"),
]
ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.04),
          ncol=4, fontsize=7.5, frameon=True, framealpha=0.9,
          handlelength=1.0, handletextpad=0.4, columnspacing=1.0)

fig.tight_layout(pad=0.2)

for d in [OUT, OUT2]:
    fig.savefig(str(d / "fig0_architecture_v3.pdf"), format="pdf",
                bbox_inches="tight", dpi=300)
    fig.savefig(str(d / "fig0_architecture_v3.png"), format="png",
                bbox_inches="tight", dpi=200)

plt.close(fig)
print("Saved fig0_architecture_v3.pdf + .png")
