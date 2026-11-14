"""Publication composite AbDev-X architecture panel (schematic + real result insets)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import image as mpimg
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "results/figures"
plt.rcParams.update({"font.size": 10})
W, H = 15.5, 8.4


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=9, tc="black", lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, color="#666666", lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 lw=lw, color=color, shrinkA=2, shrinkB=2))


def inset(fig, img_path, x, y, w, h, title=None):
    if not Path(img_path).exists():
        return
    ax = fig.add_axes([x / W, y / H, w / W, h / H]); ax.imshow(mpimg.imread(img_path)); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7.5, pad=1)


def main():
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    vir = plt.cm.viridis
    ax.text(W / 2, 8.05, "AbDev-X: multi-representation, biologically grounded, explainable stacking",
            ha="center", fontsize=13, weight="bold")

    # ---------- (a) INPUT ----------
    ax.text(0.15, 7.5, "(a) Input", fontsize=11, weight="bold")
    box(ax, 0.2, 5.35, 2.15, 1.5, "Antibody\n\nVH + VL sequence", "#ECECEC", fs=9)
    box(ax, 0.2, 3.65, 2.15, 1.35, "Predicted 3D\nstructure\n(AntiBodyBuilder3)", "#E3EEF6", fs=8.5)
    # real structure inset
    inset(fig, ROOT / "results/explain/structures/brazikumab_hic.png", 0.35, 2.0, 1.9, 1.55)

    # ---------- (b) REPRESENTATIONS ----------
    ax.text(2.75, 7.5, "(b) Six complementary representations", fontsize=11, weight="bold")
    reps = [
        ("ESM-2 650M  (general protein LM)", vir(0.12), "seq"),
        ("AbLang2  (antibody LM)", vir(0.28), "seq"),
        ("Germline / isotype (hc/lc subtype)", vir(0.44), "seq"),
        ("Physicochemical descriptors (165)", vir(0.60), "seq"),
        ("Structural features\nTAP · MOE · Aggrescan3D · AntiFold · SAP", vir(0.76), "struct"),
    ]
    ry = np.linspace(6.7, 1.05, len(reps)); rh = 0.9; rx = 2.75; rw = 3.5
    seq_src = (2.35, 6.05); struct_src = (2.35, 4.3)
    for (t, c, src), y in zip(reps, ry):
        box(ax, rx, y, rw, rh, t, c, fs=8.2, tc="white" if c[0] < 0.55 else "black")
        s = seq_src if src == "seq" else struct_src
        arrow(ax, s[0], s[1], rx, y + rh / 2, color="#888888" if src == "seq" else "#B4620A", lw=1.3)
    ax.text(2.5, 0.55, "sequence →", color="#888888", fontsize=8)
    ax.text(2.5, 0.2, "structure →", color="#B4620A", fontsize=8)

    # ---------- (c) BASE LEARNERS + STACKING ----------
    ax.text(6.5, 7.5, "(c) Base learners → stacking", fontsize=11, weight="bold")
    bx = 6.45; bw = 2.0
    for (t, c, src), y in zip(reps, ry):
        head = "Ridge  +  LightGBM" if src == "struct" else "Ridge"
        box(ax, bx, y, bw, rh, head, "#FAFAFA", fs=7.6)
        arrow(ax, rx + rw, y + rh / 2, bx, y + rh / 2)
    mx, my, mw, mh = 8.85, 3.1, 1.95, 2.4
    box(ax, mx, my, mw, mh, "Per-assay\nout-of-fold\nSTACKING\nmeta-learner\n(Ridge)\n\nleakage-free", "#DCE6F1", fs=8.8)
    for y in ry:
        arrow(ax, bx + bw, y + rh / 2, mx, my + mh / 2)
    # real meta-weight heatmap inset
    inset(fig, FIGDIR / "fig4_repweights.png", 8.7, 0.15, 3.0, 2.5, "learned meta-weights (interpretation)")

    # ---------- (d) OUTPUTS ----------
    ax.text(11.7, 7.5, "(d) Predictions", fontsize=11, weight="bold")
    outs = ["HIC", "Tm2", "AC-SINS", "PR-CHO", "Titer"]
    oy = np.linspace(6.4, 3.3, len(outs)); ox, ow, oh = 12.05, 1.9, 0.62
    for t, y in zip(outs, oy):
        box(ax, ox, y, ow, oh, t, "#E7F3E7", fs=9)
        arrow(ax, mx + mw, my + mh / 2, ox, y + oh / 2)
    ax.text(ox + ow / 2, 7.15, "Developability\n(Spearman ρ)", ha="center", fontsize=9, weight="bold")

    # ---------- (e) EXPLAINABILITY ----------
    box(ax, 12.0, 0.35, 3.3, 2.4, "", "#FFF6E9", ec="#E0A85A")
    ax.text(13.65, 2.55, "(e) Explainability", ha="center", fontsize=10, weight="bold")
    inset(fig, ROOT / "results/explain/structures/zolbetuximab_charge.png", 12.15, 0.5, 1.55, 1.75)
    inset(fig, ROOT / "results/explain/shap_HIC.png", 13.75, 0.5, 1.5, 1.9)
    ax.text(13.65, 0.2, "SHAP + residue attribution → 3D structure (PyMOL)", ha="center", fontsize=7)

    fig.savefig(FIGDIR / "fig0_architecture.png", dpi=200, bbox_inches="tight")
    print("Saved", FIGDIR / "fig0_architecture.png")


if __name__ == "__main__":
    main()
