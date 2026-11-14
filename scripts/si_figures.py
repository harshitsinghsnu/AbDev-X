"""Supplementary figures for the additional experiments: IG CDR enrichment, pooling ablation,
significance forest, cross-dataset transfer + augmentation scaling."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"; FIG = R / "figures"
ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
LBL = {"HIC": "HIC", "Tm2": "Tm2", "AC-SINS_pH7.4": "AC-SINS", "PR_CHO": "PR-CHO", "Titer": "Titer"}
def SEQ(n): return [plt.cm.viridis(v) for v in np.linspace(.12, .88, n)]


def fig_ig():
    df = pd.read_csv(R / "exp_ig.csv").set_index("assay").reindex(ASSAYS)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([LBL[a] for a in ASSAYS], df["enrichment"], color=SEQ(len(ASSAYS)))
    ax.axhline(1, color="red", ls="--", label="no enrichment (1×)")
    ax.set_ylabel("CDR hotspot enrichment (×)"); ax.set_title("Integrated Gradients: learned attribution is CDR-enriched")
    for i, v in enumerate(df["enrichment"]):
        ax.text(i, v + 0.05, f"{v:.1f}×", ha="center", fontsize=9)
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "figS6_ig_cdr.png", dpi=160); plt.close(fig)


def fig_pooling():
    # CDR and mean pooling: use exp_crossab_errorbars.csv (has per-fold std)
    # framework: point estimate only from exp_pooling.csv
    eb = pd.read_csv(R / "exp_crossab_errorbars.csv")
    pool_df = pd.read_csv(R / "exp_pooling.csv")
    mode_configs = [
        ("mean",      "VH+VL_mean", True),
        ("cdr",       "CDR_mean",   True),
        ("framework", None,         False),
    ]
    mode_labels = {"mean": "mean-pool", "cdr": "CDR-pool", "framework": "framework (pt est.)"}
    cols = SEQ(3)
    x = np.arange(len(ASSAYS)); w = 0.26

    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    for i, (mode, eb_model, has_eb) in enumerate(mode_configs):
        if has_eb:
            sub = eb[eb["model"] == eb_model].set_index("assay")
            vals = [sub.loc[a, "cv_mean"] if a in sub.index else np.nan for a in ASSAYS]
            errs = [sub.loc[a, "cv_std"]  if a in sub.index else 0      for a in ASSAYS]
            kw = {"yerr": errs, "capsize": 3, "error_kw": {"lw": 0.6}}
        else:
            row = pool_df[pool_df["pooling"] == mode].iloc[0]
            vals = [row[a] for a in ASSAYS]
            kw = {}
        ax.bar(x + (i - 1) * w, vals, w * 0.88, color=cols[i],
               label=mode_labels[mode], **kw)

    ax.set_xticks(x); ax.set_xticklabels([LBL[a] for a in ASSAYS], rotation=30, ha="right")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("CV Spearman rho (mean +/- fold std)")
    ax.set_title("ESM-2 pooling ablation (mean vs CDR vs framework pooling)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    fig.savefig(FIG / "figS7_pooling.png", dpi=160); plt.close(fig)


def fig_significance():
    df = pd.read_csv(R / "exp_significance.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = []; y = 0; cols = {"ESM2": SEQ(2)[0], "AbLang2": SEQ(2)[1]}
    for a in ASSAYS:
        for vs in ["ESM2", "AbLang2"]:
            r = df[(df.assay == a) & (df.vs == vs)].iloc[0]
            ax.errorbar(r["delta_rho"], y, xerr=[[r["delta_rho"] - r["ci_low"]], [r["ci_high"] - r["delta_rho"]]],
                        fmt="o", color=cols[vs], capsize=3)
            star = " *" if r["p_one_sided"] < 0.05 else ""
            labels.append(f"{LBL[a]} vs {vs}{star}"); y += 1
    ax.axvline(0, color="red", ls="--"); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δ Spearman ρ (AbDev-X − baseline), held-out [95% CI]")
    ax.set_title("Paired-bootstrap significance (held-out; * p<0.05)")
    fig.tight_layout(); fig.savefig(FIG / "figS8_significance.png", dpi=160); plt.close(fig)


def fig_transfer_scaling():
    tr = pd.read_csv(R / "exp_transfer.csv"); sc = pd.read_csv(R / "exp_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(tr)); w = 0.38
    axes[0].bar(x - w / 2, tr["GDPa1_to_FLAb"], w, color=SEQ(2)[0], label="GDPa1→FLAb")
    axes[0].bar(x + w / 2, tr["FLAb_to_GDPa1"], w, color=SEQ(2)[1], label="FLAb→GDPa1")
    axes[0].set_xticks(x); axes[0].set_xticklabels([LBL[a] for a in tr["assay"]]); axes[0].axhline(0, color="grey", lw=.6)
    axes[0].set_ylabel("Spearman ρ"); axes[0].set_title("Cross-dataset transfer (same assay)"); axes[0].legend()
    sc2 = sc.sort_values("delta")
    cols = ["#C44E52" if d < 0 else plt.cm.viridis(0.5) for d in sc2["delta"]]
    axes[1].barh(range(len(sc2)), sc2["delta"], color=cols)
    axes[1].set_yticks(range(len(sc2))); axes[1].set_yticklabels([f"{r.assay}: {r.study[:22]}" for _, r in sc2.iterrows()], fontsize=7)
    axes[1].axvline(0, color="grey", lw=.6); axes[1].set_xlabel("Δ held-out ρ from adding study")
    axes[1].set_title("Per-study augmentation contribution")
    fig.tight_layout(); fig.savefig(FIG / "figS9_transfer_scaling.png", dpi=160); plt.close(fig)


def main():
    fig_ig(); fig_pooling(); fig_significance(); fig_transfer_scaling()
    print("SI figures written:", [p.name for p in sorted(FIG.glob("figS*.png"))])


if __name__ == "__main__":
    main()
