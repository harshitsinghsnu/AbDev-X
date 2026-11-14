"""Generate ranking comparison figure from results/ranking_comparison.csv.

Panels:
  A) Composite Spearman rho by model (bar chart)
  B) P@5 / P@10 / P@20 for models that have them
  C) True vs predicted composite scatter (AbDev-X, loaded from antibody_ranking.csv)

Usage:
    python scripts/fig_ranking.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 200,
})

PROPS     = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
DIRECTION = {"HIC": -1, "Tm2": +1, "AC-SINS_pH7.4": -1, "PR_CHO": -1, "Titer": +1}

MODEL_ORDER_A = [
    "Random", "Descriptors", "ESM2-Ridge",
    "Desc+ESM2", "Desc+AbLang2", "All feats-Ridge", "AbDev-X (stacking)"
]
MODEL_ORDER_B = [
    "Random", "ESM2-Ridge", "AbDev-X (stacking)"
]
PALETTE_A = [
    "#bdbdbd", "#a6cee3", "#fdae61",
    "#74add1", "#d01c8b", "#4dac26", "#2166ac"
]
PALETTE_B = ["#bdbdbd", "#fdae61", "#2166ac"]
COLORS = dict(zip(MODEL_ORDER_A, PALETTE_A))


def fisher_z_ci(rho: float, n: int, alpha: float = 0.05):
    """95% CI via Fisher z-transform (asymptotic)."""
    if abs(rho) >= 1.0:
        return rho, rho
    z    = np.arctanh(rho)
    se_z = 1.0 / np.sqrt(max(n - 3, 1))
    z_crit = 1.96
    lo = np.tanh(z - z_crit * se_z)
    hi = np.tanh(z + z_crit * se_z)
    return float(lo), float(hi)


def clopper_pearson_ci(x: int, n: int, alpha: float = 0.05):
    """Exact Clopper-Pearson 95% CI for proportion x/n."""
    from scipy.stats import beta as beta_dist
    lo = beta_dist.ppf(alpha / 2,     x,     n - x + 1) if x > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, x + 1, n - x)     if x < n else 1.0
    return float(lo), float(hi)


def bootstrap_rho_ci(tc, pc, n_boot=2000, seed=42):
    """Bootstrap 95% CI for Spearman rho between tc and pc arrays."""
    from scipy.stats import spearmanr
    rng  = np.random.default_rng(seed)
    n    = len(tc)
    boot = [float(spearmanr(tc[idx := rng.integers(0, n, n)], pc[idx])[0])
            for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def composite_score(pred_df: pd.DataFrame) -> pd.Series:
    zscores = pd.DataFrame(index=pred_df.index)
    for p in PROPS:
        if p not in pred_df.columns:
            continue
        col = pred_df[p].astype(float)
        std = col.std()
        z = (col - col.mean()) / std if std > 1e-9 else col * 0.0
        zscores[p] = z * DIRECTION[p]
    return zscores.mean(axis=1)


def main():
    # ── load comparison table ─────────────────────────────────────────────────
    cmp_csv = ROOT / "results" / "ranking_comparison.csv"
    if not cmp_csv.exists():
        raise FileNotFoundError(f"Run scripts/rank_antibodies.py first: {cmp_csv}")
    cmp = pd.read_csv(cmp_csv)
    cmp = cmp.set_index("model").reindex(
        [m for m in MODEL_ORDER_A if m in cmp["model"].values])

    # ── true composite from GDPa3 (for scatter) ───────────────────────────────
    true_comp = None
    pred_comp = None
    ranking_csv = ROOT / "results" / "antibody_ranking.csv"
    g3_cfg_path = ROOT / "configs" / "gdpa3_eval.yaml"
    if ranking_csv.exists() and g3_cfg_path.exists():
        try:
            import yaml, sys
            sys.path.insert(0, str(ROOT / "src"))
            from abdev.data.dataset import resolve_column
            with open(g3_cfg_path) as f:
                g3cfg = yaml.safe_load(f)
            hcfg = g3cfg["heldout"]
            df3 = pd.read_csv(ROOT / hcfg["csv_path"])
            name_col = next((c for c in ["antibody_name", "antibody_id", "name"]
                             if c in df3.columns), None)
            true_labels = {}
            for prop, spec in hcfg["targets"].items():
                col = resolve_column(df3, spec["column"], required=False, what=prop)
                if col:
                    true_labels[prop] = df3[col].to_numpy(dtype=float)
            if true_labels:
                true_comp = composite_score(pd.DataFrame(true_labels)).values

            rk_df = pd.read_csv(ranking_csv)
            if "composite_score" in rk_df.columns and name_col:
                # align predictions to original GDPa3 row order via antibody name
                name_to_score = dict(zip(rk_df["antibody"].astype(str),
                                         rk_df["composite_score"]))
                pred_comp = df3[name_col].astype(str).map(name_to_score).values.astype(float)
            elif "composite_score" in rk_df.columns:
                # fallback: assume CSV row order matches GDPa3 original order
                pred_comp = rk_df.sort_values("rank")["composite_score"].values
        except Exception as e:
            print(f"  (scatter panel skipped: {e})")

    # ── figure ────────────────────────────────────────────────────────────────
    n_panels = 3 if (true_comp is not None and pred_comp is not None) else 2
    fig_w = 14 if n_panels == 3 else 10
    fig = plt.figure(figsize=(fig_w, 4.8))
    gs = GridSpec(1, n_panels, figure=fig, wspace=0.40,
                  left=0.06, right=0.97, top=0.88, bottom=0.20)

    models_a = [m for m in MODEL_ORDER_A if m in cmp.index]
    models_b = [m for m in MODEL_ORDER_B if m in cmp.index]

    models = models_a  # used for CI computation below
    rhos   = cmp.loc[models_a, "comp_rho"].values.astype(float)
    cols   = [COLORS.get(m, "#888888") for m in models_a]

    # ── compute CIs ───────────────────────────────────────────────────────────
    n_ab = 80
    ci_lo = np.zeros(len(models))
    ci_hi = np.zeros(len(models))
    for i, m in enumerate(models):
        rho = rhos[i]
        if m == "AbDev-X (stacking)" and true_comp is not None and pred_comp is not None:
            mv = ~(np.isnan(true_comp) | np.isnan(pred_comp))
            if mv.sum() > 5:
                ci_lo[i], ci_hi[i] = bootstrap_rho_ci(true_comp[mv], pred_comp[mv])
                continue
        ci_lo[i], ci_hi[i] = fisher_z_ci(rho, n_ab)

    # ── Panel A: Composite rho bar chart with 95% CI ──────────────────────────
    ax_a = fig.add_subplot(gs[0])
    bars = ax_a.barh(models_a, rhos, color=cols, edgecolor="white",
                     linewidth=0.5, height=0.65)
    # error bars (xerr expects [lo_err, hi_err] from centre)
    xerr_lo = rhos - ci_lo
    xerr_hi = ci_hi - rhos
    ax_a.errorbar(rhos, np.arange(len(models)),
                  xerr=[xerr_lo, xerr_hi],
                  fmt="none", ecolor="#333333", elinewidth=1.2,
                  capsize=3, capthick=1.2, zorder=5)

    for i, (m, bar) in enumerate(zip(models, bars)):
        if m == "AbDev-X (stacking)":
            bar.set_edgecolor("#1a1a1a")
            bar.set_linewidth(1.5)
        v = rhos[i]
        ax_a.text(ci_hi[i] + 0.012, i, f"{v:.3f}",
                  va="center", ha="left", fontsize=7.5,
                  fontweight="bold" if m == "AbDev-X (stacking)" else "normal")

    ax_a.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax_a.set_xlabel("Composite Spearman rho (GDPa3, n=80)\n95% CI: bootstrap (AbDev-X) / Fisher-z (others)",
                    fontsize=8)
    ax_a.set_title("(A) Composite ranking quality", fontweight="bold", pad=6)
    ax_a.set_xlim(-0.30, 0.55)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.tick_params(axis="y", labelsize=8)

    # ── Panel B: P@5/P@10/P@20 (subset of models) ────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    cols_b = [COLORS.get(m, "#888888") for m in models_b]
    p5  = cmp.loc[models_b, "P5"].apply(pd.to_numeric, errors="coerce").fillna(0).values
    p10 = cmp.loc[models_b, "P10"].apply(pd.to_numeric, errors="coerce").values \
          if "P10" in cmp.columns else np.full(len(models_b), np.nan)
    p20 = cmp.loc[models_b, "P20"].apply(pd.to_numeric, errors="coerce").values \
          if "P20" in cmp.columns else np.full(len(models_b), np.nan)

    x = np.arange(len(models_b))
    w = 0.26
    ax_b.bar(x - w, p5,  width=w, color=cols_b, alpha=0.95, edgecolor="white", linewidth=0.3, label="P@5")
    ax_b.bar(x,     p10, width=w, color=cols_b, alpha=0.65, edgecolor="white", linewidth=0.3, label="P@10")
    ax_b.bar(x + w, p20, width=w, color=cols_b, alpha=0.40, edgecolor="white", linewidth=0.3, label="P@20")

    # Clopper-Pearson 95% CI
    for pk_vals, k_val, x_offset in [(p5, 5, -w), (p10, 10, 0), (p20, 20, w)]:
        ci_lo_arr, ci_hi_arr = [], []
        for pv in pk_vals:
            if np.isnan(pv):
                ci_lo_arr.append(np.nan); ci_hi_arr.append(np.nan)
                continue
            hits = int(round(pv * k_val))
            lo, hi = clopper_pearson_ci(hits, k_val)
            ci_lo_arr.append(lo); ci_hi_arr.append(hi)
        ci_lo_arr = np.array(ci_lo_arr, dtype=float)
        ci_hi_arr = np.array(ci_hi_arr, dtype=float)
        valid = ~np.isnan(ci_lo_arr)
        if valid.any():
            ax_b.errorbar(x[valid] + x_offset, pk_vals[valid],
                          yerr=[pk_vals[valid] - ci_lo_arr[valid],
                                ci_hi_arr[valid] - pk_vals[valid]],
                          fmt="none", ecolor="#333333", elinewidth=0.9,
                          capsize=2.5, capthick=0.9, zorder=5)

    for k, y_rand, ls in [(5, 5/80, "-"), (10, 10/80, "--"), (20, 20/80, ":")]:
        ax_b.axhline(y_rand, color="gray", linewidth=0.8, linestyle=ls, alpha=0.6)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([m.replace(" (stacking)", "\n(stacking)")
                           .replace("-Ridge", "\n-Ridge")
                           for m in models_b],
                         fontsize=8, rotation=30, ha="right")
    ax_b.set_ylabel("Precision@K", fontsize=8.5)
    ax_b.set_title("(B) Precision@K — predicted top-K ∩ true top-K", fontweight="bold", pad=6)
    ax_b.set_ylim(0, 0.55)
    ax_b.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_b.spines[["top", "right"]].set_visible(False)

    # ── Panel C: True vs predicted composite (AbDev-X) ────────────────────────
    if n_panels == 3:
        from scipy.stats import spearmanr
        ax_c = fig.add_subplot(gs[2])
        m_v = ~(np.isnan(true_comp) | np.isnan(pred_comp))
        tc_v, pc_v = true_comp[m_v], pred_comp[m_v]

        true_rank = pd.Series(tc_v).rank(ascending=False).values
        norm = plt.Normalize(vmin=1, vmax=len(tc_v))
        sc = ax_c.scatter(tc_v, pc_v, c=true_rank, cmap="RdYlGn_r",
                          norm=norm, s=28, alpha=0.82,
                          edgecolors="white", linewidths=0.3, zorder=3)

        if len(tc_v) > 2:
            z = np.polyfit(tc_v, pc_v, 1)
            xr = np.linspace(tc_v.min(), tc_v.max(), 100)
            ax_c.plot(xr, np.polyval(z, xr), "k--", lw=1.0, alpha=0.6, zorder=2)

        pred_rank = pd.Series(pc_v).rank(ascending=False).astype(int).values
        n = len(pc_v)
        for idx in np.where(pred_rank <= 3)[0]:
            ax_c.annotate(f"#{pred_rank[idx]}", (tc_v[idx], pc_v[idx]),
                          fontsize=6.5, color="#2166ac",
                          xytext=(3, 2), textcoords="offset points")
        for idx in np.where(pred_rank >= n - 2)[0]:
            ax_c.annotate(f"#{pred_rank[idx]}", (tc_v[idx], pc_v[idx]),
                          fontsize=6.5, color="#d73027",
                          xytext=(3, 2), textcoords="offset points")

        rho_val = float(spearmanr(tc_v, pc_v)[0])
        ax_c.text(0.05, 0.95, f"rho = {rho_val:.3f}", transform=ax_c.transAxes,
                  fontsize=9, va="top",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        ax_c.set_xlabel("True composite score", fontsize=8.5)
        ax_c.set_ylabel("Predicted composite score (AbDev-X)", fontsize=8.5)
        ax_c.set_title("(C) AbDev-X: true vs predicted\n(green=best true rank, red=worst)",
                       fontweight="bold", pad=6)
        ax_c.spines[["top", "right"]].set_visible(False)
        cbar = fig.colorbar(sc, ax=ax_c, fraction=0.04, pad=0.04)
        cbar.set_label("True rank\n(1=best)", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)

    # ── legend strip ──────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=COLORS.get(m, "#888"), label=m) for m in models_a]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.0)

    fig.suptitle(
        "Antibody developability ranking on GDPa3 held-out (n=80). "
        "Composite score: mean z-scored sign-adjusted predictions across 5 assays.",
        fontsize=8.5, y=0.97)

    out = OUT / "fig_ranking.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved -> {out}")
    return str(out)


if __name__ == "__main__":
    main()
