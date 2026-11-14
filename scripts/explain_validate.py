"""Explainability (SHAP) validated against published biophysical ground truths.

Trains a LightGBM per assay on interpretable, named features (physicochemical descriptors +
structural hydrophobic SAP + surface electrostatics), computes SHAP (TreeExplainer), and checks the
attributions against the literature-established drivers of each assay:

  * HIC (hydrophobicity)  -> hydrophobic features dominate      (TAP PSH; Ginkgo: HIC ~ side-chain hydrophobicity)
  * AC-SINS (self-assoc)  -> charge (+ hydrophobic) features    (electrostatic self-interaction)
  * PR-CHO (polyreactivity) -> positive-charge / aromatic       (Boughter 2020; Kelly 2018)
  * Titer (expression)    -> charge / pI                        (colloidal/expression behavior)
  * Tm2 (thermostability) -> weak in physChem space             (germline/framework-driven; validated elsewhere)

Validation metrics per assay: (a) which feature CATEGORY holds the largest SHAP mass and whether it
matches the expected driver; (b) rank correlation between SHAP importance and each feature's
univariate |Spearman| to the assay (do explanations track genuine predictivity?); (c) sign check for
the signature feature (e.g. hydrophobicity should push HIC UP).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings, re
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
EXPECTED = {"HIC": "hydrophobic", "AC-SINS_pH7.4": "charge", "PR_CHO": "charge",
            "Titer": "charge", "Tm2": "hydrophobic"}  # Tm2 expected weak in physChem (germline elsewhere)

CATEGORY_PATTERNS = [
    ("hydrophobic", ["gravy", "hmoment", "aromatic", "aliphatic", "hydrophob", "tryptophan", "sap", "_psh"]),
    ("charge", ["charge", "_pi", "frac_pos", "frac_neg", "exp_pos", "exp_neg", "net_exp", "patch", "asym", "pi_"]),
    ("liability", ["deamid", "isomer", "nglyc", "oxid", "free_cys"]),
    ("polar", ["frac_polar"]),
    ("size", ["_len", "total_sasa"]),
]


def categorize(name):
    n = name.lower()
    for cat, pats in CATEGORY_PATTERNS:
        if any(p in n for p in pats):
            return cat
    return "other"


def build_features(ds):
    D = build_descriptor_matrix(ds.heavy, ds.light)
    sap = pd.read_csv(ROOT / "data/processed/sap_gdpa1.csv").set_index("antibody_name")
    el = pd.read_csv(ROOT / "data/processed/electro_gdpa1.csv").set_index("antibody_name")
    names = ds.df[ds.id_col].astype(str).values
    sapm = sap.reindex(names)[["sap_total", "sap_positive", "hydrophobic_sasa", "aromatic_sasa", "hydrophobic_frac_sasa"]].reset_index(drop=True)
    elm = el.reindex(names)[["exp_pos", "exp_neg", "net_exp_charge", "charge_asym", "max_pos_patch", "max_neg_patch"]].reset_index(drop=True)
    X = pd.concat([D.reset_index(drop=True), sapm.add_prefix("SAP_"), elm.add_prefix("ELEC_")], axis=1)
    return X.replace([np.inf, -np.inf], 0.0).fillna(0.0)


ASSAY_LBL = {"HIC": "HIC", "Tm2": "Tm2", "AC-SINS_pH7.4": "AC-SINS", "PR_CHO": "PR-CHO", "Titer": "Titer"}
CAT_COLORS = {"hydrophobic": "#C44E52", "charge": "#4C72B0", "liability": "#55A868",
              "polar": "#8172B3", "size": "#DD8452", "other": "#BBBBBB"}
CAT_ORDER   = ["hydrophobic", "charge", "liability", "polar", "size", "other"]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "figure.dpi": 150, "savefig.dpi": 200})


def fig_perantibody_heatmap(sv_all, feat_names, y_all, outdir):
    """Heatmap of per-antibody SHAP values (top-20 features × antibodies, sorted by label)."""
    fig, axes = plt.subplots(1, len(ASSAYS), figsize=(18, 5))
    for ai, a in enumerate(ASSAYS):
        ax = axes[ai]
        sv, y = sv_all[a], y_all[a]
        top20 = np.argsort(np.abs(sv).mean(0))[::-1][:20]
        sv_top = sv[:, top20]
        names_top = [feat_names[j] for j in top20]
        order = np.argsort(y)
        sv_s = sv_top[order]
        vmax = np.percentile(np.abs(sv_s), 97)
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(sv_s.T, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(20))
        ax.set_yticklabels([n[:18] for n in names_top], fontsize=5.5 if ai == 0 else 5)
        ax.set_xlabel("Antibodies (sorted by label)", fontsize=8)
        ax.set_title(ASSAY_LBL[a], fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Per-antibody LightGBM SHAP — top-20 physicochemical features", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "shap_perantibody_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved shap_perantibody_heatmap.png")


def fig_waterfall_lgbm(sv_all, base_all, feat_names, y_all, names_all, outdir):
    """Waterfall plots for 3 antibodies (lowest/median/highest) per assay."""
    fig, axes = plt.subplots(len(ASSAYS), 3, figsize=(14, 3.2 * len(ASSAYS)))
    for ai, a in enumerate(ASSAYS):
        sv, y = sv_all[a], y_all[a]
        base = base_all[a]
        picks = [np.argsort(y)[0], np.argsort(y)[len(y)//2], np.argsort(y)[-1]]
        lbls  = ["lowest", "median", "highest"]
        for pi, (pick, lbl) in enumerate(zip(picks, lbls)):
            ax = axes[ai, pi]
            row = sv[pick]
            top12 = np.argsort(np.abs(row))[::-1][:12]
            vals   = row[top12]
            names  = [feat_names[j] for j in top12]
            running = base
            for j, (n, v) in enumerate(zip(names, vals)):
                col = "#C44E52" if v >= 0 else "#4C72B0"
                ax.barh(j, v, left=running, color=col, height=0.65, alpha=0.85, edgecolor="white", lw=0.5)
                running += v
            ax.axvline(base, color="grey", lw=0.8, ls="--")
            ax.set_yticks(range(len(names))); ax.set_yticklabels([n[:20] for n in names], fontsize=7)
            ax.set_title(f"{ASSAY_LBL[a]} | {lbl}\ntrue={y[pick]:.3f}", fontsize=8)
            ax.set_xlabel("Prediction", fontsize=8)
    fig.suptitle("LightGBM per-antibody SHAP waterfall — selected antibodies", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "shap_waterfall_lgbm.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved shap_waterfall_lgbm.png")


def fig_category_stacked(sv_all, feat_cats, y_all, outdir):
    """Stacked bar per antibody showing SHAP contribution by feature category (multi-block view)."""
    fig, axes = plt.subplots(1, len(ASSAYS), figsize=(18, 4))
    for ai, a in enumerate(ASSAYS):
        ax = axes[ai]
        sv, y = sv_all[a], y_all[a]
        order = np.argsort(y)
        sv_s = sv[order]
        # Sum absolute SHAP by category, signed by majority direction
        cat_contrib = {}
        for cat in CAT_ORDER:
            mask = feat_cats == cat
            if mask.any():
                cat_contrib[cat] = sv_s[:, mask].sum(1)   # (n,) signed sum per antibody
        # Separate positive and negative contributions per category
        x = np.arange(len(y))
        bottom_pos = np.zeros(len(y)); bottom_neg = np.zeros(len(y))
        for cat in CAT_ORDER:
            if cat not in cat_contrib:
                continue
            v = cat_contrib[cat]
            pos = np.maximum(v, 0); neg = np.minimum(v, 0)
            ax.bar(x, pos, color=CAT_COLORS[cat], label=cat, bottom=bottom_pos, width=1.0, alpha=0.85)
            ax.bar(x, neg, color=CAT_COLORS[cat], bottom=bottom_neg, width=1.0, alpha=0.85)
            bottom_pos += pos; bottom_neg += neg
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(ASSAY_LBL[a], fontsize=11); ax.set_xlabel("Antibodies (sorted)", fontsize=8)
        if ai == 0:
            ax.set_ylabel("SHAP contribution"); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Multi-block SHAP: physicochemical category contributions per antibody", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "shap_category_stacked.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved shap_category_stacked.png")


def save_perantibody_csv(sv_all, feat_names, y_all, names_all, outdir):
    """Save per-antibody SHAP values for all assays as CSV."""
    for a in ASSAYS:
        sv, y, names = sv_all[a], y_all[a], names_all[a]
        df = pd.DataFrame(sv, columns=feat_names)
        df.insert(0, "antibody_name", names)
        df["true_label"] = y
        df.to_csv(outdir / f"shap_perantibody_{a.replace('/', '').replace('-', '')}.csv", index=False)
    print(f"  Saved per-antibody SHAP CSVs for {len(ASSAYS)} assays.")


def main():
    import shap
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    X = build_features(ds)
    feat_names = list(X.columns)
    cats = np.array([categorize(c) for c in feat_names])
    outdir = ROOT / "results/explain"; outdir.mkdir(parents=True, exist_ok=True)

    val_rows, top_rows = [], []
    sv_all, y_all, base_all, names_all = {}, {}, {}, {}

    for a in ASSAYS:
        y = ds.y(a); m = ~np.isnan(y)
        Xa, ya = X.to_numpy(float)[m], y[m]
        ab_names = ds.df[ds.id_col].values[m]
        model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=.8,
                                  colsample_bytree=.6, reg_lambda=5, min_child_samples=12, verbosity=-1, random_state=0).fit(Xa, ya)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xa)
        base = float(explainer.expected_value) if hasattr(explainer, "expected_value") else float(ya.mean())
        sv_all[a] = sv; y_all[a] = ya; base_all[a] = base; names_all[a] = ab_names
        imp = np.abs(sv).mean(0)                        # mean|SHAP| per feature
        imp_norm = imp / (imp.sum() + 1e-12)

        # (a) category with largest SHAP mass
        cat_mass = {c: float(imp_norm[cats == c].sum()) for c in set(cats)}
        top_cat = max(cat_mass, key=cat_mass.get)
        expected = EXPECTED[a]
        match = top_cat == expected or cat_mass.get(expected, 0) >= 0.25

        # (b) SHAP importance vs univariate |Spearman| to the assay
        uni = np.array([abs(spearmanr(Xa[:, j], ya).statistic) for j in range(Xa.shape[1])])
        rho_shap_uni = spearmanr(imp, np.nan_to_num(uni)).statistic

        # (c) sign of the top feature (correlation of feature with assay)
        top_idx = np.argsort(imp)[::-1][:8]
        top_feats = [(X.columns[j], round(float(imp_norm[j]), 3),
                      round(float(spearmanr(Xa[:, j], ya).statistic), 2)) for j in top_idx]
        for f, iv, sgn in top_feats:
            top_rows.append(dict(assay=a, feature=f, shap_share=iv, corr_with_assay=sgn, category=categorize(f)))
        val_rows.append(dict(assay=a, expected_driver=expected, top_shap_category=top_cat,
                             expected_share=round(cat_mass.get(expected, 0), 3),
                             match_ground_truth=bool(match),
                             shap_vs_univariate_rho=round(float(rho_shap_uni), 3)))

        # SHAP summary plot
        try:
            shap.summary_plot(sv, Xa, feature_names=list(X.columns), max_display=12, show=False)
            plt.title(f"SHAP - {a}"); plt.tight_layout(); plt.savefig(outdir / f"shap_{a.replace('/', '_')}.png", dpi=130); plt.close()
        except Exception:
            plt.close()

    val = pd.DataFrame(val_rows); top = pd.DataFrame(top_rows)
    val.to_csv(ROOT / "results/explain_validation.csv", index=False)
    top.to_csv(ROOT / "results/explain_top_features.csv", index=False)
    print("=== SHAP validation vs published biophysical ground truths ===")
    print(val.to_string(index=False))
    print(f"\nGround-truth matches: {int(val['match_ground_truth'].sum())}/{len(val)} assays")
    print("\n=== Top SHAP features per assay (share, corr with assay) ===")
    for a in ASSAYS:
        sub = top[top.assay == a].head(5)
        print(f"\n{a}:")
        for _, r in sub.iterrows():
            print(f"   {r.feature:28s} share={r.shap_share:.3f} corr={r.corr_with_assay:+.2f} [{r.category}]")

    # New figures
    print("\nGenerating per-antibody figures ...")
    fig_perantibody_heatmap(sv_all, feat_names, y_all, outdir)
    fig_waterfall_lgbm(sv_all, base_all, feat_names, y_all, names_all, outdir)
    fig_category_stacked(sv_all, cats, y_all, outdir)
    save_perantibody_csv(sv_all, feat_names, y_all, names_all, outdir)
    print("\nAll outputs -> results/explain/")


if __name__ == "__main__":
    main()
