"""Publication-ready explainability panel for AbDev-X.

Layout (2-row, 3-column + optional structure inset):

  A  |  B  |  C
  D  |  E  |  F

A — Block SHAP contribution fractions per assay (stacked bars)
B — Per-antibody block SHAP heatmap (HIC — most interpretable assay)
C — LightGBM top-feature bars for HIC, AC-SINS, PR_CHO (named features)
D — Per-antibody LightGBM category SHAP stacked bars (multi-block view)
E — VH per-residue ESM2-150M gradient attribution traces (all 5 assays)
F — PyMOL 3D structure surface (HIC best antibody), or CDR/FR bar if no image

Saves: results/figures/panel_explainability.png
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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import lightgbm as lgb
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix
import gdpa3_eval as GE
from abdev.models.fusion import AbDevFusion, DEFAULT_BASES, BASE_BLOCK

ASSAYS    = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
ASSAY_LBL = {"HIC": "HIC", "Tm2": "Tm2", "AC-SINS_pH7.4": "AC-SINS",
              "PR_CHO": "PR-CHO", "Titer": "Titer"}
BLOCK_LBL  = ["ESM2", "Desc", "Isotype", "Bio(ridge)", "Bio(lgbm)", "AbLang2"]
# Block colors — unified with make_combined_panels.py representation weight heatmap accent
BLOCK_COL  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#bcbd22", "#17becf"]

# Assay colors — match ASSAY_COLS in make_combined_panels.py exactly
ASSAY_COL  = {"HIC": "#1f77b4", "Tm2": "#ff7f0e", "AC-SINS_pH7.4": "#2ca02c",
              "PR_CHO": "#9467bd", "Titer": "#bcbd22"}

CAT_ORDER  = ["hydrophobic", "charge", "liability", "polar", "size", "other"]
CAT_COL    = {"hydrophobic": "#ff7f0e", "charge": "#1f77b4", "liability": "#2ca02c",
              "polar": "#9467bd", "size": "#bcbd22", "other": "#BBBBBB"}
CAT_PATS   = [
    ("hydrophobic", ["gravy", "hmoment", "aromatic", "aliphatic", "hydrophob", "tryptophan", "sap", "_psh"]),
    ("charge",      ["charge", "_pi", "frac_pos", "frac_neg", "exp_pos", "exp_neg", "net_exp",
                     "patch", "asym", "pi_"]),
    ("liability",   ["deamid", "isomer", "nglyc", "oxid", "free_cys"]),
    ("polar",       ["frac_polar"]),
    ("size",        ["_len", "total_sasa"]),
]

STRUCT_DIR = ROOT / "external/abdev-benchmark/data/structures/AntiBodyBuilder3/GDPa1"
FIG_OUT    = ROOT / "results/figures"; FIG_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 220,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ─── data helpers ─────────────────────────────────────────────────────────────

def categorize(name):
    n = name.lower()
    for cat, pats in CAT_PATS:
        if any(p in n for p in pats):
            return cat
    return "other"


def build_features(ds):
    D   = build_descriptor_matrix(ds.heavy, ds.light)
    sap = pd.read_csv(ROOT / "data/processed/sap_gdpa1.csv").set_index("antibody_name")
    el  = pd.read_csv(ROOT / "data/processed/electro_gdpa1.csv").set_index("antibody_name")
    ids = ds.df[ds.id_col].astype(str).values
    sapm = sap.reindex(ids)[["sap_total", "sap_positive", "hydrophobic_sasa",
                              "aromatic_sasa", "hydrophobic_frac_sasa"]].reset_index(drop=True)
    elm  = el.reindex(ids)[["exp_pos", "exp_neg", "net_exp_charge",
                             "charge_asym", "max_pos_patch", "max_neg_patch"]].reset_index(drop=True)
    X = pd.concat([D.reset_index(drop=True),
                   sapm.add_prefix("SAP_"), elm.add_prefix("ELEC_")], axis=1)
    return X.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_lgbm_shap(ds):
    X     = build_features(ds)
    fcols = list(X.columns)
    cats  = np.array([categorize(c) for c in fcols])
    sv_all, y_all, base_all, imp_all = {}, {}, {}, {}
    for a in ASSAYS:
        y = ds.y(a); m = ~np.isnan(y)
        Xa, ya = X.to_numpy(float)[m], y[m]
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                 subsample=.8, colsample_bytree=.6, reg_lambda=5,
                                 min_child_samples=12, verbosity=-1, random_state=0).fit(Xa, ya)
        expl = shap.TreeExplainer(mdl)
        sv   = expl.shap_values(Xa)
        sv_all[a]  = sv
        y_all[a]   = ya
        base_all[a] = float(getattr(expl, "expected_value", ya.mean()))
        imp_all[a]  = np.abs(sv).mean(0)
    return sv_all, y_all, base_all, imp_all, fcols, cats


def fit_abdevx_block_shap(ds, B1):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    kf  = KFold(5, shuffle=True, random_state=0)
    oof_all, coef_all, icpt_all = {}, {}, {}
    for assay in ASSAYS:
        y   = ds.y(assay); lab = ~np.isnan(y)
        B1m = {k: v[lab] for k, v in B1.items()}
        n   = lab.sum()
        fus = AbDevFusion(random_state=0)
        Xs  = fus._stack(B1m)
        oof = [np.full(n, np.nan) for _ in DEFAULT_BASES]
        for tr, va in kf.split(np.arange(n)):
            for bi, (name, mk) in enumerate(DEFAULT_BASES):
                oof[bi][va] = mk().fit(Xs[bi][tr], y[lab][tr]).predict(Xs[bi][va])
        Z    = np.column_stack(oof)
        meta = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(Z, y[lab])
        b    = meta.coef_
        shap_vals = b[None, :] * (Z - Z.mean(0)[None, :])
        oof_all[assay]  = shap_vals
        coef_all[assay] = b
        icpt_all[assay] = meta.intercept_
    return oof_all, coef_all, icpt_all


def load_esm150m(ds):
    pr   = np.load(ROOT / "data/processed/esm_perres_150M.npz", allow_pickle=True)
    vh_e = pr["vh_e"].astype(np.float32)
    vh_m = pr["vh_m"].astype(np.float32)
    vh_c = pr["vh_c"].astype(bool)
    vl_e = pr["vl_e"].astype(np.float32)
    vl_m = pr["vl_m"].astype(np.float32)

    def mpool(e, m):
        return (e * m[:, :, None]).sum(1) / m.sum(1, keepdims=True).clip(min=1)

    X_150m = np.nan_to_num(np.concatenate([mpool(vh_e, vh_m), mpool(vl_e, vl_m)], 1))
    vh_len = vh_m.sum(1).astype(int)
    vl_len = vl_m.sum(1).astype(int)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV

    esm_attr = {}
    for assay in ASSAYS:
        y   = ds.y(assay); lab = ~np.isnan(y)
        pipe = Pipeline([("s", StandardScaler()),
                         ("p", PCA(0.9, random_state=0, svd_solver="full")),
                         ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])
        pipe.fit(X_150m[lab], y[lab])
        ridge = pipe.named_steps["r"]
        pca   = pipe.named_steps["p"]
        sc    = pipe.named_steps["s"]
        proj_pca = pca.components_.T @ ridge.coef_
        proj = proj_pca / (sc.scale_ + 1e-30)
        proj_vh = proj[:640]; proj_vl = proj[640:]
        n = 246
        vh_attr = np.zeros((n, 140))
        for i in range(n):
            Lv = max(vh_len[i], 1)
            vh_attr[i] = (vh_e[i] @ proj_vh) / Lv
        esm_attr[assay] = {"vh_attr": vh_attr, "vh_m": vh_m.astype(bool), "vh_c": vh_c}
    return esm_attr


# ─── panel drawing functions ───────────────────────────────────────────────────

def draw_A(ax, oof_all, coef_all, ds):
    """Block SHAP contribution fractions per assay — stacked horizontal bars."""
    n_assays = len(ASSAYS)
    shap_share = np.zeros((n_assays, 6))
    for ai, assay in enumerate(ASSAYS):
        S   = oof_all[assay]
        tot = np.abs(S).sum() + 1e-9
        for bi in range(6):
            shap_share[ai, bi] = np.abs(S[:, bi]).sum() / tot

    y_pos = np.arange(n_assays)
    left  = np.zeros(n_assays)
    for bi, (bname, col) in enumerate(zip(BLOCK_LBL, BLOCK_COL)):
        ax.barh(y_pos, shap_share[:, bi], left=left, color=col,
                label=bname, height=0.65, alpha=0.88)
        left += shap_share[:, bi]

    ax.set_yticks(y_pos)
    ax.set_yticklabels([ASSAY_LBL[a] for a in ASSAYS], fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("Fraction of total |SHAP|")
    ax.set_title("A  Block contribution per assay")
    ax.legend(loc="lower right", fontsize=7, ncol=2,
              framealpha=0.7, handlelength=1.2)
    ax.invert_yaxis()


def draw_B(ax, oof_all, ds, assay="HIC"):
    """Per-antibody block SHAP heatmap for one assay."""
    y   = ds.y(assay); lab = ~np.isnan(y)
    S   = oof_all[assay]
    y_l = y[lab]; order = np.argsort(y_l)
    S_s = S[order].T      # (6, n_lab)
    vmax = np.percentile(np.abs(S_s), 97)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax.imshow(S_s, cmap="RdBu_r", norm=norm, aspect="auto",
                     interpolation="nearest")
    ax.set_yticks(range(6)); ax.set_yticklabels(BLOCK_LBL, fontsize=8)
    ax.set_xlabel("Antibodies  (sorted by label)", fontsize=8)
    ax.set_title(f"B  Per-antibody block SHAP  ({ASSAY_LBL[assay]})")
    plt.colorbar(im, ax=ax, shrink=0.8, label="SHAP", pad=0.02)


def draw_C(ax, imp_all, fcols, cats, assays_show=("HIC", "AC-SINS_pH7.4", "PR_CHO")):
    """Top-8 named features for selected assays — grouped horizontal bars."""
    n_show  = 8
    n_a     = len(assays_show)
    colors_a = [ASSAY_COL[a] for a in assays_show]
    gap     = n_show + 1           # gap between assay groups

    yticks, ylabels = [], []
    for ai, assay in enumerate(assays_show):
        imp = imp_all[assay]
        top = np.argsort(imp)[::-1][:n_show]
        imp_norm = imp / (imp.sum() + 1e-9)
        base = ai * gap
        for j, fi in enumerate(top):
            y_pos = base + j
            ax.barh(y_pos, imp_norm[fi],
                    color=colors_a[ai], alpha=0.8, height=0.75)
            yticks.append(y_pos)
            ylabels.append(fcols[fi].replace("SAP_", "").replace("ELEC_", "")
                           .replace("_pH7.4", "")[:22])

    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlabel("SHAP share")
    ax.set_title("C  Named feature importance (LightGBM SHAP)")

    patches = [mpatches.Patch(color=c, label=ASSAY_LBL[a])
               for c, a in zip(colors_a, assays_show)]
    ax.legend(handles=patches, fontsize=7, framealpha=0.7)


def draw_D(ax, sv_all, y_all, cats):
    """Category-grouped LightGBM SHAP stacked bars per antibody (HIC shown)."""
    assay = "HIC"
    sv, y = sv_all[assay], y_all[assay]
    order = np.argsort(y); sv_s = sv[order]
    x     = np.arange(len(y))

    bottom_pos = np.zeros(len(y)); bottom_neg = np.zeros(len(y))
    for cat in CAT_ORDER:
        mask = cats == cat
        if not mask.any():
            continue
        v   = sv_s[:, mask].sum(1)
        pos = np.maximum(v, 0); neg = np.minimum(v, 0)
        ax.bar(x, pos, color=CAT_COL[cat], label=cat,
               bottom=bottom_pos, width=1.0, alpha=0.85)
        ax.bar(x, neg, color=CAT_COL[cat],
               bottom=bottom_neg, width=1.0, alpha=0.85)
        bottom_pos += pos; bottom_neg += neg

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Antibodies  (sorted by HIC label)")
    ax.set_ylabel("SHAP")
    ax.set_title("D  Feature-category SHAP per antibody  (HIC)")
    ax.set_xlim(-1, len(y))
    handles = [mpatches.Patch(color=CAT_COL[c], label=c) for c in CAT_ORDER if c != "other"]
    ax.legend(handles=handles, fontsize=7, ncol=3, loc="upper left", framealpha=0.7)


def draw_E(ax, esm_attr, ds):
    """VH per-residue ESM2-150M gradient attribution — mean |IG| per position, all assays."""
    colors_a = [ASSAY_COL[a] for a in ASSAYS]
    CDR_REGIONS = {"CDR1": (27, 35), "CDR2": (51, 58), "CDR3": (97, 108)}

    # Shade CDR regions first
    cdr_shades = {"CDR1": "#F4A582", "CDR2": "#92C5DE", "CDR3": "#D6604D"}
    for cdr_name, (s, e) in CDR_REGIONS.items():
        ax.axvspan(s, e, alpha=0.18, color=cdr_shades[cdr_name],
                   label=cdr_name, zorder=1)

    for ai, assay in enumerate(ASSAYS):
        y   = ds.y(assay); lab = ~np.isnan(y)
        vh_a = esm_attr[assay]["vh_attr"][lab]
        vh_m = esm_attr[assay]["vh_m"][lab]
        n_pos = vh_a.shape[1]
        mean_abs = np.full(n_pos, np.nan)
        for t in range(n_pos):
            real = vh_m[:, t]
            if real.sum() >= 3:
                mean_abs[t] = np.abs(vh_a[:, t][real]).mean()

        valid = ~np.isnan(mean_abs)
        pos   = np.arange(n_pos)
        # Normalize each assay to [0,1] for visual comparability
        ma    = mean_abs[valid]
        if ma.max() > 0:
            ma = ma / ma.max()
        ax.plot(pos[valid], ma, color=colors_a[ai], lw=1.4,
                label=ASSAY_LBL[assay], alpha=0.9, zorder=3)

    ax.set_xlim(0, 140); ax.set_ylabel("Mean |IG|  (normalized)")
    ax.set_xlabel("VH sequence position")
    ax.set_title("E  VH per-residue gradient attribution  (ESM2-150M proxy)")
    handles_assay = [plt.Line2D([0], [0], color=colors_a[i], lw=1.8,
                                label=ASSAY_LBL[a]) for i, a in enumerate(ASSAYS)]
    handles_cdr   = [mpatches.Patch(color=cdr_shades[c], alpha=0.5, label=c)
                     for c in CDR_REGIONS]
    ax.legend(handles=handles_assay + handles_cdr,
              fontsize=7, ncol=4, loc="upper right", framealpha=0.7)


def draw_F(ax, ds):
    """Panel F: CDR vs FR mean |attribution| ratio (VH) — bar chart for all assays,
    or 3D structure image if available."""
    struct_img = None
    struct_dir = ROOT / "results/explain/structures"
    # Try to load a HIC structure image (highest-attribution antibody)
    for p in sorted(struct_dir.glob("esm_attr_*_HIC.png")):
        struct_img = p; break

    if struct_img is not None and struct_img.exists():
        img = plt.imread(str(struct_img))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("F  VH attribution — 3D surface  (HIC, ESM2-150M)")
        return

    # Fallback: load precomputed CDR/FR csv if available
    csv = ROOT / "results/explain/abdevx_perres_esm150m.csv"
    if not csv.exists():
        ax.axis("off"); ax.set_title("F  (data unavailable)"); return

    df = pd.read_csv(csv)
    df_vh = df[df["chain"] == "VH"].groupby(["assay", "region"])["mean_abs_attr"].mean().unstack("region")
    x = np.arange(len(ASSAYS)); w = 0.35
    for shift, region, color in [(-w/2, "CDR", "#C44E52"), (w/2, "FR", "#4C72B0")]:
        vals = [df_vh.loc[a, region] if a in df_vh.index and region in df_vh.columns else 0
                for a in ASSAYS]
        ax.bar(x + shift, vals, w, color=color, label=region, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ASSAY_LBL[a] for a in ASSAYS])
    ax.set_ylabel("Mean |gradient attribution|")
    ax.set_title("F  VH CDR vs FR attribution  (ESM2-150M)")
    ax.legend(fontsize=8)


# ─── assemble panel ────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading data ...")
    ds  = load_gdpa1(str(ROOT / "configs/default.yaml"))
    _, g3, B1, B3 = GE.build_blocks()

    print("Fitting LightGBM SHAP ...")
    sv_all, y_all, base_all, imp_all, fcols, cats = fit_lgbm_shap(ds)

    print("Fitting AbDev-X block SHAP ...")
    oof_all, coef_all, icpt_all = fit_abdevx_block_shap(ds, B1)

    print("Computing ESM2-150M per-residue attribution ...")
    esm_attr = load_esm150m(ds)

    # ── Build figure ──
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.42, wspace=0.38,
        left=0.06, right=0.97, top=0.93, bottom=0.07
    )

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])
    ax_F = fig.add_subplot(gs[1, 2])

    print("Drawing panels ...")
    draw_A(ax_A, oof_all, coef_all, ds)
    draw_B(ax_B, oof_all, ds, assay="HIC")
    draw_C(ax_C, imp_all, fcols, cats, assays_show=("HIC", "AC-SINS_pH7.4", "PR_CHO"))
    draw_D(ax_D, sv_all, y_all, cats)
    draw_E(ax_E, esm_attr, ds)
    draw_F(ax_F, ds)

    fig.suptitle(
        "AbDev-X Explainability — multi-level mechanistic attribution",
        fontsize=13, fontweight="bold", y=0.975
    )

    out = FIG_OUT / "panel_explainability.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
