"""AbDev-X explainability analysis — four complementary layers:

  1. Block meta-weight visualization  (enhanced representation_weights with CI)
  2. Per-antibody block contribution  (SHAP decomposition of the 6-expert meta-learner)
  3. Per-residue ESM2-150M attribution (gradient × input, mapped to VH/VL sequence positions)
  4. AbLang2 integrated gradients     (per-dimension IG, labeled by physicochemical correlates)

Figures saved to results/figures/explain_*.png
CSVs  saved to results/explain/abdevx_*.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE
from abdev.features.descriptors import build_descriptor_matrix
from abdev.models.fusion import AbDevFusion, DEFAULT_BASES, BASE_BLOCK, _ridge, _ridge_pca

ASSAYS   = GE.ASSAYS
ASSAY_LBL = {"HIC":"HIC","Tm2":"Tm2","AC-SINS_pH7.4":"AC-SINS","PR_CHO":"PR-CHO","Titer":"Titer"}
BLOCK_LBL = ["ESM2","Desc","Isotype","Bio(ridge)","Bio(lgbm)","AbLang2"]
FIG = ROOT / "results/figures"; FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results/explain"; OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "figure.dpi": 150, "savefig.dpi": 200})
ALPHAS = np.logspace(-2, 4, 12)
PALETTE = {"ESM2":"#4C72B0","Desc":"#DD8452","Isotype":"#55A868",
           "Bio(ridge)":"#C44E52","Bio(lgbm)":"#8172B3","AbLang2":"#937860"}


# ─── helpers ──────────────────────────────────────────────────────────────────

def pca_ridge(high_dim=True):
    steps = [("s", StandardScaler())]
    if high_dim:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=ALPHAS)))
    return Pipeline(steps)


def get_projection_vec(pipe, in_dim):
    """Recover the gradient ∂ŷ/∂x (shape: in_dim) from a fitted PCA+Ridge pipeline."""
    ridge = pipe.named_steps["r"]
    w = ridge.coef_                          # (n_pca_components,)
    if "p" in pipe.named_steps:
        pca = pipe.named_steps["p"]
        V   = pca.components_                # (n_components, in_dim)
        proj_pca = V.T @ w                   # (in_dim,)  before scaling
    else:
        proj_pca = w
    sc = pipe.named_steps["s"]
    proj = proj_pca / (sc.scale_ + 1e-30)   # account for StandardScaler
    return proj                              # (in_dim,)


STRUCT_DIR = ROOT / "external/abdev-benchmark/data/structures/AntiBodyBuilder3/GDPa1"

FEAT_CATEGORIES = [
    ("hydrophobic", ["gravy", "hmoment", "aromatic", "aliphatic", "hydrophob", "tryptophan"]),
    ("charge",      ["charge", "_pi", "frac_pos", "frac_neg", "pI", "pi_"]),
    ("liability",   ["deamid", "isomer", "nglyc", "oxid", "free_cys"]),
    ("polar",       ["frac_polar"]),
    ("size",        ["_len"]),
]
CAT_COLORS_D = {"hydrophobic": "#C44E52", "charge": "#4C72B0", "liability": "#55A868",
                "polar": "#8172B3", "size": "#DD8452", "other": "#BBBBBB"}


def categorize_feat(name):
    n = name.lower()
    for cat, pats in FEAT_CATEGORIES:
        if any(p in n for p in pats):
            return cat
    return "other"


# ─── Part A: Refit AbDevFusion, collect OOF + meta weights ───────────────────

def fit_abdevx_oof(B1, ds):
    """Refit AbDevFusion on each assay; return per-antibody OOF block predictions + meta weights."""
    kf = KFold(5, shuffle=True, random_state=0)
    block_names = [n for n, _ in DEFAULT_BASES]     # 6 names
    oof_all  = {}   # assay -> (n, 6) OOF predictions
    coef_all = {}   # assay -> (6,) meta weights
    icpt_all = {}   # assay -> float intercept
    oof_pred = {}   # assay -> (n,) final OOF meta prediction

    for assay in ASSAYS:
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        B1m = {k: v[lab] for k, v in B1.items()}
        n   = lab.sum()

        # Replicate AbDevFusion.fit() logic but preserve OOF arrays
        fus = AbDevFusion(random_state=0)
        Xs  = fus._stack(B1m)              # list of 6 arrays
        oof = [np.full(n, np.nan) for _ in DEFAULT_BASES]

        for tr, va in kf.split(np.arange(n)):
            for bi, (name, mk) in enumerate(DEFAULT_BASES):
                oof[bi][va] = mk().fit(Xs[bi][tr], y[lab][tr]).predict(Xs[bi][va])

        Z = np.column_stack(oof)           # (n, 6)
        meta = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(Z, y[lab])

        oof_all[assay]  = Z
        coef_all[assay] = meta.coef_
        icpt_all[assay] = meta.intercept_
        oof_pred[assay] = meta.predict(Z)

    return oof_all, coef_all, icpt_all, oof_pred


# ─── Part B: Per-antibody SHAP (linear meta-learner) ─────────────────────────

def compute_block_shap(oof_all, coef_all, icpt_all, ds):
    """SHAP for the linear meta-learner: contrib_ij = beta_j * (P_ij - mean_j(P))."""
    rows = []
    shap_arrays = {}   # assay -> (n_labeled, 6)

    for assay in ASSAYS:
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        Z   = oof_all[assay]           # (n_lab, 6)
        b   = coef_all[assay]          # (6,)
        # Linear SHAP: contrib = beta_j * (P_j - mean(P_j))
        shap_vals = b[None, :] * (Z - Z.mean(0)[None, :])  # (n_lab, 6)
        shap_arrays[assay] = shap_vals

        # Summarize per block
        for bi, bname in enumerate(BLOCK_LBL):
            rows.append(dict(assay=assay, block=bname,
                             meta_weight=round(float(b[bi]), 4),
                             mean_abs_shap=round(float(np.abs(shap_vals[:, bi]).mean()), 4),
                             shap_share=round(float(np.abs(shap_vals[:, bi]).sum() /
                                                   (np.abs(shap_vals).sum() + 1e-9)), 4)))
    return pd.DataFrame(rows), shap_arrays


# ─── Part C: Per-residue ESM2-150M attribution ───────────────────────────────

def compute_esm150m_attribution(B1, ds):
    """Train PCA+Ridge on ESM2-150M mean-pooled; compute per-residue IG per antibody per assay."""
    pr   = np.load(ROOT / "data/processed/esm_perres_150M.npz", allow_pickle=True)
    vh_e = pr["vh_e"].astype(np.float32)   # (246, 140, 640)
    vh_m = pr["vh_m"].astype(np.float32)
    vh_c = pr["vh_c"].astype(bool)
    vl_e = pr["vl_e"].astype(np.float32)
    vl_m = pr["vl_m"].astype(np.float32)
    vl_c = pr["vl_c"].astype(bool)

    def mpool(e, m):
        return (e * m[:, :, None]).sum(1) / m.sum(1, keepdims=True).clip(min=1)

    vh_pool = mpool(vh_e, vh_m)   # (246, 640)
    vl_pool = mpool(vl_e, vl_m)
    X_150m  = np.nan_to_num(np.concatenate([vh_pool, vl_pool], 1))  # (246, 1280)

    # Number of real VH/VL residues per antibody
    vh_len = vh_m.sum(1).astype(int)   # (246,)
    vl_len = vl_m.sum(1).astype(int)

    results = {}   # assay -> dict with per-antibody arrays
    per_pos_mean = {}   # assay -> (140, 140) [VH 140, VL 140]

    for assay in ASSAYS:
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        pipe = pca_ridge(high_dim=True)
        pipe.fit(X_150m[lab], y[lab])
        proj = get_projection_vec(pipe, 1280)  # (1280,) gradient w.r.t. [VH_pool, VL_pool]
        proj_vh = proj[:640]    # gradient w.r.t. VH_pool
        proj_vl = proj[640:]    # gradient w.r.t. VL_pool

        # Per-residue IG: score_t = (e_t · proj) / L  (scalar per residue per antibody)
        n = 246
        vh_attr = np.zeros((n, 140))   # attribution at each VH position
        vl_attr = np.zeros((n, 140))
        for i in range(n):
            Lv = max(vh_len[i], 1)
            Ll = max(vl_len[i], 1)
            vh_attr[i] = (vh_e[i].astype(np.float32) @ proj_vh) / Lv  # (140,)
            vl_attr[i] = (vl_e[i].astype(np.float32) @ proj_vl) / Ll  # (140,)

        results[assay] = {"vh_attr": vh_attr, "vl_attr": vl_attr,
                          "vh_m": vh_m, "vl_m": vl_m,
                          "vh_c": vh_c, "vl_c": vl_c,
                          "proj_vh": proj_vh, "proj_vl": proj_vl}
    return results


def esm_attr_to_rows(esm_results, ds):
    """Summarize per-residue attribution into CDR vs FR, per assay."""
    rows = []
    for assay, d in esm_results.items():
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        vh_a = d["vh_attr"][lab]    # (n_lab, 140)
        vl_a = d["vl_attr"][lab]
        vh_m = d["vh_m"][lab].astype(bool)
        vl_m = d["vl_m"][lab].astype(bool)
        vh_c = d["vh_c"][lab]
        vl_c = d["vl_c"][lab]

        for chain, attr, mask, cdr in [("VH", vh_a, vh_m, vh_c),
                                        ("VL", vl_a, vl_m, vl_c)]:
            for t in range(attr.shape[1]):
                is_real = mask[:, t].any()
                is_cdr  = cdr[:, t].any()
                region  = "CDR" if is_cdr else "FR"
                if is_real:
                    rows.append(dict(assay=assay, chain=chain, pos=t, region=region,
                                     mean_attr=float(attr[:, t][mask[:, t]].mean()),
                                     mean_abs_attr=float(np.abs(attr[:, t][mask[:, t]]).mean())))
    return pd.DataFrame(rows)


# ─── Part D: AbLang2 integrated gradients ────────────────────────────────────

def compute_ablang2_ig(B1, ds):
    """IG on AbLang2 480-dim embeddings: IG_i = emb_i * proj_vec (baseline=0)."""
    X_abl = np.nan_to_num(B1["ablang2"])   # (246, 480)

    # Also load descriptor names for correlation analysis
    D_df  = build_descriptor_matrix(ds.heavy, ds.light)
    D_arr = np.nan_to_num(D_df.to_numpy(float))
    D_names = list(D_df.columns)

    ig_arrays   = {}   # assay -> (n_labeled, 480) IG values
    proj_vecs   = {}   # assay -> (480,) projection vector
    dim_summary = []

    for assay in ASSAYS:
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        # Use the same _ridge() head as the actual AbLang2 block in AbDevFusion
        pipe = _ridge()
        pipe.fit(X_abl[lab], y[lab])
        proj = get_projection_vec(pipe, 480)   # (480,) gradient ∂ŷ/∂emb
        proj_vecs[assay] = proj

        # IG from baseline=0: IG_i = (emb_i - 0) * proj = emb_i * proj (element-wise)
        IG = X_abl[lab] * proj[None, :]        # (n_lab, 480) per-dim IG per antibody
        ig_arrays[assay] = IG

        mean_abs_ig = np.abs(IG).mean(0)       # (480,) importance per dimension
        top_idx = np.argsort(mean_abs_ig)[::-1][:20]

        for rank, d_idx in enumerate(top_idx):
            # Correlate this AbLang2 dimension with descriptors to find interpretation
            best_rho, best_feat = 0.0, "?"
            for di, dn in enumerate(D_names):
                rho = abs(pearsonr(X_abl[lab][:, d_idx], D_arr[lab][:, di])[0])
                if rho > best_rho:
                    best_rho, best_feat = rho, dn
            dim_summary.append(dict(assay=assay, rank=rank+1, dim=d_idx,
                                    mean_abs_ig=round(float(mean_abs_ig[d_idx]), 6),
                                    proj_weight=round(float(proj[d_idx]), 6),
                                    best_corr_feat=best_feat,
                                    best_corr_rho=round(float(best_rho), 3)))
    return ig_arrays, proj_vecs, pd.DataFrame(dim_summary)


# ─── Part E: Named feature importance from Descriptors block (linear SHAP) ───

def compute_descriptor_named_ig(B1, ds):
    """Linear SHAP on the Descriptors ridge block: IG_di = desc_di * proj_d.

    Uses the same _ridge() head as the actual Descriptors block in AbDevFusion.
    Returns: dict(assay -> (n_lab, 165) IG array) + DataFrame of top features.
    """
    X_desc = np.nan_to_num(B1["descriptors"])  # (246, 165)
    D_df   = build_descriptor_matrix(ds.heavy, ds.light)
    feat_names = list(D_df.columns)            # 165 named features
    feat_cats  = np.array([categorize_feat(f) for f in feat_names])

    ig_arrays = {}
    rows = []
    for assay in ASSAYS:
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        pipe = _ridge()                        # StandardScaler + RidgeCV (no PCA)
        pipe.fit(X_desc[lab], y[lab])
        proj = get_projection_vec(pipe, X_desc.shape[1])  # (165,)

        # IG: IG_di = desc_di * proj_d  (baseline = 0 → IG = input × gradient)
        IG = X_desc[lab] * proj[None, :]       # (n_lab, 165)
        ig_arrays[assay] = IG

        mean_abs_ig = np.abs(IG).mean(0)       # (165,) feature importance
        total = mean_abs_ig.sum() + 1e-9
        top_idx = np.argsort(mean_abs_ig)[::-1][:15]
        for rank, fi in enumerate(top_idx):
            from scipy.stats import spearmanr as sp
            rho = float(sp(X_desc[lab][:, fi], y[lab]).statistic)
            rows.append(dict(assay=assay, rank=rank+1,
                             feature=feat_names[fi],
                             category=feat_cats[fi],
                             mean_abs_ig=round(float(mean_abs_ig[fi]), 6),
                             ig_share=round(float(mean_abs_ig[fi] / total), 4),
                             proj_weight=round(float(proj[fi]), 6),
                             corr_with_assay=round(rho, 3)))

    return ig_arrays, pd.DataFrame(rows)


# ─── Part F: 3D structure mapping — ESM2-150M attribution as PDB B-factors ───

def _write_bfactor_pdb(pdb_in, vh_scores, vl_scores, pdb_out):
    """Write per-residue attribution scores into PDB B-factor column.

    vh_scores / vl_scores: 1-D arrays of real attribution values, one per real residue
    in chain-order (sequential, no padding).
    """
    from Bio.PDB import PDBParser, PDBIO
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("ab", str(pdb_in))
    chain_scores = {"H": vh_scores, "L": vl_scores}
    for chain in struct[0]:
        scores = chain_scores.get(chain.id, None)
        if scores is None:
            continue
        res_list = [r for r in chain if r.id[0] == " "]  # ATOM records only
        for ri, res in enumerate(res_list):
            v = float(scores[ri]) if ri < len(scores) else 0.0
            for atom in res:
                atom.set_bfactor(v)
    io = PDBIO(); io.set_structure(struct); io.save(str(pdb_out))


def map_esm_attr_to_structure(esm_results, ds, n_examples=5):
    """Write ESM2-150M per-residue attribution as B-factors into PDB files.

    Selects n_examples antibodies per assay (highest |mean attribution| across VH).
    Saves to results/explain/structures/esm_attr_{antibody}_{assay}.pdb
    Returns list of written PDB paths.
    """
    try:
        from Bio.PDB import PDBParser, PDBIO
    except ImportError:
        print("  BioPython not available — skipping 3D structure mapping.")
        return []

    struct_out = OUT / "structures"; struct_out.mkdir(parents=True, exist_ok=True)
    written = []

    ab_names = ds.df[ds.id_col].values  # (246,) antibody names

    for assay in ASSAYS:
        y   = ds.y(assay)
        lab = np.where(~np.isnan(y))[0]   # global indices of labeled antibodies
        vh_a = esm_results[assay]["vh_attr"]   # (246, 140)
        vl_a = esm_results[assay]["vl_attr"]
        vh_m = esm_results[assay]["vh_m"].astype(bool)   # (246, 140)
        vl_m = esm_results[assay]["vl_m"].astype(bool)

        # Score each labeled antibody by mean |VH attribution|
        mean_score = np.array([np.abs(vh_a[i, vh_m[i]]).mean() if vh_m[i].any() else 0
                                for i in lab])
        top_idx = lab[np.argsort(mean_score)[::-1][:n_examples]]

        for gi in top_idx:                  # gi = global antibody index
            name = ab_names[gi]
            pdb_in = STRUCT_DIR / f"{name}.pdb"
            if not pdb_in.exists():
                continue
            vh_scores = vh_a[gi, vh_m[gi]]  # real VH residue scores
            vl_scores = vl_a[gi, vl_m[gi]]
            pdb_out = struct_out / f"esm_attr_{name}_{assay.replace('/', '').replace('-','')}.pdb"
            try:
                _write_bfactor_pdb(pdb_in, vh_scores, vl_scores, pdb_out)
                written.append(pdb_out)
            except Exception as ex:
                print(f"    WARNING: could not write {pdb_out.name}: {ex}")

    print(f"  Wrote {len(written)} B-factor PDB files to {struct_out}")
    return written


def render_structures_pymol(pdb_paths, assay_filter=None):
    """Render B-factor PDB files as surface images using PyMOL (if available)."""
    try:
        import pymol2
    except ImportError:
        print("  PyMOL not available — PDB files written; render manually with PyMOL/ChimeraX.")
        return

    struct_out = OUT / "structures"
    rendered = 0
    for pdb in pdb_paths:
        stem = pdb.stem                    # e.g. esm_attr_trastuzumab_HIC
        png  = struct_out / f"{stem}.png"
        try:
            with pymol2.PyMOL() as p:
                c = p.cmd
                c.load(str(pdb), "ab")
                c.hide("everything"); c.show("surface")
                c.bg_color("white")
                c.set("ray_opaque_background", 0)
                # Color by B-factor: blue=negative attribution, white=zero, red=positive
                vmax = 0.05          # approximate scale; adjust per assay if needed
                c.spectrum("b", "blue_white_red", "ab", minimum=-vmax, maximum=vmax)
                c.set("ray_shadows", 0)
                c.png(str(png), width=600, height=500, ray=1)
            rendered += 1
        except Exception as ex:
            print(f"    WARNING: PyMOL render failed for {pdb.name}: {ex}")

    print(f"  PyMOL rendered {rendered}/{len(pdb_paths)} structures.")


# ─── Figures ─────────────────────────────────────────────────────────────────

def fig_meta_weights(df_shap):
    """Figure 1: Meta-weights + SHAP share per assay (double panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: meta weights (signed)
    ax = axes[0]
    x = np.arange(len(ASSAYS)); w = 0.13
    colors = list(PALETTE.values())
    for bi, bname in enumerate(BLOCK_LBL):
        sub = df_shap[df_shap["block"] == bname].set_index("assay").reindex(ASSAYS)
        ax.bar(x + (bi - 2.5) * w, sub["meta_weight"].fillna(0), w,
               color=colors[bi], label=bname)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([ASSAY_LBL[a] for a in ASSAYS])
    ax.set_ylabel("Ridge meta-weight (β)"); ax.set_title("A  AbDev-X meta-learner weights per assay")
    ax.legend(fontsize=9, ncol=2)

    # Panel B: SHAP share (fraction of total |SHAP|)
    ax = axes[1]
    bottom = np.zeros(len(ASSAYS))
    for bi, bname in enumerate(BLOCK_LBL):
        sub = df_shap[df_shap["block"] == bname].set_index("assay").reindex(ASSAYS)
        vals = sub["shap_share"].fillna(0).to_numpy()
        ax.bar(x, vals, label=bname, color=colors[bi], bottom=bottom)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([ASSAY_LBL[a] for a in ASSAYS])
    ax.set_ylabel("Fraction of total |SHAP|"); ax.set_title("B  Per-block SHAP contribution share")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG / "explain_1_meta_weights.png")
    plt.close(fig)
    print("  Saved explain_1_meta_weights.png")


def fig_perantibody_heatmap(shap_arrays, ds):
    """Figure 2: Per-antibody block contribution heatmap (one panel per assay)."""
    fig, axes = plt.subplots(1, len(ASSAYS), figsize=(18, 5), sharey=False)
    for ai, assay in enumerate(ASSAYS):
        ax = axes[ai]
        y    = ds.y(assay)
        lab  = ~np.isnan(y)
        S    = shap_arrays[assay]            # (n_lab, 6)
        y_s  = y[lab]
        order = np.argsort(y_s)             # sort antibodies by true value
        S_s  = S[order]
        vmax = np.percentile(np.abs(S_s), 97)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im   = ax.imshow(S_s.T, cmap="RdBu_r", norm=norm, aspect="auto",
                         interpolation="nearest")
        ax.set_yticks(range(6)); ax.set_yticklabels(BLOCK_LBL if ai == 0 else [], fontsize=9)
        ax.set_xlabel("Antibodies (sorted by label)", fontsize=9)
        ax.set_title(ASSAY_LBL[assay], fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.7, label="SHAP" if ai == len(ASSAYS)-1 else "")

    fig.suptitle("Per-antibody block SHAP contributions (AbDev-X meta-learner)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "explain_2_perantibody_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved explain_2_perantibody_heatmap.png")


def fig_waterfall_examples(shap_arrays, oof_pred, ds, icpt_all, n_show=3):
    """Figure 3: Waterfall plots for selected antibodies (highest/lowest/median)."""
    fig, axes = plt.subplots(len(ASSAYS), n_show, figsize=(14, 3.5 * len(ASSAYS)))
    colors_pos = "#C44E52"; colors_neg = "#4C72B0"

    for ai, assay in enumerate(ASSAYS):
        y    = ds.y(assay)
        lab  = ~np.isnan(y)
        y_l  = y[lab]
        S    = shap_arrays[assay]   # (n_lab, 6)
        base = float(icpt_all[assay])
        # pick n_show antibodies: lowest, median, highest by true label
        idx_sort = np.argsort(y_l)
        picks = [idx_sort[0], idx_sort[len(idx_sort)//2], idx_sort[-1]]
        labels_pick = ["lowest", "median", "highest"]

        for pi, (pick, lbl) in enumerate(zip(picks, labels_pick)):
            ax = axes[ai, pi]
            sv = S[pick]                  # (6,) SHAP values for this antibody
            total = base + sv.sum()
            # Sort by abs value descending for waterfall
            order = np.argsort(np.abs(sv))[::-1]
            names  = [BLOCK_LBL[i] for i in order]
            vals   = sv[order]
            cols   = [colors_pos if v >= 0 else colors_neg for v in vals]
            # Horizontal bars
            running = base
            for j, (n_b, v, c) in enumerate(zip(names, vals, cols)):
                ax.barh(j, v, left=running, color=c, alpha=0.85, height=0.65,
                        edgecolor="white", linewidth=0.5)
                running += v
                ax.text(running + (0.002 if v >= 0 else -0.002), j,
                        f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                        fontsize=7)
            ax.axvline(base, color="grey", lw=0.8, ls="--", label=f"base={base:.2f}")
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("Prediction", fontsize=8)
            ax.set_title(f"{ASSAY_LBL[assay]} | {lbl}\ntrue={y_l[pick]:.3f}  pred={total:.3f}",
                         fontsize=8)

    fig.suptitle("AbDev-X prediction waterfall — selected antibodies", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "explain_3_waterfall.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved explain_3_waterfall.png")


def fig_perres_attribution(esm_results, ds):
    """Figure 4: Per-residue ESM2-150M gradient attribution along VH sequence."""
    fig, axes = plt.subplots(len(ASSAYS), 1, figsize=(14, 3.2 * len(ASSAYS)), sharex=False)
    cdr_colors = {"CDR1": "#FDDBC7", "CDR2": "#D1E5F0", "CDR3": "#B2182B"}

    # Canonical CDR-H boundaries (Chothia, positions within our 140-slot padding)
    # From the data: CDR positions roughly 27-35 (CDR1), 51-58 (CDR2), 97-108 (CDR3)
    CDR_REGIONS = {"CDR1": (27, 35), "CDR2": (51, 58), "CDR3": (97, 108)}

    for ai, assay in enumerate(ASSAYS):
        ax = axes[ai]
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        vh_a = esm_results[assay]["vh_attr"][lab]         # (n_lab, 140) signed attribution
        vh_m = esm_results[assay]["vh_m"][lab].astype(bool)

        # Mean |attribution| per position (over labeled antibodies that have a real residue there)
        n_ab, n_pos = vh_a.shape
        mean_abs = np.full(n_pos, np.nan)
        std_abs  = np.full(n_pos, np.nan)
        for t in range(n_pos):
            real = vh_m[:, t]
            if real.sum() >= 3:
                v = np.abs(vh_a[:, t][real])
                mean_abs[t] = v.mean(); std_abs[t] = v.std()

        pos = np.arange(n_pos)
        valid = ~np.isnan(mean_abs)
        ax.plot(pos[valid], mean_abs[valid], color="#333333", lw=1.2, zorder=3)
        ax.fill_between(pos[valid],
                        np.maximum(0, mean_abs[valid] - std_abs[valid]),
                        mean_abs[valid] + std_abs[valid],
                        alpha=0.25, color="#333333", zorder=2)

        # Shade CDR regions
        for cdr_name, (s, e) in CDR_REGIONS.items():
            ax.axvspan(s, e, alpha=0.25,
                       color={"CDR1": "#F4A582", "CDR2": "#92C5DE", "CDR3": "#D6604D"}[cdr_name],
                       label=cdr_name, zorder=1)

        ax.set_xlim(0, n_pos); ax.set_ylabel("|IG|", fontsize=9)
        ax.set_title(f"{ASSAY_LBL[assay]} — VH per-residue gradient attribution (ESM2-150M)", fontsize=10)
        if ai == len(ASSAYS) - 1:
            ax.set_xlabel("VH sequence position")
        if ai == 0:
            ax.legend(fontsize=8, loc="upper right", ncol=3)

    fig.tight_layout()
    fig.savefig(FIG / "explain_4_perres_esm150m.png")
    plt.close(fig)
    print("  Saved explain_4_perres_esm150m.png")


def fig_ablang2_ig(ig_arrays, dim_summary, ds):
    """Figure 5: AbLang2 top-20 IG dimensions per assay, labeled by descriptor correlate."""
    fig, axes = plt.subplots(1, len(ASSAYS), figsize=(18, 5), sharey=False)
    for ai, assay in enumerate(ASSAYS):
        ax = axes[ai]
        sub = dim_summary[dim_summary["assay"] == assay].head(20)
        y_pos = np.arange(len(sub))
        cols = ["#C44E52" if w > 0 else "#4C72B0" for w in sub["proj_weight"]]
        ax.barh(y_pos, sub["mean_abs_ig"].values, color=cols, alpha=0.85)
        labels = [f"dim{row.dim}\n({row.best_corr_feat[:12]})" for _, row in sub.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5 if ai == 0 else 6)
        ax.set_xlabel("Mean |IG|", fontsize=9)
        ax.set_title(ASSAY_LBL[assay], fontsize=11)
        ax.invert_yaxis()

    red_patch = mpatches.Patch(color="#C44E52", label="proj_weight > 0 (increases prediction)")
    blue_patch = mpatches.Patch(color="#4C72B0", label="proj_weight < 0 (decreases prediction)")
    fig.legend(handles=[red_patch, blue_patch], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("AbLang2 integrated gradients — top-20 latent dimensions per assay\n(labeled by best-correlated descriptor)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "explain_5_ablang2_ig.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved explain_5_ablang2_ig.png")


def fig_cdr_fr_split(esm_attr_df):
    """Figure 6: CDR vs FR attribution ratio per assay (VH chain)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ASSAYS)); w = 0.35
    for shift, region, color in [(-w/2, "CDR", "#C44E52"), (w/2, "FR", "#4C72B0")]:
        vals = []
        for assay in ASSAYS:
            sub = esm_attr_df[(esm_attr_df["assay"] == assay) &
                               (esm_attr_df["chain"] == "VH") &
                               (esm_attr_df["region"] == region)]
            vals.append(sub["mean_abs_attr"].mean() if len(sub) else 0)
        ax.bar(x + shift, vals, w, color=color, label=region, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([ASSAY_LBL[a] for a in ASSAYS])
    ax.set_ylabel("Mean |gradient attribution|"); ax.set_title("VH CDR vs Framework gradient attribution (ESM2-150M)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "explain_6_cdr_fr_split.png")
    plt.close(fig)
    print("  Saved explain_6_cdr_fr_split.png")


def fig_shap_summary(shap_arrays, ds):
    """Figure 7: Beeswarm-style SHAP summary per assay using SHAP library."""
    for assay in ASSAYS:
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        S   = shap_arrays[assay]   # (n_lab, 6)
        Z_base = None
        # Create SHAP Explanation object
        exp = shap.Explanation(values=S,
                               base_values=np.zeros(lab.sum()),
                               feature_names=BLOCK_LBL)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        shap.plots.beeswarm(exp, max_display=6, show=False, ax=ax)
        ax.set_title(f"AbDev-X block SHAP — {ASSAY_LBL[assay]}", fontsize=11)
        fig.tight_layout()
        fname = FIG / f"explain_7_shap_beeswarm_{assay.replace('/','').replace('-','')}.png"
        fig.savefig(fname)
        plt.close(fig)
    print("  Saved explain_7_shap_beeswarm_*.png (5 files)")


def fig_descriptor_named_ig(desc_ig_df):
    """Figure 9: Named feature importance from Descriptors block (linear SHAP/IG)."""
    cat_colors = CAT_COLORS_D
    fig, axes = plt.subplots(1, len(ASSAYS), figsize=(18, 6), sharey=False)
    for ai, assay in enumerate(ASSAYS):
        ax = axes[ai]
        sub = desc_ig_df[desc_ig_df["assay"] == assay].head(15)
        y_pos = np.arange(len(sub))
        # Color by category, bar direction by proj_weight sign
        cols = [cat_colors.get(row.category, "#BBBBBB") for _, row in sub.iterrows()]
        ax.barh(y_pos, sub["mean_abs_ig"].values, color=cols, alpha=0.85)
        labels = [f"{row.feature[:22]}\n({row.category[:5]}, ρ={row.corr_with_assay:+.2f})"
                  for _, row in sub.iterrows()]
        ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Mean |IG|", fontsize=9)
        ax.set_title(ASSAY_LBL[assay], fontsize=11)
        ax.invert_yaxis()

    handles = [mpatches.Patch(color=c, label=cat) for cat, c in cat_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Descriptors block — named feature importance (linear IG)\n"
                 "colored by physicochemical category, ρ = Spearman with assay label",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "explain_9_descriptor_named_ig.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved explain_9_descriptor_named_ig.png")


def fig_ablang2_ig_perantibody(ig_arrays, ds):
    """Figure 8: Per-antibody AbLang2 IG trace — top-5 dims per assay, sorted by predicted value."""
    fig, axes = plt.subplots(1, len(ASSAYS), figsize=(17, 4.5))
    for ai, assay in enumerate(ASSAYS):
        ax = axes[ai]
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        IG  = ig_arrays[assay]          # (n_lab, 480)
        y_l = y[lab]
        order = np.argsort(y_l)
        IG_s  = IG[order]               # sort by true label
        # Top-5 dims by mean |IG|
        top5 = np.argsort(np.abs(IG_s).mean(0))[::-1][:5]
        colors5 = plt.cm.tab10(np.linspace(0, 0.5, 5))
        for ki, di in enumerate(top5):
            ax.plot(IG_s[:, di], color=colors5[ki], alpha=0.8, lw=1.2, label=f"dim{di}")
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.set_title(ASSAY_LBL[assay], fontsize=11)
        ax.set_xlabel("Antibody (sorted by label)", fontsize=9)
        if ai == 0:
            ax.set_ylabel("IG value")
        ax.legend(fontsize=7, ncol=1)

    fig.suptitle("AbLang2 per-antibody integrated gradients — top-5 latent dimensions", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "explain_8_ablang2_ig_perantibody.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved explain_8_ablang2_ig_perantibody.png")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading data ...")
    ds, g3, B1, B3 = GE.build_blocks()

    # ── A: Refit AbDevFusion OOF ──
    print("Refitting AbDevFusion (OOF + meta-weights) ...")
    oof_all, coef_all, icpt_all, oof_pred = fit_abdevx_oof(B1, ds)

    # ── B: Block SHAP decomposition ──
    print("Computing per-antibody block SHAP ...")
    df_shap, shap_arrays = compute_block_shap(oof_all, coef_all, icpt_all, ds)
    df_shap.to_csv(OUT / "abdevx_block_shap.csv", index=False)
    print(f"  SHAP summary:\n{df_shap.pivot_table(index='block', columns='assay', values='shap_share').round(3).to_string()}")

    # Save per-antibody SHAP arrays
    for assay in ASSAYS:
        y   = ds.y(assay)
        lab = ~np.isnan(y)
        S   = shap_arrays[assay]
        df_ab = pd.DataFrame(S, columns=BLOCK_LBL)
        df_ab.insert(0, "antibody_name", ds.df[ds.id_col].values[lab])
        df_ab["true_label"] = ds.y(assay)[lab]
        df_ab["pred_label"] = oof_pred[assay]
        df_ab.to_csv(OUT / f"abdevx_perantibody_{assay.replace('/', '').replace('-', '')}.csv", index=False)
    print(f"  Per-antibody CSVs saved for {len(ASSAYS)} assays.")

    # ── C: Per-residue ESM2-150M attribution ──
    print("Computing per-residue ESM2-150M gradient attribution ...")
    esm_results   = compute_esm150m_attribution(B1, ds)
    esm_attr_df   = esm_attr_to_rows(esm_results, ds)
    esm_attr_df.to_csv(OUT / "abdevx_perres_esm150m.csv", index=False)
    print(f"  Saved per-residue attribution for {len(esm_attr_df)} position-assay pairs.")

    # CDR vs FR summary
    cdr_fr = (esm_attr_df.groupby(["assay", "chain", "region"])["mean_abs_attr"]
              .mean().unstack("region").round(4))
    print("\n  CDR vs FR mean |attribution| (VH):")
    print(cdr_fr.xs("VH", level="chain").to_string())

    # ── D: AbLang2 IG ──
    print("\nComputing AbLang2 integrated gradients ...")
    ig_arrays, proj_vecs, dim_summary = compute_ablang2_ig(B1, ds)
    dim_summary.to_csv(OUT / "abdevx_ablang2_ig_dims.csv", index=False)
    print(f"  Top-3 dims per assay:")
    for assay in ASSAYS:
        top3 = dim_summary[dim_summary["assay"] == assay].head(3)
        items = [f"dim{r.dim}({r.best_corr_feat[:10]},rho={r.best_corr_rho:.2f})"
                 for _, r in top3.iterrows()]
        print(f"    {assay}: {', '.join(items)}")

    # ── E: Named feature importance — Descriptors block ──
    print("\nComputing Descriptors block named feature IG ...")
    desc_ig_arrays, desc_ig_df = compute_descriptor_named_ig(B1, ds)
    desc_ig_df.to_csv(OUT / "abdevx_descriptor_named_ig.csv", index=False)
    print("  Top-3 named features per assay:")
    for assay in ASSAYS:
        top3 = desc_ig_df[desc_ig_df["assay"] == assay].head(3)
        items = [f"{r.feature}(ρ={r.corr_with_assay:+.2f})" for _, r in top3.iterrows()]
        print(f"    {assay}: {', '.join(items)}")

    # ── F: 3D structure mapping ──
    print("\nMapping ESM2-150M attribution to PDB B-factors ...")
    written_pdbs = map_esm_attr_to_structure(esm_results, ds, n_examples=3)
    render_structures_pymol(written_pdbs)

    # ── Figures ──
    print("\nGenerating figures ...")
    fig_meta_weights(df_shap)
    fig_perantibody_heatmap(shap_arrays, ds)
    fig_waterfall_examples(shap_arrays, oof_pred, ds, icpt_all)
    fig_perres_attribution(esm_results, ds)
    fig_ablang2_ig(ig_arrays, dim_summary, ds)
    fig_cdr_fr_split(esm_attr_df)
    try:
        fig_shap_summary(shap_arrays, ds)
    except Exception as e:
        print(f"  SHAP beeswarm skipped ({e})")
    fig_ablang2_ig_perantibody(ig_arrays, ds)
    fig_descriptor_named_ig(desc_ig_df)

    print(f"\nDone. All figures saved to {FIG}")
    print(f"All CSVs  saved to {OUT}")


if __name__ == "__main__":
    main()
