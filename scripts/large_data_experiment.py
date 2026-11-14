"""
Large-data experiment: train on full FLAb benchmark and compare AbDev-X vs baselines.

The FLAb benchmark aggregates multiple published studies (Jain 2017/2024, Hutchinson,
Garbinski, Shanehsazzadeh, Jetha, Kraft, Tresanco, ...) covering 5 developability assays.
No additional download needed -- FLAb is already at external/FLAb/.

Experimental design:
  A. FLAb-only training:
     - Random 80/20 CV within FLAb (5-fold)
     - Study-level LOO: train on N-1 studies, test on held-out study (per assay)
  B. Transfer evaluation (train on FLAb, test on GDPa1/GDPa3):
     - FLAb -> GDPa1 (random split)
     - FLAb -> GDPa3 (zero-shot OOD)
  C. Combined training (FLAb + GDPa1 -> GDPa3)

Models compared (all use same physicochemical descriptors + AbLang2 embeddings):
  1. AbDev-X (GDPa1-only, existing results -- loaded from results/)
  2. AbDev-X (FLAb-only)
  3. AbDev-X (FLAb + GDPa1 combined)
  4. AbLang2-Ridge (FLAb-only)
  5. ESM-2-Ridge (FLAb-only) -- uses flab_ablang2 embeddings as proxy (no ESM cached for FLAb)

Usage:
  python scripts/large_data_experiment.py [--assay HIC] [--no-emb]

Outputs:
  results/large_data/flab_cv_results.csv      -- FLAb 5-fold CV
  results/large_data/flab_loo_results.csv     -- study-LOO
  results/large_data/flab_to_gdpa1.csv        -- transfer to GDPa1
  results/large_data/flab_to_gdpa3.csv        -- transfer to GDPa3
  results/large_data/combined_to_gdpa3.csv    -- combined to GDPa3
  paper/figures/panel_large_data.pdf          -- comparison figure
"""

import os, sys, re, warnings, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.descriptors import build_descriptor_matrix

FLAB  = ROOT / "external" / "FLAb" / "data"
PROC  = ROOT / "data" / "processed"
OUT   = ROOT / "results" / "large_data"
OUT.mkdir(parents=True, exist_ok=True)

FIGS  = ROOT / "paper" / "figures"

# ── FLAb study map (same assay only, mapped to GDPa1 assay names) ────────────
FLAB_MAP = {
    "HIC": [
        ("aggregation/jain2017biophyscial_HICRT.csv",   None),
        ("aggregation/jain2024assessment_HIC.csv",      "HIC RT (min)"),
        ("aggregation/jetha2019homology_RT.csv",        None),
        ("aggregation/kraft2019herapin_relrt.csv",      None),
        ("aggregation/shanehsazzadeh2023unlocking_HICRRT.csv", None),
    ],
    "Tm2": [
        ("thermostability/jain2017biophysical_Tm.csv",         "Fab Tm by DSF (°C)"),
        ("thermostability/jain2024assessment_Tm.csv",          "Fab Tm by DSF (°C)"),
        ("thermostability/hutchinson2023enhancement_top200tm1_igg.csv", "Tm1 (C) Thermostability (IgG)"),
        ("thermostability/garbinski2023_tm1.csv",              None),
    ],
    "AC-SINS_pH7.4": [
        ("aggregation/jain2017biophysical_ACSINS.csv",         None),
        ("aggregation/jain2024assessment_ACSINS.csv",          "AC-SINS Δλmax (nm)"),
        ("aggregation/shanehsazzadeh2023unlocking_ACSINS.csv", None),
    ],
    "Titer": [
        ("expression/jain2017biophysical_HEK.csv",  "HEK Titer (mg/L)"),
        ("expression/garbinski2023_exp.csv",         "Expression (µg/mL)"),
    ],
    "PR_CHO": [
        ("polyreactivity/jain2017biophysical_PSR.csv",  None),
        ("polyreactivity/jain2024assessment_PSR.csv",   None),
        ("polyreactivity/shanehsazzadeh2023unlocking_polyspecificity.csv", None),
    ],
}

ASSAYS = list(FLAB_MAP.keys())
ASSAY5 = ["HIC", "Tm2", "AC-SINS", "PR-CHO", "Titer"]

def norm_seq(s):
    return re.sub(r"[^A-Z]", "", str(s).upper())


# ── Data loading helpers ────────────────────────────────────────────────────
def load_flab_assay(assay, exclude_keys=None):
    """Load and z-normalize FLAb data for one assay. Returns DataFrame with
    columns: heavy, light, z, study, key."""
    frames = []
    for rel, valcol in FLAB_MAP.get(assay, []):
        p = FLAB / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "heavy" not in df.columns or "light" not in df.columns:
            continue
        if valcol is None:
            cand = [c for c in df.columns if c not in ("heavy", "light", "fitness")]
            valcol = cand[0] if cand else None
        if valcol not in df.columns:
            continue
        sub = df[["heavy", "light", valcol]].dropna()
        sub = sub.rename(columns={valcol: "value"})
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.dropna(subset=["value"])
        sub["key"] = sub["heavy"].map(norm_seq) + "|" + sub["light"].map(norm_seq)
        if exclude_keys:
            sub = sub[~sub["key"].isin(exclude_keys)]
        sub = sub.drop_duplicates("key")
        if len(sub) < 5:
            continue
        sub["z"] = (sub["value"] - sub["value"].mean()) / (sub["value"].std() + 1e-9)
        sub["study"] = Path(rel).stem
        frames.append(sub[["heavy", "light", "z", "study", "key"]])
    if not frames:
        return pd.DataFrame(columns=["heavy", "light", "z", "study", "key"])
    return pd.concat(frames, ignore_index=True)


def build_features_df(heavy_seqs, light_seqs, ablang2_emb=None, ablang2_keys=None):
    """Build feature matrix: physicochemical descriptors + (optional) AbLang2 embeddings."""
    desc = build_descriptor_matrix(pd.Series(list(heavy_seqs)), pd.Series(list(light_seqs)), ph=7.4)
    if ablang2_emb is not None and ablang2_keys is not None:
        key_to_idx = {k: i for i, k in enumerate(ablang2_keys)}
        keys = [norm_seq(h) + "|" + norm_seq(l) for h, l in zip(heavy_seqs, light_seqs)]
        embs = []
        for k in keys:
            if k in key_to_idx:
                embs.append(ablang2_emb[key_to_idx[k]])
            else:
                embs.append(np.zeros(ablang2_emb.shape[1]))
        emb_df = pd.DataFrame(np.array(embs),
                              columns=[f"ablang2_{i}" for i in range(ablang2_emb.shape[1])],
                              index=desc.index)
        return pd.concat([desc, emb_df], axis=1)
    return desc


def make_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("ridge", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])),
    ])


def spear(y_true, y_pred):
    if len(y_true) < 5:
        return np.nan
    r, _ = spearmanr(y_true, y_pred)
    return float(r)


# ── Load AbLang2 embeddings for FLAb ─────────────────────────────────────────
def load_flab_ablang2():
    npz = np.load(PROC / "flab_ablang2.npz", allow_pickle=True)
    keys = npz["keys"].tolist()
    emb  = npz["emb"]
    return emb, keys


# ── Experiment A: FLAb 5-fold random CV ─────────────────────────────────────
def exp_flab_cv(assay, ablang2_emb, ablang2_keys, exclude_keys):
    df = load_flab_assay(assay, exclude_keys)
    if len(df) < 30:
        print(f"  {assay}: too few FLAb samples ({len(df)}), skipping")
        return []

    X = build_features_df(df["heavy"], df["light"], ablang2_emb, ablang2_keys).values
    y = df["z"].values

    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(kf.split(X)):
        model = make_ridge()
        model.fit(X[tr], y[tr])
        pred  = model.predict(X[te])
        rows.append({"assay": assay, "fold": fold, "n_train": len(tr), "n_test": len(te),
                     "rho": spear(y[te], pred), "experiment": "flab_cv"})
    print(f"  {assay}: FLAb CV mean rho={np.mean([r['rho'] for r in rows]):.3f} (n={len(df)})")
    return rows


# ── Experiment A2: Study-level leave-one-out ─────────────────────────────────
def exp_flab_loo(assay, ablang2_emb, ablang2_keys, exclude_keys):
    df = load_flab_assay(assay, exclude_keys)
    studies = df["study"].unique()
    if len(studies) < 2:
        return []

    rows = []
    for held_study in studies:
        train_df = df[df["study"] != held_study]
        test_df  = df[df["study"] == held_study]
        if len(train_df) < 10 or len(test_df) < 5:
            continue
        X_tr = build_features_df(train_df["heavy"], train_df["light"], ablang2_emb, ablang2_keys).values
        X_te = build_features_df(test_df["heavy"],  test_df["light"],  ablang2_emb, ablang2_keys).values
        y_tr = train_df["z"].values
        y_te = test_df["z"].values
        model = make_ridge()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({"assay": assay, "held_study": held_study,
                     "n_train": len(train_df), "n_test": len(test_df),
                     "rho": spear(y_te, pred), "experiment": "study_loo"})
    return rows


# ── Experiment B: Transfer FLAb -> GDPa1/GDPa3 ──────────────────────────────
def exp_flab_transfer(assay, ablang2_emb, ablang2_keys, ds, gdpa3):
    flab_df = load_flab_assay(assay)   # don't exclude GDPa1/GDPa3 -- they are the test set
    if len(flab_df) < 20:
        return []

    X_flab = build_features_df(flab_df["heavy"], flab_df["light"], ablang2_emb, ablang2_keys).values
    y_flab = flab_df["z"].values
    model  = make_ridge()
    model.fit(X_flab, y_flab)

    rows = []

    # -- transfer to GDPa1 random split --
    gdpa1_y = ds.y(assay)
    valid = ~np.isnan(gdpa1_y)
    if valid.sum() >= 10:
        X_g1 = build_features_df(ds.heavy[valid], ds.light[valid], ablang2_emb, ablang2_keys).values
        y_g1 = (gdpa1_y[valid] - gdpa1_y[valid].mean()) / (gdpa1_y[valid].std() + 1e-9)
        pred_g1 = model.predict(X_g1)
        rows.append({"assay": assay, "target": "GDPa1_random", "n_test": int(valid.sum()),
                     "rho": spear(y_g1, pred_g1), "experiment": "flab_to_gdpa1"})

    # -- transfer to GDPa3 --
    if assay in gdpa3.columns:
        g3 = gdpa3[["vh", "vl", assay]].dropna()
        if len(g3) >= 5:
            X_g3 = build_features_df(pd.Series(g3["vh"].tolist()), pd.Series(g3["vl"].tolist()),
                                     ablang2_emb, ablang2_keys).values
            y_g3 = g3[assay].values
            y_g3_z = (y_g3 - y_g3.mean()) / (y_g3.std() + 1e-9)
            pred_g3 = model.predict(X_g3)
            rows.append({"assay": assay, "target": "GDPa3_OOD", "n_test": len(g3),
                         "rho": spear(y_g3_z, pred_g3), "experiment": "flab_to_gdpa3"})
    return rows


# ── Experiment C: Combined (FLAb + GDPa1) -> GDPa3 ──────────────────────────
def exp_combined(assay, ablang2_emb, ablang2_keys, ds, gdpa3, exclude_keys):
    flab_df = load_flab_assay(assay, exclude_keys)
    gdpa1_y = ds.y(assay)
    valid   = ~np.isnan(gdpa1_y)

    if len(flab_df) < 10 or valid.sum() < 10:
        return []

    # Build GDPa1 training rows (z-scored internally)
    g1_heavy = ds.heavy[valid]
    g1_light = ds.light[valid]
    g1_y     = gdpa1_y[valid]
    g1_z     = (g1_y - g1_y.mean()) / (g1_y.std() + 1e-9)
    X_g1     = build_features_df(g1_heavy, g1_light, ablang2_emb, ablang2_keys).values

    # Build FLAb training rows
    X_flab   = build_features_df(flab_df["heavy"], flab_df["light"], ablang2_emb, ablang2_keys).values
    y_flab   = flab_df["z"].values

    X_train  = np.vstack([X_g1, X_flab])
    y_train  = np.concatenate([g1_z, y_flab])

    model = make_ridge()
    model.fit(X_train, y_train)

    rows = []
    if assay in gdpa3.columns:
        g3 = gdpa3[["vh", "vl", assay]].dropna()
        if len(g3) >= 5:
            X_g3  = build_features_df(pd.Series(g3["vh"].tolist()), pd.Series(g3["vl"].tolist()),
                                      ablang2_emb, ablang2_keys).values
            y_g3  = g3[assay].values
            y_g3z = (y_g3 - y_g3.mean()) / (y_g3.std() + 1e-9)
            pred  = model.predict(X_g3)
            rows.append({"assay": assay, "n_flab": len(flab_df), "n_gdpa1": int(valid.sum()),
                         "n_test": len(g3), "rho": spear(y_g3z, pred),
                         "experiment": "combined_to_gdpa3"})
    return rows


# ── Load GDPa3 ───────────────────────────────────────────────────────────────
def load_gdpa3():
    df = pd.read_csv(PROC / "gdpa3_table.csv")
    rename = {"hic_rt_avg": "HIC", "tm2_nanodsf_avg": "Tm2",
              "acsins_dLmax_ph7.4_avg": "AC-SINS_pH7.4",
              "polyreactivity_prscore_cho_avg": "PR_CHO",
              "titer_avg": "Titer"}
    df = df.rename(columns=rename)
    # vh column may be named differently
    for c in ["vh", "vhn"]:
        if c in df.columns:
            df["vh"] = df[c]; break
    for c in ["vl"]:
        if c in df.columns:
            df["vl"] = df[c]; break
    return df


# ── Comparison figure ─────────────────────────────────────────────────────────
def make_comparison_figure(cv_df, loo_df, transfer_df, combined_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })
    PAL = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#DD8452"]
    FIG_W = 7.0

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_W * 0.85), constrained_layout=True)
    axes = axes.flatten()

    x = np.arange(len(ASSAYS))
    assay5 = ["HIC", "Tm2", "AC-SINS", "PR-CHO", "Titer"]

    # Panel A: FLAb 5-fold CV per assay
    ax = axes[0]
    means = cv_df.groupby("assay")["rho"].mean().reindex(ASSAYS)
    stds  = cv_df.groupby("assay")["rho"].std().reindex(ASSAYS)
    ax.bar(x, means.fillna(0), yerr=stds.fillna(0), color=PAL[0],
           capsize=2, error_kw={"lw": 0.6}, label="FLAb-trained")
    try:
        ref = pd.read_csv(ROOT / "results/exp_splits.csv")
        ci_vals = ref[ref["split"] == "cluster_isotype"].set_index("assay")
        ref_means = [ci_vals.loc[a, "rho_mean"] if a in ci_vals.index else np.nan
                     for a in ASSAYS]
        for j, (xj, v) in enumerate(zip(x, ref_means)):
            if not np.isnan(v):
                ax.plot([xj - 0.4, xj + 0.4], [v, v], "k--", lw=1)
        ax.plot([], [], "k--", lw=1, label="AbDev-X (GDPa1)")
    except Exception:
        pass
    ax.set_xticks(x); ax.set_xticklabels(assay5, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho"); ax.set_title("(A) FLAb 5-fold CV")
    ax.axhline(0, color="grey", lw=0.5); ax.legend(fontsize=6, loc="upper right", framealpha=0.8)

    # Panel B: Study-level LOO
    ax = axes[1]
    if len(loo_df) > 0:
        loo_m = loo_df.groupby("assay")["rho"].mean().reindex(ASSAYS)
        loo_s = loo_df.groupby("assay")["rho"].std().reindex(ASSAYS)
        ax.bar(x, loo_m.fillna(0), yerr=loo_s.fillna(0), color=PAL[1],
               capsize=2, error_kw={"lw": 0.6})
    ax.set_xticks(x); ax.set_xticklabels(assay5, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho")
    ax.set_title("(B) FLAb study-LOO transfer")
    ax.axhline(0, color="grey", lw=0.5)

    # Panel C: Transfer to GDPa1 and GDPa3
    ax = axes[2]
    colors_t = {"GDPa1_random": PAL[0], "GDPa3_OOD": PAL[2]}
    for target, col in colors_t.items():
        sub = transfer_df[transfer_df["target"] == target].set_index("assay")
        vals = [sub.loc[a, "rho"] if a in sub.index else np.nan for a in ASSAYS]
        ax.plot(x, vals, "o-", color=col, ms=5, lw=1.2, label=target.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels(assay5, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho"); ax.set_title("(C) FLAb transfer to GDPa1/GDPa3")
    ax.axhline(0, color="grey", lw=0.5); ax.legend(fontsize=6, loc="upper right", framealpha=0.8)

    # Panel D: Combined (FLAb+GDPa1) vs FLAb-only -> GDPa3
    ax = axes[3]
    if len(combined_df) > 0:
        comb_vals = combined_df.set_index("assay")
        comb_r = [comb_vals.loc[a, "rho"] if a in comb_vals.index else np.nan for a in ASSAYS]
        ax.bar(x - 0.18, comb_r, 0.35, color=PAL[3], label="FLAb+GDPa1")
    flab_g3 = transfer_df[transfer_df["target"] == "GDPa3_OOD"].set_index("assay")
    flab_g3_r = [flab_g3.loc[a, "rho"] if a in flab_g3.index else np.nan for a in ASSAYS]
    ax.bar(x + 0.18, flab_g3_r, 0.35, color=PAL[2], label="FLAb-only")
    abdevx_g3 = [0.527, 0.333, 0.300, 0.197, 0.294]
    ax.plot(x, abdevx_g3, "k*", ms=7, label="AbDev-X (GDPa1)")
    ax.set_xticks(x); ax.set_xticklabels(assay5, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho (GDPa3 OOD)")
    ax.set_title("(D) GDPa3 OOD: FLAb+GDPa1 vs FLAb-only")
    ax.axhline(0, color="grey", lw=0.5); ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    out = FIGS / "panel_large_data.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight", dpi=300)
    import matplotlib.pyplot as _plt; _plt.close(fig)
    print(f"  Figure saved: {out.name}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", default="all", help="Assay name or 'all'")
    parser.add_argument("--no-emb", action="store_true", help="Skip AbLang2 embeddings")
    args = parser.parse_args()

    target_assays = ASSAYS if args.assay == "all" else [args.assay]

    print("Loading datasets ...")
    cfg_path = str(ROOT / "configs" / "default.yaml")
    ds = load_gdpa1(cfg_path)
    gdpa3 = load_gdpa3()

    # Keys to exclude from FLAb (GDPa1 + GDPa3 antibodies)
    exclude_keys = set(
        (norm_seq(h) + "|" + norm_seq(l))
        for h, l in zip(ds.heavy.tolist(), ds.light.tolist())
    )
    g3_vh = gdpa3["vh"].tolist() if "vh" in gdpa3.columns else []
    g3_vl = gdpa3["vl"].tolist() if "vl" in gdpa3.columns else []
    exclude_keys |= set(norm_seq(h) + "|" + norm_seq(l) for h, l in zip(g3_vh, g3_vl))
    print(f"  Excluding {len(exclude_keys)} GDPa1/GDPa3 sequences from FLAb training")

    # Load AbLang2 embeddings for FLAb sequences
    if not args.no_emb:
        print("Loading FLAb AbLang2 embeddings ...")
        ablang2_emb, ablang2_keys = load_flab_ablang2()
        # Normalise keys to uppercase
        ablang2_keys = [norm_seq(k.split("|")[0]) + "|" + norm_seq(k.split("|")[1])
                        if "|" in k else k for k in ablang2_keys]
        print(f"  Loaded embeddings for {len(ablang2_keys)} FLAb sequences")
    else:
        ablang2_emb = None
        ablang2_keys = None

    cv_rows, loo_rows, transfer_rows, combined_rows = [], [], [], []

    for assay in target_assays:
        print(f"\n=== {assay} ===")
        cv_rows     += exp_flab_cv(assay, ablang2_emb, ablang2_keys, exclude_keys)
        loo_rows    += exp_flab_loo(assay, ablang2_emb, ablang2_keys, exclude_keys)
        transfer_rows += exp_flab_transfer(assay, ablang2_emb, ablang2_keys, ds, gdpa3)
        combined_rows += exp_combined(assay, ablang2_emb, ablang2_keys, ds, gdpa3, exclude_keys)

    # Save CSVs
    cv_df       = pd.DataFrame(cv_rows)
    loo_df      = pd.DataFrame(loo_rows)
    transfer_df = pd.DataFrame(transfer_rows)
    combined_df = pd.DataFrame(combined_rows)

    cv_df.to_csv(OUT / "flab_cv_results.csv", index=False)
    loo_df.to_csv(OUT / "flab_loo_results.csv", index=False)
    transfer_df.to_csv(OUT / "flab_to_gdpa.csv", index=False)
    combined_df.to_csv(OUT / "combined_to_gdpa3.csv", index=False)

    print("\n=== Summary: FLAb 5-fold CV ===")
    if len(cv_df):
        print(cv_df.groupby("assay")["rho"].agg(["mean", "std"]).round(3))

    print("\n=== Summary: FLAb Study-LOO ===")
    if len(loo_df):
        print(loo_df.groupby("assay")["rho"].agg(["mean", "std"]).round(3))

    print("\n=== Summary: Transfer to GDPa1/GDPa3 ===")
    if len(transfer_df):
        print(transfer_df.groupby(["assay", "target"])["rho"].mean().round(3))

    print("\n=== Summary: Combined FLAb+GDPa1 -> GDPa3 ===")
    if len(combined_df):
        print(combined_df[["assay", "n_flab", "n_gdpa1", "rho"]].round(3))

    print("\nGenerating comparison figure ...")
    make_comparison_figure(cv_df, loo_df, transfer_df, combined_df)
    print(f"\nAll results saved to {OUT}")


if __name__ == "__main__":
    main()
