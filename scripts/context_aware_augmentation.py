"""
Context-Aware / Batch-Robust Multimodal Augmentation with FLAb.

Problem confirmed by large_data_experiment.py:
  - FLAb-trained model transfers poorly to GDPa1 despite large n
  - Study-LOO collapses (HIC rho=-0.049) showing inter-study batch effects
  - Combined FLAb+GDPa1 hurts Titer and Tm2 on GDPa3

This script implements progressively stronger batch correction strategies:
  1. Raw concatenation (baseline -- same as large_data_experiment)
  2. Per-study z-score normalization of labels (ComBat-lite)
  3. Study one-hot conditioning: study ID one-hot added as input feature
  4. Inverse-study-size weighting: down-weight large studies in ridge loss
  5. Study-level residual adapters: train a per-study offset from the base model
  6. Held-out study: train with all-but-one study, test on held-out (LOO)

Each is evaluated on:
  A. FLAb 5-fold CV (within-FLAb generalization)
  B. FLAb -> GDPa1 transfer (cross-domain with protocol shift)
  C. FLAb+GDPa1 -> GDPa3 (zero-shot OOD)

Usage:
  python scripts/context_aware_augmentation.py [--assay HIC]

Outputs:
  results/context_aware/context_cv.csv
  results/context_aware/context_transfer.csv
  paper/figures/panel_context_aware.pdf
"""
import os, sys, warnings, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RES  = ROOT / "results" / "context_aware"
FIGS = ROOT / "paper" / "figures"
RES.mkdir(parents=True, exist_ok=True)

PROC = ROOT / "data" / "processed"
FLAB = ROOT / "external" / "FLAb" / "data"

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
ASSAY5 = ["HIC", "Tm2", "AC-SINS", "PR-CHO", "Titer"]
DPI    = 300
FIG_W  = 7.0

# Same FLAb file map as large_data_experiment
FLAB_MAP = {
    "HIC": [
        ("aggregation/jain2017biophyscial_HICRT.csv",        None),
        ("aggregation/jain2024assessment_HIC.csv",            "HIC RT (min)"),
        ("aggregation/jetha2019homology_RT.csv",              None),
        ("aggregation/kraft2019herapin_relrt.csv",            None),
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


# ── helpers ──────────────────────────────────────────────────────────────────
def spear(a, b):
    if len(a) < 5: return np.nan
    r, _ = spearmanr(a, b); return float(r)

def norm_seq(s):
    import re
    return re.sub(r"[^A-Z]", "", str(s).upper())

def make_ridge(alphas=None):
    alphas = alphas or [0.01, 0.1, 1, 10, 100, 1000]
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=0.95, svd_solver="full")),
                     ("r", RidgeCV(alphas=alphas))])

def fold_iter(n, k=5, seed=42):
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)):
        yield tr, te


# ── Data loading ──────────────────────────────────────────────────────────────
def load_flab_with_study(assay, gdpa1_keys=None):
    """Load FLAb for one assay, keeping study labels. Returns DataFrame with
    columns: heavy, light, value, study, key."""
    import re
    frames = []
    for rel, valcol in FLAB_MAP.get(assay, []):
        p = FLAB / rel
        if not p.exists(): continue
        df = pd.read_csv(p)
        if "heavy" not in df.columns or "light" not in df.columns: continue
        if valcol is None:
            cand = [c for c in df.columns if c not in ("heavy", "light", "fitness")]
            valcol = cand[0] if cand else None
        if valcol not in df.columns: continue
        sub = df[["heavy", "light", valcol]].dropna()
        sub = sub.rename(columns={valcol: "value"})
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.dropna(subset=["value"])
        sub["key"] = sub["heavy"].map(norm_seq) + "|" + sub["light"].map(norm_seq)
        if gdpa1_keys:
            sub = sub[~sub["key"].isin(gdpa1_keys)]
        sub = sub.drop_duplicates("key")
        if len(sub) < 5: continue
        sub["study"] = Path(rel).stem
        frames.append(sub[["heavy", "light", "value", "study", "key"]])
    if not frames:
        return pd.DataFrame(columns=["heavy", "light", "value", "study", "key"])
    return pd.concat(frames, ignore_index=True)


def build_desc_features(heavy_series, light_series):
    from abdev.features.descriptors import build_descriptor_matrix
    return build_descriptor_matrix(
        pd.Series(list(heavy_series)), pd.Series(list(light_series))
    ).values


# ── Normalization strategies ─────────────────────────────────────────────────
def apply_normalization(df, method):
    """Normalize labels in df (which has 'value' and 'study' columns).
    Returns z-scored 'y' array and per-sample weights."""
    y = df["value"].values.copy().astype(float)
    w = np.ones(len(df))

    if method == "raw_z":
        y = (y - y.mean()) / (y.std() + 1e-9)

    elif method == "study_z":
        # Per-study z-score (ComBat-lite)
        for study, grp in df.groupby("study"):
            idx = grp.index
            v = y[idx]
            y[idx] = (v - v.mean()) / (v.std() + 1e-9)

    elif method == "study_z_iw":
        # Per-study z-score + inverse-size weighting
        study_sizes = df.groupby("study").size()
        for study, grp in df.groupby("study"):
            idx = grp.index
            v = y[idx]
            y[idx] = (v - v.mean()) / (v.std() + 1e-9)
            w[idx] = 1.0 / np.sqrt(study_sizes[study])
        w /= w.sum() / len(w)  # re-normalize mean weight to 1

    elif method == "study_rank_z":
        # Per-study rank-then-z: removes scale but preserves ordering
        for study, grp in df.groupby("study"):
            idx = grp.index
            v = y[idx]
            ranks = pd.Series(v).rank().values
            y[idx] = (ranks - ranks.mean()) / (ranks.std() + 1e-9)

    return y, w


def build_study_onehots(df):
    """One-hot encode study labels. Returns (n, n_studies) matrix."""
    le = LabelEncoder()
    study_idx = le.fit_transform(df["study"].values)
    n_studies = len(le.classes_)
    onehot = np.zeros((len(df), n_studies), dtype=float)
    onehot[np.arange(len(df)), study_idx] = 1.0
    return onehot, le.classes_


# ── Experiment A: FLAb 5-fold CV with different normalizations ───────────────
def exp_flab_cv_context(assay, gdpa1_keys):
    df = load_flab_with_study(assay, gdpa1_keys)
    if len(df) < 30:
        return []

    n_studies = df["study"].nunique()
    X_desc = build_desc_features(df["heavy"], df["light"])
    X_onehot, _ = build_study_onehots(df)

    rows = []
    for norm_method in ["raw_z", "study_z", "study_z_iw", "study_rank_z"]:
        y, w = apply_normalization(df.copy(), norm_method)
        # Without study conditioning
        oof = np.zeros(len(df))
        for tr, te in fold_iter(len(df)):
            m = make_ridge()
            # Weighted fit (sample_weight not supported in Pipeline directly)
            sc = StandardScaler(); Xtr = sc.fit_transform(X_desc[tr]); Xte = sc.transform(X_desc[te])
            pca = PCA(0.95, svd_solver="full"); Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
            rid = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
            rid.fit(Xtr, y[tr], sample_weight=w[tr])
            oof[te] = rid.predict(Xte)
        rows.append({"assay": assay, "norm": norm_method, "study_cond": False,
                     "n_studies": n_studies, "n": len(df), "rho": spear(y, oof)})

        # With study one-hot conditioning (study-aware input)
        X_aug = np.hstack([X_desc, X_onehot])
        oof_aug = np.zeros(len(df))
        for tr, te in fold_iter(len(df)):
            sc = StandardScaler(); Xtr = sc.fit_transform(X_aug[tr]); Xte = sc.transform(X_aug[te])
            pca = PCA(0.95, svd_solver="full"); Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
            rid = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
            rid.fit(Xtr, y[tr], sample_weight=w[tr])
            oof_aug[te] = rid.predict(Xte)
        rows.append({"assay": assay, "norm": norm_method, "study_cond": True,
                     "n_studies": n_studies, "n": len(df), "rho": spear(y, oof_aug)})

    return rows


# ── Experiment B: Study-level residual adapter ────────────────────────────────
def exp_residual_adapter(assay, gdpa1_keys):
    """Train base model on pooled z-scored FLAb, then fit per-study residuals."""
    df = load_flab_with_study(assay, gdpa1_keys)
    if len(df) < 30: return []

    y_raw, _ = apply_normalization(df.copy(), "raw_z")
    y_study, _ = apply_normalization(df.copy(), "study_z")
    X = build_desc_features(df["heavy"], df["light"])

    rows = []
    # Baseline: raw_z OOF
    oof_base = np.zeros(len(df))
    for tr, te in fold_iter(len(df)):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        pca = PCA(0.95, svd_solver="full"); Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
        rid = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]); rid.fit(Xtr, y_raw[tr])
        oof_base[te] = rid.predict(Xte)
    rows.append({"assay": assay, "method": "raw_z_no_adapter", "rho": spear(y_raw, oof_base)})

    # Residual adapter: base_pred -> per-study mean-shift correction
    oof_adapted = oof_base.copy()
    for study, grp in df.groupby("study"):
        idx = grp.index.tolist()
        if len(idx) < 3: continue
        residuals = y_raw[idx] - oof_base[idx]
        study_bias = residuals.mean()
        oof_adapted[idx] = oof_base[idx] + study_bias
    rows.append({"assay": assay, "method": "residual_adapter", "rho": spear(y_raw, oof_adapted)})

    # Study-z OOF
    oof_sz = np.zeros(len(df))
    for tr, te in fold_iter(len(df)):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        pca = PCA(0.95, svd_solver="full"); Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
        rid = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]); rid.fit(Xtr, y_study[tr])
        oof_sz[te] = rid.predict(Xte)
    rows.append({"assay": assay, "method": "study_z_oof", "rho": spear(y_study, oof_sz)})
    return rows


# ── Experiment C: Context-conditioned FLAb -> GDPa1 transfer ─────────────────
def exp_context_transfer(assay, gdpa1_keys, ds, gdpa3):
    df = load_flab_with_study(assay, gdpa1_keys)
    if len(df) < 20: return []

    X_flab = build_desc_features(df["heavy"], df["light"])
    rows = []

    for norm_method in ["raw_z", "study_z", "study_z_iw"]:
        y, w = apply_normalization(df.copy(), norm_method)

        # Fit on FLAb
        sc = StandardScaler(); Xtr = sc.fit_transform(X_flab)
        pca = PCA(0.95, svd_solver="full"); Xtr = pca.fit_transform(Xtr)
        rid = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]); rid.fit(Xtr, y, sample_weight=w)

        # Test on GDPa1
        gdpa1_y = ds.y(assay); valid = ~np.isnan(gdpa1_y)
        if valid.sum() >= 10:
            X_g1 = build_desc_features(ds.heavy[valid], ds.light[valid])
            Xte = pca.transform(sc.transform(X_g1))
            pred = rid.predict(Xte)
            y_g1_z = (gdpa1_y[valid] - gdpa1_y[valid].mean()) / (gdpa1_y[valid].std() + 1e-9)
            rows.append({"assay": assay, "norm": norm_method, "target": "GDPa1",
                         "rho": spear(y_g1_z, pred)})

        # Test on GDPa3
        if assay in gdpa3.columns:
            g3 = gdpa3[["vh", "vl", assay]].dropna()
            if len(g3) >= 5:
                X_g3 = build_desc_features(g3["vh"], g3["vl"])
                Xte3 = pca.transform(sc.transform(X_g3))
                pred3 = rid.predict(Xte3)
                y_g3 = g3[assay].values
                y_g3z = (y_g3 - y_g3.mean()) / (y_g3.std() + 1e-9)
                rows.append({"assay": assay, "norm": norm_method, "target": "GDPa3",
                             "rho": spear(y_g3z, pred3)})
    return rows


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(cv_df, adapter_df, transfer_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                                 "axes.labelsize": 9, "axes.titlesize": 9, "pdf.fonttype": 42})
    PAL = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#DD8452", "#64B5CD"]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_W * 0.52), constrained_layout=True)

    # Panel A: FLAb CV by normalization method (no study cond)
    ax = axes[0]
    if len(cv_df):
        sub = cv_df[~cv_df["study_cond"]].copy()
        norms = sub["norm"].unique()
        x = np.arange(len(ASSAYS)); w = 0.65 / max(len(norms), 1)
        for i, nm in enumerate(norms):
            s = sub[sub["norm"] == nm].set_index("assay")
            vals = [s.loc[a, "rho"] if a in s.index else np.nan for a in ASSAYS]
            ax.bar(x + (i - len(norms)/2 + 0.5)*w, vals, w,
                   label=nm.replace("_", " "), color=PAL[i], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(ASSAY5, fontsize=6, rotation=35, ha="right")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("Spearman rho"); ax.set_title("(A) FLAb CV: normalization")
        ax.legend(fontsize=5, ncol=1, loc="upper right", framealpha=0.8)

    # Panel B: Study-cond vs no-cond comparison (study_z method)
    ax = axes[1]
    if len(cv_df):
        nc = cv_df[(cv_df["norm"] == "study_z") & ~cv_df["study_cond"]].set_index("assay")
        sc_ = cv_df[(cv_df["norm"] == "study_z") & cv_df["study_cond"]].set_index("assay")
        x = np.arange(len(ASSAYS))
        nv = [nc.loc[a, "rho"] if a in nc.index else np.nan for a in ASSAYS]
        sv = [sc_.loc[a, "rho"] if a in sc_.index else np.nan for a in ASSAYS]
        ax.bar(x - 0.18, nv, 0.35, color=PAL[0], label="no cond")
        ax.bar(x + 0.18, sv, 0.35, color=PAL[1], label="+ onehot")
        ax.set_xticks(x); ax.set_xticklabels(ASSAY5, fontsize=6, rotation=35, ha="right")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title("(B) Study one-hot conditioning")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)

    # Panel C: Transfer to GDPa1/GDPa3 by norm method
    ax = axes[2]
    if len(transfer_df):
        markers = {"GDPa1": "o-", "GDPa3": "s--"}
        norm_colors = {"raw_z": PAL[0], "study_z": PAL[1], "study_z_iw": PAL[2]}
        for norm in transfer_df["norm"].unique():
            for tgt in transfer_df["target"].unique():
                sub = transfer_df[(transfer_df["norm"] == norm) & (transfer_df["target"] == tgt)].set_index("assay")
                vals = [sub.loc[a, "rho"] if a in sub.index else np.nan for a in ASSAYS]
                lbl = f"{norm.replace('_z','')}/{tgt}"
                ax.plot(np.arange(len(ASSAYS)), vals,
                        markers.get(tgt, "o-"), color=norm_colors.get(norm, "grey"),
                        ms=4, lw=1.0, label=lbl)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(np.arange(len(ASSAYS))); ax.set_xticklabels(ASSAY5, fontsize=6, rotation=35, ha="right")
        ax.set_ylabel("Spearman rho"); ax.set_title("(C) FLAb transfer")
        ax.legend(fontsize=5, ncol=1, loc="upper right", framealpha=0.8)
    out = FIGS / "panel_context_aware.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight", dpi=DPI)
    import matplotlib.pyplot as _p; _p.close(fig)
    print(f"  Figure: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", default="all")
    args = parser.parse_args()

    import gdpa3_eval as GE
    ds, g3, _, _ = GE.build_blocks()
    gdpa3 = pd.read_csv(ROOT / "data" / "processed" / "gdpa3_table.csv")

    # GDPa1 keys to exclude from FLAb training
    gdpa1_keys = set(
        norm_seq(h) + "|" + norm_seq(l)
        for h, l in zip(ds.heavy.tolist(), ds.light.tolist())
    )

    target_assays = ASSAYS if args.assay == "all" else [args.assay]

    cv_rows, adapter_rows, transfer_rows = [], [], []
    for assay in target_assays:
        print(f"\n=== {assay} ===")
        cv_rows      += exp_flab_cv_context(assay, gdpa1_keys)
        adapter_rows += exp_residual_adapter(assay, gdpa1_keys)
        transfer_rows += exp_context_transfer(assay, gdpa1_keys, ds, gdpa3)

    cv_df       = pd.DataFrame(cv_rows)
    adapter_df  = pd.DataFrame(adapter_rows)
    transfer_df = pd.DataFrame(transfer_rows)

    cv_df.to_csv(RES / "context_cv.csv", index=False)
    adapter_df.to_csv(RES / "context_adapter.csv", index=False)
    transfer_df.to_csv(RES / "context_transfer.csv", index=False)

    print("\n=== FLAb CV by normalization (no study cond) ===")
    if len(cv_df):
        sub = cv_df[~cv_df["study_cond"]]
        print(sub.pivot(index="assay", columns="norm", values="rho").round(3))

    print("\n=== Study one-hot conditioning effect (study_z) ===")
    if len(cv_df):
        s_no = cv_df[(cv_df["norm"] == "study_z") & ~cv_df["study_cond"]].set_index("assay")["rho"]
        s_yes = cv_df[(cv_df["norm"] == "study_z") & cv_df["study_cond"]].set_index("assay")["rho"]
        delta = (s_yes - s_no).rename("delta")
        print(pd.concat([s_no.rename("no_cond"), s_yes.rename("study_onehot"), delta], axis=1).round(3))

    print("\n=== Residual adapter effect ===")
    if len(adapter_df):
        print(adapter_df.pivot(index="assay", columns="method", values="rho").round(3))

    print("\n=== Transfer to GDPa1/GDPa3 by normalization ===")
    if len(transfer_df):
        print(transfer_df.groupby(["assay", "norm", "target"])["rho"].mean().round(3))

    make_figure(cv_df, adapter_df, transfer_df)
    print(f"\nResults in {RES}")


if __name__ == "__main__":
    main()
