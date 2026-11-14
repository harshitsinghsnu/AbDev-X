"""Exp 3: Structure uncertainty robustness.

Two analyses:
  (A) Structural source comparison:
      - MOE_properties features (3D structure from MOE)
      - Aggrescan3D features (solubility patches from structure)
      - SAP + electrostatic patches (sap_gdpa1.csv, electro_gdpa1.csv)
      - Combined structural block
      vs AbLang2 / ESM2 (sequence-only)

  (B) SASA/patch noise injection:
      Inject Gaussian noise (sigma = 0.1 to 1.0 x feature std) into structural features,
      refit Ridge, compare Spearman drop on held-out fold (cluster_isotype).
      Reports HIC and AC-SINS as primary assays (structure-sensitive).

All structural features are used frozen / as-cached; no external tool calls.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE

ASSAYS = GE.ASSAYS
STRUCT_SENSITIVE = ["HIC", "AC-SINS_pH7.4"]   # structure drives these two most
BENCH = ROOT / "external/abdev-benchmark/data/features/processed_features"
NOISE_SIGMAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_BOOT = 30


def _head(high_dim=False):
    steps = [("s", StandardScaler())]
    if high_dim:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def load_structural_blocks(ds):
    """Load all structural feature blocks for GDPa1."""
    blocks = {}

    # MOE_properties
    moe_p = BENCH / "GDPa1" / "MOE_properties.csv"
    if moe_p.exists():
        moe = pd.read_csv(moe_p)
        base = pd.DataFrame({"antibody_name": ds.df[ds.id_col].values})
        moe_num = moe.select_dtypes("number").copy()
        moe_num["antibody_name"] = moe["antibody_name"].values
        merged = base.merge(moe_num, on="antibody_name", how="left")
        X = np.nan_to_num(merged.drop(columns=["antibody_name"]).to_numpy(float))
        blocks["MOE"] = X

    # Aggrescan3D
    agg_p = BENCH / "GDPa1" / "Aggrescan3D.csv"
    if agg_p.exists():
        agg = pd.read_csv(agg_p)
        base = pd.DataFrame({"antibody_name": ds.df[ds.id_col].values})
        agg_num = agg.select_dtypes("number").copy()
        agg_num["antibody_name"] = agg["antibody_name"].values
        merged = base.merge(agg_num, on="antibody_name", how="left")
        X = np.nan_to_num(merged.drop(columns=["antibody_name"]).to_numpy(float))
        blocks["Aggrescan3D"] = X

    # SAP patches
    sap_p = ROOT / "data/processed/sap_gdpa1.csv"
    sap_patch_p = ROOT / "data/processed/sap_patch_gdpa1.csv"
    if sap_p.exists() and sap_patch_p.exists():
        sap = pd.read_csv(sap_p)
        sp = pd.read_csv(sap_patch_p)
        sap_num = sap.select_dtypes("number")
        sp_num = sp.select_dtypes("number")
        X = np.nan_to_num(np.concatenate([sap_num.to_numpy(float), sp_num.to_numpy(float)], axis=1))
        blocks["SAP"] = X

    # Electrostatic patches
    el_p = ROOT / "data/processed/electro_gdpa1.csv"
    if el_p.exists():
        el = pd.read_csv(el_p)
        el_num = el.select_dtypes("number")
        blocks["Electro"] = np.nan_to_num(el_num.to_numpy(float))

    # Combined structural
    avail = [v for v in [blocks.get("MOE"), blocks.get("Aggrescan3D"),
                          blocks.get("SAP"), blocks.get("Electro")] if v is not None]
    if avail:
        blocks["Combined_struct"] = np.concatenate(avail, axis=1)

    return blocks


def eval_cv_single(X, ds, assay, split="cluster_isotype", high_dim=True):
    folds = ds.folds(split)
    ufolds = sorted(set(f for f in folds.tolist() if f == f))
    y = ds.y(assay)
    lab = ~np.isnan(y)
    scores = []
    for f in ufolds:
        tr = np.where((folds != f) & lab)[0]
        te = np.where((folds == f) & lab)[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        pipe = _head(high_dim=high_dim)
        preds = pipe.fit(X[tr], y[tr]).predict(X[te])
        scores.append(spearmanr(y[te], preds).statistic)
    return float(np.nanmean(scores)) if scores else np.nan


def noise_sensitivity(X_struct, ds, assay, n_boot=N_BOOT, rng=None):
    """Inject multiplicative noise at each sigma level; return mean Spearman per sigma."""
    if rng is None:
        rng = np.random.RandomState(42)
    folds = ds.folds("cluster_isotype")
    ufolds = sorted(set(f for f in folds.tolist() if f == f))
    y = ds.y(assay)
    lab = ~np.isnan(y)
    col_stds = X_struct.std(0).clip(min=1e-8)

    results = []
    for sigma in NOISE_SIGMAS:
        boot_scores = []
        for _ in range(n_boot if sigma > 0 else 1):
            X_noisy = X_struct.copy()
            if sigma > 0:
                noise = rng.normal(0, sigma, X_struct.shape) * col_stds[None, :]
                X_noisy = X_struct + noise
            scores = []
            for f in ufolds:
                tr = np.where((folds != f) & lab)[0]
                te = np.where((folds == f) & lab)[0]
                if len(te) < 3 or len(tr) < 10:
                    continue
                pipe = _head(high_dim=False)
                preds = pipe.fit(X_noisy[tr], y[tr]).predict(X_noisy[te])
                scores.append(spearmanr(y[te], preds).statistic)
            boot_scores.append(np.nanmean(scores) if scores else np.nan)
        results.append({
            "sigma": sigma,
            "rho_mean": round(float(np.nanmean(boot_scores)), 4),
            "rho_std": round(float(np.nanstd(boot_scores)), 4),
        })
    return pd.DataFrame(results)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading data ...")
    ds, g3, B1, B3 = GE.build_blocks()

    print("Loading structural blocks ...")
    struct_blocks = load_structural_blocks(ds)
    print(f"Structural blocks: {[(k, v.shape) for k, v in struct_blocks.items()]}")

    # Sequence baselines
    seq_blocks = {
        "ESM2_650M": np.nan_to_num(B1["esm2"]),
        "AbLang2": np.nan_to_num(B1["ablang2"]),
        "Descriptors": np.nan_to_num(B1["descriptors"]),
    }
    all_blocks = {**struct_blocks, **seq_blocks}

    # ---- Part A: CV comparison ----
    print("\n=== Part A: Structural vs sequence features (cluster_isotype CV) ===")
    rows_a = []
    for blk_name, X in all_blocks.items():
        high = X.shape[1] >= 480
        for assay in ASSAYS:
            cv = eval_cv_single(X, ds, assay, "cluster_isotype", high_dim=high)
            rows_a.append({"block": blk_name, "assay": assay, "cv_cluster_isotype": round(cv, 3),
                            "is_structural": blk_name not in seq_blocks})

    df_a = pd.DataFrame(rows_a)
    pivot_a = df_a.pivot_table(index="block", columns="assay", values="cv_cluster_isotype")
    pivot_a["MEAN"] = pivot_a.mean(axis=1)
    pivot_a = pivot_a.sort_values("MEAN", ascending=False)
    print(pivot_a.round(3).to_string())

    # ---- Part B: Noise sensitivity ----
    print("\n=== Part B: Noise sensitivity of structural blocks (HIC and AC-SINS) ===")
    rows_b = []
    for assay in STRUCT_SENSITIVE:
        for blk_name in ["Combined_struct", "SAP", "MOE", "Aggrescan3D"]:
            if blk_name not in struct_blocks:
                continue
            X = struct_blocks[blk_name]
            noise_df = noise_sensitivity(X, ds, assay, n_boot=N_BOOT)
            noise_df["assay"] = assay
            noise_df["block"] = blk_name
            rows_b.append(noise_df)
            # Summary line
            base = noise_df[noise_df["sigma"] == 0.0]["rho_mean"].values[0]
            max_drop = noise_df["rho_mean"].min()
            print(f"  {blk_name} | {assay}: base={base:.3f}, drop at max sigma={max_drop:.3f}, "
                  f"delta={base-max_drop:.3f}")

    if rows_b:
        df_b = pd.concat(rows_b, ignore_index=True)
        df_b.to_csv(ROOT / "results/exp_struct_noise.csv", index=False)

    df_a.to_csv(ROOT / "results/exp_struct_comparison.csv", index=False)
    print("\nSaved results/exp_struct_comparison.csv and results/exp_struct_noise.csv")


if __name__ == "__main__":
    main()
