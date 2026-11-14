"""Bootstrap 95% CIs for AbDev-X GDPa3 held-out Spearman rho.

For each assay, trains on GDPa1 with the OOD-CV selected block, predicts GDPa3,
then bootstraps (resample n=80 with replacement, 10000 iterations) to get
bias-corrected accelerated (BCa) 95% CIs. Compares to competition per-assay best.
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

ASSAYS   = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
LABEL_MAP = {
    "HIC":           "hic_rt_avg",
    "Tm2":           "tm2_nanodsf_avg",
    "AC-SINS_pH7.4": "acsins_dLmax_ph7.4_avg",
    "PR_CHO":        "polyreactivity_prscore_cho_avg",
    "Titer":         "titer_avg",
}
COMP = {"HIC": 0.708, "Tm2": 0.392, "AC-SINS_pH7.4": 0.356, "PR_CHO": 0.337, "Titer": 0.310}
N_BOOT = 10_000
RNG    = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

def build_blocks():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")

    esm1  = next(np.load(p)["emb"] for p in (ROOT / "data/processed").glob("esm2_*.npz")
                 if np.load(p)["emb"].shape == (len(ds.df), 2560))
    abl1  = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    desc1 = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
    iso1  = _isotype_block(ds.df)
    bio1, cols = _bio_block(ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1",
                            ds.df["antibody_name"])

    esm3  = np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"]
    abl3  = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]
    desc3 = np.nan_to_num(build_descriptor_matrix(g3["vh"], g3["vl"]).to_numpy(float))
    iso3  = _isotype_block(g3)
    bio3, _ = _bio_block(ROOT / "external/abdev-benchmark/data/features/processed_features/heldout_test",
                         g3["antibody_name"], cols=cols)

    B1 = {"esm2": np.nan_to_num(esm1), "descriptors": desc1,
          "isotype": iso1, "bio": bio1, "ablang2": np.nan_to_num(abl1)}
    B3 = {"esm2": np.nan_to_num(esm3), "descriptors": desc3,
          "isotype": iso3, "bio": bio3, "ablang2": np.nan_to_num(abl3)}
    return ds, g3, B1, B3


ISO_COLS = ["hc_subtype_IgG1","hc_subtype_IgG2","hc_subtype_IgG4",
            "lc_subtype_Kappa","lc_subtype_Lambda"]
BIO_FILES = ["TAP.csv","MOE_properties.csv","Aggrescan3D.csv","AntiFold.csv"]


def _isotype_block(df):
    d = pd.get_dummies(df[["hc_subtype","lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS, fill_value=0).to_numpy(np.float32)


def _bio_block(feat_dir, names, cols=None):
    base = pd.DataFrame({"antibody_name": list(names)})
    for f in BIO_FILES:
        p = Path(feat_dir) / f
        if not p.exists(): continue
        d = pd.read_csv(p)
        num = d.select_dtypes("number").copy()
        num["antibody_name"] = d["antibody_name"].values
        num = num.rename(columns={c: f"{f[:-4]}::{c}" for c in num.columns if c != "antibody_name"})
        base = base.merge(num, on="antibody_name", how="left")
    X = base.drop(columns=["antibody_name"])
    if cols is None: cols = list(X.columns)
    return np.nan_to_num(X.reindex(columns=cols, fill_value=0.0).to_numpy(float)), cols


def _head(blk):
    steps = [("s", StandardScaler())]
    if blk in ("esm2", "ablang2"):
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Bootstrap BCa CI
# ---------------------------------------------------------------------------

def bca_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = N_BOOT,
           alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected accelerated bootstrap CI for Spearman rho."""
    from scipy.stats import norm

    n = len(y_true)
    obs = spearmanr(y_true, y_pred).statistic

    # bootstrap distribution
    boot = np.array([
        spearmanr(y_true[idx := RNG.integers(0, n, n)],
                  y_pred[idx]).statistic
        for _ in range(n_boot)
    ])
    boot = boot[np.isfinite(boot)]

    # bias correction z0
    z0 = norm.ppf(np.mean(boot < obs))

    # acceleration a (jackknife)
    jk = np.array([spearmanr(np.delete(y_true, i), np.delete(y_pred, i)).statistic
                   for i in range(n)])
    jk_mean = jk.mean()
    num = np.sum((jk_mean - jk) ** 3)
    den = 6 * (np.sum((jk_mean - jk) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)

    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo = np.percentile(boot, 100 * adj(z_lo))
    hi = np.percentile(boot, 100 * adj(z_hi))
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ds, g3, B1, B3 = build_blocks()
    folds = ds.folds("cluster_isotype")

    rows = []
    for a in ASSAYS:
        y1 = ds.y(a)
        y3 = g3[LABEL_MAP[a]].to_numpy(float)
        ok3 = ~np.isnan(y3)

        # OOD-CV: pick best block by cluster_isotype leave-one-fold-out
        cv_scores = {}
        for blk, X1 in B1.items():
            per = []
            for f in sorted(set(folds.tolist())):
                tr = np.where((folds != f) & ~np.isnan(y1))[0]
                te = np.where((folds == f) & ~np.isnan(y1))[0]
                if len(te) < 3 or len(tr) < 10: continue
                p = _head(blk).fit(X1[tr], y1[tr]).predict(X1[te])
                per.append(spearmanr(y1[te], p).statistic)
            cv_scores[blk] = float(np.nanmean(per))

        best_blk = max(cv_scores, key=cv_scores.get)
        m1 = ~np.isnan(y1)
        pred3 = _head(best_blk).fit(B1[best_blk][m1], y1[m1]).predict(B3[best_blk])

        rho = spearmanr(y3[ok3], pred3[ok3]).statistic
        lo, hi = bca_ci(y3[ok3], pred3[ok3])

        comp_val = COMP[a]
        comp_in_ci = lo <= comp_val <= hi

        rows.append({
            "assay":        a,
            "best_block":   best_blk,
            "rho":          round(float(rho), 3),
            "ci_lo":        round(lo, 3),
            "ci_hi":        round(hi, 3),
            "comp_best":    comp_val,
            "comp_in_ci":   comp_in_ci,
        })
        print(f"{a:<15} block={best_blk:<12} rho={rho:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"comp={comp_val}  comp_in_CI={comp_in_ci}")

    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("AbDev-X GDPa3 held-out: bootstrap BCa 95% CI vs competition best")
    print("="*80)
    print(df.to_string(index=False))
    n_overlap = df["comp_in_ci"].sum()
    print(f"\nCompetition best inside AbDev-X 95% CI: {n_overlap}/5 assays")
    df.to_csv(ROOT / "results/bootstrap_ci.csv", index=False)
    print(f"Saved to results/bootstrap_ci.csv")


if __name__ == "__main__":
    main()
