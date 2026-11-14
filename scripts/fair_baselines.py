"""Fair same-split comparison: reproduce 3 recent baselines IN OUR HARNESS.

All models are evaluated with the identical protocol used for AbDev-CAPLM and AbDev-X:
  * 5-fold CV under each generalization split (random, cluster, cluster_isotype)
  * aggregated Spearman rho (pool out-of-fold predictions), per assay
so the comparison is strictly controlled (same folds, same metric, same code path).

Three recent representative models:
  * ESM-2 (650M) mean-pool + Ridge            -- general protein LM (Lin et al. 2023)
  * AbLang2 (paired) seqcoding + ElasticNet    -- antibody-specific LM (Olsen et al. 2024)
  * SaProt (structure-aware) features + Ridge  -- structure-aware LM (Su et al. 2024)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
import experiment_bench as EB

ASSAYS = EB.ASSAYS
SPLITS = ["random", "cluster", "cluster_isotype"]


def ridge(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])
def enet(): return Pipeline([("s", StandardScaler()), ("r", ElasticNetCV(l1_ratio=0.5, n_alphas=20, cv=3, max_iter=2000, random_state=0))])


def saprot_features(ds):
    """SaProt VH structure-aware features from the official benchmark (stability, solubility)."""
    p = ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1/Saprot_VH.csv"
    df = pd.read_csv(p)
    base = pd.DataFrame({"antibody_name": ds.df[ds.id_col].values}).merge(df, on="antibody_name", how="left")
    return np.nan_to_num(base.drop(columns=["antibody_name"]).select_dtypes("number").to_numpy(float))


def agg_eval(ds, X, mk, split):
    res = {}
    for a in ASSAYS:
        y = ds.y(a); pooled = np.full(len(y), np.nan)
        for _, tr, te in fold_iter(ds, a, split):
            pooled[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        m = ~np.isnan(y) & ~np.isnan(pooled)
        res[a] = float(spearmanr(y[m], pooled[m]).statistic)
    res["MEAN"] = float(np.mean([res[a] for a in ASSAYS]))
    return res


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    ESM = np.nan_to_num(EB.load_esm())
    ABL = np.nan_to_num(EB.load_ablang2())
    SAP = saprot_features(ds)
    models = [("ESM2-Ridge", ESM, ridge), ("AbLang2-ElasticNet", ABL, enet), ("SaProt-Ridge", SAP, ridge)]

    rows = []
    for split in SPLITS:
        if split not in ds.fold_cols:
            continue
        for name, X, mk in models:
            r = agg_eval(ds, X, mk, split)
            rows.append(dict(model=name, split=split, **r))
            print(f"{split:16s} {name:20s} " + " ".join(f"{a}={r[a]:.3f}" for a in ASSAYS) + f"  MEAN={r['MEAN']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/fair_baselines.csv", index=False)
    print("\nSaved results/fair_baselines.csv")


if __name__ == "__main__":
    main()
