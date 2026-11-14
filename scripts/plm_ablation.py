"""PLM ablation: ESM-2, AbLang2, AntiBERTy, ProtBert, ProGen2 under one protocol.

Each PLM's embeddings -> StandardScaler -> PCA(0.9) -> RidgeCV; GDPa1 5-fold cluster-isotype CV;
per-assay Spearman (mean +/- std across folds) + MEAN. Antibody-specific vs general PLMs.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
PROC = ROOT / "data/processed"


def esm2_mean():
    for p in PROC.glob("esm2_*.npz"):
        e = np.load(p)["emb"]
        if e.shape == (246, 1280):
            return e
    # fall back: first 1280 dims of the mean+cdr cache
    for p in PROC.glob("esm2_*.npz"):
        e = np.load(p)["emb"]
        if e.shape[1] == 2560:
            return e[:, :1280]
    return None


PLMS = {
    "ESM-2 (650M)": esm2_mean(),
    "AbLang2 (Ab)": np.load(PROC / "ablang2_gdpa1.npz")["emb"],
    "AntiBERTy (Ab)": np.load(PROC / "antiberty_gdpa1.npz")["emb"],
    "ProtBert": np.load(PROC / "protbert_gdpa1.npz")["emb"],
    "ProGen2 (small)": np.load(PROC / "progen2_gdpa1.npz")["emb"],
}


def head(): return Pipeline([("s", StandardScaler()), ("p", PCA(0.9, random_state=0, svd_solver="full")), ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    rows = []
    for name, X in PLMS.items():
        if X is None:
            continue
        X = np.nan_to_num(X)
        rec = {"PLM": name, "dim": X.shape[1]}
        means = []
        for a in ASSAYS:
            y = ds.y(a); per = []
            for _, tr, te in fold_iter(ds, a, "cluster_isotype"):
                per.append(spearmanr(y[te], head().fit(X[tr], y[tr]).predict(X[te])).statistic)
            rec[a] = round(float(np.mean(per)), 3); rec[a + "_std"] = round(float(np.std(per)), 3)
            means.append(np.mean(per))
        rec["MEAN"] = round(float(np.mean(means)), 3)
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("MEAN", ascending=False)
    cols = ["PLM", "dim"] + ASSAYS + ["MEAN"]
    print("=== PLM ablation: GDPa1 cluster-isotype CV Spearman (mean across folds) ===")
    print(df[cols].to_string(index=False))
    df.to_csv(ROOT / "results/plm_ablation.csv", index=False)

    # figure with error bars
    fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(ASSAYS)); n = len(df); w = 0.8 / n
    for i, (_, r) in enumerate(df.iterrows()):
        ax.bar(x + (i - n / 2 + 0.5) * w, [r[a] for a in ASSAYS], w,
               yerr=[r[a + "_std"] for a in ASSAYS], capsize=2, label=f"{r['PLM']} ({r['MEAN']})")
    ax.set_xticks(x); ax.set_xticklabels(ASSAYS, rotation=20, ha="right")
    ax.set_ylabel("Spearman rho (CV)"); ax.set_title("PLM ablation (GDPa1 cluster-isotype CV, ridge head)")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(ROOT / "results/fig_plm_ablation.png", dpi=140)
    print("Figure: results/fig_plm_ablation.png")


if __name__ == "__main__":
    main()
