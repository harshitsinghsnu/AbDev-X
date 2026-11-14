"""Augmented per-assay specialists: GDPa1 + external same-assay data (AbLang2 features) -> GDPa3.

Extracts AbLang2 for external sequences, pools GDPa1 (z-scored) + external (per-study z), trains a
regularized head, predicts GDPa3. Compares augmented vs GDPa1-only vs competition champion.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1

COMP = {"HIC": 0.708, "Tm2": 0.392, "AC-SINS_pH7.4": 0.356, "PR_CHO": 0.337, "Titer": 0.310}
LABEL = {"HIC": "hic_rt_avg", "Tm2": "tm2_nanodsf_avg", "AC-SINS_pH7.4": "acsins_dLmax_ph7.4_avg",
         "PR_CHO": "polyreactivity_prscore_cho_avg", "Titer": "titer_avg"}


def extract_external_ablang2():
    cache = ROOT / "data/processed/flab_ablang2.npz"
    uni = pd.read_csv(ROOT / "data/processed/flab_unique_seqs.csv")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(zip(z["keys"], z["emb"]))
    import ablang2
    model = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1,
                               device="cuda" if torch.cuda.is_available() else "cpu")
    pairs = [[str(h), str(l)] for h, l in zip(uni["heavy"], uni["light"])]
    emb = []
    for i in range(0, len(pairs), 16):
        emb.append(np.asarray(model(pairs[i:i + 16], mode="seqcoding")))
        print(f"  ext {min(i+16,len(pairs))}/{len(pairs)}", flush=True)
    emb = np.vstack(emb)
    np.savez_compressed(cache, keys=uni["key"].to_numpy(), emb=emb)
    return dict(zip(uni["key"].tolist(), emb))


def ridge(): return Pipeline([("s", StandardScaler()), ("p", PCA(0.9, random_state=0, svd_solver="full")), ("r", RidgeCV(alphas=np.logspace(-2, 4, 14)))])
def gbm(): return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, subsample=.8, colsample_bytree=.6, reg_lambda=3, min_child_samples=10, subsample_freq=1, verbosity=-1, random_state=0)


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]
    ext_emb = extract_external_ablang2()

    rows = []
    for a in ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]:
        y1 = ds.y(a); lab = ~np.isnan(y1)
        y3 = g3[LABEL[a]].to_numpy(float); ok = ~np.isnan(y3)
        mu, sd = np.nanmean(y1[lab]), np.nanstd(y1[lab]) + 1e-9
        X1, Y1 = abl1[lab], (y1[lab] - mu) / sd

        fp = ROOT / f"data/processed/flab_{a.replace('/', '_')}.csv"
        rec = {"assay": a, "champion": COMP[a], "n_ext": 0}
        for tag, mk in [("ridge", ridge), ("gbm", gbm)]:
            # GDPa1-only
            base = spearmanr(y3[ok], mk().fit(X1, Y1).predict(abl3)[ok]).statistic
            rec[f"gdpa1_{tag}"] = round(float(base), 3)
            # augmented
            if fp.exists():
                ext = pd.read_csv(fp)
                Xe = np.vstack([ext_emb[k] for k in ext["key"] if k in ext_emb])
                ze = ext[[k in ext_emb for k in ext["key"]]]["z"].to_numpy(float)
                rec["n_ext"] = len(Xe)
                Xa = np.vstack([X1, Xe]); Ya = np.concatenate([Y1, ze])
                aug = spearmanr(y3[ok], mk().fit(Xa, Ya).predict(abl3)[ok]).statistic
                rec[f"aug_{tag}"] = round(float(aug), 3)
        rows.append(rec)
    df = pd.DataFrame(rows)
    print("\n=== Augmented specialists (AbLang2) on GDPa3 held-out ===")
    print(df.to_string(index=False))
    df["best_aug"] = df[[c for c in df.columns if c.startswith("aug_")]].max(axis=1)
    df["beat"] = df["best_aug"] > df["champion"]
    print(f"\nChampions beaten: {int(df['beat'].sum())}/5")
    df.to_csv(ROOT / "results/flab_specialist.csv", index=False)


if __name__ == "__main__":
    main()
