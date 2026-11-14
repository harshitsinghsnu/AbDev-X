"""Robustness check: does the augmented Titer specialist beat champion (0.310) across seeds?"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1

ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]
z = np.load(ROOT / "data/processed/flab_ablang2.npz", allow_pickle=True)
ext_emb = dict(zip(z["keys"], z["emb"]))
ext = pd.read_csv(ROOT / "data/processed/flab_Titer.csv")

y1 = ds.y("Titer"); lab = ~np.isnan(y1)
y3 = g3["titer_avg"].to_numpy(float); ok = ~np.isnan(y3)
mu, sd = np.nanmean(y1[lab]), np.nanstd(y1[lab]) + 1e-9
X1, Y1 = abl1[lab], (y1[lab] - mu) / sd
Xe = np.vstack([ext_emb[k] for k in ext["key"] if k in ext_emb])
ze = ext[[k in ext_emb for k in ext["key"]]]["z"].to_numpy(float)
Xa, Ya = np.vstack([X1, Xe]), np.concatenate([Y1, ze])

def gbm(seed): return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, subsample=.8, colsample_bytree=.6, reg_lambda=3, min_child_samples=10, subsample_freq=1, verbosity=-1, random_state=seed)

aug, base, ens = [], [], []
for s in range(15):
    pa = gbm(s).fit(Xa, Ya).predict(abl3)
    pb = gbm(s).fit(X1, Y1).predict(abl3)
    aug.append(spearmanr(y3[ok], pa[ok]).statistic)
    base.append(spearmanr(y3[ok], pb[ok]).statistic)
    ens.append(spearmanr(y3[ok], (pa + pb)[ok]).statistic)
aug, base, ens = np.array(aug), np.array(base), np.array(ens)
print(f"Titer champion = 0.310")
print(f"GDPa1-only  GBM: {base.mean():.3f} +/- {base.std():.3f}")
print(f"Augmented   GBM: {aug.mean():.3f} +/- {aug.std():.3f}  (beats champ in {int((aug>0.310).sum())}/15 seeds)")
print(f"Aug+base ensemble: {ens.mean():.3f} +/- {ens.std():.3f}  (beats champ in {int((ens>0.310).sum())}/15 seeds)")
