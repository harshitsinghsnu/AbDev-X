"""Experiments (6) cross-dataset transfer, (13) augmentation scaling, (9) uncertainty/calibration."""
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
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE
from abdev.data.dataset import load_gdpa1
from abdev.models.fusion import AbDevFusion, DEFAULT_BASES
ASSAYS = GE.ASSAYS
FIG = ROOT / "results/figures"


def rpca(): return Pipeline([("s", StandardScaler()), ("p", PCA(0.9, random_state=0, svd_solver="full")), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])
def gbm(seed=0): return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, subsample=.8, colsample_bytree=.6, reg_lambda=3, min_child_samples=10, subsample_freq=1, verbosity=-1, random_state=seed)


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]
    z = np.load(ROOT / "data/processed/flab_ablang2.npz", allow_pickle=True)
    ext_emb = dict(zip(z["keys"], z["emb"]))
    Y3 = {a: g3[GE.LABEL_MAP[a]].to_numpy(float) for a in ASSAYS}

    # ---- (6) cross-dataset transfer (GDPa1 <-> FLAb external, same assay) ----
    tr_rows = []
    for a in ASSAYS:
        fp = ROOT / f"data/processed/flab_{a.replace('/', '_')}.csv"
        if not fp.exists():
            continue
        ext = pd.read_csv(fp); keep = [k in ext_emb for k in ext["key"]]
        Xe = np.vstack([ext_emb[k] for k in ext["key"] if k in ext_emb]); ze = ext[keep]["z"].to_numpy(float)
        y1 = ds.y(a); m = ~np.isnan(y1)
        # GDPa1 -> FLAb
        p = rpca().fit(abl1[m], y1[m]).predict(Xe); g1_to_flab = spearmanr(ze, p).statistic
        # FLAb -> GDPa1
        p2 = rpca().fit(Xe, ze).predict(abl1[m]); flab_to_g1 = spearmanr(y1[m], p2).statistic
        tr_rows.append(dict(assay=a, n_ext=len(ze), GDPa1_to_FLAb=round(float(g1_to_flab), 3),
                            FLAb_to_GDPa1=round(float(flab_to_g1), 3)))
    tr = pd.DataFrame(tr_rows); tr.to_csv(ROOT / "results/exp_transfer.csv", index=False)
    print("=== (6) Cross-dataset transfer (AbLang2 ridge, same assay) ===")
    print(tr.to_string(index=False))

    # ---- (13) augmentation scaling: per-study contribution to held-out ----
    sc_rows = []
    for a in ["Titer", "AC-SINS_pH7.4", "PR_CHO", "Tm2", "HIC"]:
        fp = ROOT / f"data/processed/flab_{a.replace('/', '_')}.csv"
        if not fp.exists():
            continue
        ext = pd.read_csv(fp); y1 = ds.y(a); m = ~np.isnan(y1); ok = ~np.isnan(Y3[a])
        mu, sd = np.nanmean(y1[m]), np.nanstd(y1[m]) + 1e-9
        base = spearmanr(Y3[a][ok], gbm().fit(abl1[m], (y1[m] - mu) / sd).predict(abl3)[ok]).statistic
        for study, sub in ext.groupby("study"):
            keep = [k in ext_emb for k in sub["key"]]
            Xe = np.vstack([ext_emb[k] for k in sub["key"] if k in ext_emb]); ze = sub[keep]["z"].to_numpy(float)
            Xa = np.vstack([abl1[m], Xe]); Ya = np.concatenate([(y1[m] - mu) / sd, ze])
            aug = spearmanr(Y3[a][ok], gbm().fit(Xa, Ya).predict(abl3)[ok]).statistic
            sc_rows.append(dict(assay=a, study=study, n=len(ze), base=round(float(base), 3),
                                augmented=round(float(aug), 3), delta=round(float(aug - base), 3)))
    sc = pd.DataFrame(sc_rows); sc.to_csv(ROOT / "results/exp_scaling.csv", index=False)
    print("\n=== (13) Per-study augmentation contribution to held-out ===")
    print(sc.to_string(index=False))

    # ---- (9) uncertainty / calibration (base-learner ensemble variance) ----
    from abdev.models.fusion import BASE_BLOCK
    ds2, g3b, B1, B3 = GE.build_blocks()
    B1 = {k: np.nan_to_num(v) for k, v in B1.items()}; B3 = {k: np.nan_to_num(v) for k, v in B3.items()}
    cal_rows = []; all_err = []; all_std = []
    for a in ASSAYS:
        y1 = ds.y(a); m = ~np.isnan(y1); ok = ~np.isnan(Y3[a])
        model = AbDevFusion().fit({k: v[m] for k, v in B1.items()}, y1[m])
        base_preds = np.column_stack([fit.predict(B3[BASE_BLOCK[name]])
                                      for (name, _), fit in zip(model.bases, model.fitted_)])
        pred = model.predict(B3); std = base_preds.std(1)
        err = np.abs(pred - Y3[a])
        rho = spearmanr(std[ok], err[ok]).statistic
        cal_rows.append(dict(assay=a, calib_rho_std_vs_error=round(float(rho), 3)))
        all_err.append((err[ok] - np.nanmean(err[ok])) / (np.nanstd(err[ok]) + 1e-9))
        all_std.append((std[ok] - np.nanmean(std[ok])) / (np.nanstd(std[ok]) + 1e-9))
    cal = pd.DataFrame(cal_rows); cal.to_csv(ROOT / "results/exp_calibration.csv", index=False)
    print("\n=== (9) Calibration: Spearman(ensemble std, |error|) on held-out ===")
    print(cal.to_string(index=False))
    ae = np.concatenate(all_err); as_ = np.concatenate(all_std)
    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.scatter(as_, ae, s=14, alpha=.5, color=plt.cm.viridis(0.4))
    ax.set_xlabel("ensemble std (z)"); ax.set_ylabel("|prediction error| (z)")
    ax.set_title(f"Held-out calibration (pooled ρ={spearmanr(as_, ae).statistic:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "figS5_calibration.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
