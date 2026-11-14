"""Error bars for the paper.

(A) GDPa1 cross-validation: per-assay Spearman per fold -> mean +/- std across 5 folds
    (cluster_isotype), for AbDev-X and recent baselines.
(B) GDPa3 held-out: bootstrap (1000x resample of the 80 antibodies) -> mean + 95% CI for each
    assay's best specialist, compared to the competition champion (esp. the marginal Titer win).
Saves CSVs and matplotlib figures.
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
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.descriptors import build_descriptor_matrix

ASSAYS = GE.ASSAYS
COMP = GE.COMP_HELDOUT
SAP = ["sap_total", "sap_positive", "hydrophobic_sasa", "aromatic_sasa", "total_sasa", "hydrophobic_frac_sasa"]


def rpca(): return Pipeline([("s", StandardScaler()), ("p", PCA(0.9, random_state=0, svd_solver="full")), ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])
def rid(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])
def lgbm(seed=0): return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, subsample=.8, colsample_bytree=.6, reg_lambda=3, min_child_samples=10, subsample_freq=1, verbosity=-1, random_state=seed)


def sap_mat(csv, names):
    d = pd.read_csv(csv); b = pd.DataFrame({"antibody_name": list(names)}).merge(d, on="antibody_name", how="left")
    return np.nan_to_num(b[SAP].to_numpy(float))


# ---------------- (A) CV error bars ----------------
def cv_error_bars(ds):
    names = ds.df[ds.id_col]
    DESC = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
    CAT = np.nan_to_num(pd.get_dummies(ds.df[["hc_subtype", "lc_subtype"]].astype(str)).to_numpy(float))
    BIO = np.nan_to_num(GE.bio_block(GE.BENCH / "GDPa1", names)[0])
    ESM = np.nan_to_num(next(np.load(p)["emb"] for p in (ROOT / "data/processed").glob("esm2_*.npz") if np.load(p)["emb"].shape == (len(ds.df), 2560)))
    ABL = np.nan_to_num(np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"])
    bases = [(ESM, rpca), (DESC, rid), (CAT, rid), (BIO, rid), (BIO, lambda: lgbm(0)), (ABL, rid)]

    rows = []
    for a in ASSAYS:
        y = ds.y(a)
        # AbDev-X per-fold
        per_x = []
        for _, tr, te in fold_iter(ds, a, "cluster_isotype"):
            oof = [np.full(len(tr), np.nan) for _ in bases]; kf = KFold(5, shuffle=True, random_state=0)
            for itr, iva in kf.split(np.arange(len(tr))):
                for bi, (X, mk) in enumerate(bases):
                    oof[bi][iva] = mk().fit(X[tr[itr]], y[tr[itr]]).predict(X[tr[iva]])
            meta = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(np.column_stack(oof), y[tr])
            fit = [mk().fit(X[tr], y[tr]) for X, mk in bases]
            pred = meta.predict(np.column_stack([f.predict(X[te]) for f, (X, mk) in zip(fit, bases)]))
            per_x.append(spearmanr(y[te], pred).statistic)
        # ESM2-ridge baseline per-fold
        per_b = []
        for _, tr, te in fold_iter(ds, a, "cluster_isotype"):
            per_b.append(spearmanr(y[te], rpca().fit(ESM[tr], y[tr]).predict(ESM[te])).statistic)
        rows.append(dict(assay=a, model="AbDev-X", mean=np.mean(per_x), std=np.std(per_x)))
        rows.append(dict(assay=a, model="ESM2-Ridge", mean=np.mean(per_b), std=np.std(per_b)))
    return pd.DataFrame(rows)


# ---------------- (B) Held-out bootstrap ----------------
def heldout_predictions(ds, g3, B1, B3):
    """Best per-assay held-out predictions (train GDPa1, predict GDPa3)."""
    preds = {}
    # HIC: structural SAP
    sap1 = sap_mat(ROOT / "data/processed/sap_gdpa1.csv", ds.df[ds.id_col]); sap3 = sap_mat(ROOT / "data/processed/sap_heldout.csv", g3["antibody_name"])
    y = ds.y("HIC"); m = ~np.isnan(y); preds["HIC"] = rid().fit(sap1[m], y[m]).predict(sap3)
    # Tm2: isotype (light-chain subtype survives on all-IgG1)
    y = ds.y("Tm2"); m = ~np.isnan(y); preds["Tm2"] = rid().fit(np.nan_to_num(B1["isotype"])[m], y[m]).predict(np.nan_to_num(B3["isotype"]))
    # AC-SINS/PR/Titer: AbLang2 + FLAb2 augmentation
    zc = np.load(ROOT / "data/processed/flab_ablang2.npz", allow_pickle=True)
    ext_emb = dict(zip(zc["keys"], zc["emb"]))
    abl1, abl3 = np.nan_to_num(B1["ablang2"]), np.nan_to_num(B3["ablang2"])
    for a, use_gbm in [("AC-SINS_pH7.4", False), ("PR_CHO", False), ("Titer", True)]:
        y = ds.y(a); m = ~np.isnan(y); mu, sd = np.nanmean(y[m]), np.nanstd(y[m]) + 1e-9
        X1, Y1 = abl1[m], (y[m] - mu) / sd
        fp = ROOT / f"data/processed/flab_{a.replace('/', '_')}.csv"
        ext = pd.read_csv(fp)
        Xe = np.vstack([ext_emb[k] for k in ext["key"] if k in ext_emb])
        ze = ext[[k in ext_emb for k in ext["key"]]]["z"].to_numpy(float)
        Xa, Ya = np.vstack([X1, Xe]), np.concatenate([Y1, ze])
        if use_gbm:  # Titer: aug+base GBM ensemble over seeds
            ps = []
            for s in range(15):
                ps.append(lgbm(s).fit(Xa, Ya).predict(abl3) + lgbm(s).fit(X1, Y1).predict(abl3))
            preds[a] = np.mean(ps, 0)
        else:
            preds[a] = rpca().fit(Xa, Ya).predict(abl3)
    return preds


def bootstrap_ci(y, pred, n=1000, seed=0):
    rng = np.random.RandomState(seed); ok = ~np.isnan(y)
    y, pred = y[ok], pred[ok]; idx = np.arange(len(y)); vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 3:
            continue
        vals.append(spearmanr(y[b], pred[b]).statistic)
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ds, g3, B1, B3 = GE.build_blocks()

    cv = cv_error_bars(ds)
    cv.to_csv(ROOT / "results/cv_errorbars.csv", index=False)
    print("=== CV (cluster_isotype) mean +/- std across folds ===")
    print(cv.round(3).to_string(index=False))

    preds = heldout_predictions(ds, g3, B1, B3)
    rows = []
    for a in ASSAYS:
        y3 = g3[GE.LABEL_MAP[a]].to_numpy(float)
        mean, lo, hi = bootstrap_ci(y3, preds[a])
        rows.append(dict(assay=a, heldout_mean=round(mean, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
                         champion=COMP[a], champion_in_CI=bool(lo <= COMP[a] <= hi), beat_point=bool(mean > COMP[a])))
    hb = pd.DataFrame(rows)
    hb.to_csv(ROOT / "results/heldout_errorbars.csv", index=False)
    print("\n=== GDPa3 held-out: bootstrap mean [95% CI] vs champion ===")
    print(hb.to_string(index=False))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ASSAYS)); w = 0.35
    for i, mdl in enumerate(["AbDev-X", "ESM2-Ridge"]):
        sub = cv[cv.model == mdl].set_index("assay").loc[ASSAYS]
        ax.bar(x + (i - 0.5) * w, sub["mean"], w, yerr=sub["std"], capsize=3, label=mdl)
    ax.set_xticks(x); ax.set_xticklabels(ASSAYS, rotation=20, ha="right"); ax.set_ylabel("Spearman rho (CV)")
    ax.set_title("GDPa1 cross-validation (cluster-isotype): mean +/- std across folds"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "results/fig_cv_errorbars.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    err = np.array([[hb.heldout_mean[i] - hb.ci_low[i], hb.ci_high[i] - hb.heldout_mean[i]] for i in range(len(hb))]).T
    ax.bar(x, hb["heldout_mean"], 0.5, yerr=err, capsize=4, color="#4C72B0", label="AbDev-X (held-out)")
    ax.scatter(x, hb["champion"], color="red", zorder=5, marker="D", label="Competition champion")
    ax.set_xticks(x); ax.set_xticklabels(ASSAYS, rotation=20, ha="right"); ax.set_ylabel("Spearman rho (GDPa3 held-out)")
    ax.set_title("GDPa3 held-out: AbDev-X bootstrap mean [95% CI] vs champion"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "results/fig_heldout_errorbars.png", dpi=140); plt.close(fig)
    print("\nFigures: results/fig_cv_errorbars.png, results/fig_heldout_errorbars.png")


if __name__ == "__main__":
    main()
