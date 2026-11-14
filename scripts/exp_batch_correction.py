"""Exp 5: External augmentation (FLAb) batch correction strategies.

Compares approaches for combining GDPa1 (n=246, assay-specific) with external FLAb data.

Strategies:
  1. no_augment       : Train only on GDPa1
  2. current_z        : GDPa1 z-scored + FLAb z-scored, equal weight, PCA on combined
  3. study_fixed_eff  : Add one-hot study dummies to FLAb; PCA on combined
  4. inv_size_weight  : Real sample weights (Ridge sample_weight); FLAb weighted by 1/sqrt(n_study)
  5. gdpa1_pca_only   : PCA fitted on GDPa1 only, FLAb projected into that basis (OOD-safe)

NOTE on why naive strategies give identical Spearman:
  Spearman is rank-invariant. Strategies that differ only by a monotone linear label
  rescaling (e.g. scale_to_gdpa1 vs current_z) produce identically ranked predictions
  from PCA+RidgeCV. Only strategies that change the feature matrix, the sample weights,
  or the PCA basis produce genuinely different results.

Reports: per-fold Spearman (mean +- std), GDPa3 heldout (point + bootstrap 95% CI).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE

ASSAYS = GE.ASSAYS
ALPHAS = np.logspace(-2, 4, 16)
N_BOOT = 1000
STRATEGIES = ["no_augment", "current_z", "study_fixed_eff", "inv_size_weight", "gdpa1_pca_only"]


def _pca_head(X_fit):
    """Fit StandardScaler + PCA(90% var) on X_fit, return (sc, pca)."""
    sc = StandardScaler().fit(X_fit)
    pca = PCA(0.9, random_state=0, svd_solver="full").fit(sc.transform(X_fit))
    return sc, pca


def _weighted_ridge(X_pca, y, w=None):
    """Fit Ridge with sample weights via cross-validated alpha selection."""
    if w is None:
        return RidgeCV(alphas=ALPHAS).fit(X_pca, y)
    # Manual CV for alpha when using sample_weight
    best_alpha, best_score = ALPHAS[0], -np.inf
    n = len(y)
    kf_idx = np.arange(n)
    np.random.seed(0)
    cv_idx = np.array_split(np.random.permutation(n), 5)
    for alpha in ALPHAS:
        fold_scores = []
        for vi in cv_idx:
            ti = np.setdiff1d(kf_idx, vi)
            if len(vi) < 2:
                continue
            r = Ridge(alpha=alpha).fit(X_pca[ti], y[ti],
                                       sample_weight=w[ti] if w is not None else None)
            p = r.predict(X_pca[vi])
            if np.std(y[vi]) > 1e-8 and np.std(p) > 1e-8:
                fold_scores.append(spearmanr(y[vi], p).statistic)
        sc = np.nanmean(fold_scores) if fold_scores else -np.inf
        if sc > best_score:
            best_score = sc; best_alpha = alpha
    return Ridge(alpha=best_alpha).fit(X_pca, y,
                                       sample_weight=w if w is not None else None)


def load_flab(assay, emb_key_to_idx, flab_emb):
    p = ROOT / f"data/processed/flab_{assay}.csv"
    if not p.exists():
        return np.zeros((0, flab_emb.shape[1])), np.zeros(0), np.array([])
    df = pd.read_csv(p)
    y_z = df["z"].to_numpy(float)
    study = df["study"].to_numpy()
    idxs = np.array([emb_key_to_idx.get(k, -1) for k in df["key"].tolist()])
    good = idxs >= 0
    return flab_emb[idxs[good]], y_z[good], study[good]


def bootstrap_rho(y_true, y_pred, n_boot=N_BOOT, rng=None):
    """Bootstrap 95% CI for Spearman correlation (paired resampling)."""
    if rng is None:
        rng = np.random.RandomState(42)
    ok = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(yt), len(yt), replace=True)   # same index for both
        boots.append(spearmanr(yt[idx], yp[idx]).statistic)
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def fit_predict_strategy(strategy, X_tr, y_tr, X_flab, y_flab, study_flab, X_te):
    """Fit model under given strategy and return predictions on X_te.

    X_te can be either GDPa1 validation fold or GDPa3 heldout (480-dim ablang2).
    Returns predictions in original GDPa1 label scale.
    """
    n_flab = len(X_flab)

    if strategy == "no_augment" or n_flab == 0:
        sc, pca = _pca_head(X_tr)
        X_tr_p = pca.transform(sc.transform(X_tr))
        X_te_p = pca.transform(sc.transform(X_te))
        ridge = _weighted_ridge(X_tr_p, y_tr)
        return ridge.predict(X_te_p)

    mu, sg = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_z = (y_tr - mu) / sg

    if strategy == "current_z":
        X_all = np.concatenate([X_tr, X_flab], 0)
        y_all = np.concatenate([y_tr_z, y_flab], 0)
        sc, pca = _pca_head(X_all)
        X_all_p = pca.transform(sc.transform(X_all))
        X_te_p = pca.transform(sc.transform(X_te))
        ridge = _weighted_ridge(X_all_p, y_all)
        return ridge.predict(X_te_p) * sg + mu

    if strategy == "study_fixed_eff":
        studies = sorted(set(study_flab))
        d_s = len(studies)
        sm = {s: i for i, s in enumerate(studies)}
        flab_oh = np.zeros((n_flab, d_s))
        for i, s in enumerate(study_flab):
            flab_oh[i, sm[s]] = 1.0
        X_tr_ext = np.concatenate([X_tr, np.zeros((len(X_tr), d_s))], 1)
        X_flab_ext = np.concatenate([X_flab, flab_oh], 1)
        X_all = np.concatenate([X_tr_ext, X_flab_ext], 0)
        y_all = np.concatenate([y_tr_z, y_flab], 0)
        sc, pca = _pca_head(X_all)
        X_all_p = pca.transform(sc.transform(X_all))
        # Test features: GDPa1 gets zero study dummies
        X_te_ext = np.concatenate([X_te, np.zeros((len(X_te), d_s))], 1)
        X_te_p = pca.transform(sc.transform(X_te_ext))
        ridge = _weighted_ridge(X_all_p, y_all)
        return ridge.predict(X_te_p) * sg + mu

    if strategy == "inv_size_weight":
        X_all = np.concatenate([X_tr, X_flab], 0)
        y_all = np.concatenate([y_tr_z, y_flab], 0)
        # Actual sample weights: FLAb weighted by 1/sqrt(study_size), normalized
        studies_u, counts = np.unique(study_flab, return_counts=True)
        sw = {s: 1.0 / np.sqrt(c) for s, c in zip(studies_u, counts)}
        flab_w = np.array([sw[s] for s in study_flab])
        flab_w = flab_w * (len(X_tr) / flab_w.sum())   # normalize total FLAb weight = n_gdpa1
        w_all = np.concatenate([np.ones(len(X_tr)), flab_w], 0)
        sc, pca = _pca_head(X_all)
        X_all_p = pca.transform(sc.transform(X_all))
        X_te_p = pca.transform(sc.transform(X_te))
        ridge = _weighted_ridge(X_all_p, y_all, w=w_all)
        return ridge.predict(X_te_p) * sg + mu

    if strategy == "gdpa1_pca_only":
        # PCA fitted on GDPa1 training data only -> FLAb projected into GDPa1 basis
        sc, pca = _pca_head(X_tr)
        X_tr_p = pca.transform(sc.transform(X_tr))
        X_flab_p = pca.transform(sc.transform(X_flab))
        X_te_p = pca.transform(sc.transform(X_te))
        X_all_p = np.concatenate([X_tr_p, X_flab_p], 0)
        y_all = np.concatenate([y_tr_z, y_flab], 0)
        ridge = _weighted_ridge(X_all_p, y_all)
        return ridge.predict(X_te_p) * sg + mu

    raise ValueError(f"Unknown strategy: {strategy}")


def eval_cv_strategy(X_gdpa1, y_gdpa1, X_flab, y_flab, study_flab, ds, strategy):
    """Return (mean_rho, std_rho) across cluster_isotype folds."""
    folds = ds.folds("cluster_isotype")
    ufolds = sorted(set(f for f in folds.tolist() if f == f))
    lab = ~np.isnan(y_gdpa1)
    scores = []
    for f in ufolds:
        tr = np.where((folds != f) & lab)[0]
        te = np.where((folds == f) & lab)[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        pred = fit_predict_strategy(strategy, X_gdpa1[tr], y_gdpa1[tr],
                                    X_flab, y_flab, study_flab, X_gdpa1[te])
        scores.append(spearmanr(y_gdpa1[te], pred).statistic)
    return (float(np.nanmean(scores)), float(np.nanstd(scores))) if scores else (np.nan, np.nan)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading data ...")
    ds, g3, B1, B3 = GE.build_blocks()

    flab_data = np.load(ROOT / "data/processed/flab_ablang2.npz", allow_pickle=True)
    flab_emb = flab_data["emb"]
    key_to_idx = {k: i for i, k in enumerate(flab_data["keys"])}

    X_gdpa1 = np.nan_to_num(B1["ablang2"])
    X_gdpa3 = np.nan_to_num(B3["ablang2"])

    rows = []
    for assay in ASSAYS:
        y1 = ds.y(assay)
        lab = ~np.isnan(y1)
        y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        ok3 = ~np.isnan(y3)

        X_flab, y_flab, study_flab = load_flab(assay, key_to_idx, flab_emb)
        print(f"  {assay}: n_flab={len(y_flab)}, "
              f"studies={list(set(study_flab)) if len(study_flab) else []}")

        for strategy in STRATEGIES:
            cv_mean, cv_std = eval_cv_strategy(X_gdpa1, y1, X_flab, y_flab,
                                               study_flab, ds, strategy)
            pred3 = fit_predict_strategy(strategy, X_gdpa1[lab], y1[lab],
                                         X_flab, y_flab, study_flab, X_gdpa3)
            rho3 = float(spearmanr(y3[ok3], pred3[ok3]).statistic) if ok3.sum() >= 3 else np.nan
            ci_lo, ci_hi = bootstrap_rho(y3, pred3) if ok3.sum() >= 3 else (np.nan, np.nan)
            rows.append({
                "strategy": strategy, "assay": assay,
                "n_flab": len(y_flab),
                "cv_mean": round(cv_mean, 3),
                "cv_std": round(cv_std, 3),
                "gdpa3": round(rho3, 3),
                "gdpa3_ci95_lo": round(ci_lo, 3),
                "gdpa3_ci95_hi": round(ci_hi, 3),
            })

    df = pd.DataFrame(rows)

    print("\n=== Exp 5: Batch correction - CV mean +- std (cluster_isotype) ===")
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy]
        parts = [f"{row['assay']}: {row['cv_mean']:.3f}+-{row['cv_std']:.3f}"
                 for _, row in sub.iterrows()]
        mean_cv = sub["cv_mean"].mean()
        print(f"  {strategy:20s}  MEAN={mean_cv:.3f}  |  {', '.join(parts)}")

    print("\n=== Exp 5: Batch correction - GDPa3 heldout (rho [95% CI]) ===")
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy]
        parts = [f"{row['assay']}: {row['gdpa3']:.3f}[{row['gdpa3_ci95_lo']:.3f},{row['gdpa3_ci95_hi']:.3f}]"
                 for _, row in sub.iterrows()]
        mean3 = sub["gdpa3"].mean()
        print(f"  {strategy:20s}  MEAN={mean3:.3f}  |  {', '.join(parts)}")

    df.to_csv(ROOT / "results/exp_batch_correction.csv", index=False)
    print("\nSaved results/exp_batch_correction.csv")


if __name__ == "__main__":
    main()
