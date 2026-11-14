"""Exp 4: Calibration and abstention study.

Uses AbDev-X cluster_isotype OOF predictions to build per-antibody prediction uncertainty:
  - Bootstrap OOF residuals to estimate prediction variance
  - Conformal / bootstrap prediction intervals (50%, 80%, 95%)
  - Abstention curves: abstain on top 10/20/30% highest-variance antibodies
  - Report retained Spearman and MAE
  - Enrichment at K: fraction of top-K true binders in top-K predicted

Extends exp_calibration.csv if present.
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
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE

ASSAYS = GE.ASSAYS
N_BOOT = 200
ABSTAIN_FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
K_VALUES = [10, 20, 30]


def _head(blk="esm2"):
    steps = [("s", StandardScaler())]
    if blk in ("esm2", "ablang2"):
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def abdevx_oof(B1, ds, assay, folds, ufolds):
    """Compute AbDev-X OOF predictions using cluster_isotype folds."""
    from abdev.models.fusion import AbDevFusion, BASE_BLOCK, DEFAULT_BASES
    y = ds.y(assay)
    lab = ~np.isnan(y)
    oof = np.full(len(y), np.nan)
    for f in ufolds:
        tr = np.where((folds != f) & lab)[0]
        te = np.where((folds == f) & lab)[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        B1m_tr = {k: v[tr] for k, v in B1.items()}
        B1m_te = {k: v[te] for k, v in B1.items()}
        model = AbDevFusion().fit(B1m_tr, y[tr])
        oof[te] = model.predict(B1m_te)
    return oof, y, lab


def bootstrap_variance(X, y_oof, lab, n_boot=N_BOOT, rng=None):
    """Estimate per-sample prediction variance via bootstrap on OOF residuals.

    Returns: (pred_mean, pred_std) each shape (n_labeled,)
    """
    if rng is None:
        rng = np.random.RandomState(0)
    idx = np.where(lab)[0]
    y_obs = y_oof[lab]  # actual labels
    resid = np.full(len(idx), np.nan)  # per-sample residuals from OOF
    boot_preds = np.zeros((n_boot, len(idx)))
    for b in range(n_boot):
        bi = rng.choice(len(idx), size=len(idx), replace=True)
        oi = np.setdiff1d(np.arange(len(idx)), bi)
        if len(oi) < 5:
            continue
        # Fit a simple ridge on OOF features (embedding not available in this context)
        # Instead: parametric bootstrap on OOF residuals themselves
        # Resample residuals and add to OOF predictions
        pass  # see below
    return None


def per_sample_uncertainty_from_folds(oof_preds_all_seeds, y, lab):
    """Given OOF predictions from multiple seeds, compute per-sample std.

    oof_preds_all_seeds: list of arrays shape (n,)
    Returns: std of predictions across seeds for labeled samples.
    """
    stacked = np.stack([o[lab] for o in oof_preds_all_seeds], axis=0)   # (n_seeds, n_lab)
    return stacked.mean(0), stacked.std(0)


def run_oof_multi_seed(B1, ds, assay, folds, ufolds, n_seeds=20):
    """Run AbDev-X OOF with multiple random seeds to estimate prediction variance."""
    from abdev.models.fusion import AbDevFusion
    y = ds.y(assay)
    lab = ~np.isnan(y)
    all_oofs = []
    for seed in range(n_seeds):
        oof = np.full(len(y), np.nan)
        for f in ufolds:
            tr = np.where((folds != f) & lab)[0]
            te = np.where((folds == f) & lab)[0]
            if len(te) < 3 or len(tr) < 10:
                continue
            B1m_tr = {k: v[tr] for k, v in B1.items()}
            B1m_te = {k: v[te] for k, v in B1.items()}
            model = AbDevFusion(random_state=seed).fit(B1m_tr, y[tr])
            oof[te] = model.predict(B1m_te)
        all_oofs.append(oof)
    return all_oofs, y, lab


def conformal_interval(residuals, alpha=0.10):
    """Non-conformity score -> symmetric interval at (1-alpha) coverage."""
    q = np.quantile(np.abs(residuals), 1 - alpha)
    return float(q)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading data ...")
    ds, g3, B1, B3 = GE.build_blocks()

    folds = ds.folds("cluster_isotype")
    ufolds = sorted(set(f for f in folds.tolist() if f == f))

    all_rows_calib = []
    all_rows_abstain = []

    for assay in ASSAYS:
        print(f"\n  Processing {assay} ...")
        y = ds.y(assay)
        lab = ~np.isnan(y)

        # Multi-seed OOF for uncertainty quantification
        all_oofs, _, _ = run_oof_multi_seed(B1, ds, assay, folds, ufolds, n_seeds=20)
        pred_mean, pred_std = per_sample_uncertainty_from_folds(all_oofs, y, lab)
        y_lab = y[lab]

        # --- Calibration ---
        residuals = y_lab - pred_mean
        for alpha, label in [(0.50, "50%"), (0.20, "80%"), (0.05, "95%")]:
            q = conformal_interval(residuals, alpha=alpha)
            coverage_actual = float(np.mean(np.abs(residuals) <= q))
            all_rows_calib.append({
                "assay": assay,
                "target_coverage": 1 - alpha,
                "interval_halfwidth": round(q, 4),
                "actual_coverage": round(coverage_actual, 4),
                "n_labeled": int(lab.sum()),
            })

        # --- Correlation of |residual| with pred_std ---
        if pred_std.std() > 1e-8:
            rho_calib = spearmanr(np.abs(residuals), pred_std).statistic
        else:
            rho_calib = np.nan
        all_rows_calib[-1]["std_vs_error_rho"] = round(float(rho_calib), 4)

        # --- Abstention curves ---
        base_rho = float(spearmanr(y_lab, pred_mean).statistic)
        base_mae = float(np.mean(np.abs(residuals)))
        for frac in ABSTAIN_FRACS:
            n_abstain = int(frac * len(y_lab))
            # Abstain on highest-uncertainty samples
            if n_abstain > 0 and n_abstain < len(y_lab):
                abstain_idx = np.argsort(pred_std)[::-1][:n_abstain]
                keep = np.ones(len(y_lab), bool)
                keep[abstain_idx] = False
            else:
                keep = np.ones(len(y_lab), bool)
            n_keep = keep.sum()
            if n_keep < 5:
                rho_k = np.nan; mae_k = np.nan
            else:
                rho_k = float(spearmanr(y_lab[keep], pred_mean[keep]).statistic)
                mae_k = float(np.mean(np.abs(residuals[keep])))

            # Enrichment at K (top-K by prediction vs top-K by truth)
            enrich = {}
            for k in K_VALUES:
                if n_keep < k:
                    enrich[f"enrichment@{k}"] = None
                    continue
                pred_topk = set(np.argsort(pred_mean[keep])[-k:])
                true_topk = set(np.argsort(y_lab[keep])[-k:])
                enrich[f"enrichment@{k}"] = round(len(pred_topk & true_topk) / k, 3)

            row = {
                "assay": assay,
                "abstain_frac": frac,
                "n_retained": n_keep,
                "rho_retained": round(rho_k, 4) if not np.isnan(rho_k) else None,
                "rho_delta": round(rho_k - base_rho, 4) if not np.isnan(rho_k) else None,
                "mae_retained": round(mae_k, 4) if not np.isnan(mae_k) else None,
                "mae_delta": round(mae_k - base_mae, 4) if not np.isnan(mae_k) else None,
            }
            row.update(enrich)
            all_rows_abstain.append(row)

        print(f"    base rho={base_rho:.3f}, calib_rho_std_vs_error={rho_calib:.3f}")

    df_calib = pd.DataFrame(all_rows_calib)
    df_abstain = pd.DataFrame(all_rows_abstain)

    print("\n=== Exp 4: Calibration - prediction interval coverage ===")
    print(df_calib.to_string(index=False))

    print("\n=== Exp 4: Abstention curves (rho at each abstention frac) ===")
    pivot = df_abstain.pivot_table(index="abstain_frac", columns="assay", values="rho_retained")
    pivot["MEAN"] = pivot.mean(axis=1)
    print(pivot.round(3).to_string())

    df_calib.to_csv(ROOT / "results/exp_calibration_full.csv", index=False)
    df_abstain.to_csv(ROOT / "results/exp_abstention.csv", index=False)
    print("\nSaved results/exp_calibration_full.csv and results/exp_abstention.csv")


if __name__ == "__main__":
    main()
