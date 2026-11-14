"""AbDev-X-MO: Reliability-Aware Multi-Objective Pareto Selection.

Stages 3-4 of the AbDev-X-MO proposal:
  Stage 3 -- Bootstrap uncertainty estimation per assay
  Stage 4 -- Constrained Pareto ranking with OOD-distance score

Stages 1-2 (representations + per-assay predictors) reuse the existing
OOD-CV block-selection logic from gdpa3_strategies.py.

Usage:
    conda run -n base python scripts/abdevx_mo.py
    conda run -n base python scripts/abdevx_mo.py --n-bootstrap 100 --kappa 0.5

No GDPa3 labels are used for block selection, kappa choice, or threshold tuning.
GDPa3 labels are only used in the final evaluation section.

Outputs (results/):
    abdevx_mo_predictions.csv   per-candidate per-assay mu, sigma
    abdevx_mo_pareto.csv        Pareto membership + OOD score, per kappa
    abdevx_mo_strategies.csv    strategy comparison table
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gdpa3_eval as GE
from abdev.pareto import (
    conservative_objectives,
    hypervolume,
    knn_ood_distance,
    multi_assay_hit_rate,
    nondominated_rank,
    pareto_front,
    standardize_ood,
    top_k_recovery,
)

ASSAYS = GE.ASSAYS  # ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
BLOCKS = ["esm2", "descriptors", "isotype", "bio", "ablang2"]

# Maximize convention: lower raw value is better for these assays -> negate
DIRECTIONS = {
    "HIC":          "lower",   # hydrophobic retention -- want low
    "Tm2":          "higher",  # melting temp -- want high
    "AC-SINS_pH7.4": "lower",  # self-association -- want low
    "PR_CHO":       "lower",   # polyreactivity -- want low
    "Titer":        "higher",  # expression -- want high
}
SIGN = {a: (-1.0 if DIRECTIONS[a] == "lower" else 1.0) for a in ASSAYS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_head(blk: str) -> Pipeline:
    steps: list = [("scaler", StandardScaler())]
    if blk in ("esm2", "ablang2"):
        steps.append(("pca", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("ridge", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Stage 2b: OOD-CV block selection (no held-out peeking)
# ---------------------------------------------------------------------------

def ood_cv_select(ds, B1: dict, folds: np.ndarray) -> dict[str, str]:
    """Return best block per assay by GDPa1 cluster-isotype OOD cross-validation."""
    selected: dict[str, str] = {}
    cv_table: dict[str, dict[str, float]] = {}
    for a in ASSAYS:
        y1 = ds.y(a)
        cv_scores: dict[str, float] = {}
        for blk in BLOCKS:
            X1 = np.nan_to_num(B1[blk])
            per_fold: list[float] = []
            for f in sorted(set(folds.tolist())):
                tr = np.where((folds != f) & ~np.isnan(y1))[0]
                te = np.where((folds == f) & ~np.isnan(y1))[0]
                if len(te) < 3 or len(tr) < 10:
                    continue
                pred = make_head(blk).fit(X1[tr], y1[tr]).predict(X1[te])
                per_fold.append(float(spearmanr(y1[te], pred).statistic))
            cv_scores[blk] = float(np.nanmean(per_fold)) if per_fold else 0.0
        selected[a] = max(cv_scores, key=cv_scores.__getitem__)
        cv_table[a] = cv_scores
    return selected, cv_table


# ---------------------------------------------------------------------------
# Stage 3: Bootstrap uncertainty
# ---------------------------------------------------------------------------

def bootstrap_predict(
    X1: np.ndarray, y1: np.ndarray, X3: np.ndarray,
    blk: str, n_bootstrap: int, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit n_bootstrap resampled models; return (mu, sigma) arrays of shape (len(X3),)."""
    rng = np.random.default_rng(seed)
    n_train = len(y1)
    preds = np.empty((n_bootstrap, len(X3)))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_train, size=n_train)
        X_b, y_b = X1[idx], y1[idx]
        try:
            preds[b] = make_head(blk).fit(X_b, y_b).predict(X3)
        except Exception:
            preds[b] = np.nan
    valid = ~np.isnan(preds).all(axis=1)
    preds = preds[valid]
    return preds.mean(axis=0), preds.std(axis=0)


# ---------------------------------------------------------------------------
# OOD distance (Stage 4 input)
# ---------------------------------------------------------------------------

def compute_ood_distance(B1: dict, B3: dict, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """PCA-reduced ESM-2 kNN distance; returns (d_self_ood, d_test_ood).

    d_self_ood: kNN(k+1) self-distances on GDPa1 (for standardisation reference)
    d_test_ood: kNN(k) distances from each GDPa3 candidate to GDPa1
    """
    X_emb = np.nan_to_num(B1["esm2"])
    X_emb3 = np.nan_to_num(B3["esm2"])
    pca = PCA(n_components=50, random_state=0, svd_solver="full").fit(X_emb)
    Z1 = pca.transform(X_emb)
    Z3 = pca.transform(X_emb3)
    # self-distances (leave-one-out approximation: use k+1, skip nearest=self)
    d_self = knn_ood_distance(Z1, Z1, k=k + 1)   # includes self -> small
    d_test = knn_ood_distance(Z1, Z3, k=k)
    return d_self, d_test


# ---------------------------------------------------------------------------
# Strategy comparison helpers
# ---------------------------------------------------------------------------

def composite_score(mu_std: np.ndarray) -> np.ndarray:
    """Equal-weight sum of standardised, sign-corrected predictions -> higher = better."""
    return mu_std.sum(axis=1)


def pareto_top_k(q_matrix: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k candidates by Pareto rank then composite score tie-break."""
    ranks = nondominated_rank(q_matrix)
    tiebreak = q_matrix.sum(axis=1)
    order = np.lexsort((-tiebreak, ranks))
    return order[:k]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=50)
    ap.add_argument("--kappa", type=float, default=0.5,
                    help="conservatism; 0=mean only, higher penalises uncertainty more")
    ap.add_argument("--k-ood", type=int, default=5, help="kNN for OOD distance")
    ap.add_argument("--top-k", type=int, default=5, help="candidates to select per strategy")
    args = ap.parse_args()

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    ds, g3, B1, B3 = GE.build_blocks()
    folds = ds.folds("cluster_isotype")
    n3 = len(g3)
    print(f"  GDPa1: {len(ds.df)}  GDPa3: {n3}")

    # ---- Stage 2b: OOD-CV block selection ----
    print("\n[Stage 2b] OOD-CV block selection ...")
    selected, cv_table = ood_cv_select(ds, B1, folds)
    print("  Selected blocks:")
    for a in ASSAYS:
        best_blk = selected[a]
        cv_scores_str = "  ".join(f"{b}={cv_table[a][b]:.3f}" for b in BLOCKS)
        print(f"    {a}: {best_blk} (CV: {cv_scores_str})")

    # ---- Stage 3: Bootstrap uncertainty ----
    print(f"\n[Stage 3] Bootstrap uncertainty (n_bootstrap={args.n_bootstrap}) ...")
    mu_raw = np.empty((n3, len(ASSAYS)))   # raw scale, per-assay
    sigma_raw = np.empty_like(mu_raw)

    for j, a in enumerate(ASSAYS):
        blk = selected[a]
        y1 = ds.y(a)
        ok = ~np.isnan(y1)
        X1 = np.nan_to_num(B1[blk])[ok]
        X3 = np.nan_to_num(B3[blk])
        mu_raw[:, j], sigma_raw[:, j] = bootstrap_predict(
            X1, y1[ok], X3, blk, n_bootstrap=args.n_bootstrap, seed=42
        )
        print(f"    {a}: mu range [{mu_raw[:, j].min():.3f}, {mu_raw[:, j].max():.3f}]"
              f"  sigma mean {sigma_raw[:, j].mean():.4f}")

    # ---- Standardise using GDPa1 training stats per assay ----
    # Fit standardisation on full-data (GDPa1) predictions to get comparable scale
    print("\n[Stage 3b] Standardising predictions to GDPa1 training distribution ...")
    mu_std = np.empty_like(mu_raw)    # standardised + sign-corrected (higher=better)
    sigma_std = np.empty_like(mu_raw)

    train_stats: dict[str, tuple[float, float]] = {}
    for j, a in enumerate(ASSAYS):
        blk = selected[a]
        y1 = ds.y(a)
        ok = ~np.isnan(y1)
        X1 = np.nan_to_num(B1[blk])
        # predict GDPa1 with full-data model (no leakage -- this is for scale only)
        pred_train = make_head(blk).fit(X1[ok], y1[ok]).predict(X1[ok])
        t_mean, t_std = float(pred_train.mean()), float(pred_train.std() + 1e-9)
        train_stats[a] = (t_mean, t_std)
        # standardise and flip sign so higher = better
        mu_std[:, j] = SIGN[a] * (mu_raw[:, j] - t_mean) / t_std
        sigma_std[:, j] = sigma_raw[:, j] / t_std   # uncertainty on same scale, always positive

    # ---- OOD distance ----
    print(f"\n[Stage 4a] Computing OOD distance (k={args.k_ood}) ...")
    d_self, d_test = compute_ood_distance(B1, B3, k=args.k_ood)
    ood_z = standardize_ood(d_self, d_test)
    print(f"  OOD z-score: min={ood_z.min():.2f}  max={ood_z.max():.2f}"
          f"  n_flagged(z>2)={int((ood_z > 2.0).sum())}")

    # ---- Conservative objectives ----
    print(f"\n[Stage 4b] Conservative objectives (kappa={args.kappa}) ...")
    q = conservative_objectives(mu_std, sigma_std, kappa=args.kappa)
    # kappa=0 baseline (mean only)
    q0 = conservative_objectives(mu_std, sigma_std, kappa=0.0)

    # ---- Pareto ranking ----
    ranks_q = nondominated_rank(q)
    ranks_q0 = nondominated_rank(q0)
    pareto_mask_q = pareto_front(q)
    pareto_mask_q0 = pareto_front(q0)
    print(f"  Pareto front size: kappa=0.0: {pareto_mask_q0.sum()}  "
          f"kappa={args.kappa}: {pareto_mask_q.sum()}")

    # ---- Save predictions ----
    pred_df = pd.DataFrame({
        "antibody_name": g3["antibody_name"].values,
        "ood_distance_z": ood_z.round(4),
        "ood_flag": (ood_z > 2.0),
    })
    for j, a in enumerate(ASSAYS):
        pred_df[f"{a}_mu_raw"] = mu_raw[:, j].round(5)
        pred_df[f"{a}_sigma_raw"] = sigma_raw[:, j].round(5)
        pred_df[f"{a}_mu_std"] = mu_std[:, j].round(4)
        pred_df[f"{a}_sigma_std"] = sigma_std[:, j].round(4)
        pred_df[f"{a}_q_kappa{args.kappa}"] = q[:, j].round(4)
    pred_df.to_csv(results_dir / "abdevx_mo_predictions.csv", index=False)
    print(f"\nSaved predictions -> results/abdevx_mo_predictions.csv")

    # ---- Save Pareto rankings ----
    pareto_df = pred_df[["antibody_name", "ood_distance_z", "ood_flag"]].copy()
    pareto_df["pareto_rank_mean"] = ranks_q0
    pareto_df[f"pareto_rank_kappa{args.kappa}"] = ranks_q
    pareto_df["composite_score_mean"] = composite_score(mu_std).round(4)
    pareto_df[f"composite_score_kappa{args.kappa}"] = composite_score(q).round(4)
    pareto_df["on_pareto_front_mean"] = pareto_mask_q0
    pareto_df[f"on_pareto_front_kappa{args.kappa}"] = pareto_mask_q
    pareto_df.sort_values("pareto_rank_mean").to_csv(
        results_dir / "abdevx_mo_pareto.csv", index=False)
    print(f"Saved Pareto ranking -> results/abdevx_mo_pareto.csv")

    # ---- Evaluation on GDPa3 labels ----
    print("\n" + "=" * 70)
    print("  EVALUATION ON GDPa3 LABELS (held-out)")
    print("=" * 70)

    # True assay values (sign-corrected to higher=better for consistency)
    true_raw = np.column_stack([
        g3[GE.LABEL_MAP[a]].to_numpy(float) for a in ASSAYS
    ])
    valid_mask = ~np.isnan(true_raw).any(axis=1)
    true_std = np.empty_like(true_raw)
    for j, a in enumerate(ASSAYS):
        t_mean, t_std = train_stats[a]
        true_std[:, j] = SIGN[a] * (true_raw[:, j] - t_mean) / t_std

    # Per-assay Spearman rho (from bootstrap mean predictions)
    print("\nPer-assay Spearman rho (bootstrap mean vs true):")
    rhos: list[float] = []
    for j, a in enumerate(ASSAYS):
        ok = ~np.isnan(true_raw[:, j])
        rho = float(spearmanr(true_raw[ok, j], mu_raw[ok, j]).statistic)
        rhos.append(rho)
        print(f"  {a:<20} rho={rho:.3f}  (selected block: {selected[a]})")
    print(f"  {'MEAN':<20} rho={np.mean(rhos):.3f}")

    # Strategy comparison
    k = args.top_k
    strategies: dict[str, np.ndarray] = {}

    # 1. Random
    rng = np.random.default_rng(0)
    strategies["Random"] = rng.choice(n3, size=k, replace=False)

    # 2. Equal-weight composite (current baseline)
    strategies["Composite_mean"] = np.argsort(-composite_score(mu_std))[:k]
    strategies[f"Composite_kappa{args.kappa}"] = np.argsort(-composite_score(q))[:k]

    # 3. Single-assay top-k (best-per-assay)
    for j, a in enumerate(ASSAYS):
        strategies[f"SingleAssay_{a}"] = np.argsort(-mu_std[:, j])[:k]

    # 4. Pareto top-k (by rank then composite tie-break)
    strategies["Pareto_mean"] = pareto_top_k(q0, k)
    strategies[f"Pareto_kappa{args.kappa}"] = pareto_top_k(q, k)

    # true_std is already sign-corrected: higher = better for all objectives
    all_higher = ["higher"] * len(ASSAYS)

    print(f"\nStrategy comparison (top-{k} selection):")
    print(f"{'Strategy':<35} {'Multi-hit%':>10} {'Top-k HIC':>10} {'Top-k Tm2':>10}"
          f" {'n_OOD_flag':>10}")
    rows = []
    for name, idx in strategies.items():
        mhr = multi_assay_hit_rate(idx, true_std, all_higher, top_frac=0.2)
        # per-assay top-k recovery: how many of strategy's top-k match true top-k for that assay
        hic_rec = top_k_recovery(mu_std[:, 0], true_std[:, 0], k=k, direction="higher")
        tm2_rec = top_k_recovery(mu_std[:, 1], true_std[:, 1], k=k, direction="higher")
        n_ood = int(ood_z[idx].max() > 2.0)
        rows.append(dict(strategy=name, multi_hit_rate=round(mhr, 3),
                         hic_topk=round(hic_rec, 2), tm2_topk=round(tm2_rec, 2),
                         n_ood_flagged=int((ood_z[idx] > 2.0).sum())))
        print(f"  {name:<35} {mhr:>10.3f} {hic_rec:>10.2f} {tm2_rec:>10.2f}"
              f" {int((ood_z[idx] > 2.0).sum()):>10}")

    strat_df = pd.DataFrame(rows)
    strat_df.to_csv(results_dir / "abdevx_mo_strategies.csv", index=False)
    print(f"\nSaved strategy comparison -> results/abdevx_mo_strategies.csv")

    # Hypervolume of each strategy's selected candidates (valid rows only)
    print(f"\nHypervolume (selected top-{k} on complete-measurement antibodies):")
    valid_true = true_std[valid_mask]
    ref_point = valid_true.min(axis=0) - 0.5   # nadir minus margin
    for name, idx in strategies.items():
        # restrict to selected candidates with complete measurements
        valid_idx = idx[valid_mask[idx]]
        if len(valid_idx) == 0:
            print(f"  {name:<35} HV=N/A (no complete-measurement candidates)")
            continue
        hv = hypervolume(true_std[valid_idx], reference=ref_point)
        print(f"  {name:<35} HV={hv:.4f}  (n_valid={len(valid_idx)})")

    # Composite rho (current paper metric: Spearman of composite score vs composite true)
    print("\nComposite ranking Spearman rho (composite_score vs true_composite_score):")
    true_comp = true_std.sum(axis=1)
    for tag, sc in [("mean_composite", composite_score(mu_std)),
                    (f"conservative_kappa{args.kappa}", composite_score(q))]:
        ok = ~np.isnan(true_comp)
        rho = float(spearmanr(true_comp[ok], sc[ok]).statistic)
        print(f"  {tag:<40} rho={rho:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
