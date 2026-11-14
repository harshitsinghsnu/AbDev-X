"""Exp 6: Seed and fold-count robustness of block selection.

Repeats per-assay block selection (best single block by OOD CV score) over:
  - 10 random seeds for cluster assignment
  - 3, 5, 10 fold counts

Uses cluster_isotype-style partitioning with varied seeds (KMeans re-cluster on ESM2 embeddings).

Reports:
  - Block selection frequency (how often each block wins per assay)
  - Kendall's W (concordance) of block rankings across seeds
  - GDPa3 heldout Spearman of the consensus-selected block vs seed-varying selections
  - Stability index: fraction of (seed, assay) pairs where the same block wins as the modal choice
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
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE

ASSAYS = GE.ASSAYS
BLOCKS = ["esm2", "descriptors", "isotype", "bio", "ablang2"]
N_SEEDS = 10
FOLD_COUNTS = [3, 5, 10]


def _head(blk="esm2"):
    steps = [("s", StandardScaler())]
    if blk in ("esm2", "ablang2"):
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def make_cluster_isotype_folds(ds, seed: int, n_folds: int, X_cluster: np.ndarray):
    """Re-cluster antibodies with given seed and assign fold labels.

    Uses KMeans on ESM2 embeddings stratified by isotype (approximate cluster_isotype).
    Returns fold array of length n, with values 0..n_folds-1.
    """
    n = len(X_cluster)
    # Cluster with fresh seed
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(X_cluster)   # (n,)

    # Isotype stratification: assign each (cluster, isotype) group to a fold
    # Simple: use cluster label directly as fold id (approximately balanced)
    isotype = ds.df.get("hc_subtype", pd.Series(["IgG1"] * n, name="hc_subtype")).to_numpy()
    # Combine cluster and isotype -> assign folds round-robin
    groups = pd.DataFrame({"cluster": cluster_labels, "isotype": isotype})
    groups["group_id"] = pd.factorize(groups["cluster"].astype(str) + "_" + groups["isotype"].astype(str))[0]
    group_ids = groups["group_id"].to_numpy()
    n_groups = group_ids.max() + 1
    # Round-robin assignment of groups to folds
    group_fold = {g: g % n_folds for g in range(n_groups)}
    folds = np.array([group_fold[g] for g in group_ids])
    return folds


def eval_block_cv(X, y, folds, n_folds, blk_name):
    """Evaluate a block's CV Spearman on given fold assignment."""
    lab = ~np.isnan(y)
    ufolds = list(range(n_folds))
    scores = []
    for f in ufolds:
        tr = np.where((folds != f) & lab)[0]
        te = np.where((folds == f) & lab)[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        pipe = _head(blk=blk_name)
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        scores.append(spearmanr(y[te], pred).statistic)
    return float(np.nanmean(scores)) if scores else np.nan


def kendalls_w(rankings):
    """Compute Kendall's W (coefficient of concordance) for a matrix of rankings.

    rankings: (n_judges, n_items) - lower rank = better
    Returns W in [0, 1], 1 = perfect concordance.
    """
    m, n = rankings.shape
    if m < 2 or n < 2:
        return np.nan
    R = rankings.sum(0)   # sum of ranks per item
    S = np.sum((R - R.mean()) ** 2)
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    return float(W)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading data ...")
    ds, g3, B1, B3 = GE.build_blocks()

    # Use ESM2 embeddings for clustering (dimensionality reduced)
    X_esm = np.nan_to_num(B1["esm2"])
    pca_clust = PCA(20, random_state=0, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(X_esm)
    )  # (246, 20)

    # GDPa3 held-out predictions per block (trained on full GDPa1)
    held_preds = {}
    for blk in BLOCKS:
        X3 = np.nan_to_num(B3[blk])
        held_preds[blk] = {}
        for assay in ASSAYS:
            y1 = ds.y(assay)
            lab = ~np.isnan(y1)
            pipe = _head(blk=blk).fit(np.nan_to_num(B1[blk])[lab], y1[lab])
            held_preds[blk][assay] = pipe.predict(X3)

    rows = []
    # selection_matrix[assay][fold_count] = list of (seed, winner_block)
    selections = {a: {nf: [] for nf in FOLD_COUNTS} for a in ASSAYS}

    for n_folds in FOLD_COUNTS:
        print(f"\n  n_folds={n_folds}")
        for seed in range(N_SEEDS):
            folds = make_cluster_isotype_folds(ds, seed=seed, n_folds=n_folds, X_cluster=pca_clust)
            for assay in ASSAYS:
                y1 = ds.y(assay)
                block_scores = {}
                for blk in BLOCKS:
                    X1 = np.nan_to_num(B1[blk])
                    cv = eval_block_cv(X1, y1, folds, n_folds, blk_name=blk)
                    block_scores[blk] = cv
                winner = max(block_scores, key=block_scores.get)
                selections[assay][n_folds].append(winner)
                # GDPa3 Spearman with winning block
                y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
                ok = ~np.isnan(y3)
                rho3 = float(spearmanr(y3[ok], held_preds[winner][assay][ok]).statistic) if ok.sum() >= 3 else np.nan
                rows.append({
                    "n_folds": n_folds,
                    "seed": seed,
                    "assay": assay,
                    "winner_block": winner,
                    "cv_winner_score": round(block_scores[winner], 3),
                    "gdpa3_winner": round(rho3, 3),
                    **{f"cv_{b}": round(block_scores[b], 3) for b in BLOCKS},
                })

    df = pd.DataFrame(rows)

    # --- Block selection frequency ---
    print("\n=== Exp 6: Block selection frequency (n_folds=5) ===")
    df5 = df[df["n_folds"] == 5]
    freq = df5.groupby(["assay", "winner_block"]).size().unstack(fill_value=0)
    freq["total"] = freq.sum(axis=1)
    print(freq.to_string())

    # --- Stability: modal winner per assay × fold_count ---
    print("\n=== Stability: modal winner per assay per fold-count ===")
    stab_rows = []
    for assay in ASSAYS:
        for nf in FOLD_COUNTS:
            wins = selections[assay][nf]
            modal = max(set(wins), key=wins.count)
            frac = wins.count(modal) / len(wins)
            stab_rows.append({"assay": assay, "n_folds": nf, "modal_block": modal, "modal_frac": round(frac, 3)})
    df_stab = pd.DataFrame(stab_rows)
    print(df_stab.pivot_table(index="assay", columns="n_folds", values="modal_frac").round(3).to_string())

    # --- Kendall's W of block rankings per assay ---
    print("\n=== Kendall's W (concordance of block rankings across seeds, n_folds=5) ===")
    for assay in ASSAYS:
        df5a = df5[df5["assay"] == assay]
        cv_cols = [f"cv_{b}" for b in BLOCKS]
        ranking_mat = df5a[cv_cols].rank(axis=1, ascending=False).to_numpy()   # (n_seeds, n_blocks)
        W = kendalls_w(ranking_mat)
        print(f"  {assay}: W={W:.3f}")

    # --- GDPa3 transfer: modal vs any-seed selection ---
    print("\n=== GDPa3: modal-block (n_folds=5) vs mean across all seeds ===")
    for assay in ASSAYS:
        modal_b = max(set(selections[assay][5]), key=selections[assay][5].count)
        y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        ok = ~np.isnan(y3)
        rho_modal = float(spearmanr(y3[ok], held_preds[modal_b][assay][ok]).statistic)
        rho_all = [float(spearmanr(y3[ok], held_preds[b][assay][ok]).statistic) for b in BLOCKS]
        print(f"  {assay}: modal({modal_b})={rho_modal:.3f}, mean_all_blocks={np.mean(rho_all):.3f}")

    df.to_csv(ROOT / "results/exp_seed_fold_robustness.csv", index=False)
    df_stab.to_csv(ROOT / "results/exp_seed_fold_stability.csv", index=False)
    print("\nSaved results/exp_seed_fold_robustness.csv and exp_seed_fold_stability.csv")


if __name__ == "__main__":
    main()
