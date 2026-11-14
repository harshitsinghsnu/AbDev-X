"""Pareto dominance utilities for multi-objective candidate selection.

All functions assume a maximization convention: higher q_j is better for every objective.
Convert lower-is-better assays (HIC, AC-SINS, PR-CHO) to negative before calling.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Dominance and fronts
# ---------------------------------------------------------------------------

def is_dominated(objectives: np.ndarray) -> np.ndarray:
    """Return boolean mask: True iff row i is dominated by at least one other row.

    Candidate a dominates b if a >= b on all objectives and a > b on at least one.
    O(n^2 * m) — fine for n <= 2000.
    """
    n = len(objectives)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        diff = objectives - objectives[i]         # (n, m)
        all_ge = np.all(diff >= 0.0, axis=1)
        any_gt = np.any(diff > 0.0, axis=1)
        all_ge[i] = False                         # exclude self
        if (all_ge & any_gt).any():
            dominated[i] = True
    return dominated


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated (Pareto-optimal) rows."""
    return ~is_dominated(objectives)


def nondominated_rank(objectives: np.ndarray) -> np.ndarray:
    """Return integer Pareto rank per row (rank 1 = Pareto front, rank 2 = next layer, …)."""
    n = len(objectives)
    ranks = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    rank = 1
    while remaining.any():
        idx = np.where(remaining)[0]
        sub = objectives[idx]
        front_mask = pareto_front(sub)
        ranks[idx[front_mask]] = rank
        remaining[idx[front_mask]] = False
        rank += 1
    return ranks


# ---------------------------------------------------------------------------
# Conservative objectives
# ---------------------------------------------------------------------------

def conservative_objectives(mu: np.ndarray, sigma: np.ndarray, kappa: float) -> np.ndarray:
    """q = mu - kappa * sigma (all higher-is-better).

    Parameters
    ----------
    mu : (n, m)  predictive mean per candidate per objective (standardized, higher = better)
    sigma : (n, m)  predictive std per candidate per objective (>= 0)
    kappa : float  conservatism; 0 = use mean only
    """
    return mu - kappa * np.abs(sigma)


# ---------------------------------------------------------------------------
# OOD distance
# ---------------------------------------------------------------------------

def knn_ood_distance(X_train: np.ndarray, X_test: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean distance to k nearest training neighbors for each test point.

    Uses Euclidean distance in the provided embedding space.  Caller should
    apply PCA or normalisation before passing X.
    Returns array of shape (n_test,), higher = further from training distribution.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_train)
    dists, _ = nn.kneighbors(X_test)
    return dists.mean(axis=1)


def standardize_ood(d_train: np.ndarray, d_test: np.ndarray) -> np.ndarray:
    """Z-score OOD distances using training self-distances as reference.

    Train self-distance: kNN on GDPa1 with k+1 neighbors, discard nearest (self).
    Caller passes those distances as d_train.
    """
    mu = np.median(d_train)
    mad = np.median(np.abs(d_train - mu)) + 1e-9
    return (d_test - mu) / (1.4826 * mad)   # robust z-score


# ---------------------------------------------------------------------------
# Hypervolume (exact, for m <= 5, n <= 200)
# ---------------------------------------------------------------------------

def _hv_2d_min(pts: np.ndarray, ref: np.ndarray) -> float:
    """2D hypervolume for minimization (lower is better). ref is the nadir/worst point.

    Correct sweep: sort by x ascending, accumulate (next_x - p_x) * (ref_y - p_y).
    """
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for i, p in enumerate(pts):
        if p[1] >= prev_y:
            continue   # dominated by an earlier (leftward) point
        next_x = pts[i + 1, 0] if i + 1 < len(pts) else ref[0]
        hv += (next_x - p[0]) * (ref[1] - p[1])
        prev_y = p[1]
    return float(hv)


def _hv_recursive(pts: np.ndarray, ref: np.ndarray, dim: int) -> float:
    """WFG-style recursive hypervolume for minimization (lower is better)."""
    if dim == 1:
        return float(max(0.0, ref[0] - pts[:, 0].min()))
    if dim == 2:
        return _hv_2d_min(pts, ref)
    # Slice along the last dimension: sort ascending (best = smallest first)
    pts = pts[np.argsort(pts[:, dim - 1])]
    hv = 0.0
    prev = ref[dim - 1]
    for i in range(len(pts) - 1, -1, -1):  # process from worst (largest) to best (smallest)
        if pts[i, dim - 1] >= ref[dim - 1]:
            continue
        slice_h = prev - pts[i, dim - 1]
        # All points with dim-1 index up to i share this z-level
        sub = pts[:i + 1, :dim - 1]
        sub_ref = ref[:dim - 1]
        hv += slice_h * _hv_recursive(sub, sub_ref, dim - 1)
        prev = pts[i, dim - 1]
    return hv


def hypervolume(objectives: np.ndarray, reference: np.ndarray | None = None) -> float:
    """Exact hypervolume indicator (WFG). Works for m <= 5 and n <= ~200.

    objectives : (n, m) — higher is better (maximization convention)
    reference  : (m,)  — worst/nadir reference point in maximization space;
                         defaults to nadir - 0.1 * range

    Internally converts to minimization (negation) so the WFG recursion applies.
    """
    if reference is None:
        nadir = objectives.min(axis=0)
        ideal = objectives.max(axis=0)
        reference = nadir - 0.1 * (ideal - nadir + 1e-9)
    # Convert maximization -> minimization: negate everything.
    # In minimization space: neg_ref is the worst (highest) value; neg_obj < neg_ref means better.
    neg_obj = -objectives
    neg_ref = -reference
    m = objectives.shape[1]
    # keep points strictly better than reference (neg_obj < neg_ref in each dim)
    mask = np.all(neg_obj < neg_ref, axis=1)
    pts = neg_obj[mask]
    if len(pts) == 0:
        return 0.0
    return _hv_recursive(pts, neg_ref, m)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def top_k_recovery(scores: np.ndarray, true_y: np.ndarray, k: int, direction: str = "higher") -> float:
    """Fraction of top-k by `scores` that are in top-k by `true_y`.

    direction: 'higher' = larger true_y is better; 'lower' = smaller is better.
    """
    if direction == "lower":
        true_rank = np.argsort(true_y)[:k]
    else:
        true_rank = np.argsort(-true_y)[:k]
    pred_rank = np.argsort(-scores)[:k]
    return len(set(true_rank) & set(pred_rank)) / k


def multi_assay_hit_rate(selected_idx: np.ndarray, true_ys: np.ndarray,
                          directions: list[str], top_frac: float = 0.2) -> float:
    """Fraction of selected antibodies that are in the top `top_frac` for at least half the assays.

    true_ys : (n, m) true assay values
    directions : list of 'higher' or 'lower' per column
    """
    n, m = true_ys.shape
    k = max(1, int(np.floor(top_frac * n)))
    good = np.zeros(n, dtype=int)
    for j, d in enumerate(directions):
        if d == "lower":
            thresh_idx = np.argsort(true_ys[:, j])[:k]
        else:
            thresh_idx = np.argsort(-true_ys[:, j])[:k]
        good[thresh_idx] += 1
    is_hit = good[selected_idx] >= max(1, m // 2)
    return float(is_hit.mean())
