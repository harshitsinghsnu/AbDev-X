"""Scoring utilities. The competition metric is Spearman's rho, per property."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def summarize_folds(per_fold: list[float]) -> dict:
    arr = np.asarray([x for x in per_fold if x == x], dtype=float)
    return {"mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
            "n_folds": int(arr.size)}
