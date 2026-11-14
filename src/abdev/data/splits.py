"""Fold iteration honoring the official GDPa1 fold columns.

The competition uses precomputed fold assignments (so cluster-aware grouping is fixed and
reproducible across teams). We never re-derive folds; we only iterate the shipped column,
restricted to rows that have a label for the property being modeled.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from .dataset import GDPa1


def fold_iter(ds: GDPa1, prop: str, fold_logical: str) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold_id, train_idx, test_idx) over rows labeled for `prop`.

    Indices are positional into the *full* dataframe so feature matrices align 1:1.
    """
    if fold_logical not in ds.fold_cols:
        raise KeyError(f"Fold '{fold_logical}' not available. Have: {list(ds.fold_cols)}")
    folds = ds.folds(fold_logical)
    labeled = ds.labeled_mask(prop)
    unique = [f for f in sorted(set(folds.tolist())) if f == f]  # drop NaN
    for fid in unique:
        test_mask = (folds == fid) & labeled
        train_mask = (folds != fid) & labeled
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        yield fid, train_idx, test_idx
