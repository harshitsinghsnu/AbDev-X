"""Cross-validation evaluation harness honoring the official GDPa1 folds.

Produces a per-property, per-fold Spearman table for any model factory, plus mean +/- std.
The same harness scores both the baseline and our ensemble so comparisons are apples-to-apples
on identical splits. Set `mt_tasks` to enable the multi-task base learner (the model factory
must then accept fit(X, y, Y_multi=, task_index=)).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .data.dataset import GDPa1
from .data.splits import fold_iter
from .metrics import spearman, summarize_folds

ModelFactory = Callable[[], object]


def evaluate_model(ds: GDPa1, X: np.ndarray, model_factory: ModelFactory,
                   properties: List[str], fold_logical: str,
                   mt_tasks: Optional[List[str]] = None) -> pd.DataFrame:
    """Return tidy DataFrame: rows = (property, fold, spearman, n_test)."""
    Y_multi = None
    if mt_tasks:
        Y_multi = np.column_stack([ds.y(t) for t in mt_tasks])

    records = []
    for prop in properties:
        y_all = ds.y(prop)
        for fid, tr, te in fold_iter(ds, prop, fold_logical):
            model = model_factory()
            if mt_tasks:
                model.fit(X[tr], y_all[tr], Y_multi=Y_multi[tr],  # type: ignore[call-arg]
                          task_index=mt_tasks.index(prop))
            else:
                model.fit(X[tr], y_all[tr])
            pred = model.predict(X[te])
            records.append({"property": prop, "fold": fid,
                            "spearman": spearman(y_all[te], pred), "n_test": len(te)})
    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prop, g in df.groupby("property"):
        s = summarize_folds(g["spearman"].tolist())
        rows.append({"property": prop, "spearman_mean": s["mean"],
                     "spearman_std": s["std"], "n_folds": s["n_folds"]})
    return pd.DataFrame(rows).sort_values("property").reset_index(drop=True)


def compare(baseline: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    b = aggregate(baseline).rename(columns={"spearman_mean": "baseline_mean"})
    m = aggregate(model).rename(columns={"spearman_mean": "model_mean"})
    out = b[["property", "baseline_mean"]].merge(m[["property", "model_mean"]], on="property")
    out["delta"] = out["model_mean"] - out["baseline_mean"]
    out["wins"] = out["delta"] > 0
    return out
