"""Reproduce the released competition baseline: Ridge on ESM-2 mean-pooled embeddings.

Running this on the official folds gives us the exact numbers we must beat -- computed
in-repo on identical splits, which is the only fair, verifiable comparison (the private
80-antibody holdout is not public).
"""
from __future__ import annotations

import pandas as pd

from .data.dataset import GDPa1, load_gdpa1
from .evaluate import aggregate, evaluate_model
from .features.plm import esm2_embeddings
from .models.heads import make_ridge


def run_esm2_ridge_baseline(ds: GDPa1, cfg: dict, fold_logical: str) -> pd.DataFrame:
    fv = (ds.heavy + ds.light).tolist()
    X = esm2_embeddings(fv, cfg["features"]["esm2_model"], pool=("mean",),
                        cache_dir=cfg["data"]["processed_dir"])
    props = list(ds.target_cols.keys())
    per_fold = evaluate_model(ds, X, make_ridge, props, fold_logical)
    return per_fold


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("configs/default.yaml"))
    ds = load_gdpa1()
    fold = cfg["data"]["primary_fold"]
    per_fold = run_esm2_ridge_baseline(ds, cfg, fold)
    print(aggregate(per_fold).to_string(index=False))
