"""Fast offline model search using CACHED ESM-2 embeddings (no GPU, seconds per config).

Loads dataset + cached ESM-2 (mean and mean+cdr) + interpretable descriptors, then evaluates
many tiny-data-appropriate modeling strategies per property on the official folds, printing
Spearman means so we can pick a design that beats the ridge baseline before committing a full run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.descriptors import build_descriptor_matrix
from abdev.metrics import spearman

CFG = yaml.safe_load(open(ROOT / "configs/default.yaml"))
FOLD = CFG["data"]["primary_fold"]


def load_cached_esm():
    proc = ROOT / "data/processed"
    mean_v, meancdr_v = None, None
    for f in proc.glob("esm2_*.npz"):
        arr = np.load(f)["emb"]
        if arr.shape[1] == 1280:
            mean_v = arr
        elif arr.shape[1] == 2560:
            meancdr_v = arr
    return mean_v, meancdr_v


def eval_config(ds, X, make_model, props):
    out = {}
    for prop in props:
        y = ds.y(prop)
        scores = []
        for _, tr, te in fold_iter(ds, prop, FOLD):
            m = make_model()
            m.fit(X[tr], y[tr])
            scores.append(spearman(y[te], m.predict(X[te])))
        out[prop] = float(np.nanmean(scores))
    return out


def ridge():
    return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])


def pca_ridge(k):
    return Pipeline([("s", StandardScaler()), ("p", PCA(k, random_state=0)),
                     ("r", RidgeCV(alphas=np.logspace(-1, 4, 12)))])


def pls(k):
    return PLSRegression(n_components=k)


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    props = list(ds.target_cols.keys())
    mean_v, meancdr_v = load_cached_esm()
    desc = build_descriptor_matrix(ds.heavy, ds.light).to_numpy(dtype=float)
    print(f"ESM mean {None if mean_v is None else mean_v.shape} | "
          f"mean+cdr {None if meancdr_v is None else meancdr_v.shape} | desc {desc.shape}")

    configs = {
        "baseline: ridge(ESM-mean)": (mean_v, ridge),
        "ridge(ESM-mean+desc)": (np.hstack([mean_v, desc]), ridge),
        "ridge(desc only)": (desc, ridge),
        "PCA64+ridge(ESM-mean)": (mean_v, lambda: pca_ridge(64)),
        "PCA128+ridge(ESM-mean)": (mean_v, lambda: pca_ridge(128)),
        "PCA64+ridge(ESM-mean+desc)": (np.hstack([mean_v, desc]), lambda: pca_ridge(64)),
        "PCA128+ridge(mean+cdr)": (meancdr_v, lambda: pca_ridge(128)),
        "PCA128+ridge(mean+cdr+desc)": (np.hstack([meancdr_v, desc]), lambda: pca_ridge(128)),
        "PLS10(ESM-mean)": (mean_v, lambda: pls(10)),
        "PLS15(mean+cdr+desc)": (np.hstack([meancdr_v, desc]), lambda: pls(15)),
    }

    rows = []
    for name, (X, mk) in configs.items():
        res = eval_config(ds, np.nan_to_num(X), mk, props)
        res["config"] = name
        res["MEAN"] = float(np.mean([res[p] for p in props]))
        rows.append(res)
    df = pd.DataFrame(rows)[["config"] + props + ["MEAN"]]
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
