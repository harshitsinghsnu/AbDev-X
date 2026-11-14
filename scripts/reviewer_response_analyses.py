"""Reviewer-response analyses for AbDev-X.

Adds lightweight checks requested after the first manuscript draft:

1. Stability of per-assay representation selection under bootstrap resampling of the five
   official cluster-isotype folds.
2. GDPa3 transfer for top-k representation averaging, using only GDPa1 CV to pick top-k.
3. Ridge vs non-negative vs simplex stacking meta-learners on GDPa3.
4. Decision-centric GDPa3 metrics: Precision@K and enrichment over random.

All analyses use cached GDPa1/GDPa3/FLAb2 assets; no PLM extraction or network access.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer_response"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abdev.data.dataset import load_gdpa1  # noqa: E402
from abdev.data.splits import fold_iter  # noqa: E402
from abdev.features.descriptors import build_descriptor_matrix  # noqa: E402
import gdpa3_eval as GE  # noqa: E402

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
LOWER_IS_BETTER = {"HIC", "AC-SINS_pH7.4", "PR_CHO"}


def rpca():
    return Pipeline(
        [
            ("s", StandardScaler()),
            ("p", PCA(0.9, random_state=0, svd_solver="full")),
            ("r", RidgeCV(alphas=np.logspace(-2, 4, 12))),
        ]
    )


def ridge():
    return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])


def rank_metric(y_true: np.ndarray, pred: np.ndarray, assay: str, k: int) -> Tuple[float, float]:
    ok = ~np.isnan(y_true) & ~np.isnan(pred)
    y = y_true[ok]
    p = pred[ok]
    n = len(y)
    if n == 0:
        return np.nan, np.nan
    k = min(k, n)
    if assay in LOWER_IS_BETTER:
        pred_top = np.argsort(p)[:k]
        true_top = set(np.argsort(y)[:k].tolist())
    else:
        pred_top = np.argsort(-p)[:k]
        true_top = set(np.argsort(-y)[:k].tolist())
    hits = sum(int(i in true_top) for i in pred_top)
    precision = hits / k
    enrichment = precision / (k / n)
    return float(precision), float(enrichment)


def build_blocks():
    ds, g3, b1, b3 = GE.build_blocks()
    blocks = {
        "esm2": (np.nan_to_num(b1["esm2"]), np.nan_to_num(b3["esm2"]), rpca),
        "descriptors": (np.nan_to_num(b1["descriptors"]), np.nan_to_num(b3["descriptors"]), ridge),
        "isotype": (np.nan_to_num(b1["isotype"]), np.nan_to_num(b3["isotype"]), ridge),
        "bio": (np.nan_to_num(b1["bio"]), np.nan_to_num(b3["bio"]), ridge),
        "ablang2": (np.nan_to_num(b1["ablang2"]), np.nan_to_num(b3["ablang2"]), ridge),
    }
    return ds, g3, blocks


def block_cv_scores(ds, blocks: Dict[str, Tuple[np.ndarray, np.ndarray, callable]], split="cluster_isotype"):
    rows = []
    for assay in ASSAYS:
        y = ds.y(assay)
        for block, (x1, _, mk) in blocks.items():
            for fid, tr, te in fold_iter(ds, assay, split):
                pred = mk().fit(x1[tr], y[tr]).predict(x1[te])
                rho = spearmanr(y[te], pred).statistic
                rows.append({"assay": assay, "block": block, "fold": fid, "rho": float(rho)})
    return pd.DataFrame(rows)


def selection_stability(cv: pd.DataFrame, n_boot=2000, seed=13):
    rng = np.random.default_rng(seed)
    rows = []
    topk_rows = []
    for assay, sub in cv.groupby("assay"):
        folds = sorted(sub["fold"].unique())
        blocks = sorted(sub["block"].unique())
        picks = []
        for _ in range(n_boot):
            sampled = rng.choice(folds, size=len(folds), replace=True)
            means = {}
            for block in blocks:
                b = sub[sub["block"] == block]
                means[block] = np.mean([b[b["fold"] == f]["rho"].iloc[0] for f in sampled])
            picks.append(max(means, key=means.get))
        counts = pd.Series(picks).value_counts(normalize=True)
        for block in blocks:
            rows.append(
                {
                    "assay": assay,
                    "block": block,
                    "selection_fraction": round(float(counts.get(block, 0.0)), 3),
                    "mean_cv": round(float(sub[sub["block"] == block]["rho"].mean()), 3),
                    "std_cv": round(float(sub[sub["block"] == block]["rho"].std(ddof=1)), 3),
                }
            )
        ordered = (
            sub.groupby("block")["rho"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .assign(assay=assay)
        )
        for k in (1, 2, 3):
            topk_rows.append({"assay": assay, "k": k, "blocks": "+".join(ordered.head(k)["block"])})
    return pd.DataFrame(rows), pd.DataFrame(topk_rows)


def train_block_predictions(ds, g3, blocks, cv):
    pred_rows = []
    pred_by_assay = {}
    for assay in ASSAYS:
        y1 = ds.y(assay)
        m = ~np.isnan(y1)
        y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        pred_by_assay[assay] = {}
        mu, sd = np.nanmean(y1[m]), np.nanstd(y1[m]) + 1e-9
        for block, (x1, x3, mk) in blocks.items():
            model = mk().fit(x1[m], (y1[m] - mu) / sd)
            pred = model.predict(x3) * sd + mu
            pred_by_assay[assay][block] = pred
            ok = ~np.isnan(y3)
            pred_rows.append(
                {
                    "assay": assay,
                    "model": block,
                    "rho": round(float(spearmanr(y3[ok], pred[ok]).statistic), 3),
                }
            )
    return pd.DataFrame(pred_rows), pred_by_assay


def topk_averaging(ds, g3, cv, pred_by_assay):
    rows = []
    for assay in ASSAYS:
        y1 = ds.y(assay)
        y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        ok = ~np.isnan(y3)
        ordered = cv[cv["assay"] == assay].groupby("block")["rho"].mean().sort_values(ascending=False)
        mu, sd = np.nanmean(y1[~np.isnan(y1)]), np.nanstd(y1[~np.isnan(y1)]) + 1e-9
        for k in (1, 2, 3):
            chosen = list(ordered.head(k).index)
            zpred = np.column_stack([(pred_by_assay[assay][b] - mu) / sd for b in chosen]).mean(1)
            pred = zpred * sd + mu
            rows.append(
                {
                    "assay": assay,
                    "k": k,
                    "blocks": "+".join(chosen),
                    "rho": round(float(spearmanr(y3[ok], pred[ok]).statistic), 3),
                }
            )
    out = pd.DataFrame(rows)
    mean_rows = out.groupby("k", as_index=False)["rho"].mean()
    mean_rows.insert(0, "assay", "MEAN")
    mean_rows["blocks"] = ""
    return pd.concat([out, mean_rows[out.columns]], ignore_index=True)


def constrained_stacking(ds, g3, blocks):
    rows = []
    for assay in ASSAYS:
        y1 = ds.y(assay)
        m = ~np.isnan(y1)
        idx = np.where(m)[0]
        y = y1[m]
        y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        ok = ~np.isnan(y3)
        names = list(blocks)
        oof = np.full((len(idx), len(names)), np.nan)
        held = np.zeros((len(y3), len(names)))
        for bi, name in enumerate(names):
            x1, x3, mk = blocks[name]
            for _, tr_full, te_full in fold_iter(ds, assay, "cluster_isotype"):
                tr = np.array([np.where(idx == t)[0][0] for t in tr_full if t in set(idx)])
                te = np.array([np.where(idx == t)[0][0] for t in te_full if t in set(idx)])
                oof[te, bi] = mk().fit(x1[idx[tr]], y[tr]).predict(x1[idx[te]])
            held[:, bi] = mk().fit(x1[m], y).predict(x3)
        ridge_meta = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(oof, y)
        pred_ridge = ridge_meta.predict(held)
        w_nnls, _ = nnls(np.column_stack([oof, np.ones(len(oof))]), y)
        pred_nnls = np.column_stack([held, np.ones(len(held))]) @ w_nnls
        w_simplex, _ = nnls(oof, y)
        w_simplex = w_simplex / (w_simplex.sum() + 1e-9)
        pred_simplex = held @ w_simplex
        for model, pred in (
            ("ridge_meta", pred_ridge),
            ("nnls_meta", pred_nnls),
            ("simplex_meta", pred_simplex),
        ):
            rows.append(
                {
                    "assay": assay,
                    "model": model,
                    "rho": round(float(spearmanr(y3[ok], pred[ok]).statistic), 3),
                }
            )
    out = pd.DataFrame(rows)
    mean = out.groupby("model", as_index=False)["rho"].mean()
    mean.insert(0, "assay", "MEAN")
    return pd.concat([out, mean[out.columns]], ignore_index=True)


def decision_metrics(g3, pred_by_assay, selected):
    rows = []
    for assay in ASSAYS:
        y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        block = selected[selected["assay"] == assay].sort_values("k").iloc[0]["blocks"]
        pred = pred_by_assay[assay][block]
        for k in (5, 10, 20):
            p_at_k, enrich = rank_metric(y3, pred, assay, k)
            rows.append(
                {
                    "assay": assay,
                    "model": f"best_single_{block}",
                    "K": k,
                    "precision_at_k": round(p_at_k, 3),
                    "enrichment": round(enrich, 3),
                }
            )
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds, g3, blocks = build_blocks()
    cv = block_cv_scores(ds, blocks)
    cv.to_csv(OUT / "representation_cv_by_fold.csv", index=False)
    stability, topk = selection_stability(cv)
    stability.to_csv(OUT / "selection_stability.csv", index=False)
    topk.to_csv(OUT / "topk_selected_blocks.csv", index=False)
    held_blocks, pred_by_assay = train_block_predictions(ds, g3, blocks, cv)
    held_blocks.to_csv(OUT / "gdpa3_single_block.csv", index=False)
    topk_perf = topk_averaging(ds, g3, cv, pred_by_assay)
    topk_perf.to_csv(OUT / "gdpa3_topk_averaging.csv", index=False)
    meta = constrained_stacking(ds, g3, blocks)
    meta.to_csv(OUT / "gdpa3_constrained_stacking.csv", index=False)
    dec = decision_metrics(g3, pred_by_assay, topk)
    dec.to_csv(OUT / "gdpa3_decision_metrics.csv", index=False)

    print("=== Selection stability: most stable block per assay ===")
    print(
        stability.sort_values(["assay", "selection_fraction"], ascending=[True, False])
        .groupby("assay")
        .head(2)
        .to_string(index=False)
    )
    print("\n=== GDPa3 top-k averaging ===")
    print(topk_perf.pivot_table(index="assay", columns="k", values="rho").round(3).to_string())
    print("\n=== GDPa3 constrained stacking ===")
    print(meta.pivot_table(index="assay", columns="model", values="rho").round(3).to_string())
    print("\n=== GDPa3 decision metrics, best-single representation ===")
    print(dec.to_string(index=False))
    print(f"\nSaved reviewer-response CSVs under {OUT}")


if __name__ == "__main__":
    main()
