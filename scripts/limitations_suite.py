"""Run reviewer-requested limitation analyses for AbDev-X.

This script is intentionally cache-first: it reuses GDPa1/GDPa3 embeddings and
feature blocks already present in data/processed and external/abdev-benchmark.
It writes small CSV files under results/limitations_suite.

Analyses covered:
  * structure-source robustness summary
  * standardized stacking coefficient bootstrap stability
  * representation-selection multiple-comparison diagnostics
  * GDPa3 non-inferiority bootstrap against published champion point scores
  * decision-centric Precision@K plus uncertainty-abstention tradeoffs
  * external augmentation batch/study sensitivity summary
  * optional extra frozen-encoder baselines from user-provided .npz embeddings
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "limitations_suite"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abdev.data.splits import fold_iter  # noqa: E402
import gdpa3_eval as GE  # noqa: E402

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
LOWER_IS_BETTER = {"HIC", "AC-SINS_pH7.4", "PR_CHO"}
CHAMPIONS = {"HIC": 0.708, "Tm2": 0.392, "AC-SINS_pH7.4": 0.356, "PR_CHO": 0.337, "Titer": 0.310}


def rpca() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(0.90, random_state=0, svd_solver="full")),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 12))),
        ]
    )


def ridge() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 12)))])


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    out = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        out[idx] = min(running, 1.0)
    return out.tolist()


def safe_rho(y: np.ndarray, pred: np.ndarray) -> float:
    ok = ~np.isnan(y) & ~np.isnan(pred)
    if ok.sum() < 3:
        return np.nan
    return float(spearmanr(y[ok], pred[ok]).statistic)


def rank_metric(y_true: np.ndarray, pred: np.ndarray, assay: str, k: int) -> tuple[float, float]:
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
    precision = sum(int(i in true_top) for i in pred_top) / k
    return float(precision), float(precision / (k / n))


def load_blocks(
    extra_embeddings: list[str],
) -> tuple[object, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray, Callable[[], Pipeline]]], list[str]]:
    ds, g3, b1, b3 = GE.build_blocks()
    blocks: dict[str, tuple[np.ndarray, np.ndarray, Callable[[], Pipeline]]] = {
        "esm2": (np.nan_to_num(b1["esm2"]), np.nan_to_num(b3["esm2"]), rpca),
        "descriptors": (np.nan_to_num(b1["descriptors"]), np.nan_to_num(b3["descriptors"]), ridge),
        "isotype": (np.nan_to_num(b1["isotype"]), np.nan_to_num(b3["isotype"]), ridge),
        "bio": (np.nan_to_num(b1["bio"]), np.nan_to_num(b3["bio"]), ridge),
        "ablang2": (np.nan_to_num(b1["ablang2"]), np.nan_to_num(b3["ablang2"]), ridge),
    }

    for spec in extra_embeddings:
        name, gdpa1_path, gdpa3_path = spec.split("=", 1)[0], None, None
        rhs = spec.split("=", 1)[1] if "=" in spec else ""
        parts = rhs.split(":")
        if len(parts) != 2:
            raise ValueError("--extra-embedding must be name=gdpa1.npz:gdpa3.npz")
        gdpa1_path, gdpa3_path = (ROOT / parts[0], ROOT / parts[1])
        missing = [str(p.relative_to(ROOT)) for p in (gdpa1_path, gdpa3_path) if not p.exists()]
        if missing:
            print(
                f"[warn] skipping optional embedding '{name}' because file(s) are missing: "
                + ", ".join(missing),
                flush=True,
            )
            print(
                "       Create them with scripts/extract_extra_encoder.py, or rerun without --extra-embedding.",
                flush=True,
            )
            continue
        x1 = np.load(gdpa1_path)["emb"]
        x3 = np.load(gdpa3_path)["emb"]
        if len(x1) != len(ds.df) or len(x3) != len(g3):
            raise ValueError(
                f"Embedding '{name}' has incompatible row counts: "
                f"GDPa1 {x1.shape}, GDPa3 {x3.shape}; expected {len(ds.df)} and {len(g3)} rows."
            )
        blocks[name] = (np.nan_to_num(x1), np.nan_to_num(x3), rpca)
    loaded_extra = [s.split("=", 1)[0] for s in extra_embeddings if s.split("=", 1)[0] in blocks]
    return ds, g3, blocks, loaded_extra


def cv_block_scores(ds, blocks, split: str) -> pd.DataFrame:
    rows = []
    for assay in ASSAYS:
        y = ds.y(assay)
        for block, (x1, _, make_model) in blocks.items():
            for fold, tr, te in fold_iter(ds, assay, split):
                pred = make_model().fit(x1[tr], y[tr]).predict(x1[te])
                rows.append({"assay": assay, "block": block, "fold": fold, "rho": safe_rho(y[te], pred)})
    return pd.DataFrame(rows)


def full_block_predictions(ds, g3, blocks) -> dict[str, dict[str, np.ndarray]]:
    preds: dict[str, dict[str, np.ndarray]] = {}
    for assay in ASSAYS:
        y = ds.y(assay)
        train = ~np.isnan(y)
        preds[assay] = {}
        for block, (x1, x3, make_model) in blocks.items():
            preds[assay][block] = make_model().fit(x1[train], y[train]).predict(x3)
    return preds


def selection_multiple_tests(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for assay, sub in cv.groupby("assay"):
        wide = sub.pivot(index="fold", columns="block", values="rho").dropna(axis=1)
        means = wide.mean().sort_values(ascending=False)
        best = means.index[0]
        fried_stat, fried_p = friedmanchisquare(*(wide[c].to_numpy() for c in wide.columns))
        pvals = []
        pairs = []
        for other in means.index[1:]:
            try:
                p = float(wilcoxon(wide[best], wide[other], zero_method="wilcox").pvalue)
            except ValueError:
                p = 1.0
            pvals.append(p)
            pairs.append(other)
        adjusted = holm(pvals)
        for other, p, p_holm in zip(pairs, pvals, adjusted):
            rows.append(
                {
                    "assay": assay,
                    "best_block": best,
                    "other_block": other,
                    "best_mean_cv": means[best],
                    "other_mean_cv": means[other],
                    "friedman_p": fried_p,
                    "wilcoxon_p": p,
                    "wilcoxon_p_holm_within_assay": p_holm,
                }
            )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm_global"] = holm(out["wilcoxon_p"].tolist())
    return out


def standardized_oof_meta(ds, blocks, split: str) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    out = {}
    names = list(blocks)
    for assay in ASSAYS:
        y_all = ds.y(assay)
        valid = np.where(~np.isnan(y_all))[0]
        pos = {idx: i for i, idx in enumerate(valid)}
        y = y_all[valid]
        oof = np.full((len(valid), len(names)), np.nan)
        for bi, name in enumerate(names):
            x1, _, make_model = blocks[name]
            for _, tr_full, te_full in fold_iter(ds, assay, split):
                tr = np.array([pos[i] for i in tr_full if i in pos])
                te = np.array([pos[i] for i in te_full if i in pos])
                if len(te) == 0:
                    continue
                oof[te, bi] = make_model().fit(x1[valid[tr]], y[tr]).predict(x1[valid[te]])
        keep = ~np.isnan(oof).any(axis=1)
        scaler = StandardScaler().fit(oof[keep])
        out[assay] = (scaler.transform(oof[keep]), y[keep], names)
    return out


def meta_weight_stability(ds, blocks, split: str, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for assay, (x_meta, y, names) in standardized_oof_meta(ds, blocks, split).items():
        fitted = RidgeCV(alphas=np.logspace(-2, 4, 12)).fit(x_meta, y)
        boot = np.zeros((n_boot, len(names)))
        for b in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            boot[b] = RidgeCV(alphas=np.logspace(-2, 4, 12)).fit(x_meta[idx], y[idx]).coef_
        for i, name in enumerate(names):
            coef = boot[:, i]
            point = float(fitted.coef_[i])
            rows.append(
                {
                    "assay": assay,
                    "block": name,
                    "standardized_weight": point,
                    "boot_mean": float(np.mean(coef)),
                    "boot_sd": float(np.std(coef, ddof=1)),
                    "ci_low": float(np.quantile(coef, 0.025)),
                    "ci_high": float(np.quantile(coef, 0.975)),
                    "negative_fraction": float(np.mean(coef < 0)),
                    "sign_stability": float(max(np.mean(coef >= 0), np.mean(coef < 0))),
                }
            )
    return pd.DataFrame(rows)


def noninferiority_bootstrap(g3, pred_by_assay, selected: pd.DataFrame, margin: float, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for assay in ASSAYS:
        block = selected[selected["assay"] == assay].iloc[0]["best_block"]
        y = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        pred = pred_by_assay[assay][block]
        ok = ~np.isnan(y) & ~np.isnan(pred)
        y = y[ok]
        pred = pred[ok]
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            boot.append(safe_rho(y[idx], pred[idx]))
        boot = np.array(boot)
        threshold = CHAMPIONS[assay] - margin
        rows.append(
            {
                "assay": assay,
                "selected_block": block,
                "rho": safe_rho(y, pred),
                "champion": CHAMPIONS[assay],
                "noninferiority_margin": margin,
                "champion_minus_margin": threshold,
                "ci_low": float(np.nanquantile(boot, 0.025)),
                "ci_high": float(np.nanquantile(boot, 0.975)),
                "bootstrap_p_h0_rho_le_champion_minus_margin": float(np.nanmean(boot <= threshold)),
                "noninferior_by_ci": bool(np.nanquantile(boot, 0.025) > threshold),
            }
        )
    return pd.DataFrame(rows)


def decision_and_abstention(g3, pred_by_assay, selected: pd.DataFrame, abstain_fracs: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_rows = []
    abstain_rows = []
    for assay in ASSAYS:
        block = selected[selected["assay"] == assay].iloc[0]["best_block"]
        y = g3[GE.LABEL_MAP[assay]].to_numpy(float)
        pred = pred_by_assay[assay][block]
        for k in (5, 10, 20):
            precision, enrichment = rank_metric(y, pred, assay, k)
            decision_rows.append(
                {
                    "assay": assay,
                    "selected_block": block,
                    "k": k,
                    "precision_at_k": precision,
                    "enrichment": enrichment,
                }
            )

        all_preds = np.column_stack([pred_by_assay[assay][b] for b in pred_by_assay[assay]])
        zpred = StandardScaler().fit_transform(all_preds)
        uncertainty = np.nanstd(zpred, axis=1)
        ok = ~np.isnan(y) & ~np.isnan(pred) & ~np.isnan(uncertainty)
        for frac in abstain_fracs:
            cutoff = np.nanquantile(uncertainty[ok], 1.0 - frac)
            keep = ok & (uncertainty <= cutoff)
            p10, e10 = rank_metric(y[keep], pred[keep], assay, 10)
            abstain_rows.append(
                {
                    "assay": assay,
                    "selected_block": block,
                    "abstain_fraction": frac,
                    "n_retained": int(keep.sum()),
                    "retained_spearman": safe_rho(y[keep], pred[keep]),
                    "retained_precision_at_10": p10,
                    "retained_enrichment_at_10": e10,
                }
            )
    return pd.DataFrame(decision_rows), pd.DataFrame(abstain_rows)


def batch_effect_summary() -> pd.DataFrame:
    path = ROOT / "results" / "exp_scaling.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for assay, sub in df.groupby("assay"):
        rows.append(
            {
                "assay": assay,
                "n_studies": len(sub),
                "mean_delta": float(sub["delta"].mean()),
                "inverse_n_weighted_delta": float(np.average(sub["delta"], weights=1.0 / sub["n"])),
                "best_study": sub.loc[sub["delta"].idxmax(), "study"],
                "best_delta": float(sub["delta"].max()),
                "worst_study": sub.loc[sub["delta"].idxmin(), "study"],
                "worst_delta": float(sub["delta"].min()),
                "fraction_harmful": float(np.mean(sub["delta"] < 0)),
            }
        )
    return pd.DataFrame(rows)


def structure_source_summary() -> pd.DataFrame:
    path = ROOT / "results" / "exp_structure_source.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        [{"note": "results/exp_structure_source.csv not found; run scripts/exp_core.py or structure-source analysis first."}]
    )


def extra_encoder_summary(cv: pd.DataFrame, extra_names: list[str]) -> pd.DataFrame:
    if not extra_names:
        return pd.DataFrame()
    rows = []
    for name in extra_names:
        sub = cv[cv["block"] == name]
        if sub.empty:
            continue
        for assay, a in sub.groupby("assay"):
            rows.append({"encoder": name, "assay": assay, "mean_cv": float(a["rho"].mean()), "std_cv": float(a["rho"].std(ddof=1))})
        rows.append({"encoder": name, "assay": "MEAN", "mean_cv": float(sub.groupby("assay")["rho"].mean().mean()), "std_cv": np.nan})
    return pd.DataFrame(rows)


def embedding_sanity(blocks, extra_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in extra_names:
        if name not in blocks:
            continue
        x1, x3, _ = blocks[name]
        for dataset, x in (("GDPa1", x1), ("GDPa3", x3)):
            centered_norm = np.linalg.norm(x - x.mean(axis=0, keepdims=True), axis=1)
            rows.append(
                {
                    "encoder": name,
                    "dataset": dataset,
                    "shape": "x".join(map(str, x.shape)),
                    "global_std": float(np.std(x)),
                    "mean_centered_row_norm": float(np.mean(centered_norm)),
                    "fraction_near_constant_dims": float(np.mean(np.std(x, axis=0) < 1e-8)),
                    "warning": bool(np.mean(centered_norm) < 1e-3),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="cluster_isotype")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--noninferiority-margin", type=float, default=0.10)
    parser.add_argument(
        "--extra-embedding",
        action="append",
        default=[],
        help="Optional frozen encoder as name=path/to/gdpa1.npz:path/to/gdpa3.npz; arrays must be stored as 'emb'.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory relative to repo root. Default changes when extra embeddings are loaded.")
    args = parser.parse_args()

    ds, g3, blocks, extra_names = load_blocks(args.extra_embedding)
    out_dir = ROOT / args.out_dir if args.out_dir else OUT
    if args.out_dir is None and extra_names:
        out_dir = ROOT / "results" / ("limitations_suite_" + "_".join(extra_names))
    out_dir.mkdir(parents=True, exist_ok=True)

    cv = cv_block_scores(ds, blocks, args.split)
    cv.to_csv(out_dir / "representation_cv_by_fold.csv", index=False)

    selected = (
        cv.groupby(["assay", "block"], as_index=False)["rho"]
        .mean()
        .sort_values(["assay", "rho"], ascending=[True, False])
        .groupby("assay", as_index=False)
        .first()
        .rename(columns={"block": "best_block", "rho": "best_mean_cv"})
    )
    selected.to_csv(out_dir / "selected_blocks.csv", index=False)

    pred_by_assay = full_block_predictions(ds, g3, blocks)

    outputs = {
        "structure_source_summary.csv": structure_source_summary(),
        "selection_multiple_tests.csv": selection_multiple_tests(cv),
        "meta_weight_bootstrap.csv": meta_weight_stability(ds, blocks, args.split, args.n_boot, args.seed),
        "noninferiority_bootstrap.csv": noninferiority_bootstrap(
            g3, pred_by_assay, selected, args.noninferiority_margin, args.n_boot, args.seed
        ),
        "batch_effect_summary.csv": batch_effect_summary(),
    }
    decision, abstention = decision_and_abstention(g3, pred_by_assay, selected, [0.0, 0.1, 0.2, 0.3])
    outputs["decision_metrics.csv"] = decision
    outputs["abstention_tradeoff.csv"] = abstention
    outputs["extra_encoder_cv_summary.csv"] = extra_encoder_summary(cv, extra_names)
    outputs["extra_embedding_sanity.csv"] = embedding_sanity(blocks, extra_names)

    for name, df in outputs.items():
        df.to_csv(out_dir / name, index=False)

    print(f"Saved limitation analyses to {out_dir}")
    print("\nSelected OOD-CV blocks:")
    print(selected.round(3).to_string(index=False))
    print("\nNon-inferiority bootstrap:")
    print(outputs["noninferiority_bootstrap.csv"].round(3).to_string(index=False))
    print("\nAbstention tradeoff (first rows):")
    print(abstention.round(3).head(12).to_string(index=False))
    if extra_names:
        print("\nExtra encoder CV summary:")
        print(outputs["extra_encoder_cv_summary.csv"].round(3).to_string(index=False))
        print("\nExtra embedding sanity:")
        print(outputs["extra_embedding_sanity.csv"].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
