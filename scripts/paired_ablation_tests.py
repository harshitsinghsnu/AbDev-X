"""Paired statistical tests for AbDev-X ablation methods.

By default this uses fold-level Spearman values from
results/abdevx2_ablation/fold_metrics.csv and treats each matched
split x fold x assay score as one paired observation.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
DEFAULT_METHODS = ["representation_fusion", "multitask_projection", "cdr_framework_moe"]
LABELS = {
    "representation_fusion": "AbDev-X",
    "abdevx2_full": "AbDev-X 2.0 full",
    "multitask_projection": "Multitask projection",
    "cdr_framework_moe": "CDR-framework MoE",
}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in original order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def load_long_scores(path: Path, methods: list[str], include_mean: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [m for m in methods if m not in set(df["method"])]
    if missing:
        raise ValueError(f"Missing method(s) in {path}: {missing}")

    key_cols = ["split"]
    if "fold" in df.columns:
        key_cols.append("fold")

    value_cols = ASSAYS + (["MEAN"] if include_mean else [])
    missing_cols = [c for c in key_cols + value_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing column(s) in {path}: {missing_cols}")

    long = df[df["method"].isin(methods)].melt(
        id_vars=["method", *key_cols],
        value_vars=value_cols,
        var_name="assay",
        value_name="score",
    )
    long = long.dropna(subset=["score"])
    return long


def paired_matrix(long: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, list[str]]:
    key_cols = [c for c in ["split", "fold", "assay"] if c in long.columns]
    wide = long.pivot_table(index=key_cols, columns="method", values="score", aggfunc="first")
    wide = wide.dropna(subset=methods)
    if wide.empty:
        raise ValueError("No complete paired observations found for the requested methods.")
    return wide[methods], key_cols


def pairwise_tests(wide: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for a, b in combinations(methods, 2):
        diff = wide[a].to_numpy() - wide[b].to_numpy()
        t = ttest_rel(wide[a], wide[b], nan_policy="omit")
        try:
            w = wilcoxon(wide[a], wide[b], zero_method="wilcox", alternative="two-sided")
            wilcoxon_p = float(w.pvalue)
            wilcoxon_stat = float(w.statistic)
        except ValueError:
            wilcoxon_p = np.nan
            wilcoxon_stat = np.nan
        rows.append(
            {
                "method_a": LABELS.get(a, a),
                "method_b": LABELS.get(b, b),
                "n_pairs": int(len(diff)),
                "mean_a": float(np.mean(wide[a])),
                "mean_b": float(np.mean(wide[b])),
                "mean_delta_a_minus_b": float(np.mean(diff)),
                "median_delta_a_minus_b": float(np.median(diff)),
                "paired_t_stat": float(t.statistic),
                "paired_t_p": float(t.pvalue),
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
            }
        )

    out = pd.DataFrame(rows)
    out["paired_t_p_holm"] = holm_adjust(out["paired_t_p"].tolist())
    if out["wilcoxon_p"].notna().all():
        out["wilcoxon_p_holm"] = holm_adjust(out["wilcoxon_p"].tolist())
    else:
        out["wilcoxon_p_holm"] = np.nan
    return out


def omnibus_test(wide: pd.DataFrame, methods: list[str]) -> dict[str, float]:
    stat, p = friedmanchisquare(*(wide[m].to_numpy() for m in methods))
    return {"friedman_chi2": float(stat), "friedman_p": float(p), "n_blocks": int(len(wide))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/abdevx2_ablation/fold_metrics.csv",
        help="CSV with method, split, optional fold, and assay score columns.",
    )
    parser.add_argument(
        "--methods",
        nargs=3,
        default=DEFAULT_METHODS,
        help="Exactly three method IDs to compare.",
    )
    parser.add_argument(
        "--include-mean",
        action="store_true",
        help="Also include the MEAN column as a paired observation. Default: assays only.",
    )
    parser.add_argument(
        "--out",
        default="results/abdevx2_ablation/paired_tests.csv",
        help="Where to write pairwise test results.",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    long = load_long_scores(input_path, args.methods, args.include_mean)
    wide, key_cols = paired_matrix(long, args.methods)
    omnibus = omnibus_test(wide, args.methods)
    pairwise = pairwise_tests(wide, args.methods)
    pairwise.to_csv(out_path, index=False)

    labels = [LABELS.get(m, m) for m in args.methods]
    print("Methods:", " vs ".join(labels))
    print("Paired unit:", " x ".join(key_cols))
    print(f"Complete paired observations: {len(wide)}")
    print(
        "Friedman omnibus: "
        f"chi2={omnibus['friedman_chi2']:.4g}, p={omnibus['friedman_p']:.4g}"
    )
    print()
    print(pairwise.round(4).to_string(index=False))
    print(f"\nSaved pairwise results to {out_path}")


if __name__ == "__main__":
    main()
