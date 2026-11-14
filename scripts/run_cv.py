"""End-to-end cross-validation: baseline vs AbDev-X on the official GDPa1 folds.

    python scripts/run_cv.py --config configs/default.yaml --fold cluster_isotype

Outputs to results/:
  * baseline_<fold>.csv        per-fold Spearman for ESM-2 ridge baseline
  * abdevx_<fold>.csv          per-fold Spearman for our ensemble
  * comparison_<fold>.csv      per-property mean + delta + win flag
Honors the fairness protocol in PLAN.md (identical splits, in-fold preprocessing).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.pipeline import build_features
from abdev.evaluate import evaluate_model, aggregate, compare
from abdev.models.heads import make_ridge
from abdev.models.ensemble import StackingRegressor
from abdev.baselines import run_esm2_ridge_baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fold", default=None, help="logical fold name; default = config primary_fold")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--multitask", action="store_true", help="enable multi-task base learner")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    fold = args.fold or cfg["data"]["primary_fold"]
    results_dir = Path(cfg["eval"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    ds = load_gdpa1(args.config)
    props = list(ds.target_cols.keys())
    print(f"Loaded {len(ds.df)} antibodies. Scoring properties: {props}")
    print(f"Using fold column: {fold} -> {ds.fold_cols.get(fold)}")

    # --- Our fused feature matrix (descriptors [+ PLM if enabled]) ---
    fm = build_features(ds, cfg)
    print(f"Feature matrix: {fm.X.shape[0]} x {fm.X.shape[1]} "
          f"(blocks: { {k: len(v) for k, v in fm.blocks.items()} })")
    X = fm.values

    # --- Baseline (ESM-2 ridge) on identical folds ---
    baseline_df = None
    if not args.no_baseline:
        try:
            baseline_df = run_esm2_ridge_baseline(ds, cfg, fold)
            baseline_df.to_csv(results_dir / f"baseline_{fold}.csv", index=False)
            print("\n=== Baseline (ESM-2 ridge) ===")
            print(aggregate(baseline_df).to_string(index=False))
        except Exception as e:
            print(f"[warn] baseline skipped ({type(e).__name__}: {e}). "
                  f"Run with PLM deps + GPU for the baseline.")

    # --- AbDev-X ensemble ---
    base_heads = [h for h in cfg["model"]["heads"] if h in ("ridge", "lgbm")]
    mt_tasks = None
    if args.multitask:
        mt_tasks = props + [t for t in ds.aux_target_cols if t not in props]

    def factory():
        return StackingRegressor(base_heads, random_state=cfg["model"]["random_state"],
                                 use_multitask=bool(args.multitask))

    model_df = evaluate_model(ds, X, factory, props, fold, mt_tasks=mt_tasks)
    model_df.to_csv(results_dir / f"abdevx_{fold}.csv", index=False)
    print("\n=== AbDev-X (fused stacking ensemble) ===")
    print(aggregate(model_df).to_string(index=False))

    if baseline_df is not None:
        cmp = compare(baseline_df, model_df)
        cmp.to_csv(results_dir / f"comparison_{fold}.csv", index=False)
        print("\n=== Comparison (model - baseline) ===")
        print(cmp.to_string(index=False))
        print(f"\nWins on {int(cmp['wins'].sum())}/{len(cmp)} properties.")


if __name__ == "__main__":
    main()
