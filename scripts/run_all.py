"""One-shot research-article pipeline: data -> features -> CV -> comparison -> SHAP.

Prerequisite: authenticate to the gated GDPa1 dataset once (see README), then:

    python scripts/run_all.py --config configs/default.yaml --fold cluster_isotype

Produces, in results/:
  * baseline_<fold>.csv, abdevx_<fold>.csv         per-fold Spearman
  * comparison_<fold>.csv                          per-property: baseline, AbDev-X, delta,
                                                    + published competition test-set reference
  * explain/shap_<property>.{csv,png}, block_importance_<property>.csv
This is the table + figures for the manuscript. No leaderboard submission is made.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.pipeline import build_features
from abdev.evaluate import evaluate_model, aggregate, compare
from abdev.models.ensemble import StackingRegressor
from abdev.models.heads import LGBMHead, make_ridge
from abdev.baselines import run_esm2_ridge_baseline
from abdev.reference import add_reference, REFERENCE_NOTE
from abdev.explain import shap_global, block_importance, liability_card


def ensure_data(config: str) -> None:
    raw = Path(yaml.safe_load(open(config))["data"]["raw_dir"])
    if (raw / "ACTIVE_CSV.txt").exists() or any(raw.glob("*.csv")):
        return
    print("No local CSV; attempting download (requires HF auth)...")
    subprocess.run([sys.executable, "-m", "abdev.data.download", "--config", config],
                   check=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fold", default=None)
    ap.add_argument("--multitask", action="store_true", default=True)
    ap.add_argument("--no-multitask", dest="multitask", action="store_false")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    fold = args.fold or cfg["data"]["primary_fold"]
    results = Path(cfg["eval"]["results_dir"]); results.mkdir(parents=True, exist_ok=True)
    explain_dir = results / "explain"

    ensure_data(args.config)
    ds = load_gdpa1(args.config)
    props = list(ds.target_cols.keys())
    print(f"\nAntibodies: {len(ds.df)} | properties: {props} | fold: {fold} -> {ds.fold_cols.get(fold)}")
    print(ds.summary().to_string(index=False))

    fm = build_features(ds, cfg)
    print(f"\nFeature matrix: {fm.X.shape[0]} x {fm.X.shape[1]} "
          f"blocks={ {k: len(v) for k, v in fm.blocks.items()} }")
    X = fm.values

    # Baseline on identical folds
    baseline_df = run_esm2_ridge_baseline(ds, cfg, fold)
    baseline_df.to_csv(results / f"baseline_{fold}.csv", index=False)
    print("\n=== Baseline (ESM-2 ridge) ===\n" + aggregate(baseline_df).to_string(index=False))

    # AbDev-X
    base_heads = [h for h in cfg["model"]["heads"] if h in ("ridge", "lgbm")]
    mt_tasks = (props + [t for t in ds.aux_target_cols if t not in props]) if args.multitask else None
    factory = lambda: StackingRegressor(base_heads, random_state=cfg["model"]["random_state"],
                                        use_multitask=bool(args.multitask))
    model_df = evaluate_model(ds, X, factory, props, fold, mt_tasks=mt_tasks)
    model_df.to_csv(results / f"abdevx_{fold}.csv", index=False)
    print("\n=== AbDev-X (fused stacking ensemble) ===\n" + aggregate(model_df).to_string(index=False))

    # Comparison table for the manuscript
    cmp = add_reference(compare(baseline_df, model_df))
    cmp.to_csv(results / f"comparison_{fold}.csv", index=False)
    print("\n=== Manuscript comparison table ===\n" + cmp.to_string(index=False))
    print(f"\nNote: {REFERENCE_NOTE}")
    print(f"\nAbDev-X beats in-repo baseline on {int(cmp['wins'].sum())}/{len(cmp)} properties.")

    # Explainability (LightGBM on descriptors, refit on full data, per property)
    desc_cols = fm.blocks.get("descriptors", [])
    if desc_cols:
        Xd = ds.df.index.to_numpy()  # placeholder to keep alignment explicit
        Xdesc = fm.X[desc_cols].to_numpy(dtype=float)
        for prop in props:
            y = ds.y(prop); mask = ~np.isnan(y)
            head = LGBMHead(random_state=cfg["model"]["random_state"]).fit(Xdesc[mask], y[mask])
            sdf = shap_global(head, Xdesc[mask], desc_cols, str(explain_dir), prop)
            if sdf is not None:
                bi = block_importance(
                    pd.concat([sdf, pd.DataFrame({"feature": fm.names,
                              "mean_abs_shap": 0.0})]).drop_duplicates("feature"),
                    fm.blocks)
                bi.to_csv(explain_dir / f"block_importance_{prop}.csv", index=False)
        print(f"\nSHAP + block-importance written to {explain_dir}")

    # Per-antibody liability cards
    cards = [liability_card(h, l, n) for h, l, n in
             zip(ds.heavy, ds.light, ds.df[ds.id_col].astype(str))]
    pd.DataFrame(cards).to_json(results / "liability_cards.json", orient="records", indent=2)
    print(f"Liability cards -> {results / 'liability_cards.json'}")


if __name__ == "__main__":
    main()
