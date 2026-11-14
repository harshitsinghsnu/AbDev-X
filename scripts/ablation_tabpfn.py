"""Ablation: compare AbDev-X with and without TabPFN as a stacking head.

Usage:
    python scripts/ablation_tabpfn.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from abdev.data.dataset import load_gdpa1
from abdev.pipeline import build_features
from abdev.models.ensemble import StackingRegressor
from abdev.evaluate import evaluate_model, aggregate

CONFIG = ROOT / "configs" / "default.yaml"
PROPS = ["HIC", "PR_CHO", "AC-SINS_pH7.4", "Tm2", "Titer"]
FOLD = "cluster_isotype"
N_SEEDS = 5


def make_factory(base_names, use_mt, seed_offset=0):
    def factory():
        return StackingRegressor(
            base_names=[n for n in base_names if n != "multitask_mlp"],
            random_state=42 + seed_offset,
            use_multitask=use_mt,
        )
    return factory


def multi_seed_evaluate(ds, X, base_names, use_mt, props, n_seeds=N_SEEDS):
    """Average OOF predictions over multiple seeds, then compute Spearman."""
    from abdev.data.splits import fold_iter
    from abdev.metrics import spearman

    mt_tasks = props if use_mt else None
    Y_multi = np.column_stack([ds.y(t) for t in props]) if use_mt else None

    records = []
    for prop in props:
        y_all = ds.y(prop)
        for fid, tr, te in fold_iter(ds, prop, FOLD):
            seed_preds = []
            for s in range(n_seeds):
                model = StackingRegressor(
                    base_names=[n for n in base_names if n != "multitask_mlp"],
                    random_state=42 + s,
                    use_multitask=use_mt,
                )
                if use_mt:
                    model.fit(X[tr], y_all[tr], Y_multi=Y_multi[tr],
                              task_index=props.index(prop))
                else:
                    model.fit(X[tr], y_all[tr])
                seed_preds.append(model.predict(X[te]))
            avg_pred = np.mean(seed_preds, axis=0)
            records.append({"property": prop, "fold": fid,
                            "spearman": spearman(y_all[te], avg_pred),
                            "n_test": len(te)})
    return pd.DataFrame(records)


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    print("Loading GDPa1...")
    ds = load_gdpa1(str(CONFIG))
    print("Building features...")
    fm = build_features(ds, cfg)
    X = fm.X.to_numpy(dtype=float)

    experiments = {
        "current (ridge+lgbm+mlp)": (["ridge", "lgbm", "multitask_mlp"], True),
        "+ tabpfn":                  (["ridge", "lgbm", "multitask_mlp", "tabpfn"], True),
        "tabpfn only":               (["tabpfn"], False),
    }

    rows = []
    for label, (heads, use_mt) in experiments.items():
        print(f"\n=== {label} ===")
        df = multi_seed_evaluate(ds, X, heads, use_mt, PROPS)
        agg = aggregate(df)
        for _, row in agg.iterrows():
            print(f"  {row['property']}: {row['spearman_mean']:.3f} ± {row['spearman_std']:.3f}")
        mean_r = agg["spearman_mean"].mean()
        print(f"  mean rho: {mean_r:.3f}")
        row_dict = {"config": label, "mean": mean_r}
        row_dict.update(dict(zip(agg["property"], agg["spearman_mean"])))
        rows.append(row_dict)

    out = ROOT / "results" / "ablation_tabpfn.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
