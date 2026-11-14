"""End-to-end wiring test on synthetic data (no network, no GPU, no PLM/LGBM required).

Generates a tiny fake GDPa1-like frame, builds descriptor features, and runs the CV harness
with a ridge stacking ensemble to prove data->features->folds->scoring is correctly wired.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abdev.data.dataset import GDPa1
from abdev.evaluate import evaluate_model, aggregate
from abdev.features.descriptors import build_descriptor_matrix
from abdev.models.ensemble import StackingRegressor

AAs = "ACDEFGHIKLMNPQRSTVWY"


def _rand_seq(rng, n):
    return "".join(rng.choice(list(AAs), size=n))


def _make_dataset(n=60, seed=0):
    rng = np.random.default_rng(seed)
    heavy = [_rand_seq(rng, 118) for _ in range(n)]
    light = [_rand_seq(rng, 107) for _ in range(n)]
    # Target correlated with heavy-chain aromatic content so a model can learn signal.
    arom = np.array([sum(c in "FWY" for c in h) for h in heavy], dtype=float)
    y = arom + rng.normal(0, 1.0, n)
    folds = rng.integers(0, 5, n)
    df = pd.DataFrame({"antibody_name": [f"ab{i}" for i in range(n)],
                       "heavy_chain": heavy, "light_chain": light,
                       "HIC": y, "random_fold": folds})
    return GDPa1(df=df, id_col="antibody_name", heavy_col="heavy_chain",
                 light_col="light_chain", fold_cols={"random": "random_fold"},
                 target_cols={"HIC": "HIC"}, directions={"HIC": "lower"})


def test_end_to_end_ridge():
    ds = _make_dataset()
    X = build_descriptor_matrix(ds.heavy, ds.light).to_numpy(dtype=float)
    factory = lambda: StackingRegressor(["ridge"], random_state=0)
    res = evaluate_model(ds, X, factory, ["HIC"], "random")
    agg = aggregate(res)
    print(agg.to_string(index=False))
    assert res["fold"].nunique() >= 2
    assert np.isfinite(agg.loc[0, "spearman_mean"])


if __name__ == "__main__":
    test_end_to_end_ridge()
    print("Synthetic end-to-end pipeline test passed.")
