"""Build a leaderboard-format prediction file for the GDPa1 held-out test set.

The private 80-antibody test set is not public, so this trains AbDev-X on ALL public GDPa1
antibodies and emits predictions for a user-supplied test sequence CSV (heavy/light columns),
in the column layout the HuggingFace leaderboard expects. The user submits the result; only
then is a private-holdout SOTA claim verifiable.

    python scripts/make_submission.py --test path/to/test_sequences.csv --out results/submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abdev.data.dataset import load_gdpa1, resolve_column
from abdev.pipeline import build_features
from abdev.features.descriptors import build_descriptor_matrix
from abdev.features.plm import build_plm_matrix
from abdev.models.ensemble import StackingRegressor


def _features_for(heavy, light, cfg):
    parts = []
    if cfg["features"].get("use_descriptors", True):
        parts.append(build_descriptor_matrix(heavy, light, ph=cfg["features"].get("ph", 7.4)))
    if cfg["features"].get("use_esm2") or cfg["features"].get("use_ablang2"):
        plm = build_plm_matrix(heavy, light, cfg)
        if plm.shape[1]:
            parts.append(plm)
    X = pd.concat(parts, axis=1).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--test", required=True, help="CSV with heavy/light sequence columns")
    ap.add_argument("--out", default="results/submission.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    ds = load_gdpa1(args.config)
    fm = build_features(ds, cfg)
    train_cols = fm.names
    Xtr = fm.X

    test = pd.read_csv(args.test)
    h = resolve_column(test, cfg["data"]["heavy_aliases"], what="heavy")
    l = resolve_column(test, cfg["data"]["light_aliases"], what="light")
    Xte = _features_for(test[h].astype(str), test[l].astype(str), cfg)
    Xte = Xte.reindex(columns=train_cols, fill_value=0.0)  # align to train schema

    out = {}
    for prop in ds.target_cols:
        y = ds.y(prop)
        mask = ~pd.isna(y)
        model = StackingRegressor([h for h in cfg["model"]["heads"] if h in ("ridge", "lgbm")],
                                  random_state=cfg["model"]["random_state"])
        model.fit(Xtr.to_numpy()[mask], y[mask])
        out[prop] = model.predict(Xte.to_numpy())

    sub = pd.DataFrame(out)
    id_col = resolve_column(test, cfg["data"]["id_aliases"], required=False, what="id")
    if id_col:
        sub.insert(0, id_col, test[id_col].values)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"Wrote {len(sub)} predictions x {len(out)} properties -> {args.out}")


if __name__ == "__main__":
    main()
