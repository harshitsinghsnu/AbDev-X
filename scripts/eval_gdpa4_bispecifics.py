"""Evaluate AbDev-X on GDPa4 bispecifics (N3) using a parent-arm composition strategy.

Strategy: predict developability for each parent arm (VH1/VL1 and VH2/VL2) using
AbDev-X trained on GDPa1. Combine arm predictions via average. Compare to measured
bispecific values from the GDPa4 tall assay table.

This is a zero-shot transfer test: AbDev-X was never trained on bispecifics.

Usage:
    python scripts/eval_gdpa4_bispecifics.py

Outputs to results/heldout_gdpa4_bispecifics/:
    arm_predictions.csv      per-arm predictions
    bispecific_predictions.csv  averaged arm predictions
    spearman_summary.csv
    bootstrap_ci.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.pipeline import build_features
from abdev.features.descriptors import build_descriptor_matrix
from abdev.features.plm import build_plm_matrix
from abdev.models.ensemble import StackingRegressor
from abdev.models.heads import make_ridge


def spearman_rho(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def bootstrap_ci(y_true, y_pred, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    n_obs = len(yt)
    if n_obs < 3:
        return float("nan"), float("nan")
    rhos = [float(spearmanr(yt[rng.integers(0, n_obs, n_obs)],
                             yp[rng.integers(0, n_obs, n_obs)])[0])
            for _ in range(n)]
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def build_test_features(heavy, light, fm_train, cfg):
    parts = []
    if cfg["features"].get("use_descriptors", True):
        parts.append(build_descriptor_matrix(heavy, light, ph=cfg["features"].get("ph", 7.4)))
    if cfg["features"].get("use_esm2") or cfg["features"].get("use_ablang2"):
        plm = build_plm_matrix(heavy, light, cfg)
        if plm.shape[1]:
            parts.append(plm)
    Xte = pd.concat(parts, axis=1).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    return Xte.reindex(columns=fm_train.names, fill_value=0.0).to_numpy(dtype=float)


def predict_arm(ds, fm_train, Xte, props, cfg, n_seeds=15):
    """Train AbDev-X on GDPa1 and predict on one arm's test features."""
    Xtr = fm_train.X.to_numpy(dtype=float)
    heads = [h for h in cfg["model"]["heads"] if h in ("ridge", "lgbm")]
    preds = {}
    for prop in props:
        if prop not in ds.target_cols:
            continue
        y = ds.y(prop)
        mask = ~np.isnan(y)
        arm_preds = []
        for seed in range(n_seeds):
            model = StackingRegressor(heads, random_state=seed)
            model.fit(Xtr[mask], y[mask])
            arm_preds.append(model.predict(Xte))
        preds[prop] = np.mean(arm_preds, axis=0)
    return preds


def main():
    cfg_path = ROOT / "configs/default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    results_dir = ROOT / "results/heldout_gdpa4_bispecifics"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load training data (GDPa1)
    print("Loading GDPa1 (train)...")
    ds = load_gdpa1(str(cfg_path))
    fm_train = build_features(ds, cfg)
    print(f"GDPa1: {len(ds.df)} antibodies | features: {fm_train.X.shape}")

    # Load bispecific test data
    bsab_path = ROOT / "data/processed/gdpa4_bispecifics.csv"
    bsabs = pd.read_csv(bsab_path)
    print(f"\nGDPa4 bispecifics: {bsabs.shape}")

    # Properties to evaluate (overlap with GDPa1)
    eval_props = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "Tm2"]
    gdpa1_prop_map = {
        "HIC": "HIC",
        "AC-SINS_pH7.4": "AC-SINS_pH7.4",
        "PR_CHO": "PR_CHO",
        "Tm2": "Tm2",
    }
    # Columns in bsab CSV
    bsab_target_cols = {
        "HIC": "HIC",
        "AC-SINS_pH7.4": "AC-SINS",
        "PR_CHO": "PR_CHO",
        "Tm2": "Tm2",
    }

    # Build features for arm 1 and arm 2
    print("\nBuilding features for arm 1...")
    heavy1 = bsabs["vh_protein_sequence-1"].astype(str).str.strip()
    light1 = bsabs["vl_protein_sequence-1"].astype(str).str.strip()
    Xte1 = build_test_features(heavy1, light1, fm_train, cfg)

    print("Building features for arm 2...")
    heavy2 = bsabs["vh_protein_sequence-2"].astype(str).str.strip()
    light2 = bsabs["vl_protein_sequence-2"].astype(str).str.strip()
    Xte2 = build_test_features(heavy2, light2, fm_train, cfg)

    # Predict for each arm
    print("\nPredicting arm 1...")
    preds1 = predict_arm(ds, fm_train, Xte1, list(ds.target_cols.keys()), cfg)
    print("Predicting arm 2...")
    preds2 = predict_arm(ds, fm_train, Xte2, list(ds.target_cols.keys()), cfg)

    # Average arm predictions (bispecific composition strategy)
    preds_avg = {p: (preds1[p] + preds2[p]) / 2
                 for p in preds1 if p in preds2}
    # Minimum arm strategy (pessimistic)
    preds_min = {p: np.minimum(preds1[p], preds2[p])
                 for p in preds1 if p in preds2}

    # Save arm predictions
    arm_df = pd.DataFrame({"bsab_name": bsabs.get("antibody_name", bsabs.index),
                            "arm1_name": bsabs.get("antibody_name-1", ""),
                            "arm2_name": bsabs.get("antibody_name-2", "")})
    for p in preds1:
        arm_df[f"arm1_{p}"] = preds1[p]
        arm_df[f"arm2_{p}"] = preds2[p]
        arm_df[f"avg_{p}"] = preds_avg.get(p, float("nan"))
    arm_df.to_csv(results_dir / "arm_predictions.csv", index=False)

    # Evaluate against measured bispecific values
    print("\n=== Bispecific held-out Spearman rho ===")
    summary_rows = []
    ci_rows = []

    for strategy, preds_dict in [("AbDev-X-avg", preds_avg), ("AbDev-X-min", preds_min)]:
        row = {"method": strategy}
        for eval_prop, gdpa1_prop in gdpa1_prop_map.items():
            bsab_col = bsab_target_cols.get(eval_prop, eval_prop)
            if bsab_col not in bsabs.columns or gdpa1_prop not in preds_dict:
                row[eval_prop] = float("nan")
                continue
            yt = bsabs[bsab_col].to_numpy(dtype=float)
            yp = preds_dict[gdpa1_prop]
            rho = spearman_rho(yt, yp)
            lo, hi = bootstrap_ci(yt, yp)
            row[eval_prop] = rho
            ci_rows.append({"method": strategy, "property": eval_prop,
                            "rho": rho, "ci_low": lo, "ci_high": hi})
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    ci_df = pd.DataFrame(ci_rows)
    summary_df.to_csv(results_dir / "spearman_summary.csv", index=False)
    ci_df.to_csv(results_dir / "bootstrap_ci.csv", index=False)

    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n95% CI:")
    print(ci_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nOutputs -> {results_dir}")


if __name__ == "__main__":
    main()
