"""Option 1: Train AbDev-X on combined GDPa1+GDPa2.1+GDPa5, evaluate on GDPa3.

Comparisons:
  A) GDPa1-only baseline (current AbDev-X)
  B) Combined IgG+VHH-Fc training (Option 1)
  C) Combined + format feature (format-aware variant)

Usage:
    python scripts/eval_combined_training.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from abdev.data.dataset import load_gdpa1, GDPa1, resolve_column
from abdev.features.descriptors import build_descriptor_matrix
from abdev.features.plm import build_plm_matrix
from abdev.models.ensemble import StackingRegressor
from abdev.models.heads import make_ridge
from abdev.pipeline import build_features

CONFIG = ROOT / "configs" / "default.yaml"
GDPA3_CONFIG = ROOT / "configs" / "gdpa3_eval.yaml"
COMBINED_CSV = ROOT / "data" / "processed" / "combined_train.csv"
PROPS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
# Map GDPa3 target key -> PROPS key used internally
PROP_ALIAS = {"AC-SINS": "AC-SINS_pH7.4"}
N_SEEDS = 5


def spearman_rho(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    if yt.shape != yp.shape or len(yt) == 0:
        return float("nan")
    m = ~(np.isnan(yt) | np.isnan(yp))
    return float(spearmanr(yt[m], yp[m])[0]) if m.sum() > 2 else float("nan")


def safe_lookup(preds: dict, prop: str) -> np.ndarray | None:
    """Look up predictions by prop name, trying PROP_ALIAS if not found."""
    if prop in preds:
        return preds[prop]
    alt = PROP_ALIAS.get(prop)
    if alt and alt in preds:
        return preds[alt]
    return None


def bootstrap_ci(yt, yp, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    m = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[m], yp[m]
    if len(yt) < 3:
        return float("nan"), float("nan")
    boot = [spearmanr(yt[ix := rng.integers(0, len(yt), len(yt))], yp[ix])[0] for _ in range(n)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def build_features_df(heavy, light, cfg: dict) -> pd.DataFrame:
    """Build descriptor + PLM features for a sequence list/Series."""
    # build_descriptor_matrix expects pd.Series; normalise inputs
    heavy_s = pd.Series(list(heavy)).reset_index(drop=True)
    light_s = pd.Series(list(light)).reset_index(drop=True)
    parts = []
    if cfg["features"].get("use_descriptors", True):
        parts.append(build_descriptor_matrix(heavy_s, light_s, ph=cfg["features"].get("ph", 7.4)))
    if cfg["features"].get("use_esm2") or cfg["features"].get("use_ablang2"):
        plm = build_plm_matrix(heavy_s, light_s, cfg)
        if plm.shape[1]:
            parts.append(plm)
    X = pd.concat(parts, axis=1).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    return X


def load_gdpa3(gdpa3_cfg: dict) -> tuple:
    """Return (df, heavy, light, ids, targets_dict)."""
    hcfg = gdpa3_cfg["heldout"]
    df = pd.read_csv(ROOT / hcfg["csv_path"])
    h = resolve_column(df, hcfg["heavy_aliases"])
    l = resolve_column(df, hcfg["light_aliases"])
    heavy = df[h].astype(str).str.strip()
    light = df[l].fillna("").astype(str).str.replace("nan", "", regex=False).str.strip()
    targets = {}
    for prop, spec in hcfg["targets"].items():
        col = resolve_column(df, spec["column"], required=False, what=prop)
        if col:
            targets[prop] = df[col].to_numpy(dtype=float)
    return df, heavy, light, targets


def run_abdevx(Xtr: np.ndarray, y_train: dict, Xte: np.ndarray,
               heads: list, use_mt: bool, props: list, n_seeds: int = N_SEEDS) -> dict:
    """Train multi-seed AbDev-X on (Xtr, y_train), predict on Xte."""
    Y_multi = np.column_stack([y_train.get(p, np.full(Xtr.shape[0], np.nan)) for p in props])
    seed_preds = {p: [] for p in props}
    for s in range(n_seeds):
        for i, prop in enumerate(props):
            y = y_train.get(prop, np.full(Xtr.shape[0], np.nan))
            mask = ~np.isnan(y)
            if mask.sum() < 5:
                continue
            model = StackingRegressor(
                base_names=[h for h in heads if h != "multitask_mlp"],
                random_state=42 + s,
                use_multitask=use_mt,
            )
            if use_mt:
                model.fit(Xtr[mask], y[mask], Y_multi=Y_multi[mask], task_index=i)
            else:
                model.fit(Xtr[mask], y[mask])
            seed_preds[prop].append(model.predict(Xte))
    return {p: np.mean(v, axis=0) for p, v in seed_preds.items() if v}


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    with open(GDPA3_CONFIG) as f:
        gdpa3_cfg = yaml.safe_load(f)

    heads = [h for h in cfg["model"]["heads"]]
    use_mt = "multitask_mlp" in heads

    # ── Load GDPa3 test set ──────────────────────────────────────────────────
    print("Loading GDPa3 held-out...")
    _, gdpa3_h, gdpa3_l, gdpa3_targets = load_gdpa3(gdpa3_cfg)
    print(f"  GDPa3: {len(gdpa3_h)} antibodies")
    print(f"  Properties: {list(gdpa3_targets.keys())}")

    print("\nBuilding GDPa3 features...")
    Xte_df = build_features_df(gdpa3_h, gdpa3_l, cfg)
    print(f"  Xte shape: {Xte_df.shape}")

    results = {}

    # ── Option A: GDPa1-only (current baseline) ──────────────────────────────
    print("\n=== Option A: GDPa1 only (current) ===")
    ds = load_gdpa1(str(CONFIG))
    fm_train = build_features(ds, cfg)
    Xtr_g1 = fm_train.X.to_numpy(dtype=float)
    Xte_df = build_features_df(gdpa3_h, gdpa3_l, cfg)
    Xte_aligned = Xte_df.reindex(columns=fm_train.names, fill_value=0.0).to_numpy(dtype=float)
    y_train_g1 = {p: ds.y(p) for p in PROPS if p in ds.target_cols}
    preds_a = run_abdevx(Xtr_g1, y_train_g1, Xte_aligned, heads, use_mt, PROPS)
    results["GDPa1-only"] = {
        p: spearman_rho(gdpa3_targets[p], v)
        for p in gdpa3_targets
        if (v := safe_lookup(preds_a, p)) is not None
    }

    # ── Option B: Combined IgG+VHH training ──────────────────────────────────
    if not COMBINED_CSV.exists():
        print("\nBuilding combined dataset first...")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_combined_dataset.py")], check=True)

    print("\n=== Option B: Combined GDPa1+GDPa2.1+GDPa5 ===")
    combined = pd.read_csv(COMBINED_CSV)
    print(f"  Combined: {len(combined)} antibodies")
    c_heavy = combined["vhn"].astype(str).str.strip()
    c_light = combined["vl"].fillna("").astype(str).str.replace("nan", "", regex=False).str.strip()

    print("  Building combined features...")
    Xtr_combined_df = build_features_df(c_heavy, c_light, cfg)
    Xtr_combined = Xtr_combined_df.to_numpy(dtype=float)
    Xte_combined = Xte_df.reindex(columns=Xtr_combined_df.columns, fill_value=0.0).to_numpy(dtype=float)
    y_train_comb = {}
    for prop in PROPS:
        col = {"AC-SINS_pH7.4": "AC-SINS_pH7.4"}.get(prop, prop)
        if col in combined.columns:
            y_train_comb[prop] = combined[col].to_numpy(dtype=float)
    preds_b = run_abdevx(Xtr_combined, y_train_comb, Xte_combined, heads, use_mt, PROPS)
    results["Combined (GDPa1+2.1+5)"] = {
        p: spearman_rho(gdpa3_targets[p], v)
        for p in gdpa3_targets
        if (v := safe_lookup(preds_b, p)) is not None
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS: GDPa3 Spearman rho")
    print("="*60)
    props_avail = list(gdpa3_targets.keys())
    header = f"{'Method':<35}" + "".join(f"{p:>10}" for p in props_avail) + f"{'Mean':>8}"
    print(header)
    print("-"*len(header))
    for method, rhos in results.items():
        vals = [rhos.get(p, float("nan")) for p in props_avail]
        mean = float(np.nanmean(vals))
        row = f"{method:<35}" + "".join(f"{v:>10.3f}" for v in vals) + f"{mean:>8.3f}"
        print(row)

    # Bootstrap CIs for combined
    print("\n95% CI for Combined method:")
    for p in props_avail:
        yt = gdpa3_targets.get(p)
        yp = safe_lookup(preds_b, p)
        if yt is not None and yp is not None and len(yp) == len(yt):
            lo, hi = bootstrap_ci(yt, yp)
            print(f"  {p}: {spearman_rho(yt, yp):.3f} [{lo:.2f}, {hi:.2f}]")

    # Save
    out_dir = ROOT / "results" / "combined_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, rhos in results.items():
        vals = [rhos.get(p, float("nan")) for p in props_avail]
        rows.append({"method": method, **dict(zip(props_avail, vals)),
                     "mean": float(np.nanmean(vals))})
    pd.DataFrame(rows).to_csv(out_dir / "spearman_summary.csv", index=False)
    print(f"\nSaved -> {out_dir}/spearman_summary.csv")


if __name__ == "__main__":
    main()
