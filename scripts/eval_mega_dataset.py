"""Evaluate AbDev-X trained on progressively larger pooled datasets vs GDPa3 held-out.

Conditions compared (GDPa3, n=80):
  A) GDPa1-only          (n=244, IgG)
  B) mega_ginkgo         (n=430, IgG+VHH-Fc, raw values)
  C) mega_full           (n=591, Ginkgo+FLAb, per-source z-scored)
  D) mega_full + DOTAD   (n=591+DOTAD, if data/processed/dotad_clean.csv exists)

Per-source z-scoring is applied when mixing datasets with different assay scales
(FLAb uses pre-z-scored values; Ginkgo sources are raw). All targets are z-scored
within source before pooling so the model is always predicting z-scores, which is
rank-preserving and compatible with Spearman-rho evaluation on GDPa3.

Usage:
    python scripts/eval_mega_dataset.py
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
from abdev.pipeline import build_features

CONFIG     = ROOT / "configs" / "default.yaml"
GDPA3_CFG  = ROOT / "configs" / "gdpa3_eval.yaml"
PROPS      = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
PROP_ALIAS = {"AC-SINS": "AC-SINS_pH7.4"}
N_SEEDS    = 5

# ── utilities ─────────────────────────────────────────────────────────────────

def spearman_rho(yt, yp) -> float:
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    if yt.shape != yp.shape or len(yt) == 0:
        return float("nan")
    m = ~(np.isnan(yt) | np.isnan(yp))
    return float(spearmanr(yt[m], yp[m])[0]) if m.sum() > 2 else float("nan")


def bootstrap_ci(yt, yp, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    m = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[m], yp[m]
    if len(yt) < 3:
        return float("nan"), float("nan")
    boot = [spearmanr(yt[ix := rng.integers(0, len(yt), len(yt))], yp[ix])[0]
            for _ in range(n)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def safe_lookup(preds: dict, prop: str) -> np.ndarray | None:
    if prop in preds:
        return preds[prop]
    alt = PROP_ALIAS.get(prop)
    if alt and alt in preds:
        return preds[alt]
    return None


def build_features_df(heavy: pd.Series, light: pd.Series, cfg: dict) -> pd.DataFrame:
    heavy_s = pd.Series(list(heavy)).reset_index(drop=True)
    light_s = pd.Series(list(light)).reset_index(drop=True)
    parts = []
    if cfg["features"].get("use_descriptors", True):
        parts.append(build_descriptor_matrix(heavy_s, light_s,
                                             ph=cfg["features"].get("ph", 7.4)))
    if cfg["features"].get("use_esm2") or cfg["features"].get("use_ablang2"):
        plm = build_plm_matrix(heavy_s, light_s, cfg)
        if plm.shape[1]:
            parts.append(plm)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def load_gdpa3(cfg: dict) -> tuple[pd.Series, pd.Series, dict]:
    hcfg = cfg["heldout"]
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
    return heavy, light, targets


def zscore_per_source(df: pd.DataFrame, props: list[str]) -> pd.DataFrame:
    """Z-score each property within each source group (in-place copy)."""
    df = df.copy()
    for src, grp in df.groupby("source"):
        for p in props:
            if p not in df.columns:
                continue
            vals = df.loc[grp.index, p]
            std = vals.std(skipna=True)
            mu = vals.mean(skipna=True)
            if std > 1e-9:
                df.loc[grp.index, p] = (vals - mu) / std
    return df


def run_abdevx(Xtr: np.ndarray, y_train: dict, Xte: np.ndarray,
               heads: list, use_mt: bool, props: list,
               n_seeds: int = N_SEEDS) -> dict:
    Y_multi = np.column_stack(
        [y_train.get(p, np.full(Xtr.shape[0], np.nan)) for p in props]
    )
    seed_preds: dict = {p: [] for p in props}
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    with open(GDPA3_CFG) as f:
        gdpa3_cfg = yaml.safe_load(f)

    heads  = list(cfg["model"]["heads"])
    use_mt = "multitask_mlp" in heads

    print("Loading GDPa3 held-out (n=80)...")
    gdpa3_h, gdpa3_l, gdpa3_targets = load_gdpa3(gdpa3_cfg)
    print(f"  Properties: {list(gdpa3_targets.keys())}")

    print("Building GDPa3 features...")
    Xte_df = build_features_df(gdpa3_h, gdpa3_l, cfg)
    print(f"  Xte: {Xte_df.shape}")

    results: dict[str, dict] = {}
    preds_all: dict[str, dict] = {}

    # ── A: GDPa1 only ─────────────────────────────────────────────────────────
    print("\n=== A: GDPa1-only (n=244) ===")
    ds = load_gdpa1(str(CONFIG))
    fm = build_features(ds, cfg)
    Xtr_g1 = fm.X.to_numpy(dtype=float)
    Xte_g1 = Xte_df.reindex(columns=fm.names, fill_value=0.0).to_numpy(dtype=float)
    y_g1 = {p: ds.y(p) for p in PROPS if p in ds.target_cols}
    preds_a = run_abdevx(Xtr_g1, y_g1, Xte_g1, heads, use_mt, PROPS)
    preds_all["GDPa1-only (n=244)"] = preds_a
    results["GDPa1-only (n=244)"] = {
        p: spearman_rho(gdpa3_targets[p], v)
        for p in gdpa3_targets
        if (v := safe_lookup(preds_a, p)) is not None
    }

    def eval_mega(label: str, csv_path: Path) -> None:
        """Train on mega CSV (z-scored per source), eval on GDPa3."""
        print(f"\n=== {label} ===")
        df = pd.read_csv(csv_path)
        print(f"  Loaded: {len(df)} rows, sources: {dict(df['source'].value_counts())}")

        # per-source z-score so FLAb (pre-z) and Ginkgo (raw) are on same scale
        df = zscore_per_source(df, PROPS)

        heavy = df["vhn"].astype(str).str.strip()
        light = df["vl"].fillna("").astype(str).str.replace("nan", "", regex=False).str.strip()

        print("  Building train features...")
        Xtr_df = build_features_df(heavy, light, cfg)
        Xtr    = Xtr_df.to_numpy(dtype=float)
        Xte    = Xte_df.reindex(columns=Xtr_df.columns, fill_value=0.0).to_numpy(dtype=float)

        y_train: dict = {}
        for p in PROPS:
            if p in df.columns:
                y_train[p] = df[p].to_numpy(dtype=float)

        preds = run_abdevx(Xtr, y_train, Xte, heads, use_mt, PROPS)
        preds_all[label] = preds
        results[label] = {
            p: spearman_rho(gdpa3_targets[p], v)
            for p in gdpa3_targets
            if (v := safe_lookup(preds, p)) is not None
        }

    # ── B: mega_ginkgo (G1+G2.1+G5, n=430) ──────────────────────────────────
    eval_mega("mega_ginkgo (n=430)", ROOT / "data" / "processed" / "mega_ginkgo.csv")

    # ── C: mega_full (Ginkgo+FLAb, n=591) ────────────────────────────────────
    eval_mega("mega_full (n=591)", ROOT / "data" / "processed" / "mega_full.csv")

    # ── D: mega_full + DOTAD (if available) ──────────────────────────────────
    dotad_path = ROOT / "data" / "processed" / "dotad_clean.csv"
    if dotad_path.exists():
        eval_mega(f"mega+DOTAD", ROOT / "data" / "processed" / "mega_dotad.csv")
    else:
        print("\n[Skip D] DOTAD not yet processed — run scripts/process_dotad.py first")

    # ── print table ───────────────────────────────────────────────────────────
    props_avail = list(gdpa3_targets.keys())
    print("\n" + "=" * 65)
    print("RESULTS: GDPa3 Spearman rho")
    print("=" * 65)
    header = f"{'Condition':<35}" + "".join(f"{p[:7]:>9}" for p in props_avail) + f"{'Mean':>7}"
    print(header)
    print("-" * len(header))
    for cond, rhos in results.items():
        vals = [rhos.get(p, float("nan")) for p in props_avail]
        mean = float(np.nanmean(vals))
        row = f"{cond:<35}" + "".join(f"{v:>9.3f}" for v in vals) + f"{mean:>7.3f}"
        print(row)

    # ── bootstrap CIs for best condition ─────────────────────────────────────
    best = max(results, key=lambda k: np.nanmean(list(results[k].values())))
    print(f"\n95% bootstrap CI for '{best}':")
    for p in props_avail:
        yt = gdpa3_targets.get(p)
        yp = safe_lookup(preds_all[best], p)
        if yt is not None and yp is not None and len(yp) == len(yt):
            lo, hi = bootstrap_ci(yt, yp)
            print(f"  {p}: {spearman_rho(yt, yp):.3f} [{lo:.2f}, {hi:.2f}]")

    # ── save CSV ──────────────────────────────────────────────────────────────
    out_dir = ROOT / "results" / "mega_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond, rhos in results.items():
        vals = [rhos.get(p, float("nan")) for p in props_avail]
        rows.append({"condition": cond, **dict(zip(props_avail, vals)),
                     "mean": float(np.nanmean(vals))})
    out_csv = out_dir / "spearman_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
