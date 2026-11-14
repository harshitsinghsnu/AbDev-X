"""Held-out evaluation: train on GDPa1, predict on GDPa3 (or GDPa4), compare all baselines.

Usage:
    python scripts/eval_heldout.py --dataset gdpa3          # uses configs/gdpa3_eval.yaml
    python scripts/eval_heldout.py --dataset gdpa4          # uses configs/gdpa4_eval.yaml
    python scripts/eval_heldout.py --test path/to/test.csv  # custom held-out CSV

Outputs to results/heldout_<dataset>/:
    predictions.csv          per-antibody predictions from every method
    spearman_summary.csv     method x property Spearman rho
    bootstrap_ci.csv         95% CI via paired bootstrap (n=1000) per method x property
    comparison_table.tex     LaTeX table for manuscript
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1, resolve_column
from abdev.pipeline import build_features
from abdev.features.descriptors import build_descriptor_matrix
from abdev.features.plm import build_plm_matrix, esm2_embeddings
from abdev.models.ensemble import StackingRegressor
from abdev.models.heads import make_ridge


# ──────────────────────────────────────────────────────────────────────────────
# Competition champion reference (GDPa3, from mAbs 2026 paper)
# ──────────────────────────────────────────────────────────────────────────────
CHAMPION = {
    "HIC":    {"rho": 0.708, "method": "structural SAP"},
    "Tm2":    {"rho": 0.392, "method": "germline / light-chain subtype"},
    "AC-SINS": {"rho": 0.356, "method": "AbLang2 + FLAb augmentation"},
    "PR_CHO": {"rho": 0.337, "method": "AbLang2 + FLAb augmentation"},
    "Titer":  {"rho": 0.310, "method": "AbLang2 + FLAb augmentation"},
}


def spearman_rho(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def bootstrap_ci(y_true, y_pred, n=1000, seed=42):
    """Paired bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    n_obs = len(yt)
    if n_obs < 3:
        return float("nan"), float("nan")
    rhos = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        r, _ = spearmanr(yt[idx], yp[idx])
        rhos.append(float(r))
    rhos = np.array(rhos)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def features_for_test(heavy, light, cfg):
    """Build feature matrix for a test set using same feature config as train."""
    parts = []
    if cfg["features"].get("use_descriptors", True):
        parts.append(build_descriptor_matrix(heavy, light, ph=cfg["features"].get("ph", 7.4)))
    if cfg["features"].get("use_esm2") or cfg["features"].get("use_ablang2"):
        plm = build_plm_matrix(heavy, light, cfg)
        if plm.shape[1]:
            parts.append(plm)
    X = pd.concat(parts, axis=1).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    return X


def load_heldout(heldout_cfg, global_cfg):
    """Load held-out CSV and return (df, heavy_series, light_series, {prop: y_array})."""
    path = Path(heldout_cfg["csv_path"])
    if not path.exists():
        raise FileNotFoundError(f"Held-out CSV not found: {path}")
    df = pd.read_csv(path)
    print(f"[info] loaded held-out: {path.name}: {len(df)} rows x {df.shape[1]} cols")

    h_col = resolve_column(df, heldout_cfg["heavy_aliases"], what="heavy chain")
    l_col = resolve_column(df, heldout_cfg["light_aliases"], what="light chain")
    heavy = df[h_col].astype(str).str.strip()
    light = df[l_col].fillna("").astype(str).str.replace("nan", "", regex=False).str.strip()

    targets = {}
    for prop, spec in heldout_cfg["targets"].items():
        col = resolve_column(df, spec["column"], required=False, what=f"target {prop}")
        if col:
            targets[prop] = df[col].to_numpy(dtype=float)
        else:
            print(f"[warn] target '{prop}' not in held-out CSV; skipping.")

    id_col = resolve_column(df, heldout_cfg["id_aliases"], required=False, what="id")
    ids = df[id_col].astype(str) if id_col else pd.Series(range(len(df)), name="idx")
    return df, heavy, light, ids, targets


# ──────────────────────────────────────────────────────────────────────────────
# Baseline runners (train on GDPa1 full, predict on held-out)
# ──────────────────────────────────────────────────────────────────────────────

def run_esm2_ridge(ds, cfg, Xte_esm2, props):
    """ESM-2 Ridge: train on full GDPa1, predict on held-out."""
    from sklearn.linear_model import RidgeCV
    heavy_light_train = (ds.heavy + ds.light).tolist()
    Xtr = esm2_embeddings(heavy_light_train, cfg["features"]["esm2_model"],
                          pool=("mean",), cache_dir=cfg["data"]["processed_dir"])
    preds = {}
    for prop in props:
        y = ds.y(prop)
        mask = ~np.isnan(y)
        m = make_ridge()
        m.fit(Xtr[mask], y[mask])
        preds[prop] = m.predict(Xte_esm2)
    return preds


def run_descriptors_ridge(ds, fm_train, train_cols, Xte, props):
    """Descriptor-only Ridge (interpretable baseline): train on full GDPa1."""
    desc_cols = fm_train.blocks.get("descriptors", [])
    if not desc_cols:
        return {}
    Xtr = fm_train.X[desc_cols].to_numpy(dtype=float)
    Xte_desc = Xte[desc_cols].to_numpy(dtype=float) if all(c in Xte.columns for c in desc_cols) else None
    if Xte_desc is None:
        return {}
    preds = {}
    for prop in props:
        y = ds.y(prop)
        mask = ~np.isnan(y)
        m = make_ridge()
        m.fit(Xtr[mask], y[mask])
        preds[prop] = m.predict(Xte_desc)
    return preds


def run_ablang2_elasticnet(ds, cfg, Xte_ablang2, props):
    """AbLang2 ElasticNet: train on full GDPa1, predict on held-out."""
    from sklearn.linear_model import ElasticNetCV
    ablang2_cols = [c for c in cfg.get("_ablang2_cols", []) if c]
    if not ablang2_cols:
        return {}
    preds = {}
    for prop in props:
        y = ds.y(prop)
        mask = ~np.isnan(y)
        m = ElasticNetCV(cv=5, max_iter=5000, random_state=42)
        m.fit(Xte_ablang2[mask], y[mask])
        preds[prop] = m.predict(Xte_ablang2)
    return preds


def run_abdevx(ds, cfg, fm_train, Xte, props):
    """AbDev-X full stacking: train on GDPa1, predict on held-out."""
    Xtr = fm_train.X.to_numpy(dtype=float)
    train_cols = fm_train.names
    # Align test columns to train schema
    Xte_aligned = Xte.reindex(columns=train_cols, fill_value=0.0).to_numpy(dtype=float)
    heads = [h for h in cfg["model"]["heads"] if h in ("ridge", "lgbm")]
    preds = {}
    for prop in props:
        y = ds.y(prop)
        mask = ~np.isnan(y)
        model = StackingRegressor(heads, random_state=cfg["model"]["random_state"])
        model.fit(Xtr[mask], y[mask])
        preds[prop] = model.predict(Xte_aligned)
    return preds


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table formatter
# ──────────────────────────────────────────────────────────────────────────────

def to_latex_table(summary_df, ci_df, dataset_name, out_path):
    """Write a LaTeX table: methods x properties with Spearman rho [CI]."""
    props = [c for c in summary_df.columns if c != "method"]
    methods = summary_df["method"].tolist()

    lines = [
        r"\begin{table}[h]\centering\small",
        r"\caption{Held-out Spearman $\rho$ on " + dataset_name +
        r": AbDev-X vs.\ baselines. Bootstrap 95\% CI in brackets.}",
        r"\label{tab:heldout_" + dataset_name.lower().replace(" ", "_") + r"}",
        r"\begin{tabular}{l" + "c" * len(props) + r"}",
        r"\toprule",
        "Method & " + " & ".join(props) + r" \\",
        r"\midrule",
    ]

    for m in methods:
        row_vals = []
        for p in props:
            rho = summary_df.loc[summary_df["method"] == m, p].values
            rho = rho[0] if len(rho) else float("nan")
            ci_row = ci_df[(ci_df["method"] == m) & (ci_df["property"] == p)]
            if not ci_row.empty and not np.isnan(rho):
                lo, hi = ci_row["ci_low"].values[0], ci_row["ci_high"].values[0]
                val = f"{rho:.3f} [{lo:.2f}, {hi:.2f}]"
            elif not np.isnan(rho):
                val = f"{rho:.3f}"
            else:
                val = "---"
            row_vals.append(val)
        bold = "AbDev-X" in m
        prefix = r"\textbf{" + m + r"}" if bold else m
        lines.append(prefix + " & " + " & ".join(row_vals) + r" \\")

    # Champion row
    champ_vals = []
    for p in props:
        if p in CHAMPION:
            champ_vals.append(f"{CHAMPION[p]['rho']:.3f}")
        else:
            champ_vals.append("---")
    lines.append(r"\midrule")
    lines.append(r"Competition champion & " + " & ".join(champ_vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX table -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gdpa3",
                    help="'gdpa3' | 'gdpa4' | path to a custom eval config yaml")
    ap.add_argument("--config", default=None, help="Override train config (default: configs/default.yaml)")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=15, help="Training seeds to average over")
    args = ap.parse_args()

    # Resolve config paths
    if args.dataset in ("gdpa3", "gdpa4", "gdpa5"):
        eval_cfg_path = ROOT / f"configs/{args.dataset}_eval.yaml"
    else:
        eval_cfg_path = Path(args.dataset)
    if not eval_cfg_path.exists():
        raise FileNotFoundError(f"Eval config not found: {eval_cfg_path}")

    eval_cfg = yaml.safe_load(eval_cfg_path.read_text())
    train_cfg_path = args.config or (ROOT / "configs/default.yaml")
    train_cfg = yaml.safe_load(open(train_cfg_path))
    # Merge: heldout section comes from eval_cfg
    train_cfg["heldout"] = eval_cfg["heldout"]

    dataset_label = args.dataset.upper()
    results_dir = ROOT / "results" / f"heldout_{args.dataset}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load training data ──────────────────────────────────────────────────
    print("\n=== Loading GDPa1 (train) ===")
    ds = load_gdpa1(str(train_cfg_path))
    props_train = list(ds.target_cols.keys())

    # Map GDPa1 property names to GDPa3 names (may differ in capitalization)
    heldout_props = list(eval_cfg["heldout"]["targets"].keys())
    # Build intersection aligned on normalized names
    prop_map = {}  # heldout_prop -> gdpa1_prop
    for hp in heldout_props:
        norm_hp = hp.replace("-", "_").upper()
        for gp in props_train:
            if gp.replace("-", "_").upper() == norm_hp:
                prop_map[hp] = gp
                break
        if hp not in prop_map:
            prop_map[hp] = hp  # assume same name; will fail gracefully if absent

    print(f"Property mapping (held-out -> GDPa1): {prop_map}")

    # ── Load held-out data ──────────────────────────────────────────────────
    print(f"\n=== Loading {dataset_label} (held-out) ===")
    df_te, heavy_te, light_te, ids_te, y_te = load_heldout(eval_cfg["heldout"], train_cfg)
    eval_props = list(y_te.keys())
    print(f"Held-out: {len(df_te)} antibodies. Properties with labels: {eval_props}")

    # ── Build features ──────────────────────────────────────────────────────
    print("\n=== Building features ===")
    fm_train = build_features(ds, train_cfg)
    Xte_df = features_for_test(heavy_te, light_te, train_cfg)
    train_cols = fm_train.names
    # Align columns
    Xte_aligned = Xte_df.reindex(columns=train_cols, fill_value=0.0)
    print(f"Train features: {fm_train.X.shape} | Test features: {Xte_aligned.shape}")

    # ── Run all methods ─────────────────────────────────────────────────────
    print("\n=== Running methods ===")
    all_preds = {}  # method_name -> {prop: np.array of predictions}

    # AbDev-X (multi-seed for stability)
    print("  AbDev-X (multi-seed)...")
    abdevx_by_seed = []
    Xtr = fm_train.X.to_numpy(dtype=float)
    Xte_np = Xte_aligned.to_numpy(dtype=float)
    heads = [h for h in train_cfg["model"]["heads"] if h in ("ridge", "lgbm")]
    for seed in range(args.seeds):
        seed_preds = {}
        for hp in eval_props:
            gp = prop_map.get(hp, hp)
            if gp not in ds.target_cols:
                continue
            y = ds.y(gp)
            mask = ~np.isnan(y)
            model = StackingRegressor(heads, random_state=seed)
            model.fit(Xtr[mask], y[mask])
            seed_preds[hp] = model.predict(Xte_np)
        abdevx_by_seed.append(seed_preds)
    # Average across seeds
    abdevx_preds = {}
    for hp in eval_props:
        seed_arrs = [s[hp] for s in abdevx_by_seed if hp in s]
        if seed_arrs:
            abdevx_preds[hp] = np.mean(seed_arrs, axis=0)
    all_preds["AbDev-X"] = abdevx_preds

    # ESM-2 Ridge baseline
    print("  ESM-2 Ridge...")
    try:
        esm2_cols = fm_train.blocks.get("esm2", [])
        if esm2_cols:
            Xtr_esm2 = fm_train.X[esm2_cols].to_numpy(dtype=float)
            Xte_esm2 = Xte_aligned[esm2_cols].to_numpy(dtype=float)
            esm2_preds = {}
            for hp in eval_props:
                gp = prop_map.get(hp, hp)
                if gp not in ds.target_cols:
                    continue
                y = ds.y(gp)
                mask = ~np.isnan(y)
                m = make_ridge()
                m.fit(Xtr_esm2[mask], y[mask])
                esm2_preds[hp] = m.predict(Xte_esm2)
            all_preds["ESM2-Ridge"] = esm2_preds
        else:
            print("    [warn] no esm2 block in features — skipping ESM2-Ridge")
    except Exception as e:
        print(f"    [warn] ESM2-Ridge failed: {e}")

    # AbLang2 ElasticNet
    print("  AbLang2 ElasticNet...")
    try:
        from sklearn.linear_model import ElasticNetCV
        ablang2_cols = fm_train.blocks.get("ablang2", [])
        if ablang2_cols:
            Xtr_ab = fm_train.X[ablang2_cols].to_numpy(dtype=float)
            Xte_ab = Xte_aligned[ablang2_cols].to_numpy(dtype=float)
            ab_preds = {}
            for hp in eval_props:
                gp = prop_map.get(hp, hp)
                if gp not in ds.target_cols:
                    continue
                y = ds.y(gp)
                mask = ~np.isnan(y)
                m = ElasticNetCV(cv=5, max_iter=5000, random_state=42)
                m.fit(Xtr_ab[mask], y[mask])
                ab_preds[hp] = m.predict(Xte_ab)
            all_preds["AbLang2-ElasticNet"] = ab_preds
        else:
            print("    [warn] no ablang2 block in features — skipping AbLang2-ElasticNet")
    except Exception as e:
        print(f"    [warn] AbLang2-ElasticNet failed: {e}")

    # Descriptor-only Ridge (interpretable baseline)
    print("  Descriptor-Ridge...")
    try:
        desc_cols = fm_train.blocks.get("descriptors", [])
        if desc_cols:
            Xtr_desc = fm_train.X[desc_cols].to_numpy(dtype=float)
            Xte_desc = Xte_aligned[desc_cols].to_numpy(dtype=float)
            desc_preds = {}
            for hp in eval_props:
                gp = prop_map.get(hp, hp)
                if gp not in ds.target_cols:
                    continue
                y = ds.y(gp)
                mask = ~np.isnan(y)
                m = make_ridge()
                m.fit(Xtr_desc[mask], y[mask])
                desc_preds[hp] = m.predict(Xte_desc)
            all_preds["Descriptor-Ridge"] = desc_preds
    except Exception as e:
        print(f"    [warn] Descriptor-Ridge failed: {e}")

    # ── Compute Spearman rho + bootstrap CIs ────────────────────────────────
    print("\n=== Computing Spearman rho + bootstrap CIs ===")
    summary_rows = []
    ci_rows = []
    pred_records = {"antibody_id": ids_te.values}

    for method, preds in all_preds.items():
        row = {"method": method}
        for hp in eval_props:
            if hp not in preds or hp not in y_te:
                row[hp] = float("nan")
                continue
            yp = preds[hp]
            yt = y_te[hp]
            rho = spearman_rho(yt, yp)
            lo, hi = bootstrap_ci(yt, yp, n=args.n_bootstrap, seed=42)
            row[hp] = rho
            ci_rows.append({"method": method, "property": hp,
                            "rho": rho, "ci_low": lo, "ci_high": hi})
            pred_records[f"{method}_{hp}"] = yp
            if hp not in pred_records:
                pred_records[f"true_{hp}"] = yt
        summary_rows.append(row)

    # Add ground truth to predictions CSV
    for hp in eval_props:
        pred_records[f"true_{hp}"] = y_te[hp]

    summary_df = pd.DataFrame(summary_rows)
    ci_df = pd.DataFrame(ci_rows)

    summary_df.to_csv(results_dir / "spearman_summary.csv", index=False)
    ci_df.to_csv(results_dir / "bootstrap_ci.csv", index=False)
    pd.DataFrame(pred_records).to_csv(results_dir / "predictions.csv", index=False)

    # ── Print results table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"HELD-OUT RESULTS: {dataset_label}  (n={len(df_te)})")
    print(f"{'='*70}")
    # Wide format print
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n--- Competition champion reference ---")
    for hp in eval_props:
        if hp in CHAMPION:
            c = CHAMPION[hp]
            abdevx_rho = summary_df.loc[summary_df["method"] == "AbDev-X", hp].values
            rho_str = f"AbDev-X={abdevx_rho[0]:.3f}" if len(abdevx_rho) else ""
            print(f"  {hp:12s}: champion={c['rho']:.3f} ({c['method']})  {rho_str}")

    # ── LaTeX table ─────────────────────────────────────────────────────────
    to_latex_table(summary_df, ci_df, dataset_label, results_dir / "comparison_table.tex")

    print(f"\nAll outputs -> {results_dir}")


if __name__ == "__main__":
    main()
