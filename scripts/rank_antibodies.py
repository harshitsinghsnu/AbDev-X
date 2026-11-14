"""Rank antibodies by predicted developability using AbDev-X.

Trains on GDPa1, predicts on a target CSV (default: GDPa3 held-out),
then builds a composite developability score and ranked table.

Sign convention (higher composite = better developability):
  HIC      lower RT = less hydrophobic = better  → flip
  AC-SINS  lower = less self-association = better → flip
  Tm2      higher = more stable = better          → keep
  PR_CHO   lower = less polyreactive = better     → flip
  Titer    higher = more expression = better      → keep

Composite score: mean of z-scored, sign-adjusted per-assay predictions
(NaN assays excluded from each antibody's mean).

Usage:
    python scripts/rank_antibodies.py                        # GDPa3
    python scripts/rank_antibodies.py --input seqs.csv       # custom input
    python scripts/rank_antibodies.py --top 20 --bottom 20
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from abdev.data.dataset import load_gdpa1, resolve_column
from abdev.features.descriptors import build_descriptor_matrix
from abdev.features.plm import build_plm_matrix
from abdev.models.ensemble import StackingRegressor
from abdev.pipeline import build_features

CONFIG    = ROOT / "configs" / "default.yaml"
GDPA3_CFG = ROOT / "configs" / "gdpa3_eval.yaml"
PROPS     = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
N_SEEDS   = 5

# direction: +1 = higher predicted value is BETTER, -1 = lower is better
DIRECTION = {
    "HIC":          -1,   # lower RT = less hydrophobic = better
    "Tm2":          +1,   # higher Tm = more stable = better
    "AC-SINS_pH7.4": -1,  # lower = less self-association = better
    "PR_CHO":       -1,   # lower = less polyreactive = better
    "Titer":        +1,   # higher = more expression = better
}
PROP_ALIAS = {"AC-SINS": "AC-SINS_pH7.4"}


def build_features_df(heavy, light, cfg):
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


def train_and_predict(cfg, heads, use_mt, Xtr, y_train, Xte, n_seeds=N_SEEDS):
    Y_multi = np.column_stack(
        [y_train.get(p, np.full(Xtr.shape[0], np.nan)) for p in PROPS]
    )
    seed_preds = {p: [] for p in PROPS}
    for s in range(n_seeds):
        for i, prop in enumerate(PROPS):
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


def composite_score(pred_df: pd.DataFrame) -> pd.Series:
    """Z-score + flip + mean across available assays."""
    zscores = pd.DataFrame(index=pred_df.index)
    for p in PROPS:
        if p not in pred_df.columns:
            continue
        col = pred_df[p].astype(float)
        std = col.std()
        if std > 1e-9:
            z = (col - col.mean()) / std
        else:
            z = col * 0.0
        zscores[p] = z * DIRECTION[p]
    return zscores.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None,
                        help="CSV with vhn/vh + vl columns and optional true labels")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top (best) antibodies to show")
    parser.add_argument("--bottom", type=int, default=10,
                        help="Number of bottom (worst) antibodies to show")
    parser.add_argument("--out", default=None,
                        help="Save full ranked table to this CSV path")
    args = parser.parse_args()

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    heads  = list(cfg["model"]["heads"])
    use_mt = "multitask_mlp" in heads

    # ── load target sequences ─────────────────────────────────────────────────
    if args.input is None:
        # default: GDPa3 held-out
        with open(GDPA3_CFG) as f:
            gdpa3_cfg = yaml.safe_load(f)
        hcfg = gdpa3_cfg["heldout"]
        df_te = pd.read_csv(ROOT / hcfg["csv_path"])
        h_col = resolve_column(df_te, hcfg["heavy_aliases"])
        l_col = resolve_column(df_te, hcfg["light_aliases"])
        te_heavy = df_te[h_col].astype(str).str.strip()
        te_light = df_te[l_col].fillna("").astype(str).str.replace("nan", "", regex=False).str.strip()
        name_col = next((c for c in ["antibody_name", "antibody_id", "name"] if c in df_te.columns), None)
        te_names = df_te[name_col].astype(str) if name_col else pd.Series(
            [f"Ab-{i+1}" for i in range(len(df_te))])
        # load true labels if available
        true_labels = {}
        for prop, spec in hcfg["targets"].items():
            col = resolve_column(df_te, spec["column"], required=False, what=prop)
            if col:
                true_labels[prop] = df_te[col].to_numpy(dtype=float)
        print(f"Target: GDPa3 held-out ({len(df_te)} antibodies)")
    else:
        df_te = pd.read_csv(args.input)
        for h_alias in ["vhn", "vh", "vh_protein_sequence", "heavy", "VH"]:
            if h_alias in df_te.columns:
                te_heavy = df_te[h_alias].astype(str).str.strip()
                break
        for l_alias in ["vl", "vl_protein_sequence", "light", "VL"]:
            if l_alias in df_te.columns:
                te_light = df_te[l_alias].fillna("").astype(str).str.replace("nan", "", regex=False).str.strip()
                break
        name_col = next((c for c in ["antibody_name", "antibody_id", "name"] if c in df_te.columns), None)
        te_names = df_te[name_col].astype(str) if name_col else pd.Series(
            [f"Ab-{i+1}" for i in range(len(df_te))])
        true_labels = {}
        for p in PROPS:
            alias = next((k for k, v in PROP_ALIAS.items() if v == p), p)
            col = next((c for c in [p, alias] if c in df_te.columns), None)
            if col:
                true_labels[p] = pd.to_numeric(df_te[col], errors="coerce").to_numpy(dtype=float)
        print(f"Target: {args.input} ({len(df_te)} antibodies)")

    # ── build test features ───────────────────────────────────────────────────
    print("Building target features...")
    Xte_df = build_features_df(te_heavy, te_light, cfg)

    # ── train on GDPa1 ────────────────────────────────────────────────────────
    print("Training AbDev-X on GDPa1...")
    ds = load_gdpa1(str(CONFIG))
    fm = build_features(ds, cfg)
    Xtr = fm.X.to_numpy(dtype=float)
    Xte = Xte_df.reindex(columns=fm.names, fill_value=0.0).to_numpy(dtype=float)
    y_train = {p: ds.y(p) for p in PROPS if p in ds.target_cols}
    preds = train_and_predict(cfg, heads, use_mt, Xtr, y_train, Xte)

    # ── build ranked table ────────────────────────────────────────────────────
    table = pd.DataFrame({"antibody": te_names.values})
    for p in PROPS:
        if p in preds:
            table[f"pred_{p}"] = preds[p]
        if p in true_labels:
            table[f"true_{p}"] = true_labels[p]

    # composite score and rank
    pred_cols = {p: f"pred_{p}" for p in PROPS if f"pred_{p}" in table.columns}
    pred_df = table[[pred_cols[p] for p in pred_cols]].rename(columns={v: k for k, v in pred_cols.items()})
    table["composite_score"] = composite_score(pred_df)
    table["rank"] = table["composite_score"].rank(ascending=False).astype(int)
    table = table.sort_values("rank")

    # ── Spearman rank agreement with true composite (if labels available) ─────
    if true_labels:
        true_df = pd.DataFrame({p: true_labels[p] for p in true_labels})
        true_comp = composite_score(true_df)
        pred_comp = table["composite_score"].values[table.index.argsort()]  # reorder to original index
        # align by original index
        true_comp_aligned = true_comp.values
        pred_comp_aligned = table.set_index(table.index)["composite_score"].reindex(range(len(te_heavy))).values
        m = ~(np.isnan(true_comp_aligned) | np.isnan(pred_comp_aligned))
        if m.sum() > 2:
            rho_comp = spearmanr(true_comp_aligned[m], pred_comp_aligned[m])[0]
            print(f"\nComposite ranking agreement (Spearman ρ): {rho_comp:.3f}  (n={m.sum()})")

        # per-assay Spearman
        print("\nPer-assay ranking agreement (predicted vs true):")
        print(f"  {'Assay':<18} {'ρ':>7}  {'Direction'}")
        print(f"  {'-'*40}")
        for p in PROPS:
            if p in preds and p in true_labels:
                yt = true_labels[p].astype(float)
                yp = preds[p].astype(float)
                m = ~(np.isnan(yt) | np.isnan(yp))
                if m.sum() > 2:
                    rho = spearmanr(yt[m], yp[m])[0]
                    dirn = "higher=worse" if DIRECTION[p] == -1 else "higher=better"
                    print(f"  {p:<18} {rho:>7.3f}  ({dirn})")

    # ── print top/bottom ──────────────────────────────────────────────────────
    disp_cols = ["rank", "antibody", "composite_score"] + [f"pred_{p}" for p in PROPS if f"pred_{p}" in table.columns]

    print(f"\n{'='*70}")
    print(f"TOP {args.top} — best predicted developability")
    print(f"{'='*70}")
    top = table.head(args.top)[disp_cols]
    print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*70}")
    print(f"BOTTOM {args.bottom} — worst predicted developability")
    print(f"{'='*70}")
    bot = table.tail(args.bottom)[disp_cols]
    print(bot.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── precision@K: do top-K predicted match top-K true? ────────────────────
    if true_labels:
        print(f"\nPrecision@K (predicted top-K ∩ true top-K):")
        true_comp = composite_score(pd.DataFrame({p: true_labels[p] for p in true_labels}))
        pred_comp = table["composite_score"].reindex(range(len(te_heavy))).values
        true_rank = true_comp.rank(ascending=False).astype(int).values
        pred_rank = pd.Series(pred_comp).rank(ascending=False).astype(int).values
        for k in [5, 10, 20]:
            true_top = set(np.where(true_rank <= k)[0])
            pred_top = set(np.where(pred_rank <= k)[0])
            if true_top:
                prec = len(true_top & pred_top) / k
                print(f"  P@{k:<3}: {prec:.2f}  ({len(true_top & pred_top)}/{k} overlap)")

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = args.out or str(ROOT / "results" / "antibody_ranking.csv")
    table.to_csv(out_path, index=False)
    print(f"\nFull ranked table saved -> {out_path}")


if __name__ == "__main__":
    main()
