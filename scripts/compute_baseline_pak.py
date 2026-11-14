"""Compute P@5, P@10, P@20 and composite rho for all baseline models on GDPa3.

Updates results/ranking_comparison.csv with complete P@K values.

Usage:
    python scripts/compute_baseline_pak.py
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from abdev.data.dataset import load_gdpa1, resolve_column
from abdev.features.descriptors import build_descriptor_matrix
from abdev.features.plm import build_plm_matrix
from abdev.pipeline import build_features

ROOT   = Path(__file__).resolve().parents[1]
PROPS  = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
DIRECTION = {"HIC": -1, "Tm2": +1, "AC-SINS_pH7.4": -1, "PR_CHO": -1, "Titer": +1}


def composite(preds: dict) -> np.ndarray:
    cols = []
    for p, v in preds.items():
        std = np.nanstd(v)
        if std > 1e-9:
            cols.append((v - np.nanmean(v)) / std * DIRECTION[p])
    return np.nanmean(np.column_stack(cols), axis=1) if cols else np.zeros(1)


def ridge_predict(Xtr, y, Xte):
    mask = ~np.isnan(y)
    if mask.sum() < 5:
        return np.full(Xte.shape[0], np.nan)
    sc = StandardScaler()
    m  = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
    m.fit(sc.fit_transform(Xtr[mask]), y[mask])
    return m.predict(sc.transform(Xte))


def precision_at_k(true_comp, pred_comp, k):
    m = ~(np.isnan(true_comp) | np.isnan(pred_comp))
    if m.sum() < k:
        return np.nan
    tr = pd.Series(true_comp[m]).rank(ascending=False).astype(int).values
    pr = pd.Series(pred_comp[m]).rank(ascending=False).astype(int).values
    return len(set(np.where(tr <= k)[0]) & set(np.where(pr <= k)[0])) / k


def main():
    print("Loading configs and data...")
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / "configs" / "gdpa3_eval.yaml") as f:
        g3cfg = yaml.safe_load(f)

    hcfg = g3cfg["heldout"]
    df3  = pd.read_csv(ROOT / hcfg["csv_path"])
    h3   = resolve_column(df3, hcfg["heavy_aliases"])
    l3   = resolve_column(df3, hcfg["light_aliases"])
    te_h = df3[h3].astype(str).str.strip()
    te_l = df3[l3].fillna("").astype(str).str.replace("nan", "", regex=False).str.strip()

    true_labels = {}
    for prop, spec in hcfg["targets"].items():
        col = resolve_column(df3, spec["column"], required=False, what=prop)
        if col:
            true_labels[prop] = df3[col].to_numpy(dtype=float)

    print("Building GDPa3 features (descriptors + PLM)...")
    heavy_s = pd.Series(list(te_h)).reset_index(drop=True)
    light_s = pd.Series(list(te_l)).reset_index(drop=True)
    desc_te = build_descriptor_matrix(heavy_s, light_s, ph=7.4)
    plm_te  = build_plm_matrix(heavy_s, light_s, cfg)
    Xte_all = pd.concat([desc_te, plm_te], axis=1).replace([np.inf, -np.inf], 0).fillna(0)

    print("Building GDPa1 features...")
    ds  = load_gdpa1(str(ROOT / "configs" / "default.yaml"))
    fm  = build_features(ds, cfg)
    Xtr_df  = fm.X
    y_train = {p: ds.y(p) for p in PROPS if p in ds.target_cols}

    desc_names = [c for c in fm.names if c in desc_te.columns]
    esm2_names = [c for c in fm.names if c.startswith("esm2_")]
    abl2_names = [c for c in fm.names if c.startswith("ablang2_")]

    true_comp = composite(true_labels)

    baseline_configs = {
        "Descriptors":     desc_names,
        "ESM2-Ridge":      esm2_names,
        "Desc+ESM2":       desc_names + esm2_names,
        "Desc+AbLang2":    desc_names + abl2_names,
        "All feats-Ridge": list(fm.names),
    }

    results = {}
    for label, feat_names in baseline_configs.items():
        print(f"  Running {label}...")
        Xtr = Xtr_df[feat_names].to_numpy(dtype=float)
        Xte = Xte_all.reindex(columns=feat_names, fill_value=0).to_numpy(dtype=float)
        preds = {}
        for p in PROPS:
            if p in y_train:
                preds[p] = ridge_predict(Xtr, y_train[p], Xte)
        pc = composite(preds)
        m  = ~(np.isnan(true_comp) | np.isnan(pc))
        rho = float(spearmanr(true_comp[m], pc[m])[0]) if m.sum() > 2 else np.nan
        results[label] = {
            "comp_rho": round(rho, 3),
            "P5":  round(precision_at_k(true_comp, pc, 5),  3),
            "P10": round(precision_at_k(true_comp, pc, 10), 3),
            "P20": round(precision_at_k(true_comp, pc, 20), 3),
        }
        print(f"    rho={rho:.3f}  P@5={results[label]['P5']:.2f}  "
              f"P@10={results[label]['P10']:.2f}  P@20={results[label]['P20']:.2f}")

    # Random (theoretical)
    results["Random"] = {"comp_rho": 0.000, "P5": round(5/80,3), "P10": round(10/80,3), "P20": round(20/80,3)}

    # AbDev-X from existing ranking CSV
    rk = pd.read_csv(ROOT / "results" / "antibody_ranking.csv")
    name_col = next((c for c in ["antibody_name","antibody_id","name"] if c in df3.columns), None)
    if name_col:
        name_to_score = dict(zip(rk["antibody"].astype(str), rk["composite_score"]))
        abdevx_comp = df3[name_col].astype(str).map(name_to_score).values.astype(float)
    else:
        abdevx_comp = rk.sort_values("rank")["composite_score"].values
    m = ~(np.isnan(true_comp) | np.isnan(abdevx_comp))
    rho_abdevx = float(spearmanr(true_comp[m], abdevx_comp[m])[0])
    results["AbDev-X (stacking)"] = {
        "comp_rho": round(rho_abdevx, 3),
        "P5":  round(precision_at_k(true_comp, abdevx_comp, 5),  3),
        "P10": round(precision_at_k(true_comp, abdevx_comp, 10), 3),
        "P20": round(precision_at_k(true_comp, abdevx_comp, 20), 3),
    }

    # write updated CSV
    MODEL_ORDER = [
        "Random", "Descriptors", "ESM2-Ridge",
        "Desc+ESM2", "Desc+AbLang2", "All feats-Ridge", "AbDev-X (stacking)"
    ]
    rows = []
    for m in MODEL_ORDER:
        if m in results:
            rows.append({"model": m, **results[m]})
    out_csv = ROOT / "results" / "ranking_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nUpdated -> {out_csv}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
