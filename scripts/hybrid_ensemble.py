"""Hybrid AbDev-X + AbDev-MoE ensemble on GDPa3.

Strategy:
  For each assay independently, choose the source model based on OOD-CV performance
  (no peeking at held-out labels). Also compute:
    - AbDev-X only
    - MoE only
    - Simple average of both
    - Oracle hybrid (post-hoc best per assay — upper bound)

Usage:
    python scripts/hybrid_ensemble.py
"""
from __future__ import annotations
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gdpa3_eval as GE
from moe_gdpa3 import AbDevMoE, EXPERT_SPEC, _load_sap_electro, build_gate, _block_head, BASE_EXPERTS
from abdev.models.fusion import AbDevFusion

ASSAYS    = GE.ASSAYS
LABEL_MAP = GE.LABEL_MAP


def main():
    print("Loading data and building blocks...")
    ds, g3, B1_raw, B3_raw = GE.build_blocks()

    sap1 = _load_sap_electro(ds.df["antibody_name"], "gdpa1")
    sap3 = _load_sap_electro(g3["antibody_name"],    "heldout")
    B1   = {**B1_raw, "sap_electro": sap1}
    B3   = {**B3_raw, "sap_electro": sap3}

    print("Building gate features...")
    G1, G3 = build_gate(B1, B3)

    folds  = ds.folds("cluster_isotype")
    ufolds = sorted(f for f in set(folds.tolist()) if f == f)

    # Per-assay predictions
    abdevx_preds = {}   # assay → (n_gdpa3,) array
    moe_preds    = {}
    true_vals    = {}
    cv_rho_abdevx = {}
    cv_rho_moe    = {}

    for a in ASSAYS:
        print(f"\n--- {a} ---")
        y1  = ds.y(a)
        lab = ~np.isnan(y1)
        y3  = g3[LABEL_MAP[a]].to_numpy(float)
        true_vals[a] = y3

        # ── AbDev-X (fixed-weight fusion) ─────────────────────────────────────
        B1m = {k: v[lab] for k, v in B1.items()}
        abdevx_preds[a] = AbDevFusion().fit(B1m, y1[lab]).predict(B3)

        # OOD-CV rho for AbDev-X via block-level OOF (proxy for full model CV)
        blk_rhos = {}
        for blk in ["esm2", "descriptors", "isotype", "bio", "ablang2", "sap_electro"]:
            Xb1 = np.nan_to_num(B1[blk])
            per = []
            for f in ufolds:
                tr = np.where((folds != f) & lab)[0]
                te = np.where((folds == f) & lab)[0]
                if len(te) < 3 or len(tr) < 10:
                    continue
                p = _block_head(blk).fit(Xb1[tr], y1[tr]).predict(Xb1[te])
                per.append(float(spearmanr(y1[te], p).statistic))
            blk_rhos[blk] = float(np.nanmean(per)) if per else 0.0

        # Full fusion CV estimate: average single-block CV rhos (proxy)
        # Use representation_fusion ablation metric if available
        try:
            abl = pd.read_csv(ROOT / "results/abdevx2_ablation/metrics.csv")
            row = abl[(abl["method"] == "representation_fusion") &
                      (abl["split"] == "cluster_isotype")]
            prop_col = {"AC-SINS_pH7.4": "AC-SINS_pH7.4"}.get(a, a)
            cv_rho_abdevx[a] = float(row[prop_col].values[0]) if len(row) else float(np.mean(list(blk_rhos.values())))
        except Exception:
            cv_rho_abdevx[a] = float(np.mean(list(blk_rhos.values())))

        # ── AbDev-MoE (7 experts + gate) ──────────────────────────────────────
        print(f"  Fitting MoE...")
        moe = AbDevMoE()
        moe.fit({k: v[lab] for k, v in B1.items()}, y1[lab], G1[lab])
        moe_preds[a] = moe.predict(B3, G3)

        # MoE CV rho from ablation
        try:
            abl = pd.read_csv(ROOT / "results/abdevx2_ablation/metrics.csv")
            row = abl[(abl["method"] == "moe") & (abl["split"] == "cluster_isotype")]
            cv_rho_moe[a] = float(row[a].values[0]) if len(row) else 0.0
        except Exception:
            cv_rho_moe[a] = 0.0

    # ── Build hybrid predictions ───────────────────────────────────────────────
    print("\n\nBuilding hybrid ensembles...")
    print(f"\n{'Assay':<18} {'AbDev-X CV':>12} {'MoE CV':>10} {'CV winner':>12}")
    print("-" * 55)

    cv_selection = {}   # assay → "abdevx" or "moe"
    for a in ASSAYS:
        winner = "moe" if cv_rho_moe[a] > cv_rho_abdevx[a] else "abdevx"
        cv_selection[a] = winner
        print(f"  {a:<16} {cv_rho_abdevx[a]:>12.3f} {cv_rho_moe[a]:>10.3f} {winner:>12}")

    # Four strategies: abdevx / moe / average / cv-hybrid / oracle-hybrid
    results = {
        "AbDev-X":       {},
        "AbDev-MoE":     {},
        "Average":       {},
        "CV-Hybrid":     {},
        "Oracle-Hybrid": {},
    }

    for a in ASSAYS:
        y3   = true_vals[a]
        ok   = ~np.isnan(y3)
        pa   = abdevx_preds[a]
        pm   = moe_preds[a]
        pavg = (pa + pm) / 2.0
        pcv  = pm if cv_selection[a] == "moe" else pa
        # oracle: whichever gets higher rho on held-out (not usable in practice)
        rho_a = float(spearmanr(y3[ok], pa[ok]).statistic)
        rho_m = float(spearmanr(y3[ok], pm[ok]).statistic)
        por = pm if rho_m > rho_a else pa

        results["AbDev-X"][a]       = float(spearmanr(y3[ok], pa[ok]).statistic)
        results["AbDev-MoE"][a]     = float(spearmanr(y3[ok], pm[ok]).statistic)
        results["Average"][a]       = float(spearmanr(y3[ok], pavg[ok]).statistic)
        results["CV-Hybrid"][a]     = float(spearmanr(y3[ok], pcv[ok]).statistic)
        results["Oracle-Hybrid"][a] = float(spearmanr(y3[ok], por[ok]).statistic)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  GDPa3 HELD-OUT  |  Spearman rho per assay")
    print("=" * 80)
    header = f"  {'Model':<22}" + "".join(f" {a:>15}" for a in ASSAYS) + f"  {'MEAN':>7}"
    print(header)
    print("-" * 80)

    rows = []
    for name, res in results.items():
        vals = [res[a] for a in ASSAYS]
        mean = float(np.mean(vals))
        row = {"model": name, **{a: round(v, 3) for a, v in zip(ASSAYS, vals)}, "MEAN": round(mean, 3)}
        rows.append(row)
        line = f"  {name:<22}" + "".join(f" {v:>15.3f}" for v in vals) + f"  {mean:>7.3f}"
        print(line)

    print("=" * 80)
    print("\nNote: CV-Hybrid uses OOD-CV performance to select per-assay model (valid).")
    print("      Oracle-Hybrid selects based on held-out rho (upper bound only).")

    # save
    out = ROOT / "results" / "hybrid_ensemble.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    # CV selection summary
    print("\nCV-based model selection per assay:")
    for a in ASSAYS:
        src = cv_selection[a]
        cv_rho  = cv_rho_moe[a] if src == "moe" else cv_rho_abdevx[a]
        held_rho = results["CV-Hybrid"][a]
        print(f"  {a:<18} → {src}  (CV rho={cv_rho:.3f}, held-out rho={held_rho:.3f})")


if __name__ == "__main__":
    main()
