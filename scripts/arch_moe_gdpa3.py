"""Architecture-level MoE: combine AbDev-X / AbDevMoE / SparseMoE / KronLSTM as experts.

Each of the 4 trained architectures is treated as an expert for each assay.
A level-2 gate is trained on OOF expert predictions + gate features to route
per-antibody to the best combination.

Combination strategies evaluated
---------------------------------
1. AbDev-X only           (baseline)
2. Simple average         (4 experts)
3. OOD-CV static          (pick best arch per assay by cluster_isotype CV rho)
4. Sparse MoE (top-1)     (per-antibody hard routing, 1-of-4 gate)
5. Sparse MoE (top-2)     (per-antibody soft-sparse routing, top-2-of-4)
6. Oracle                 (post-hoc best arch per assay — upper bound)

Oracle: 0.347  (AbDev-X: 0.251)  —  the gap this script aims to close.

Usage:
    python scripts/arch_moe_gdpa3.py
"""
from __future__ import annotations
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gdpa3_eval as GE
from moe_gdpa3 import (AbDevMoE, EXPERT_SPEC, BASE_EXPERTS,
                        _load_sap_electro, build_gate)
from sparse_moe_gdpa3 import SparseMoE, _train_sparse_gate, _sparse_predict
from kronecker_lstm_gdpa3 import (KronLSTMModel, BlockPreprocessor,
                                   BLOCK_ORDER, D_PROJ, D_HEAD,
                                   _to_tensors, _predict as _klstm_predict)

import torch
import torch.nn as nn

ASSAYS    = GE.ASSAYS
LABEL_MAP = GE.LABEL_MAP
COMP      = GE.COMP_HELDOUT

ARCH_NAMES = ["AbDev-X", "AbDevMoE", "SparseMoE-2", "KronLSTM"]


# ── OOF generation for each architecture ────────────────────────────────────

def _oof_abdevx(Xb, y, folds, ufolds) -> np.ndarray:
    """OOF predictions for AbDev-X (fixed Ridge meta)."""
    from abdev.models.fusion import AbDevFusion
    n   = len(y)
    oof = np.full(n, np.nan)
    for f in ufolds:
        tr = np.where(folds != f)[0]
        va = np.where(folds == f)[0]
        if len(va) < 3 or len(tr) < 10:
            continue
        B_tr = {k: v[tr] for k, v in Xb.items()}
        B_va = {k: v[va] for k, v in Xb.items()}
        oof[va] = AbDevFusion().fit(B_tr, y[tr]).predict(B_va)
    return oof


def _oof_abdevmoe(Xb, y, G, folds, ufolds) -> np.ndarray:
    """OOF predictions for AbDevMoE (soft Ridge gate)."""
    n   = len(y)
    oof = np.full(n, np.nan)
    for f in ufolds:
        tr = np.where(folds != f)[0]
        va = np.where(folds == f)[0]
        if len(va) < 3 or len(tr) < 10:
            continue
        moe = AbDevMoE()
        moe.fit({k: v[tr] for k, v in Xb.items()}, y[tr], G[tr])
        oof[va] = moe.predict({k: v[va] for k, v in Xb.items()}, G[va])
    return oof


def _oof_sparsemoe(Xb, y, G, folds, ufolds, top_k=2) -> np.ndarray:
    """OOF predictions for SparseMoE (neural top-K gate)."""
    n   = len(y)
    oof = np.full(n, np.nan)
    for f in ufolds:
        tr = np.where(folds != f)[0]
        va = np.where(folds == f)[0]
        if len(va) < 3 or len(tr) < 10:
            continue
        sm = SparseMoE(top_k=top_k)
        sm.fit({k: v[tr] for k, v in Xb.items()}, y[tr], G[tr])
        oof[va] = sm.predict({k: v[va] for k, v in Xb.items()}, G[va])
    return oof


def _train_klstm_fold(B_tr, y_tr, B_va, y_va, block_dims):
    """Train one KronLSTM fold (fast mode: fewer epochs for OOF)."""
    import torch
    torch.manual_seed(0)
    model = KronLSTMModel(block_dims, D_PROJ, D_HEAD)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    bt, bv = _to_tensors(B_tr), _to_tensors(B_va)
    yt, yv = torch.FloatTensor(y_tr), torch.FloatTensor(y_va)
    best_val, best_state, no_imp = float("inf"), None, 0
    for ep in range(800):
        model.train(); opt.zero_grad()
        loss = ((model(bt) - yt) ** 2).mean()
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(((model(bv) - yv) ** 2).mean())
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 100: break
    if best_state: model.load_state_dict(best_state)
    model.eval(); return model


def _oof_kronlstm(Xb, y, folds, ufolds) -> np.ndarray:
    """OOF predictions for KronLSTM (per-fold neural training)."""
    n    = len(y)
    oof  = np.full(n, np.nan)
    avail = [b for b in BLOCK_ORDER if b in Xb]
    for f in ufolds:
        tr = np.where(folds != f)[0]
        va = np.where(folds == f)[0]
        if len(va) < 3 or len(tr) < 10:
            continue
        prep = BlockPreprocessor()
        B_tr = prep.fit_transform({b: Xb[b][tr] for b in avail})
        B_va = prep.transform({b: Xb[b][va] for b in avail})
        model = _train_klstm_fold(B_tr, y[tr], B_va, y[va], prep.dims())
        oof[va] = _klstm_predict(model, B_va)
    return oof


# ── Level-2 sparse gate ──────────────────────────────────────────────────────

def _train_arch_gate(P_oof, y, G, top_k=1, n_epochs=1500, seed=0):
    """Train architecture-level sparse gate.
    P_oof: (n, 4) OOF predictions from 4 architectures.
    G:     (n, d) gate features (isotype + descriptors + ESM2 PCs).
    """
    valid = ~np.any(np.isnan(P_oof), axis=1)
    sc = StandardScaler().fit(G[valid])
    Gs = sc.transform(G[valid])
    # Concatenate expert predictions with gate features for richer routing signal
    gate_in = np.column_stack([Gs, P_oof[valid]])   # (n_valid, d_gate + 4)
    gate = _train_sparse_gate(P_oof[valid], y[valid], gate_in,
                               top_k=top_k, n_epochs=n_epochs,
                               lam_lb=0.05, seed=seed)
    return gate, sc


def _arch_gate_predict(gate, sc, P_held, G_held, top_k):
    Gs = sc.transform(G_held)
    gate_in = np.column_stack([Gs, P_held])
    return _sparse_predict(gate, P_held, gate_in, top_k=top_k)


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        try:
            _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 70)
    print("  Architecture-level MoE: 4 experts  (AbDev-X / AbDevMoE / SparseMoE / KronLSTM)")
    print("=" * 70, flush=True)

    print("Loading data ...", flush=True)
    ds, g3, B1_raw, B3_raw = GE.build_blocks()

    sap1 = _load_sap_electro(ds.df["antibody_name"], "gdpa1")
    sap3 = _load_sap_electro(g3["antibody_name"],    "heldout")
    B1   = {**B1_raw, "sap_electro": sap1}
    B3   = {**B3_raw, "sap_electro": sap3}

    print("Building gate features ...", flush=True)
    G1, G3 = build_gate(B1, B3)

    folds  = ds.folds("cluster_isotype")
    ufolds = sorted(f for f in set(folds.tolist()) if f == f)

    strategies = ["AbDev-X", "Average", "OOD-CV static",
                  "ArchMoE top-1", "ArchMoE top-2", "Oracle"]
    results     = {s: {} for s in strategies}
    cv_rho_arch = {a: {} for a in ASSAYS}   # assay -> arch -> ood-cv rho

    for a in ASSAYS:
        print(f"\n{'='*55}\n  Assay: {a}", flush=True)
        y1  = ds.y(a)
        lab = ~np.isnan(y1)
        y3  = g3[LABEL_MAP[a]].to_numpy(float)
        ok  = ~np.isnan(y3)
        B1m = {k: v[lab] for k, v in B1.items()}
        y1l = y1[lab]
        G1l = G1[lab]
        fold_lab = folds[lab]

        # ── Generate OOF predictions for each architecture ──────────────────
        print("  Generating OOF: AbDev-X ...", flush=True)
        oof_ax = _oof_abdevx(B1m, y1l, fold_lab, ufolds)

        print("  Generating OOF: AbDevMoE ...", flush=True)
        oof_moe = _oof_abdevmoe(B1m, y1l, G1l, fold_lab, ufolds)

        print("  Generating OOF: SparseMoE-2 ...", flush=True)
        oof_sm = _oof_sparsemoe(B1m, y1l, G1l, fold_lab, ufolds, top_k=2)

        print("  Generating OOF: KronLSTM ...", flush=True)
        oof_kl = _oof_kronlstm(B1m, y1l, fold_lab, ufolds)

        OOF = {"AbDev-X": oof_ax, "AbDevMoE": oof_moe,
               "SparseMoE-2": oof_sm, "KronLSTM": oof_kl}

        # OOD-CV rho per architecture
        print("  OOD-CV rho per architecture:", flush=True)
        for arch_name, oof in OOF.items():
            valid = ~(np.isnan(y1l) | np.isnan(oof))
            rho   = float(spearmanr(y1l[valid], oof[valid]).statistic) if valid.sum() > 3 else 0.0
            cv_rho_arch[a][arch_name] = rho
            print(f"    {arch_name:<15} OOD-CV rho = {rho:.3f}", flush=True)

        # ── Retrain full models on all data → predict GDPa3 ────────────────
        print("  Full-data refit -> GDPa3 ...", flush=True)

        from abdev.models.fusion import AbDevFusion
        p_ax  = AbDevFusion().fit(B1m, y1l).predict(B3)

        moe = AbDevMoE()
        moe.fit(B1m, y1l, G1l)
        p_moe = moe.predict(B3, G3)

        sm2 = SparseMoE(top_k=2)
        sm2.fit(B1m, y1l, G1l)
        p_sm = sm2.predict(B3, G3)

        # KronLSTM full refit
        avail    = [b for b in BLOCK_ORDER if b in B1m]
        prep_f   = BlockPreprocessor()
        B_full   = prep_f.fit_transform({b: B1m[b] for b in avail})
        va_px    = np.where(fold_lab == ufolds[0])[0]
        tr_px    = np.where(fold_lab != ufolds[0])[0]
        B_va_px  = prep_f.transform({b: B1m[b][va_px] for b in avail})
        from kronecker_lstm_gdpa3 import _train as _klstm_train
        model_kl = _klstm_train(
            {b: B_full[b][tr_px] for b in B_full},
            y1l[tr_px], B_va_px, y1l[va_px],
            prep_f.dims(), n_epochs=2000, patience=200, seed=1,
        )
        B3_proc = prep_f.transform({b: B3[b] for b in avail if b in B3})
        p_kl    = _klstm_predict(model_kl, B3_proc)

        P_held = np.column_stack([p_ax, p_moe, p_sm, p_kl])  # (80, 4)
        P_oof  = np.column_stack([oof_ax, oof_moe, oof_sm, oof_kl])

        # ── Strategies ──────────────────────────────────────────────────────

        # 1. AbDev-X baseline
        results["AbDev-X"][a] = float(spearmanr(y3[ok], p_ax[ok]).statistic)

        # 2. Simple average
        p_avg = P_held.mean(axis=1)
        results["Average"][a] = float(spearmanr(y3[ok], p_avg[ok]).statistic)

        # 3. OOD-CV static: pick best arch by OOD-CV rho
        best_arch = max(cv_rho_arch[a], key=cv_rho_arch[a].get)
        p_static  = {"AbDev-X": p_ax, "AbDevMoE": p_moe,
                     "SparseMoE-2": p_sm, "KronLSTM": p_kl}[best_arch]
        results["OOD-CV static"][a] = float(spearmanr(y3[ok], p_static[ok]).statistic)
        print(f"  OOD-CV static picks: {best_arch} (CV={cv_rho_arch[a][best_arch]:.3f})", flush=True)

        # 4-5. Architecture-level sparse MoE gate (top-1 and top-2)
        print("  Training ArchMoE gate ...", flush=True)
        for top_k, strat in [(1, "ArchMoE top-1"), (2, "ArchMoE top-2")]:
            gate, sc = _train_arch_gate(P_oof, y1l, G1l, top_k=top_k)
            p_gate   = _arch_gate_predict(gate, sc, P_held, G3, top_k)
            results[strat][a] = float(spearmanr(y3[ok], p_gate[ok]).statistic)

        # 6. Oracle: best arch by held-out rho (upper bound)
        held_rhos  = {
            "AbDev-X":     float(spearmanr(y3[ok], p_ax[ok]).statistic),
            "AbDevMoE":    float(spearmanr(y3[ok], p_moe[ok]).statistic),
            "SparseMoE-2": float(spearmanr(y3[ok], p_sm[ok]).statistic),
            "KronLSTM":    float(spearmanr(y3[ok], p_kl[ok]).statistic),
        }
        oracle_arch = max(held_rhos, key=held_rhos.get)
        results["Oracle"][a] = held_rhos[oracle_arch]
        print(f"  Oracle picks: {oracle_arch} (held-out={held_rhos[oracle_arch]:.3f})", flush=True)

        # Quick summary
        print(f"\n  Strategy results for {a}:")
        for s in strategies:
            print(f"    {s:<20} {results[s][a]:.3f}")

    # ── Final report ──────────────────────────────────────────────────────────
    rows = []
    for s in strategies:
        r = {a: round(float(results[s][a]), 3) for a in ASSAYS}
        r["MEAN"]  = round(float(np.mean([results[s][a] for a in ASSAYS])), 3)
        r["Model"] = s
        rows.append(r)

    df = pd.DataFrame(rows)[["Model"] + ASSAYS + ["MEAN"]]
    pd.set_option("display.width", 240, "display.max_columns", 20)
    sep = "=" * 140
    print(f"\n\n{sep}")
    print("  Architecture-level MoE  |  GDPa3 HELD-OUT  |  Spearman rho")
    print(sep)
    print(df.to_string(index=False))
    print(sep)

    # OOD-CV summary
    print("\nOOD-CV Spearman rho per architecture per assay:")
    header = f"  {'':22}" + "".join(f"  {n:>14}" for n in ARCH_NAMES)
    print(header)
    for a in ASSAYS:
        row = f"  {a:<22}" + "".join(f"  {cv_rho_arch[a][n]:>14.3f}" for n in ARCH_NAMES)
        print(row)

    out = ROOT / "results" / "arch_moe_gdpa3.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved -> {out}")

    # Delta vs AbDev-X
    ax_mean = float(np.mean([results["AbDev-X"][a] for a in ASSAYS]))
    print("\nDelta vs AbDev-X:")
    for s in strategies[1:]:
        mean_s = float(np.mean([results[s][a] for a in ASSAYS]))
        print(f"  {s:<20} {mean_s:.3f}  (delta={mean_s - ax_mean:+.3f})")


if __name__ == "__main__":
    main()
