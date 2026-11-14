"""AbDev Sparse-MoE: top-K neural gating over Ridge/LightGBM experts on GDPa3.

Architecture
------------
Experts: 7 sklearn Ridge/LightGBM heads — identical to moe_gdpa3.py.
Gate:    Linear(13, K) gate trained with soft Gumbel routing (temp anneal 2→0.3)
         and a load-balance auxiliary loss.  At inference: hard top-K mask applied
         to soft weights, renormalised.
Pred:    ŷ = Σ_k  w_k(g_i) · expert_k(x_i)   (no Ridge on top of gate).

Motivation
----------
AbDevMoE's Ridge meta has 7 + 7×13 = 98 features for n≈246 — high-dim even for
Ridge.  Sparse routing reduces the effective combination to top-K experts per
antibody, acting as a stronger inductive bias.

Usage:
    python scripts/sparse_moe_gdpa3.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gdpa3_eval as GE
from moe_gdpa3 import (AbDevMoE, EXPERT_SPEC, BASE_EXPERTS,
                        _load_sap_electro, build_gate)

ASSAYS    = GE.ASSAYS
LABEL_MAP = GE.LABEL_MAP
COMP      = GE.COMP_HELDOUT


# ── Sparse gate (PyTorch) ─────────────────────────────────────────────────────

def _train_sparse_gate(
    P_oof:      np.ndarray,   # (n, K) — OOF expert predictions
    y:          np.ndarray,   # (n,)
    G:          np.ndarray,   # (n, d_gate)
    top_k:      int   = 2,
    n_epochs:   int   = 2000,
    lr:         float = 5e-3,
    wd:         float = 1e-3,
    temp_start: float = 2.0,
    temp_end:   float = 0.3,
    lam_lb:     float = 0.02,  # load-balance coefficient
    patience:   int   = 250,
    seed:       int   = 0,
):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    N, K = P_oof.shape
    P_t  = torch.FloatTensor(P_oof)
    y_t  = torch.FloatTensor(y)
    G_t  = torch.FloatTensor(G)

    gate = nn.Linear(G.shape[1], K)
    nn.init.zeros_(gate.weight)
    nn.init.zeros_(gate.bias)
    opt  = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=wd)

    best_loss, best_state, no_improve = float("inf"), None, 0
    for ep in range(n_epochs):
        opt.zero_grad()
        temp    = temp_start * (temp_end / temp_start) ** (ep / n_epochs)
        logits  = gate(G_t)                              # (N, K)
        weights = torch.softmax(logits / temp, dim=-1)  # (N, K)

        y_hat   = (weights * P_t).sum(dim=-1)
        mse     = ((y_hat - y_t) ** 2).mean()

        # Load-balance: penalise high variance in per-expert mean routing mass
        mean_load = weights.mean(dim=0)   # (K,)
        lb        = mean_load.var() * K

        loss = mse + lam_lb * lb
        loss.backward()
        opt.step()

        v = float(loss.item())
        if v < best_loss:
            best_loss  = v
            best_state = {k: val.clone() for k, val in gate.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        gate.load_state_dict(best_state)
    gate.eval()
    return gate


def _sparse_predict(gate, P_held: np.ndarray, G_held: np.ndarray, top_k: int = 2) -> np.ndarray:
    import torch
    with torch.no_grad():
        logits    = gate(torch.FloatTensor(G_held))          # (m, K)
        _, idx    = torch.topk(logits, top_k, dim=-1)        # hard top-K indices
        hard_mask = torch.zeros_like(logits).scatter(-1, idx, 1.0)
        soft_w    = torch.softmax(logits, dim=-1)
        weights   = hard_mask * soft_w
        weights   = weights / (weights.sum(-1, keepdim=True) + 1e-9)
        return (weights * torch.FloatTensor(P_held)).sum(-1).numpy()


# ── SparseMoE estimator ───────────────────────────────────────────────────────

class SparseMoE:
    """Top-K neural gate over Ridge/LightGBM experts (leakage-free OOF training)."""

    def __init__(self, experts: list[str] = BASE_EXPERTS,
                 inner_folds: int = 5, top_k: int = 2,
                 gate_epochs: int = 2000, random_state: int = 0):
        self.experts     = experts
        self.inner_folds = inner_folds
        self.top_k       = top_k
        self.gate_epochs = gate_epochs
        self.rs          = random_state

    def _get(self, Xb: dict, name: str) -> np.ndarray:
        return np.nan_to_num(np.asarray(Xb[EXPERT_SPEC[name][1]], float))

    def fit(self, Xb: dict, y: np.ndarray, G: np.ndarray):
        y = np.asarray(y, float)
        G = np.asarray(G, float)
        n = len(y)
        kf = KFold(self.inner_folds, shuffle=True, random_state=self.rs)

        oof = {k: np.full(n, np.nan) for k in self.experts}
        for tr, va in kf.split(np.arange(n)):
            for k in self.experts:
                mk = EXPERT_SPEC[k][0]
                oof[k][va] = mk().fit(
                    self._get(Xb, k)[tr], y[tr]
                ).predict(self._get(Xb, k)[va])

        P_oof = np.column_stack([oof[k] for k in self.experts])  # (n, K)
        valid = ~np.any(np.isnan(P_oof), axis=1)

        self.G_sc_ = StandardScaler().fit(G[valid])
        Gs = self.G_sc_.transform(G[valid])

        self.gate_ = _train_sparse_gate(
            P_oof[valid], y[valid], Gs,
            top_k=self.top_k, n_epochs=self.gate_epochs, seed=self.rs
        )

        self.fitted_ = {k: EXPERT_SPEC[k][0]().fit(self._get(Xb, k), y)
                        for k in self.experts}
        return self

    def predict(self, Xb: dict, G: np.ndarray) -> np.ndarray:
        Gs     = self.G_sc_.transform(np.asarray(G, float))
        P_held = np.column_stack([
            self.fitted_[k].predict(self._get(Xb, k))
            for k in self.experts
        ])
        return _sparse_predict(self.gate_, P_held, Gs, top_k=self.top_k)


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        try:
            _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 65)
    print("  AbDev Sparse-MoE (top-K neural gate)")
    print("=" * 65, flush=True)

    print("Loading data ...", flush=True)
    ds, g3, B1_raw, B3_raw = GE.build_blocks()

    sap1 = _load_sap_electro(ds.df["antibody_name"], "gdpa1")
    sap3 = _load_sap_electro(g3["antibody_name"],    "heldout")
    B1   = {**B1_raw, "sap_electro": sap1}
    B3   = {**B3_raw, "sap_electro": sap3}

    print("Building gate features ...", flush=True)
    G1, G3 = build_gate(B1, B3)

    results: dict[str, dict] = {
        "abdevx_fusion": {},
        "moe_soft":      {},
        "sparse_moe_k2": {},
        "sparse_moe_k3": {},
    }

    for a in ASSAYS:
        print(f"\n--- {a} ---", flush=True)
        y1  = ds.y(a)
        lab = ~np.isnan(y1)
        y3  = g3[LABEL_MAP[a]].to_numpy(float)
        ok  = ~np.isnan(y3)
        B1m = {k: v[lab] for k, v in B1.items()}

        from abdev.models.fusion import AbDevFusion
        results["abdevx_fusion"][a] = float(
            spearmanr(y3[ok], AbDevFusion().fit(B1m, y1[lab]).predict(B3)[ok]).statistic)

        print("  Fitting soft MoE ...", flush=True)
        moe = AbDevMoE()
        moe.fit(B1m, y1[lab], G1[lab])
        results["moe_soft"][a] = float(
            spearmanr(y3[ok], moe.predict(B3, G3)[ok]).statistic)

        print("  Fitting sparse MoE (top-2) ...", flush=True)
        sm2 = SparseMoE(top_k=2)
        sm2.fit(B1m, y1[lab], G1[lab])
        results["sparse_moe_k2"][a] = float(
            spearmanr(y3[ok], sm2.predict(B3, G3)[ok]).statistic)

        print("  Fitting sparse MoE (top-3) ...", flush=True)
        sm3 = SparseMoE(top_k=3)
        sm3.fit(B1m, y1[lab], G1[lab])
        results["sparse_moe_k3"][a] = float(
            spearmanr(y3[ok], sm3.predict(B3, G3)[ok]).statistic)

    model_order = ["abdevx_fusion", "moe_soft", "sparse_moe_k2", "sparse_moe_k3"]
    label = {
        "abdevx_fusion": "AbDev-X (fixed Ridge meta)",
        "moe_soft":      "AbDevMoE (soft Ridge interaction)",
        "sparse_moe_k2": "SparseMoE (top-2 neural gate)",
        "sparse_moe_k3": "SparseMoE (top-3 neural gate)",
    }
    rows = []
    for m in model_order:
        r = {a: round(float(results[m][a]), 3) for a in ASSAYS}
        r["MEAN"]  = round(float(np.mean([results[m][a] for a in ASSAYS])), 3)
        r["Model"] = label[m]
        rows.append(r)

    df = pd.DataFrame(rows)[["Model"] + ASSAYS + ["MEAN"]]
    pd.set_option("display.width", 220, "display.max_columns", 20)
    sep = "=" * 140
    print(f"\n\n{sep}")
    print("  SparseMoE  |  GDPa3 HELD-OUT  |  Spearman rho")
    print(sep)
    print(df.to_string(index=False))
    print(sep)

    out = ROOT / "results" / "sparse_moe_gdpa3.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved -> {out}")

    for m in ["sparse_moe_k2", "sparse_moe_k3"]:
        mean_m  = float(np.mean([results[m][a]           for a in ASSAYS]))
        mean_ax = float(np.mean([results["abdevx_fusion"][a] for a in ASSAYS]))
        print(f"\n{label[m]}  delta vs AbDev-X = {mean_m - mean_ax:+.3f}")
        for a in ASSAYS:
            d = results[m][a] - results["abdevx_fusion"][a]
            print(f"  {a:22s}  {results[m][a]:.3f}  (AbDev-X={results['abdevx_fusion'][a]:.3f}, d={d:+.3f})")


if __name__ == "__main__":
    main()
