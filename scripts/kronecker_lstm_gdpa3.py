"""AbDev KronLSTM: Kronecker Delta Rule memory architecture on GDPa3.

Implements the architecture from:
  "KronLSTM: Kronecker Delta Rule for Long Context Linear Attention"
  (Hartl et al., arXiv:2610.26834)

Architecture
------------
Memory update (Eq. 10 of the paper):
  S_{t+1} = S_t F_t
            - beta_t * (v_t^T S_t F_t k_t) * v_t k_t^T     [Kronecker-delta erase]
            + c_t * v_t k_t^T                                [write]

where:
  S_t  in R^{d x d}   associative memory matrix
  k_t  in R^d          key  (L2-normalised)
  v_t  in R^d          value
  q_t  in R^d          query (L2-normalised)
  F_t  = diag(f_t)     per-element forget gate, f_t in (0,1)^d
  beta_t in (0,1)      scalar erase gate
  c_t    in (0,1)      scalar write gate

Read-out:
  h_tilde_t = S_t q_t                         [memory read]
  h_t       = W_out(o_t o LayerNorm(h_tilde_t)) [gated output]
  pred      = W_head(h_T)                     [scalar prediction after last block]

Applied to antibody features: T=6 representation blocks are processed as a
sequential input.  Each block is first reduced (StandardScaler + PCA, fitted
in-fold) then projected to d_proj=8 before entering the KronLSTM cell.

Block sequence order (coarse -> fine):
  descriptors -> isotype -> bio -> sap_electro -> esm2 -> ablang2

Usage:
    python scripts/kronecker_lstm_gdpa3.py
"""
from __future__ import annotations
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gdpa3_eval as GE
from moe_gdpa3 import _load_sap_electro

ASSAYS    = GE.ASSAYS
LABEL_MAP = GE.LABEL_MAP
COMP      = GE.COMP_HELDOUT

BLOCK_ORDER = ["descriptors", "isotype", "bio", "sap_electro", "esm2", "ablang2"]
D_PROJ = 8    # per-block projection dimension (= LSTM input & memory size)
D_HEAD = 8    # memory matrix dimension: S in R^{D_HEAD x D_HEAD}


# ── KronLSTM cell (paper Eq. 10) ─────────────────────────────────────────────

class KronLSTMCell(nn.Module):
    """
    Single KronLSTM step implementing the Kronecker Delta Rule update.

    Memory S in R^{d x d} is updated as:
      S_{t+1} = S_t diag(f_t)
                - beta_t * (v_t^T S_t diag(f_t) k_t) * outer(v_t, k_t)
                + c_t * outer(v_t, k_t)
    """
    def __init__(self, d_input: int, d_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_head = d_head

        # Key / Value / Query projections
        self.W_k = nn.Linear(d_input, d_head, bias=False)
        self.W_v = nn.Linear(d_input, d_head, bias=False)
        self.W_q = nn.Linear(d_input, d_head, bias=False)

        # Forget gate (per-dimension of key space)
        self.W_f = nn.Linear(d_input, d_head)

        # Scalar gates: erase (beta) and write (c)
        self.w_beta = nn.Linear(d_input, 1)
        self.w_c    = nn.Linear(d_input, 1)

        # Output gate (per-dimension)
        self.W_o   = nn.Linear(d_input, d_head)
        self.W_out = nn.Linear(d_head, d_input)
        self.norm  = nn.LayerNorm(d_head)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_k, self.W_v, self.W_q]:
            nn.init.orthogonal_(m.weight)
        # Initialise forget gate bias so default f ~ 0.9 (mild forgetting)
        nn.init.constant_(self.W_f.bias, 2.0)
        # Initialise erase and write gates so beta ~ 0.5, c ~ 0.5
        nn.init.zeros_(self.w_beta.bias)
        nn.init.zeros_(self.w_c.bias)

    def forward(self, x: torch.Tensor, S: torch.Tensor):
        """
        Args:
          x:  (N, d_input)     current block projection
          S:  (N, d_head, d_head)  memory matrix at time t

        Returns:
          h:     (N, d_input)   output representation
          S_new: (N, d_head, d_head)  updated memory
        """
        N = x.shape[0]

        k = F.normalize(self.W_k(x), dim=-1)    # (N, d_head)
        v = self.W_v(x)                          # (N, d_head)
        q = F.normalize(self.W_q(x), dim=-1)    # (N, d_head)

        f    = torch.sigmoid(self.W_f(x))        # (N, d_head)  forget gate
        beta = torch.sigmoid(self.w_beta(x))     # (N, 1)       erase gate
        c    = torch.sigmoid(self.w_c(x))        # (N, 1)       write gate
        o    = torch.sigmoid(self.W_o(x))        # (N, d_head)  output gate

        # ── Memory update ──────────────────────────────────────────────────
        # 1. Apply forget: S_f = S diag(f)  ->  multiply each column j by f[j]
        S_f = S * f.unsqueeze(1)                  # (N, d_head, d_head)

        # 2. Erase:  beta * (v^T S_f k) * v k^T
        #   S_f k  : (N, d_head)
        Sf_k = torch.einsum("nij,nj->ni", S_f, k)           # (N, d_head)
        #   v^T S_f k: (N,) scalar per batch
        vt_Sf_k = (v * Sf_k).sum(dim=-1)                    # (N,)
        #   outer product v k^T: (N, d_head, d_head)
        outer_vk = torch.einsum("ni,nj->nij", v, k)         # (N, d_v, d_k)
        erase = (beta * vt_Sf_k.unsqueeze(-1)).unsqueeze(-1) * outer_vk  # (N, d, d)

        # 3. Write: c * v k^T
        write = c.unsqueeze(-1) * outer_vk                   # (N, d, d)

        S_new = S_f - erase + write                          # (N, d, d)

        # ── Read-out ────────────────────────────────────────────────────────
        h_tilde = torch.einsum("nij,nj->ni", S_new, q)      # (N, d_head)
        h = self.W_out(o * self.norm(h_tilde))               # (N, d_input)

        return h, S_new


# ── Full model ────────────────────────────────────────────────────────────────

class KronLSTMModel(nn.Module):
    """Per-block projection -> KronLSTM over T=6 blocks -> scalar prediction."""
    def __init__(self, block_dims: dict[str, int],
                 d_proj: int, d_head: int, dropout: float = 0.2):
        super().__init__()
        self.block_order = BLOCK_ORDER
        self.projections = nn.ModuleDict({
            b: nn.Sequential(
                nn.Linear(block_dims[b], d_proj),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for b in BLOCK_ORDER if b in block_dims
        })
        self.kron_cell = KronLSTMCell(d_proj, d_head, dropout=dropout)
        self.head      = nn.Linear(d_proj, 1)
        self.d_head    = d_head
        self.d_proj    = d_proj

    def forward(self, blocks: dict[str, torch.Tensor]) -> torch.Tensor:
        N = next(iter(blocks.values())).shape[0]
        S = torch.zeros(N, self.d_head, self.d_head)
        h = torch.zeros(N, self.d_proj)
        for blk in self.block_order:
            if blk not in blocks:
                continue
            x = self.projections[blk](blocks[blk])   # (N, d_proj)
            h, S = self.kron_cell(x, S)
        return self.head(h).squeeze(-1)               # (N,)


# ── Block preprocessor ────────────────────────────────────────────────────────

class BlockPreprocessor:
    MAX_COMP = 30

    def __init__(self):
        self.pipes_: dict = {}

    def fit_transform(self, blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for b, X in blocks.items():
            X  = np.nan_to_num(X).astype(np.float32)
            sc = StandardScaler().fit(X)
            Xs = sc.transform(X)
            nc = min(self.MAX_COMP, X.shape[1], X.shape[0] - 1)
            if nc < X.shape[1] and nc >= 1:
                pca = PCA(n_components=nc, random_state=0,
                          svd_solver="full").fit(Xs)
                Xt  = pca.transform(Xs)
            else:
                pca = None
                Xt  = Xs
            self.pipes_[b] = (sc, pca)
            out[b] = Xt.astype(np.float32)
        return out

    def transform(self, blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for b, X in blocks.items():
            X = np.nan_to_num(X).astype(np.float32)
            sc, pca = self.pipes_[b]
            Xs = sc.transform(X)
            out[b] = (pca.transform(Xs) if pca is not None else Xs).astype(np.float32)
        return out

    def dims(self) -> dict[str, int]:
        d = {}
        for b, (sc, pca) in self.pipes_.items():
            d[b] = int(pca.n_components_) if pca is not None else int(sc.mean_.shape[0])
        return d


# ── Training / inference ──────────────────────────────────────────────────────

def _to_tensors(blocks: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {b: torch.FloatTensor(v) for b, v in blocks.items()}


def _train(blocks_tr, y_tr, blocks_va, y_va, block_dims,
           n_epochs=2000, lr=1e-3, wd=1e-3, patience=200, seed=0):
    torch.manual_seed(seed)
    model = KronLSTMModel(block_dims, D_PROJ, D_HEAD)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, eta_min=lr / 20)

    bt = _to_tensors(blocks_tr)
    bv = _to_tensors(blocks_va)
    yt = torch.FloatTensor(y_tr)
    yv = torch.FloatTensor(y_va)

    best_val, best_state, no_imp = float("inf"), None, 0
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        pred = model(bt)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = float(((model(bv) - yv) ** 2).mean())
        if vl < best_val:
            best_val  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _predict(model: KronLSTMModel, blocks: dict[str, np.ndarray]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(_to_tensors(blocks)).numpy()


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        try:
            _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    n_cell_params = sum(p.numel() for p in KronLSTMCell(D_PROJ, D_HEAD).parameters())
    print("=" * 70)
    print("  AbDev KronLSTM (Kronecker Delta Rule, arXiv:2610.26834)")
    print(f"  d_proj={D_PROJ}, d_head={D_HEAD},  "
          f"KronLSTMCell params: {n_cell_params}")
    print("=" * 70, flush=True)

    print("Loading data ...", flush=True)
    ds, g3, B1_raw, B3_raw = GE.build_blocks()

    sap1 = _load_sap_electro(ds.df["antibody_name"], "gdpa1")
    sap3 = _load_sap_electro(g3["antibody_name"],    "heldout")
    B1   = {**B1_raw, "sap_electro": sap1}
    B3   = {**B3_raw, "sap_electro": sap3}

    folds  = ds.folds("cluster_isotype")
    ufolds = sorted(f for f in set(folds.tolist()) if f == f)

    results: dict[str, dict] = {
        "abdevx_fusion":  {},
        "kronecker_lstm": {},
    }
    cv_rho: dict[str, list[float]] = {a: [] for a in ASSAYS}

    for a in ASSAYS:
        print(f"\n{'='*50}\n  Assay: {a}", flush=True)
        y1  = ds.y(a)
        lab = ~np.isnan(y1)
        y3  = g3[LABEL_MAP[a]].to_numpy(float)
        ok  = ~np.isnan(y3)
        B1m = {k: v[lab] for k, v in B1.items()}
        y1l = y1[lab]

        from abdev.models.fusion import AbDevFusion
        results["abdevx_fusion"][a] = float(
            spearmanr(y3[ok], AbDevFusion().fit(B1m, y1l).predict(B3)[ok]).statistic)
        print(f"  AbDev-X rho = {results['abdevx_fusion'][a]:.3f}", flush=True)

        fold_ids  = folds[lab]
        n_lab     = lab.sum()
        avail     = [b for b in BLOCK_ORDER if b in B1m]
        oof_kl    = np.full(n_lab, np.nan)

        for fi, f in enumerate(ufolds):
            tr = np.where(fold_ids != f)[0]
            va = np.where(fold_ids == f)[0]
            if len(va) < 3 or len(tr) < 10:
                continue

            prep = BlockPreprocessor()
            B_tr = prep.fit_transform({b: B1m[b][tr] for b in avail})
            B_va = prep.transform({b: B1m[b][va] for b in avail})

            model = _train(B_tr, y1l[tr], B_va, y1l[va], prep.dims())
            oof_kl[va] = _predict(model, B_va)

            rho_f = float(spearmanr(y1l[va], oof_kl[va]).statistic)
            cv_rho[a].append(rho_f)
            print(f"  fold {fi+1}/{len(ufolds)} (tr={len(tr)}, va={len(va)})  "
                  f"rho={rho_f:.3f}", flush=True)

        mean_cv = float(np.nanmean(cv_rho[a])) if cv_rho[a] else float("nan")
        print(f"  OOD-CV mean rho = {mean_cv:.3f}", flush=True)

        # Full-data refit with fold-0 as validation proxy for early stopping
        print("  Full-data refit -> GDPa3 ...", flush=True)
        prep_full = BlockPreprocessor()
        B_full    = prep_full.fit_transform({b: B1m[b] for b in avail})

        va_proxy = np.where(fold_ids == ufolds[0])[0]
        tr_proxy = np.where(fold_ids != ufolds[0])[0]
        B_va_px  = prep_full.transform({b: B1m[b][va_proxy] for b in avail})

        model_full = _train(
            {b: B_full[b][tr_proxy] for b in B_full},
            y1l[tr_proxy],
            B_va_px, y1l[va_proxy],
            prep_full.dims(),
            n_epochs=3000, patience=300, seed=1,
        )
        B3_proc = prep_full.transform({b: B3[b] for b in avail if b in B3})
        pred3   = _predict(model_full, B3_proc)
        results["kronecker_lstm"][a] = float(spearmanr(y3[ok], pred3[ok]).statistic)
        print(f"  GDPa3 rho = {results['kronecker_lstm'][a]:.3f}", flush=True)

    # Report
    label = {
        "abdevx_fusion":  "AbDev-X (fixed Ridge meta)",
        "kronecker_lstm": f"KronLSTM (d={D_PROJ}, d_head={D_HEAD}, arXiv:2610.26834)",
    }
    rows = []
    for m in ["abdevx_fusion", "kronecker_lstm"]:
        r = {a: round(float(results[m][a]), 3) for a in ASSAYS}
        r["MEAN"]  = round(float(np.mean([results[m][a] for a in ASSAYS])), 3)
        r["Model"] = label[m]
        rows.append(r)

    df = pd.DataFrame(rows)[["Model"] + ASSAYS + ["MEAN"]]
    pd.set_option("display.width", 240, "display.max_columns", 20)
    sep = "=" * 140
    print(f"\n\n{sep}")
    print("  KronLSTM  |  GDPa3 HELD-OUT  |  Spearman rho")
    print(sep)
    print(df.to_string(index=False))
    print(sep)

    print("\nOOD-CV Spearman rho (cluster_isotype):")
    for a in ASSAYS:
        if cv_rho[a]:
            print(f"  {a:22s}  {float(np.nanmean(cv_rho[a])):.3f}")

    out = ROOT / "results" / "kronecker_lstm_gdpa3.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved -> {out}")

    mean_kl = float(np.mean([results["kronecker_lstm"][a] for a in ASSAYS]))
    mean_ax = float(np.mean([results["abdevx_fusion"][a]  for a in ASSAYS]))
    print(f"\nKronLSTM  mean rho={mean_kl:.3f}  AbDev-X={mean_ax:.3f}  "
          f"delta={mean_kl - mean_ax:+.3f}")
    for a in ASSAYS:
        d = results["kronecker_lstm"][a] - results["abdevx_fusion"][a]
        print(f"  {a:22s}  KronLSTM={results['kronecker_lstm'][a]:.3f}  "
              f"AbDev-X={results['abdevx_fusion'][a]:.3f}  d={d:+.3f}")


if __name__ == "__main__":
    main()
