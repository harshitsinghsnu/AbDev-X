"""AbDev-CAPLM (fast): chain-aware head on frozen ESM-2 per-residue embeddings.

Realizes the novel architecture from the reference papers (chain-aware VH/VL encoding, VH<->VL
cross-attention, CDR-aware attention pooling, multi-task heads, residue-attention explainability)
without backpropagating through the PLM -- appropriate for the tiny (n=246) regime and reliable/fast.

5-fold CV across generalization splits (random, cluster, cluster_isotype); aggregated Spearman.

  set KMP_DUPLICATE_LIB_OK=TRUE & set PYTHONPATH=src
  python scripts/train_caplm_fast.py --splits random cluster cluster_isotype --epochs 150
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd   # MKL libs before torch
import torch
import torch.nn as nn
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.scale = d ** -0.5

    def forward(self, H, pad, cdr):
        logits = (H @ self.q) * self.scale + self.gate * cdr
        logits = logits.masked_fill(~pad, float("-inf"))
        a = torch.softmax(logits, 1)
        return torch.einsum("bl,bld->bd", a, H), a


class ChainAwareHead(nn.Module):
    def __init__(self, in_dim=640, d=256, extra_dim=5, n_tasks=5, heads=4, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU())
        self.chain = nn.Embedding(2, d)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.pool_h, self.pool_l = AttnPool(d), AttnPool(d)
        self.trunk = nn.Sequential(nn.Linear(4 * d + extra_dim, d), nn.LayerNorm(d), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_tasks)])
        # direct un-bottlenecked path for the biological signal (isotype -> each assay)
        self.extra_head = nn.Linear(extra_dim, n_tasks)

    @staticmethod
    def _meanpool(H, pad):
        w = pad.float() / pad.float().sum(1, keepdim=True).clamp(min=1)
        return torch.einsum("bl,bld->bd", w, H)

    def forward(self, b, return_attn=False):
        Hh = self.proj(b["vh_e"]) + self.chain(torch.zeros(b["vh_e"].shape[0], dtype=torch.long, device=b["vh_e"].device))[:, None]
        Hl = self.proj(b["vl_e"]) + self.chain(torch.ones(b["vl_e"].shape[0], dtype=torch.long, device=b["vl_e"].device))[:, None]
        ph, pl = b["vh_m"].bool(), b["vl_m"].bool()
        hh, _ = self.cross(Hh, Hl, Hl, key_padding_mask=~pl)
        hl, _ = self.cross(Hl, Hh, Hh, key_padding_mask=~ph)
        Hh, Hl = self.norm(Hh + hh), self.norm(Hl + hl)
        vh, ah = self.pool_h(Hh, ph, b["vh_c"])
        vl, al = self.pool_l(Hl, pl, b["vl_c"])
        # mean-pool residual (stable, data-efficient) alongside CDR-aware attention pool
        mh, ml = self._meanpool(Hh, ph), self._meanpool(Hl, pl)
        z = self.trunk(torch.cat([vh, vl, mh, ml, b["extra"]], 1))
        pred = torch.cat([h(z) for h in self.heads], 1) + self.extra_head(b["extra"])
        return (pred, {"vh": ah, "vl": al}) if return_attn else pred


def masked_mse(p, y, m):
    return (((p - y) ** 2) * m).sum() / m.sum().clamp(min=1.0)


def load_data(ds):
    z = np.load(ROOT / "data/processed/esm_perres_150M.npz")
    T = {k: torch.tensor(z[k].astype(np.float32)) for k in ["vh_e", "vh_m", "vh_c", "vl_e", "vl_m", "vl_c"]}
    iso = pd.get_dummies(ds.df[["hc_subtype", "lc_subtype"]].astype(str)).to_numpy(np.float32)
    T["extra"] = torch.tensor(iso)
    Y = np.column_stack([ds.y(a) for a in ASSAYS]).astype(np.float32)
    return T, Y


def slice_batch(T, idx):
    return {k: T[k][idx].to(DEVICE) for k in T}


def train_fold(T, Y, tr, va, epochs, lr, patience, seed, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ChainAwareHead(extra_dim=T["extra"].shape[1], n_tasks=len(ASSAYS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mu, sd = np.nanmean(Y[tr], 0), np.nanstd(Y[tr], 0) + 1e-6
    Yn = np.nan_to_num((Y - mu) / sd); obs = (~np.isnan((Y - mu) / sd)).astype(np.float32)
    Yt = torch.tensor(Yn, device=DEVICE); Mt = torch.tensor(obs, device=DEVICE)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]; opt.zero_grad()
            loss = masked_mse(model(slice_batch(T, idx)), Yt[idx], Mt[idx])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = masked_mse(model(slice_batch(T, va)), Yt[va], Mt[va]).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and ep % 20 == 0:
            print(f"    ep{ep} val={vl:.3f}", flush=True)
        if bad >= patience:
            break
    if state:
        model.load_state_dict(state)
    return model, mu, sd


@torch.no_grad()
def predict(model, T, idx, mu, sd):
    model.eval(); out = []
    for i in range(0, len(idx), 32):
        out.append(model(slice_batch(T, idx[i:i + 32])).cpu().numpy())
    return np.vstack(out) * sd + mu


def folds_all(ds, split):
    f = ds.folds(split)
    for fid in sorted(set(f.tolist())):
        if fid == fid:
            te = np.where(f == fid)[0]; tr = np.where(f != fid)[0]
            if len(te) and len(tr):
                yield int(fid), tr, te


def run(ds, T, Y, split, epochs, lr, patience):
    pooled = np.full((len(Y), len(ASSAYS)), np.nan)
    for fid, tr, te in folds_all(ds, split):
        rng = np.random.RandomState(fid); s = rng.permutation(tr); c = int(0.85 * len(s))
        model, mu, sd = train_fold(T, Y, s[:c], s[c:], epochs, lr, patience, seed=fid)
        pooled[te] = predict(model, T, te, mu, sd)
        del model; torch.cuda.empty_cache()
    res = {}
    for j, a in enumerate(ASSAYS):
        m = ~np.isnan(Y[:, j]) & ~np.isnan(pooled[:, j])
        res[a] = float(spearmanr(Y[m, j], pooled[m, j]).statistic)
    res["MEAN"] = float(np.mean([res[a] for a in ASSAYS]))
    return res, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["random", "cluster", "cluster_isotype"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--check", action="store_true", help="verbose single-fold learn check then exit")
    args = ap.parse_args()
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    T, Y = load_data(ds)
    print(f"loaded per-residue embeddings; extra_dim={T['extra'].shape[1]}, device={DEVICE}", flush=True)

    if args.check:
        f = ds.folds("cluster_isotype"); tr = np.where(f != 0)[0]; te = np.where(f == 0)[0]
        rng = np.random.RandomState(0); s = rng.permutation(tr); c = int(0.85 * len(s))
        model, mu, sd = train_fold(T, Y, s[:c], s[c:], args.epochs, args.lr, args.patience, 0, verbose=True)
        P = predict(model, T, te, mu, sd)
        for j, a in enumerate(ASSAYS):
            m = ~np.isnan(Y[te, j]); print(f"  test {a}={spearmanr(Y[te][m, j], P[m, j]).statistic:.3f}", flush=True)
        return

    rows = []
    for split in args.splits:
        res, pooled = run(ds, T, Y, split, args.epochs, args.lr, args.patience)
        print(f"{split:16s} " + " ".join(f"{a}={res[a]:.3f}" for a in ASSAYS) + f"  MEAN={res['MEAN']:.3f}", flush=True)
        rows.append(dict(model="AbDev-CAPLM", split=split, **res))
        np.save(ROOT / f"results/caplm_pooled_{split}.npy", pooled)
    pd.DataFrame(rows).to_csv(ROOT / "results/caplm_splits.csv", index=False)
    print("Saved results/caplm_splits.csv")


if __name__ == "__main__":
    main()
