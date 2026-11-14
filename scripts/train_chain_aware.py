"""Train/evaluate the Chain-Aware LoRA-fine-tuned PLM (AbDev-CAPLM) on GDPa1.

5-fold CV under MULTIPLE generalization-split strategies (the official GDPa1 fold columns):
  * random           -> in-distribution (IID)
  * cluster          -> sequence-cluster hold-out (harder generalization)
  * cluster_isotype  -> isotype-stratified cluster (Ginkgo's recommended reporting split)

One multi-task model predicts all 5 assays (masked loss for missing labels). Aggregated Spearman
(pool out-of-fold predictions) per assay, matching the official benchmark metric.

Usage:
  set KMP_DUPLICATE_LIB_OK=TRUE & set PYTHONPATH=src
  python scripts/train_chain_aware.py --splits cluster_isotype --epochs 40
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.numbering import number_chain
from abdev.models.chain_aware import CAPLMConfig, ChainAwareDevModel, masked_mse

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cdr_vector(seq: str, L: int) -> np.ndarray:
    v = np.zeros(L, dtype=np.float32)
    reg = number_chain(seq)
    if reg is None:
        return v
    for cdr in (reg.cdr1, reg.cdr2, reg.cdr3):
        if cdr:
            i = seq.find(cdr)
            if i >= 0:
                v[i:i + len(cdr)] = 1.0
    return v


def tokenize_all(seqs, tok, max_len):
    enc = tok(list(seqs), padding="max_length", truncation=True, max_length=max_len,
              return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def build_tensors(ds, cfg, tok):
    heavy = ds.heavy.tolist(); light = ds.light.tolist()
    vh_ids, vh_mask = tokenize_all(heavy, tok, cfg.max_len)
    vl_ids, vl_mask = tokenize_all(light, tok, cfg.max_len)
    # CDR masks aligned to token positions (offset +1 for CLS)
    vh_cdr = torch.zeros_like(vh_mask, dtype=torch.float32)
    vl_cdr = torch.zeros_like(vl_mask, dtype=torch.float32)
    for i, (h, l) in enumerate(zip(heavy, light)):
        ch = cdr_vector(h, cfg.max_len - 2); cl = cdr_vector(l, cfg.max_len - 2)
        vh_cdr[i, 1:1 + len(ch)] = torch.tensor(ch)
        vl_cdr[i, 1:1 + len(cl)] = torch.tensor(cl)
    # extra: germline/isotype one-hot
    iso = pd.get_dummies(ds.df[["hc_subtype", "lc_subtype"]].astype(str)).to_numpy(np.float32)
    extra = torch.tensor(iso)
    Y = np.column_stack([ds.y(a) for a in ASSAYS]).astype(np.float32)
    return dict(vh_ids=vh_ids, vh_mask=vh_mask, vl_ids=vl_ids, vl_mask=vl_mask,
                vh_cdr=vh_cdr, vl_cdr=vl_cdr, extra=extra), Y


def batch_slice(T, idx, device):
    return {k: T[k][idx].to(device) for k in
            ["vh_ids", "vh_mask", "vl_ids", "vl_mask", "vh_cdr", "vl_cdr", "extra"]}


def train_one(T, Y, tr, va, cfg, epochs, lr, patience, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ChainAwareDevModel(cfg).to(DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    # in-fold target standardization (ignore NaN)
    mu = np.nanmean(Y[tr], 0); sd = np.nanstd(Y[tr], 0) + 1e-6
    Yn = (Y - mu) / sd
    obs = (~np.isnan(Yn)).astype(np.float32); Yn = np.nan_to_num(Yn)
    Yt = torch.tensor(Yn, device=DEVICE); Mt = torch.tensor(obs, device=DEVICE)

    best_val, best_state, bad = 1e9, None, 0
    bs = 8
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = model(batch_slice(T, idx, DEVICE))
            loss = masked_mse(pred, Yt[idx], Mt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            vp = model(batch_slice(T, va, DEVICE))
            vloss = masked_mse(vp, Yt[va], Mt[va]).item()
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, mu, sd


@torch.no_grad()
def predict(model, T, idx, mu, sd):
    model.eval()
    out = []
    for i in range(0, len(idx), 16):
        sl = idx[i:i + 16]
        p = model(batch_slice(T, sl, DEVICE)).cpu().numpy()
        out.append(p)
    return np.vstack(out) * sd + mu


def run_split(ds, T, Y, cfg, split, epochs, lr, patience):
    n = len(Y)
    pooled = np.full((n, len(ASSAYS)), np.nan)
    for fid, tr, te in fold_iter_all(ds, split):
        # inner val = last 15% of train (shuffled)
        rng = np.random.RandomState(fid)
        trs = rng.permutation(tr); cut = int(0.85 * len(trs))
        tin, vin = trs[:cut], trs[cut:]
        model, mu, sd = train_one(T, Y, tin, vin, cfg, epochs, lr, patience, seed=fid)
        pooled[te] = predict(model, T, te, mu, sd)
        del model; torch.cuda.empty_cache()
        print(f"    fold {fid} done ({len(te)} test)", flush=True)
    res = {}
    for j, a in enumerate(ASSAYS):
        m = ~np.isnan(Y[:, j]) & ~np.isnan(pooled[:, j])
        res[a] = float(spearmanr(Y[m, j], pooled[m, j]).statistic)
    res["MEAN"] = float(np.mean([res[a] for a in ASSAYS]))
    return res, pooled


def fold_iter_all(ds, split):
    """Iterate folds over ALL rows (multi-task model uses masked loss for missing labels)."""
    folds = ds.folds(split)
    for f in sorted(set(folds.tolist())):
        if f != f:
            continue
        te = np.where(folds == f)[0]; tr = np.where(folds != f)[0]
        if len(te) and len(tr):
            yield int(f), tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="+", default=["cluster_isotype"])
    ap.add_argument("--esm", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    ds = load_gdpa1(str(ROOT / args.config))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.esm)
    cfg = CAPLMConfig(esm_name=args.esm, n_tasks=len(ASSAYS))
    T, Y = build_tensors(ds, cfg, tok)
    cfg.extra_dim = T["extra"].shape[1]
    print(f"tensors ready: extra_dim={cfg.extra_dim}, device={DEVICE}, esm={args.esm}", flush=True)

    rows = []
    for split in args.splits:
        if split not in ds.fold_cols:
            print(f"[skip] split {split} not in {list(ds.fold_cols)}"); continue
        print(f"\n=== generalization split: {split} ({ds.fold_cols[split]}) ===", flush=True)
        res, pooled = run_split(ds, T, Y, cfg, split, args.epochs, args.lr, args.patience)
        print("  " + " ".join(f"{a}={res[a]:.3f}" for a in ASSAYS) + f"  MEAN={res['MEAN']:.3f}", flush=True)
        rows.append(dict(split=split, **res))
        np.save(ROOT / f"results/caplm_pooled_{split}.npy", pooled)

    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results/caplm_cv.csv", index=False)
    print("\n=== AbDev-CAPLM across generalization splits ===")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
