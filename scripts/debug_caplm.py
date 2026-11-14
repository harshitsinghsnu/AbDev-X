"""Diagnostic: can AbDev-CAPLM learn? Train on one fold, print per-epoch train loss + train/test rho."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd  # import numpy/pandas (MKL) BEFORE torch to avoid Windows OMP segfault
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from abdev.data.dataset import load_gdpa1
from abdev.models.chain_aware import CAPLMConfig, ChainAwareDevModel, masked_mse
import train_chain_aware as TC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
from transformers import AutoTokenizer
esm = "facebook/esm2_t30_150M_UR50D"
tok = AutoTokenizer.from_pretrained(esm)
cfg = CAPLMConfig(esm_name=esm, n_tasks=len(TC.ASSAYS));
T, Y = TC.build_tensors(ds, cfg, tok); cfg.extra_dim = T["extra"].shape[1]

folds = ds.folds("cluster_isotype")
tr = np.where(folds != 0)[0]; te = np.where(folds == 0)[0]
mu = np.nanmean(Y[tr], 0); sd = np.nanstd(Y[tr], 0) + 1e-6
Yn = (Y - mu) / sd; obs = (~np.isnan(Yn)).astype(np.float32); Yn = np.nan_to_num(Yn)
Yt = torch.tensor(Yn, device=DEVICE); Mt = torch.tensor(obs, device=DEVICE)

torch.manual_seed(0); np.random.seed(0)
model = ChainAwareDevModel(cfg).to(DEVICE)
# separate LRs: new modules high, LoRA lower
lora_p = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]
new_p = [p for n, p in model.named_parameters() if p.requires_grad and "lora" not in n.lower()]
print(f"trainable: lora={sum(p.numel() for p in lora_p):,} new={sum(p.numel() for p in new_p):,}", flush=True)
opt = torch.optim.AdamW([{"params": lora_p, "lr": 3e-4}, {"params": new_p, "lr": 2e-3}], weight_decay=0.01)

def rho(idx):
    model.eval()
    with torch.no_grad():
        p = []
        for i in range(0, len(idx), 16):
            p.append(model(TC.batch_slice(T, idx[i:i+16], DEVICE)).cpu().numpy())
    P = np.vstack(p)
    out = []
    for j, a in enumerate(TC.ASSAYS):
        m = ~np.isnan(Y[idx, j])
        out.append(spearmanr(Y[idx][m, j], P[m, j]).statistic)
    return np.array(out)

bs = 8
for ep in range(25):
    model.train(); perm = np.random.permutation(tr); tot = 0
    for i in range(0, len(perm), bs):
        idx = perm[i:i+bs]; opt.zero_grad()
        loss = masked_mse(model(TC.batch_slice(T, idx, DEVICE)), Yt[idx], Mt[idx])
        loss.backward(); torch.nn.utils.clip_grad_norm_(lora_p + new_p, 1.0); opt.step()
        tot += loss.item()
    if ep % 3 == 0 or ep == 24:
        rtr = rho(tr); rte = rho(te)
        print(f"ep{ep:2d} loss={tot/ (len(perm)//bs+1):.3f} | train rho HIC={rtr[0]:.2f} Tm2={rtr[1]:.2f} mean={rtr.mean():.2f}"
              f" | test HIC={rte[0]:.2f} Tm2={rte[1]:.2f} mean={rte.mean():.2f}", flush=True)
