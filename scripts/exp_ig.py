"""Experiment (10): Integrated Gradients on the chain-aware net -> per-residue attribution + CDR
enrichment (learned-attribution complement to the mechanistic SAP maps)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import train_caplm_fast as TC
from abdev.data.dataset import load_gdpa1
ASSAYS = TC.ASSAYS
DEV = TC.DEVICE


def integrated_gradients(model, batch, task, steps=24):
    """IG of output[:,task] w.r.t. vh_e and vl_e embeddings (baseline = zeros)."""
    model.eval()
    vh0, vl0 = batch["vh_e"], batch["vl_e"]
    attr = {"vh": torch.zeros_like(vh0), "vl": torch.zeros_like(vl0)}
    for s in range(1, steps + 1):
        alpha = s / steps
        b = dict(batch)
        b["vh_e"] = (alpha * vh0).clone().requires_grad_(True)
        b["vl_e"] = (alpha * vl0).clone().requires_grad_(True)
        out = model(b)[:, task].sum()
        gvh, gvl = torch.autograd.grad(out, [b["vh_e"], b["vl_e"]])
        attr["vh"] += gvh.detach() / steps
        attr["vl"] += gvl.detach() / steps
    ig_vh = (vh0 * attr["vh"]).abs().sum(-1)   # (B, L) per-residue importance
    ig_vl = (vl0 * attr["vl"]).abs().sum(-1)
    return ig_vh.cpu().numpy(), ig_vl.cpu().numpy()


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    T, Y = TC.load_data(ds)
    # train multitask chain-aware head on ALL data (short)
    tr = np.arange(len(Y)); rng = np.random.RandomState(0); rng.shuffle(tr)
    model, mu, sd = TC.train_fold(T, Y, tr[:220], tr[220:], epochs=80, lr=2e-3, patience=25, seed=0)

    rows = []
    for ti, assay in enumerate(ASSAYS):
        # IG over a sample of antibodies with a label for this assay
        idx = np.where(~np.isnan(Y[:, ti]))[0][:60]
        b = TC.slice_batch(T, idx)
        ig_vh, ig_vl = integrated_gradients(model, b, ti)
        cdr_frac, fw_frac = [], []
        for r, i in enumerate(idx):
            for ig, mask, cdr in [(ig_vh[r], T["vh_m"][i].numpy(), T["vh_c"][i].numpy()),
                                  (ig_vl[r], T["vl_m"][i].numpy(), T["vl_c"][i].numpy())]:
                valid = mask.astype(bool)
                if valid.sum() < 5:
                    continue
                v = ig.copy(); v[~valid] = -np.inf
                k = max(5, int(0.15 * valid.sum()))
                top = np.argsort(v)[::-1][:k]
                cdr_frac.append(np.mean(cdr[top] > 0.5))
        base = np.mean([(T["vh_c"][i].numpy()[T["vh_m"][i].numpy().astype(bool)] > 0.5).mean() for i in idx])
        enr = float(np.mean(cdr_frac)) / (base + 1e-9)
        rows.append(dict(assay=assay, IG_CDR_hotspot_frac=round(float(np.mean(cdr_frac)), 3),
                         CDR_baseline_frac=round(float(base), 3), enrichment=round(enr, 2)))
    df = pd.DataFrame(rows); df.to_csv(ROOT / "results/exp_ig.csv", index=False)
    print("=== (10) Integrated Gradients (chain-aware net): CDR enrichment of top residues ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
