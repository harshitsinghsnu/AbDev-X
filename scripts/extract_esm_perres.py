"""Precompute frozen ESM-2 per-residue embeddings for VH and VL (padding + CDR masks).

Cached once so the chain-aware head trains in seconds per fold (no repeated PLM forward pass).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np, pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1
from abdev.features.numbering import number_chain

ESM = "facebook/esm2_t30_150M_UR50D"
MAXLEN = 140
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cdr_vec(seq, L):
    v = np.zeros(L, dtype=np.float32)
    reg = number_chain(seq)
    if reg:
        for cdr in (reg.cdr1, reg.cdr2, reg.cdr3):
            if cdr:
                i = seq.find(cdr)
                if i >= 0:
                    v[i:i + len(cdr)] = 1.0
    return v


@torch.no_grad()
def embed(seqs, tok, model):
    N = len(seqs)
    E = np.zeros((N, MAXLEN, model.config.hidden_size), dtype=np.float16)
    M = np.zeros((N, MAXLEN), dtype=np.float32)
    C = np.zeros((N, MAXLEN), dtype=np.float32)
    for i in range(0, N, 16):
        chunk = seqs[i:i + 16]
        enc = tok(chunk, padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEVICE)
        h = model(**enc).last_hidden_state.float().cpu().numpy()  # (b, L, D)
        m = enc["attention_mask"].cpu().numpy()
        for b, s in enumerate(chunk):
            E[i + b] = h[b].astype(np.float16)
            M[i + b] = m[b]
            cv = cdr_vec(s, MAXLEN - 2)
            C[i + b, 1:1 + len(cv)] = cv
        print(f"  {min(i+16,N)}/{N}", flush=True)
    return E, M, C


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(ESM)
    model = AutoModel.from_pretrained(ESM).to(DEVICE).eval()
    print("VH...", flush=True); vh_e, vh_m, vh_c = embed(ds.heavy.tolist(), tok, model)
    print("VL...", flush=True); vl_e, vl_m, vl_c = embed(ds.light.tolist(), tok, model)
    out = ROOT / "data/processed/esm_perres_150M.npz"
    np.savez_compressed(out, vh_e=vh_e, vh_m=vh_m, vh_c=vh_c, vl_e=vl_e, vl_m=vl_m, vl_c=vl_c)
    print(f"saved {out}  VH{vh_e.shape} VL{vl_e.shape}", flush=True)


if __name__ == "__main__":
    main()
