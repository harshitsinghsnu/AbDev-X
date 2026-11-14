"""Extract AntiBERTy (OAS-trained antibody LM) embeddings for GDPa1 + GDPa3.

Per chain: mean-pool the per-residue hidden states (512-d); concatenate VH|VL -> 1024-d per antibody.
AntiBERTy is trained on 558M OAS sequences, so it should transfer well to the OAS-sampled GDPa3.
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


def embed_pairs(vh, vl):
    from antiberty import AntiBERTyRunner
    runner = AntiBERTyRunner()
    out = []
    B = 32
    seqs_h = [str(s) for s in vh]; seqs_l = [str(s) for s in vl]
    for i in range(0, len(seqs_h), B):
        eh = runner.embed(seqs_h[i:i + B]); el = runner.embed(seqs_l[i:i + B])
        for h, l in zip(eh, el):
            vh_vec = h.mean(0).cpu().numpy(); vl_vec = l.mean(0).cpu().numpy()
            out.append(np.concatenate([vh_vec, vl_vec]))
        print(f"  {min(i + B, len(seqs_h))}/{len(seqs_h)}", flush=True)
    return np.vstack(out)


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    e1 = embed_pairs(ds.heavy.tolist(), ds.light.tolist())
    np.savez_compressed(ROOT / "data/processed/antiberty_gdpa1.npz", emb=e1)
    print(f"GDPa1 AntiBERTy {e1.shape}", flush=True)

    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    e3 = embed_pairs(g3["vh"].tolist(), g3["vl"].tolist())
    np.savez_compressed(ROOT / "data/processed/antiberty_gdpa3.npz", emb=e3)
    print(f"GDPa3 AntiBERTy {e3.shape}", flush=True)


if __name__ == "__main__":
    main()
