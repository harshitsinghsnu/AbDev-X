"""Extract AbLang2 (paired) seqcoding embeddings for GDPa1, cached to npz.

Run with the OpenMP-conflict workaround on Windows:
    set KMP_DUPLICATE_LIB_OK=TRUE
    set PYTHONPATH=src
    python scripts/extract_ablang2.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    print(f"loaded {len(ds.df)} antibodies", flush=True)
    import torch, ablang2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1, device=dev)
    print(f"AbLang2 ready on {dev}", flush=True)
    pairs = [[str(h), str(l)] for h, l in zip(ds.heavy.tolist(), ds.light.tolist())]
    embs = []
    B = 16
    for i in range(0, len(pairs), B):
        embs.append(np.asarray(model(pairs[i:i + B], mode="seqcoding")))
        print(f"  {min(i + B, len(pairs))}/{len(pairs)}", flush=True)
    E = np.vstack(embs)
    out = ROOT / "data/processed/ablang2_gdpa1.npz"
    np.savez_compressed(out, emb=E)
    print(f"saved {E.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
