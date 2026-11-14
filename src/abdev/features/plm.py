"""Protein-language-model embeddings for antibody Fv sequences.

Two backends, concatenated when both are available:
  * ESM-2 (general PLM) via HuggingFace transformers  -- mirrors the competition baseline view.
  * AbLang2 (antibody-specific PLM) via the `ablang2` package -- antibody grammar.

Both produce per-residue hidden states; we pool to fixed-size vectors with mean pooling and,
when CDR boundaries are known, CDR-focused pooling (developability signal concentrates in CDRs).
Embeddings are cached to disk (npz) keyed by model + sequence hash so the (expensive) forward
pass runs once. Requires a GPU for practical throughput on ESM-2 650M.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .numbering import number_chain


def _seq_hash(model: str, seqs: List[str]) -> str:
    h = hashlib.sha1(model.encode())
    for s in seqs:
        h.update(b"\x00")
        h.update(s.encode())
    return h.hexdigest()[:16]


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _cdr_mask(seq: str, length: int) -> np.ndarray:
    """Boolean residue mask marking CDR positions (best-effort via numbering)."""
    mask = np.zeros(length, dtype=bool)
    regions = number_chain(seq)
    if regions is None:
        return mask
    for cdr in (regions.cdr1, regions.cdr2, regions.cdr3):
        if not cdr:
            continue
        idx = seq.find(cdr)
        if idx >= 0:
            mask[idx: idx + len(cdr)] = True
    return mask


# ---------------------------------------------------------------------------- ESM-2
def esm2_embeddings(seqs: List[str], model_name: str, pool=("mean", "cdr"),
                    batch_size: int = 8, cache_dir: str = "data/processed") -> np.ndarray:
    cache = Path(cache_dir) / f"esm2_{_seq_hash(model_name + str(pool), seqs)}.npz"
    if cache.exists():
        return np.load(cache)["emb"]

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = _device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i: i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(device)
            hs = model(**enc).last_hidden_state.float().cpu().numpy()  # (B, L, D)
            attn = enc["attention_mask"].cpu().numpy().astype(bool)
            for b, seq in enumerate(batch):
                valid = attn[b]
                res = hs[b][valid][1:-1]  # drop BOS/EOS
                vecs = []
                if "mean" in pool:
                    vecs.append(res.mean(0))
                if "cdr" in pool:
                    cm = _cdr_mask(seq, res.shape[0])
                    vecs.append(res[cm].mean(0) if cm.any() else res.mean(0))
                out.append(np.concatenate(vecs))
    emb = np.vstack(out)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, emb=emb)
    return emb


# ---------------------------------------------------------------------------- AbLang2
def ablang2_embeddings(heavy: List[str], light: List[str],
                       cache_dir: str = "data/processed") -> Optional[np.ndarray]:
    cache = Path(cache_dir) / f"ablang2_{_seq_hash('ablang2', [h + '|' + l for h, l in zip(heavy, light)])}.npz"
    if cache.exists():
        return np.load(cache)["emb"]
    try:
        import ablang2
    except ImportError:
        print("[warn] ablang2 not installed; skipping antibody-specific embeddings.")
        return None

    model = ablang2.pretrained()
    pairs = [[h, l] for h, l in zip(heavy, light)]
    emb = np.asarray(model(pairs, mode="seqcoding"))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, emb=emb)
    return emb


def build_plm_matrix(heavy: pd.Series, light: pd.Series, cfg: dict) -> pd.DataFrame:
    """Concatenate enabled PLM embeddings into a named feature frame (aligned to index)."""
    f = cfg["features"]
    blocks, names = [], []
    fv = (heavy.astype(str) + light.astype(str)).tolist()

    if f.get("use_esm2", True):
        e = esm2_embeddings(fv, f["esm2_model"], tuple(f.get("pool", ["mean"])),
                            cache_dir=cfg["data"]["processed_dir"])
        blocks.append(e)
        names += [f"esm2_{i}" for i in range(e.shape[1])]

    if f.get("use_ablang2", True):
        a = ablang2_embeddings(heavy.astype(str).tolist(), light.astype(str).tolist(),
                               cache_dir=cfg["data"]["processed_dir"])
        if a is not None:
            blocks.append(a)
            names += [f"ablang2_{i}" for i in range(a.shape[1])]

    if not blocks:
        return pd.DataFrame(index=heavy.index)
    mat = np.hstack(blocks)
    return pd.DataFrame(mat, columns=names, index=heavy.index)
