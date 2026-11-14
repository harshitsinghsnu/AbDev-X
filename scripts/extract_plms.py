"""Extract ProtBert + ProGen2 embeddings for GDPa1 (mean-pooled VH and VL, concatenated)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


def protbert(seqs, bs=16):
    from transformers import BertModel, BertTokenizer
    tok = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert").to(DEV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            batch = [" ".join(re.sub(r"[^A-Z]", "", s)) for s in seqs[i:i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=200).to(DEV)
            h = model(**enc).last_hidden_state
            out.append(mean_pool(h, enc["attention_mask"]).cpu().numpy())
            print(f"  protbert {min(i+bs,len(seqs))}/{len(seqs)}", flush=True)
    return np.vstack(out)


def progen2(seqs, bs=1):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "hugohrban/progen2-small"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).to(DEV).eval()
    out = []
    with torch.no_grad():
        for i, s in enumerate(seqs):  # per-sequence (no padding needed for causal LM)
            ids = tok(re.sub(r"[^A-Z]", "", s), return_tensors="pt").input_ids.to(DEV)
            h = model(input_ids=ids, output_hidden_states=True).hidden_states[-1]  # (1,L,D)
            out.append(h[0].mean(0).cpu().numpy())
            if (i + 1) % 32 == 0:
                print(f"  progen2 {i+1}/{len(seqs)}", flush=True)
    return np.vstack(out)


def build(name, fn, vh, vl):
    cache = ROOT / f"data/processed/{name}_gdpa1.npz"
    if cache.exists():
        print(f"{name} cached"); return
    try:
        h = fn(vh); l = fn(vl)
        emb = np.hstack([h, l])
        np.savez_compressed(cache, emb=emb)
        print(f"{name} {emb.shape} saved", flush=True)
    except Exception as e:
        print(f"{name} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    vh = ds.heavy.tolist(); vl = ds.light.tolist()
    build("protbert", protbert, vh, vl)
    build("progen2", progen2, vh, vl)


if __name__ == "__main__":
    main()
