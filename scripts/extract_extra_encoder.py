"""Extract paired VH/VL embeddings for an arbitrary Hugging Face encoder.

The output format matches scripts/limitations_suite.py:
  data/processed/<name>_gdpa1.npz with array key "emb"
  data/processed/<name>_gdpa3.npz with array key "emb"

Examples:
  python scripts/extract_extra_encoder.py --name igbert --model <HF_OR_LOCAL_MODEL>
  python scripts/extract_extra_encoder.py --name igt5 --model <HF_OR_LOCAL_MODEL> --trust-remote-code

This script does not hard-code IgBERT/IgT5 model IDs because checkpoint names can
change. Pass either a Hugging Face model ID or a local checkpoint directory.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1  # noqa: E402


def clean_sequence(seq: str, spaced: bool) -> str:
    seq = re.sub(r"[^A-Z]", "", str(seq).upper())
    return " ".join(seq) if spaced else seq


def should_space_sequences(tokenizer, sample_seq: str, requested_spaced: bool, no_auto_spaced: bool) -> bool:
    if requested_spaced or no_auto_spaced:
        return requested_spaced
    compact = clean_sequence(sample_seq, spaced=False)
    spaced = clean_sequence(sample_seq, spaced=True)
    compact_tokens = tokenizer.tokenize(compact)
    spaced_tokens = tokenizer.tokenize(spaced)
    compact_unknown = compact_tokens.count(getattr(tokenizer, "unk_token", "[UNK]"))
    spaced_unknown = spaced_tokens.count(getattr(tokenizer, "unk_token", "[UNK]"))
    if len(compact_tokens) <= 2 and len(spaced_tokens) > 10 and spaced_unknown <= compact_unknown:
        print(
            "[info] tokenizer appears amino-acid-token based; using space-separated sequences. "
            "Pass --no-auto-spaced to disable.",
            flush=True,
        )
        return True
    return False


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def model_hidden(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
        return outputs.encoder_last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    raise RuntimeError("Could not find hidden states in model output.")


def embed_sequences(seqs, tokenizer, model, device: str, batch_size: int, max_length: int, spaced: bool):
    vectors = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = [clean_sequence(s, spaced) for s in seqs[i : i + batch_size]]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
            outputs = model(**encoded, output_hidden_states=True)
            hidden = model_hidden(outputs)
            pooled = mean_pool(hidden, encoded["attention_mask"])
            vectors.append(pooled.cpu().numpy())
            print(f"  embedded {min(i + batch_size, len(seqs))}/{len(seqs)}", flush=True)
    return np.vstack(vectors)


def paired_embeddings(vh, vl, tokenizer, model, device: str, batch_size: int, max_length: int, spaced: bool):
    print("heavy chain", flush=True)
    h = embed_sequences(vh, tokenizer, model, device, batch_size, max_length, spaced)
    print("light chain", flush=True)
    l = embed_sequences(vl, tokenizer, model, device, batch_size, max_length, spaced)
    return np.hstack([h, l])


def embedding_sanity(name: str, emb: np.ndarray) -> None:
    centered_norm = np.linalg.norm(emb - emb.mean(axis=0, keepdims=True), axis=1)
    mean_norm = float(np.mean(centered_norm))
    constant_dims = float(np.mean(np.std(emb, axis=0) < 1e-8))
    print(
        f"{name} sanity: global_std={float(np.std(emb)):.6g}, "
        f"mean_centered_row_norm={mean_norm:.6g}, near_constant_dims={constant_dims:.3f}",
        flush=True,
    )
    if mean_norm < 1e-3:
        print(
            "[warn] embeddings are nearly identical across rows. This usually means tokenization failed; "
            "rerun with --force --spaced.",
            flush=True,
        )


def load_gdpa3():
    path = ROOT / "data" / "processed" / "gdpa3_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/gdpa3_prep.py first.")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Output prefix, e.g. igbert or igt5.")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint directory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--spaced", action="store_true", help="Tokenize as space-separated amino acids, useful for ProtBert-like tokenizers.")
    parser.add_argument("--no-auto-spaced", action="store_true", help="Disable automatic space-separated amino-acid tokenization detection.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    out1 = ROOT / "data" / "processed" / f"{args.name}_gdpa1.npz"
    out3 = ROOT / "data" / "processed" / f"{args.name}_gdpa3.npz"
    if out1.exists() and out3.exists() and not args.force:
        print(f"cached: {out1}")
        print(f"cached: {out3}")
        return

    from transformers import AutoModel, AutoTokenizer

    print(f"loading {args.model} on {args.device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).to(args.device)

    ds = load_gdpa1(str(ROOT / "configs" / "default.yaml"))
    g3 = load_gdpa3()
    spaced = should_space_sequences(tokenizer, ds.heavy.iloc[0], args.spaced, args.no_auto_spaced)

    if args.force or not out1.exists():
        emb1 = paired_embeddings(
            ds.heavy.tolist(),
            ds.light.tolist(),
            tokenizer,
            model,
            args.device,
            args.batch_size,
            args.max_length,
            spaced,
        )
        embedding_sanity(args.name + " GDPa1", emb1)
        np.savez_compressed(out1, emb=emb1)
        print(f"saved {emb1.shape} -> {out1}", flush=True)
    else:
        embedding_sanity(args.name + " GDPa1 cached", np.load(out1)["emb"])

    if args.force or not out3.exists():
        emb3 = paired_embeddings(
            g3["vh"].astype(str).tolist(),
            g3["vl"].astype(str).tolist(),
            tokenizer,
            model,
            args.device,
            args.batch_size,
            args.max_length,
            spaced,
        )
        embedding_sanity(args.name + " GDPa3", emb3)
        np.savez_compressed(out3, emb=emb3)
        print(f"saved {emb3.shape} -> {out3}", flush=True)
    else:
        embedding_sanity(args.name + " GDPa3 cached", np.load(out3)["emb"])


if __name__ == "__main__":
    main()
