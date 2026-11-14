"""Chain-Aware LoRA-fine-tuned PLM for antibody developability (AbDev-CAPLM).

Novel architecture, inspired by:
  * Singh et al., "Explainable protein-protein binding affinity prediction via fine-tuning PLMs"
    (bioRxiv 2026.03.30.715237): PEFT/LoRA adaptation of a PLM encoder + residue-level explainability.
  * Singh et al., "Antibody-Antigen Affinity Prediction with Chain-Aware Protein Language Modeling"
    (bioRxiv 2026.06.19.733375): chain-aware paired VH/VL encoding + CDR-focused (curriculum) refinement.

Design (adapted from affinity -> single-antibody developability):
  ESM-2 backbone, LoRA-adapted (parameter-efficient). VH and VL are encoded by the SHARED encoder;
  each residue stream gets a learned chain-type embedding. A VH<->VL CROSS-ATTENTION block models the
  paired-chain interface (relevant to aggregation / self-association). CDR-AWARE ATTENTION POOLING
  (attention logits biased toward CDR positions via a learnable gate on a CDR mask) collapses each
  chain to a vector. These are fused with the germline/ISOTYPE one-hot (constant-region identity that
  drives the Tm2 melt) and optional frozen AbLang2 / descriptor features, then a shared trunk feeds
  per-assay MULTI-TASK regression heads. Attention-pool weights give per-residue explainability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CAPLMConfig:
    esm_name: str = "facebook/esm2_t30_150M_UR50D"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    proj_dim: int = 256
    trunk_dim: int = 256
    n_heads: int = 4
    dropout: float = 0.3
    extra_dim: int = 0        # isotype + optional AbLang2/descriptor concat width
    n_tasks: int = 5
    max_len: int = 200


class AttentionPool(nn.Module):
    """Learned-query attention pooling with a CDR-awareness gate on the CDR mask."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.scale = dim ** -0.5
        self.cdr_gate = nn.Parameter(torch.tensor(1.0))  # learnable CDR emphasis

    def forward(self, H, pad_mask, cdr_mask):
        # H: (B,L,D)  pad_mask: (B,L) True=valid  cdr_mask: (B,L) 1=CDR
        logits = (H @ self.query) * self.scale               # (B,L)
        logits = logits + self.cdr_gate * cdr_mask           # bias toward CDRs
        logits = logits.masked_fill(~pad_mask, float("-inf"))
        attn = torch.softmax(logits, dim=1)                  # (B,L)
        pooled = torch.einsum("bl,bld->bd", attn, H)
        return pooled, attn


class ChainAwareDevModel(nn.Module):
    def __init__(self, cfg: CAPLMConfig):
        super().__init__()
        self.cfg = cfg
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model

        backbone = AutoModel.from_pretrained(cfg.esm_name)
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=["query", "value"], bias="none")
        self.encoder = get_peft_model(backbone, lora)
        d = backbone.config.hidden_size

        self.chain_emb = nn.Embedding(2, d)                   # 0=heavy, 1=light
        self.cross = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.pool_h = AttentionPool(d)
        self.pool_l = AttentionPool(d)

        fuse_in = 2 * d + cfg.extra_dim
        self.trunk = nn.Sequential(
            nn.Linear(fuse_in, cfg.trunk_dim), nn.LayerNorm(cfg.trunk_dim), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.trunk_dim, cfg.trunk_dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.heads = nn.ModuleList([nn.Linear(cfg.trunk_dim, 1) for _ in range(cfg.n_tasks)])

    def encode_chain(self, input_ids, attn_mask, chain_id):
        out = self.encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        out = out + self.chain_emb(torch.full(input_ids.shape[:1], chain_id,
                                              device=input_ids.device))[:, None, :]
        return out

    def forward(self, batch, return_attn: bool = False):
        Hh = self.encode_chain(batch["vh_ids"], batch["vh_mask"], 0)   # (B,Lh,D)
        Hl = self.encode_chain(batch["vl_ids"], batch["vl_mask"], 1)   # (B,Ll,D)
        vh_pad = batch["vh_mask"].bool(); vl_pad = batch["vl_mask"].bool()

        # VH<->VL cross-attention (paired-chain interface)
        hh, _ = self.cross(Hh, Hl, Hl, key_padding_mask=~vl_pad)
        hl, _ = self.cross(Hl, Hh, Hh, key_padding_mask=~vh_pad)
        Hh = self.cross_norm(Hh + hh)
        Hl = self.cross_norm(Hl + hl)

        vh_vec, a_h = self.pool_h(Hh, vh_pad, batch["vh_cdr"])
        vl_vec, a_l = self.pool_l(Hl, vl_pad, batch["vl_cdr"])

        feats = [vh_vec, vl_vec]
        if "extra" in batch and batch["extra"] is not None:
            feats.append(batch["extra"])
        z = self.trunk(torch.cat(feats, dim=1))
        preds = torch.cat([h(z) for h in self.heads], dim=1)          # (B, n_tasks)
        if return_attn:
            return preds, {"vh_attn": a_h, "vl_attn": a_l}
        return preds


def masked_mse(pred, target, mask):
    """Multi-task MSE ignoring NaN/missing labels (mask: 1=observed)."""
    diff = (pred - target) ** 2 * mask
    return diff.sum() / mask.sum().clamp(min=1.0)
