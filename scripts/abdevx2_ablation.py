"""Run AbDev-X 2.0 neural ablations.

This script evaluates the requested alternatives to prediction-level stacking:

* representation-level fusion with trainable projections
* cross-modal attention
* mixture-of-experts assay heads
* explicit full / framework / CDR sequence-region streams
* shallow residue-graph attention from cached per-residue ESM-2 plus sequence node features
* heteroscedastic uncertainty heads
* shared multi-task training with masked labels

It intentionally reuses cached features from data/processed and does not invoke PLMs.

Example:
  set KMP_DUPLICATE_LIB_OK=TRUE
  set PYTHONPATH=src
  python scripts/abdevx2_ablation.py --splits cluster_isotype --epochs 250
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1  # noqa: E402
from abdev.features.descriptors import _KD, _net_charge, build_descriptor_matrix  # noqa: E402

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_I = {a: i for i, a in enumerate(AA)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FeatureBundle:
    tabular: Dict[str, np.ndarray]
    tensors: Dict[str, torch.Tensor]
    y: np.ndarray
    ids: List[str]


def _clean_matrix(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _load_npz_emb(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return _clean_matrix(np.load(path)["emb"])


def load_esm2_mean() -> np.ndarray:
    """Prefer cached 1280-d ESM-2 means; otherwise derive mean+CDR halves from 2560-d cache."""
    candidates = []
    for p in PROC.glob("esm2_*.npz"):
        e = _load_npz_emb(p)
        if e is not None and e.shape[0] == 246:
            candidates.append((p, e))
    for _, e in candidates:
        if e.shape[1] == 1280:
            return e
    for _, e in candidates:
        if e.shape[1] >= 1280:
            return e[:, :1280]
    raise FileNotFoundError("No GDPa1 ESM-2 embedding cache found under data/processed/esm2_*.npz")


def _merge_feature_csv(ds, filename: str) -> Optional[pd.DataFrame]:
    path = PROC / filename
    if not path.exists() or ds.id_col not in ds.df.columns:
        return None
    df = pd.read_csv(path)
    if ds.id_col not in df.columns:
        return None
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c != ds.id_col]
    if not num:
        return None
    out = ds.df[[ds.id_col]].merge(df[[ds.id_col] + num], on=ds.id_col, how="left")
    return out[num]


def build_bio_features(ds) -> np.ndarray:
    frames = []
    for name in ("sap_gdpa1.csv", "sap_patch_gdpa1.csv", "electro_gdpa1.csv"):
        f = _merge_feature_csv(ds, name)
        if f is not None:
            frames.append(f)
    if not frames:
        return np.zeros((len(ds.df), 1), dtype=np.float32)
    x = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    return _clean_matrix(x.to_numpy())


def build_isotype(ds) -> np.ndarray:
    cols = [c for c in ("hc_subtype", "lc_subtype") if c in ds.df.columns]
    if not cols:
        return np.zeros((len(ds.df), 1), dtype=np.float32)
    return _clean_matrix(pd.get_dummies(ds.df[cols].astype(str), dummy_na=True).to_numpy())


def _region_pool(e: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    keep = (mask > 0.5).astype(np.float32) * weights.astype(np.float32)
    denom = keep.sum(1, keepdims=True).clip(min=1.0)
    return (e * keep[..., None]).sum(1) / denom


def build_region_embeddings() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, torch.Tensor]]:
    z = np.load(PROC / "esm_perres_150M.npz")
    tensors = {
        "vh_e": torch.tensor(z["vh_e"].astype(np.float32)),
        "vh_m": torch.tensor(z["vh_m"].astype(np.float32)),
        "vh_c": torch.tensor(z["vh_c"].astype(np.float32)),
        "vl_e": torch.tensor(z["vl_e"].astype(np.float32)),
        "vl_m": torch.tensor(z["vl_m"].astype(np.float32)),
        "vl_c": torch.tensor(z["vl_c"].astype(np.float32)),
    }
    vh_e, vl_e = z["vh_e"].astype(np.float32), z["vl_e"].astype(np.float32)
    vh_m, vl_m = z["vh_m"].astype(np.float32), z["vl_m"].astype(np.float32)
    vh_c, vl_c = z["vh_c"].astype(np.float32), z["vl_c"].astype(np.float32)
    vh_full = _region_pool(vh_e, vh_m, np.ones_like(vh_m))
    vl_full = _region_pool(vl_e, vl_m, np.ones_like(vl_m))
    vh_cdr = _region_pool(vh_e, vh_m, vh_c)
    vl_cdr = _region_pool(vl_e, vl_m, vl_c)
    vh_fw = _region_pool(vh_e, vh_m, 1.0 - vh_c)
    vl_fw = _region_pool(vl_e, vl_m, 1.0 - vl_c)
    return (
        _clean_matrix(np.concatenate([vh_full, vl_full], axis=1)),
        _clean_matrix(np.concatenate([vh_fw, vl_fw], axis=1)),
        _clean_matrix(np.concatenate([vh_cdr, vl_cdr], axis=1)),
        tensors,
    )


def build_node_features(ds, max_len: int = 280) -> Dict[str, torch.Tensor]:
    node_dim = 20 + 7
    nodes = np.zeros((len(ds.df), max_len, node_dim), dtype=np.float32)
    mask = np.zeros((len(ds.df), max_len), dtype=np.float32)
    cdr = np.zeros((len(ds.df), max_len), dtype=np.float32)
    chains = np.zeros((len(ds.df), max_len), dtype=np.float32)
    for i, (heavy, light) in enumerate(zip(ds.heavy.tolist(), ds.light.tolist())):
        cursor = 0
        for chain_id, seq in enumerate((heavy, light)):
            seq = str(seq).upper()
            for pos, aa in enumerate(seq[: max_len - cursor]):
                j = cursor + pos
                mask[i, j] = 1.0
                chains[i, j] = float(chain_id)
                if aa in AA_TO_I:
                    nodes[i, j, AA_TO_I[aa]] = 1.0
                kd = _KD.get(aa, 0.0)
                charge = _net_charge(aa, 7.4) if aa in AA_TO_I else 0.0
                nodes[i, j, 20:] = np.array(
                    [kd, charge, float(pos) / 140.0, float(chain_id), 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )
            cursor += min(len(seq), max_len - cursor)
            if cursor >= max_len:
                break
    if (PROC / "esm_perres_150M.npz").exists():
        z = np.load(PROC / "esm_perres_150M.npz")
        vh_c = z["vh_c"].astype(np.float32)
        vl_c = z["vl_c"].astype(np.float32)
        cdr[:, : vh_c.shape[1]] = vh_c[:, : max_len]
        start = min(vh_c.shape[1], max_len)
        end = min(start + vl_c.shape[1], max_len)
        cdr[:, start:end] = vl_c[:, : end - start]
        nodes[:, :, 24] = cdr
        nodes[:, :, 25] = 1.0 - cdr
    return {"nodes": torch.tensor(nodes), "node_mask": torch.tensor(mask), "node_cdr": torch.tensor(cdr)}


def standardize_blocks(blocks: Dict[str, np.ndarray], tr: np.ndarray, te: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    tr_out, te_out = {}, {}
    for name, x in blocks.items():
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        xt = sc.fit_transform(imp.fit_transform(x[tr]))
        xv = sc.transform(imp.transform(x[te]))
        tr_out[name], te_out[name] = _clean_matrix(xt), _clean_matrix(xv)
    return tr_out, te_out


def load_bundle(config: str) -> FeatureBundle:
    ds = load_gdpa1(str(ROOT / config))
    full, framework, cdr, tensors = build_region_embeddings()
    descriptors = build_descriptor_matrix(ds.heavy, ds.light, ph=7.4).replace([np.inf, -np.inf], 0).fillna(0)
    ablang2 = _load_npz_emb(PROC / "ablang2_gdpa1.npz")
    if ablang2 is None:
        print("[warn] missing ablang2_gdpa1.npz; using a zero AbLang2 block.")
        ablang2 = np.zeros((len(ds.df), 1), dtype=np.float32)
    tabular = {
        "esm2": load_esm2_mean(),
        "full": full,
        "framework": framework,
        "cdr": cdr,
        "ablang2": ablang2,
        "descriptors": _clean_matrix(descriptors.to_numpy()),
        "isotype": build_isotype(ds),
        "bio": build_bio_features(ds),
    }
    tensors.update(build_node_features(ds))
    y = np.column_stack([ds.y(a) for a in ASSAYS]).astype(np.float32)
    ids = ds.df[ds.id_col].astype(str).tolist()
    return FeatureBundle(tabular=tabular, tensors=tensors, y=y, ids=ids)


def fold_iter_all(ds, split: str) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
    folds = ds.folds(split)
    for fid in sorted(set(folds.tolist())):
        if fid != fid:
            continue
        te = np.where(folds == fid)[0]
        tr = np.where(folds != fid)[0]
        if len(te) and len(tr):
            yield int(fid), tr, te


class ModalProjector(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class GraphAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, pad_mask):
        y, _ = self.attn(x, x, x, key_padding_mask=pad_mask)
        x = self.norm1(x + y)
        return self.norm2(x + self.ff(x))


class ResidueGraphEncoder(nn.Module):
    """Shallow GAT-style encoder over residue nodes.

    If true 3D contacts are not cached, the graph is approximated by attention over padded
    residue nodes enriched with sequence, chain, CDR, charge and hydrophobicity features.
    """

    def __init__(self, node_dim: int, latent_dim: int, heads: int, dropout: float):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(node_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.layers = nn.ModuleList([GraphAttentionBlock(latent_dim, heads, dropout) for _ in range(2)])
        self.query = nn.Parameter(torch.randn(latent_dim) * 0.02)

    def forward(self, nodes, mask, cdr):
        x = self.inp(nodes)
        pad = mask < 0.5
        for layer in self.layers:
            x = layer(x, pad)
        logits = (x @ self.query) / np.sqrt(x.shape[-1])
        logits = logits + cdr
        logits = logits.masked_fill(pad, float("-inf"))
        w = torch.softmax(logits, dim=1)
        return torch.einsum("bl,bld->bd", w, x)


class AbDevX2(nn.Module):
    def __init__(
        self,
        block_dims: Dict[str, int],
        node_dim: int,
        latent_dim: int = 128,
        n_tasks: int = 5,
        n_experts: int = 5,
        heads: int = 4,
        dropout: float = 0.25,
        use_attention: bool = True,
        use_moe: bool = False,
        use_regions: bool = False,
        use_graph: bool = False,
        uncertainty: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_moe = use_moe
        self.use_regions = use_regions
        self.use_graph = use_graph
        self.uncertainty = uncertainty

        names = ["esm2", "ablang2", "descriptors", "isotype", "bio"]
        if use_regions:
            names += ["full", "framework", "cdr"]
        self.block_names = [n for n in names if n in block_dims]
        self.projectors = nn.ModuleDict({n: ModalProjector(block_dims[n], latent_dim, dropout) for n in self.block_names})
        if use_graph:
            self.graph = ResidueGraphEncoder(node_dim, latent_dim, heads, dropout)
            self.block_names.append("graph")
        self.type_emb = nn.Parameter(torch.randn(len(self.block_names), latent_dim) * 0.02)
        self.attn = nn.MultiheadAttention(latent_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        out_dim = 2 if uncertainty else 1
        if use_moe:
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(latent_dim, latent_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(latent_dim, out_dim),
                    )
                    for _ in range(n_experts)
                ]
            )
            self.gates = nn.ModuleList([nn.Linear(latent_dim, n_experts) for _ in range(n_tasks)])
        else:
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(latent_dim, latent_dim // 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(latent_dim // 2, out_dim),
                    )
                    for _ in range(n_tasks)
                ]
            )

    def forward(self, batch):
        toks = []
        for i, name in enumerate(self.block_names):
            if name == "graph":
                z = self.graph(batch["nodes"], batch["node_mask"], batch["node_cdr"])
            else:
                z = self.projectors[name](batch[name])
            toks.append(z + self.type_emb[i])
        x = torch.stack(toks, dim=1)
        if self.use_attention and x.shape[1] > 1:
            y, _ = self.attn(x, x, x)
            x = self.norm(x + y)
        z = self.shared(x.mean(dim=1))
        task_out = []
        if self.use_moe:
            expert = torch.stack([e(z) for e in self.experts], dim=1)
            for gate in self.gates:
                w = torch.softmax(gate(z), dim=1).unsqueeze(-1)
                task_out.append((expert * w).sum(dim=1))
        else:
            task_out = [h(z) for h in self.heads]
        out = torch.stack(task_out, dim=1)
        mu = out[..., 0]
        logvar = out[..., 1].clamp(-8.0, 6.0) if self.uncertainty else None
        return mu, logvar


METHODS = {
    "representation_fusion": dict(use_attention=True, use_moe=False, use_regions=False, use_graph=False, uncertainty=False),
    "moe": dict(use_attention=True, use_moe=True, use_regions=False, use_graph=False, uncertainty=False),
    "cdr_framework_moe": dict(use_attention=True, use_moe=True, use_regions=True, use_graph=False, uncertainty=False),
    "graph_moe": dict(use_attention=True, use_moe=True, use_regions=True, use_graph=True, uncertainty=False),
    "uncertainty_moe": dict(use_attention=True, use_moe=True, use_regions=True, use_graph=False, uncertainty=True),
    "multitask_projection": dict(use_attention=False, use_moe=False, use_regions=True, use_graph=False, uncertainty=False),
    "abdevx2_full": dict(use_attention=True, use_moe=True, use_regions=True, use_graph=True, uncertainty=True),
}


def masked_loss(mu, logvar, y, mask):
    if logvar is None:
        return (((mu - y) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
    inv_var = torch.exp(-logvar)
    loss = ((mu - y) ** 2) * inv_var + logvar
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def make_batch(tabular, tensors, idx):
    out = {k: torch.tensor(v[idx], dtype=torch.float32, device=DEVICE) for k, v in tabular.items()}
    for k in ("nodes", "node_mask", "node_cdr"):
        out[k] = tensors[k][idx].to(DEVICE)
    return out


def train_fold(
    method: str,
    tr_blocks: Dict[str, np.ndarray],
    tensors: Dict[str, torch.Tensor],
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
    latent_dim: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = np.nanmean(y[tr_idx], axis=0)
    sd = np.nanstd(y[tr_idx], axis=0) + 1e-6
    yn = (y - mu) / sd
    mask = (~np.isnan(yn)).astype(np.float32)
    yn = np.nan_to_num(yn).astype(np.float32)
    yt = torch.tensor(yn, device=DEVICE)
    mt = torch.tensor(mask, device=DEVICE)
    block_dims = {k: v.shape[1] for k, v in tr_blocks.items()}
    node_dim = int(tensors["nodes"].shape[-1])
    model = AbDevX2(block_dims, node_dim, latent_dim=latent_dim, **METHODS[method]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        for start in range(0, len(perm), 24):
            idx = perm[start : start + 24]
            opt.zero_grad()
            pred, logvar = model(make_batch(tr_blocks, tensors, idx))
            loss = masked_loss(pred, logvar, yt[idx], mt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred, logvar = model(make_batch(tr_blocks, tensors, va_idx))
            val = masked_loss(pred, logvar, yt[va_idx], mt[va_idx]).item()
        if val < best - 1e-4:
            best, bad = val, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if state is not None:
        model.load_state_dict(state)
    return model, mu, sd


@torch.no_grad()
def predict(model, blocks, tensors, idx, mu, sd):
    model.eval()
    preds, sigmas = [], []
    for start in range(0, len(idx), 64):
        sl = idx[start : start + 64]
        p, lv = model(make_batch(blocks, tensors, sl))
        preds.append(p.cpu().numpy())
        if lv is not None:
            sigmas.append(np.sqrt(np.exp(lv.cpu().numpy())) * sd)
    pred = np.vstack(preds) * sd + mu
    sigma = np.vstack(sigmas) if sigmas else None
    return pred, sigma


def score_predictions(y, p) -> Dict[str, float]:
    out = {}
    for j, assay in enumerate(ASSAYS):
        m = ~np.isnan(y[:, j]) & ~np.isnan(p[:, j])
        out[assay] = float(spearmanr(y[m, j], p[m, j]).statistic) if m.sum() > 2 else np.nan
    out["MEAN"] = float(np.nanmean([out[a] for a in ASSAYS]))
    return out


def run_method(method, split, ds, bundle, args):
    n = len(bundle.y)
    pooled = np.full((n, len(ASSAYS)), np.nan, dtype=np.float32)
    sigma_pooled = np.full_like(pooled, np.nan)
    fold_rows = []
    for fid, tr, te in fold_iter_all(ds, split):
        rng = np.random.RandomState(args.seed + fid)
        shuffled = rng.permutation(tr)
        cut = max(1, int(0.85 * len(shuffled)))
        tin, vin = shuffled[:cut], shuffled[cut:]
        if len(vin) == 0:
            vin = shuffled[-1:]
            tin = shuffled[:-1]
        _, all_blocks = standardize_blocks(bundle.tabular, tin, np.arange(n))
        model, ymu, ysd = train_fold(
            method, all_blocks, bundle.tensors, bundle.y, tin, vin,
            args.epochs, args.lr, args.patience, args.seed + fid, args.latent_dim,
        )
        pred, sig = predict(model, all_blocks, bundle.tensors, te, ymu, ysd)
        pooled[te] = pred
        if sig is not None:
            sigma_pooled[te] = sig
        fold_score = score_predictions(bundle.y[te], pred)
        fold_rows.append({"method": method, "split": split, "fold": fid, **fold_score})
        print(f"    {method:22s} {split:16s} fold {fid} MEAN={fold_score['MEAN']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return pooled, sigma_pooled, fold_rows


def save_predictions(out_dir: Path, method: str, split: str, ids: List[str], y, pred, sigma):
    df = pd.DataFrame({"antibody_id": ids})
    for j, assay in enumerate(ASSAYS):
        df[f"y_{assay}"] = y[:, j]
        df[f"pred_{assay}"] = pred[:, j]
        if not np.isnan(sigma[:, j]).all():
            df[f"sigma_{assay}"] = sigma[:, j]
    df.to_csv(out_dir / f"predictions_{method}_{split}.csv", index=False)


def fold_summary(folds: pd.DataFrame) -> pd.DataFrame:
    value_cols = ASSAYS + ["MEAN"]
    grouped = folds.groupby(["method", "split"], as_index=False)
    mean = grouped[value_cols].mean()
    std = grouped[value_cols].std(ddof=1).fillna(0.0)
    std = std.rename(columns={c: f"{c}_std" for c in value_cols})
    return mean.merge(std, on=["method", "split"]).sort_values("MEAN", ascending=False)


def plot_comparison(metrics: pd.DataFrame, out: Path):
    df = metrics.sort_values("MEAN", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    yerr = df["MEAN_std"] if "MEAN_std" in df.columns else None
    ax.bar(x, df["MEAN"], yerr=yerr, capsize=4 if yerr is not None else 0, color="#2f6f73")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"] + "\n" + df["split"], rotation=35, ha="right")
    ax.set_ylabel("Mean Spearman rho")
    ax.set_title("AbDev-X 2.0 ablation comparison")
    ax.axhline(0.0, color="black", linewidth=0.8)
    for i, v in enumerate(df["MEAN"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_assay_errorbars(summary: pd.DataFrame, out: Path):
    df = summary.sort_values("MEAN", ascending=False)
    labels = df["method"] + "\n" + df["split"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.ravel()
    for ax, assay in zip(axes, ASSAYS + ["MEAN"]):
        x = np.arange(len(df))
        ax.bar(x, df[assay], yerr=df[f"{assay}_std"], capsize=3, color="#5b8c5a")
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_title(assay)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, ha="right", fontsize=7)
    fig.suptitle("AbDev-X 2.0 fold CV Spearman with error bars")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="+", default=["cluster_isotype"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/abdevx2_ablation")
    ap.add_argument("--plot-only", action="store_true", help="Read fold_metrics.csv from --out-dir and regenerate error-bar summaries/plots.")
    args = ap.parse_args()

    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(METHODS)}")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        folds_path = out_dir / "fold_metrics.csv"
        if not folds_path.exists():
            raise FileNotFoundError(f"{folds_path} not found.")
        folds = pd.read_csv(folds_path)
        summary = fold_summary(folds)
        summary.to_csv(out_dir / "metrics_fold_mean_std.csv", index=False)
        plot_comparison(summary, out_dir / "comparison_errorbars.png")
        plot_assay_errorbars(summary, out_dir / "assay_errorbars.png")
        print(summary.round(3).to_string(index=False))
        print(f"saved {out_dir / 'metrics_fold_mean_std.csv'}")
        print(f"saved {out_dir / 'comparison_errorbars.png'}")
        print(f"saved {out_dir / 'assay_errorbars.png'}")
        return

    ds = load_gdpa1(str(ROOT / args.config))
    bundle = load_bundle(args.config)
    print(f"device={DEVICE}; methods={args.methods}; splits={args.splits}", flush=True)

    metric_rows, fold_rows = [], []
    for split in args.splits:
        if split not in ds.fold_cols:
            print(f"[skip] split {split} not available. Have {list(ds.fold_cols)}")
            continue
        for method in args.methods:
            pooled, sigmas, fr = run_method(method, split, ds, bundle, args)
            scores = score_predictions(bundle.y, pooled)
            metric_rows.append({"method": method, "split": split, **scores})
            fold_rows.extend(fr)
            save_predictions(out_dir, method, split, bundle.ids, bundle.y, pooled, sigmas)
            np.save(out_dir / f"pooled_{method}_{split}.npy", pooled)
            if not np.isnan(sigmas).all():
                np.save(out_dir / f"sigma_{method}_{split}.npy", sigmas)
            print(
                f"  {method:22s} {split:16s} "
                + " ".join(f"{a}={scores[a]:.3f}" for a in ASSAYS)
                + f" MEAN={scores['MEAN']:.3f}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    folds = pd.DataFrame(fold_rows)
    summary = fold_summary(folds)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    folds.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "metrics_fold_mean_std.csv", index=False)
    plot_comparison(metrics, out_dir / "comparison.png")
    plot_comparison(summary, out_dir / "comparison_errorbars.png")
    plot_assay_errorbars(summary, out_dir / "assay_errorbars.png")
    print(f"saved {out_dir / 'metrics.csv'}")
    print(f"saved {out_dir / 'metrics_fold_mean_std.csv'}")
    print(f"saved {out_dir / 'comparison.png'}")
    print(f"saved {out_dir / 'comparison_errorbars.png'}")
    print(f"saved {out_dir / 'assay_errorbars.png'}")


if __name__ == "__main__":
    main()
