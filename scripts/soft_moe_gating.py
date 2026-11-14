"""
Soft Biological Mixture-of-Experts (Soft MoE) gating for AbDev-X.

Replaces the hard OOD-CV block selection + ridge stacking with a learnable
soft gating network whose inputs are biologically motivated:
  - Isotype one-hot (IgG1 / IgG2 / IgG4)
  - CDR-H3 length bin (short / medium / long)
  - Germline distance proxy

Gate output: simplex weights over the AbDev-X expert blocks.

Usage:
  python scripts/soft_moe_gating.py [--assay all] [--split cluster_isotype]

Outputs:
  results/soft_moe/soft_moe_cv.csv
  results/soft_moe/gate_weights.csv
  paper/figures/panel_soft_moe.pdf
"""
import os, sys, warnings, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.special import softmax
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RES  = ROOT / "results" / "soft_moe"
FIGS = ROOT / "paper" / "figures"
RES.mkdir(parents=True, exist_ok=True)

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
ASSAY5 = ["HIC", "Tm2", "AC-SINS", "PR-CHO", "Titer"]
DPI    = 300
FIG_W  = 7.0


# ── helpers ──────────────────────────────────────────────────────────────────
def spear(a, b):
    if len(a) < 5: return np.nan
    r, _ = spearmanr(a, b); return float(r)

def fold_iter(n, k=5, seed=42):
    from sklearn.model_selection import KFold
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)):
        yield tr, te

def make_ridge():
    return Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=0.95, svd_solver="full")),
                     ("r", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]))])


# ── gate context features ─────────────────────────────────────────────────────
def build_gate_context(ds):
    """Biologically motivated context features for the gating network."""
    df  = ds.df
    feats = {}
    # 1. Isotype one-hot
    if "hc_subtype" in df.columns:
        for iso in ["IgG1", "IgG2", "IgG4"]:
            feats[f"iso_{iso}"] = (df["hc_subtype"].fillna("").str.contains(iso)).astype(float).values
    else:
        feats["iso_unknown"] = np.ones(len(df))
    # 2. CDR-H3 length proxy (approximate from VH length)
    vh_len = ds.heavy.str.len().values.astype(float)
    cdr3_proxy = np.clip(vh_len - 110, 0, 30)  # rough CDR3 contribution
    feats["cdrh3_short"]  = (cdr3_proxy <= 8).astype(float)
    feats["cdrh3_medium"] = ((cdr3_proxy > 8) & (cdr3_proxy <= 14)).astype(float)
    feats["cdrh3_long"]   = (cdr3_proxy > 14).astype(float)
    # 3. Germline distance proxy (normalised VH length deviation)
    feats["vh_len_norm"]  = (vh_len - vh_len.mean()) / (vh_len.std() + 1e-8)
    return pd.DataFrame(feats)


# ── Soft gate ────────────────────────────────────────────────────────────────
class SoftGate:
    """Linear softmax gate Z -> simplex weights over blocks."""
    def __init__(self, n_blocks, temperature=1.0, l2=0.1):
        self.n_blocks    = n_blocks
        self.temperature = temperature
        self.l2          = l2
        self.W           = None

    def _weights(self, Z):
        return softmax((Z @ self.W.T) / self.temperature, axis=1)

    def fit(self, Z, B_preds, y, n_iter=300, lr=0.005):
        n, k = B_preds.shape
        Zsc  = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        self.W = np.zeros((k, Zsc.shape[1]))
        best_loss, best_W = np.inf, self.W.copy()
        uniform = np.ones(k) / k
        for _ in range(n_iter):
            w     = self._weights(Zsc)
            y_hat = (w * B_preds).sum(1)
            err   = y_hat - y
            mse   = (err**2).mean()
            kl    = (w * np.log(w / uniform + 1e-9)).sum(1).mean()
            loss  = mse + self.l2 * kl
            # gradient through softmax
            d_w   = (2 * err / n)[:, None] * B_preds
            sw    = w
            dlog  = d_w - (d_w * sw).sum(1, keepdims=True) * sw
            dlog /= self.temperature
            dkl   = (1 + np.log(w / uniform + 1e-9)) * sw
            dkl  -= (dkl * sw).sum(1, keepdims=True) * sw
            dlog += self.l2 * dkl / n
            self.W -= lr * (dlog[:, :, None] * Zsc[:, None, :]).mean(0)
            if loss < best_loss: best_loss, best_W = loss, self.W.copy()
        self.W = best_W
        return self

    def predict(self, Z, B_preds):
        Zsc = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        w   = self._weights(Zsc)
        return (w * B_preds).sum(1), w


# ── Main experiment ───────────────────────────────────────────────────────────
def run_soft_moe(assay, ds, B1, context_df):
    y = ds.y(assay)
    m = ~np.isnan(y); y_m = y[m]; n = int(m.sum())
    if n < 30: return [], []

    block_names = list(B1.keys())
    B_m = {k: v[m] for k, v in B1.items()}   # subset to labeled
    Z_m = context_df.values[m]                # gate context
    X_concat = np.hstack([v for v in B_m.values()])

    # Per-block OOF predictions
    oof_per_block = {}
    for bname, Xb in B_m.items():
        oof = np.zeros(n)
        for tr, te in fold_iter(n):
            mod = make_ridge(); mod.fit(Xb[tr], y_m[tr]); oof[te] = mod.predict(Xb[te])
        oof_per_block[bname] = oof
    B_oof = np.column_stack([oof_per_block[b] for b in block_names])  # (n, k)

    # Ridge stacking baseline (concatenated blocks)
    oof_ridge = np.zeros(n)
    for tr, te in fold_iter(n):
        mod = make_ridge(); mod.fit(X_concat[tr], y_m[tr]); oof_ridge[te] = mod.predict(X_concat[te])
    rho_ridge = spear(y_m, oof_ridge)

    # Soft gate OOF
    oof_sg   = np.zeros(n)
    all_wts  = np.zeros((n, len(block_names)))
    for tr, te in fold_iter(n):
        gate = SoftGate(len(block_names), temperature=1.0, l2=0.1)
        gate.fit(Z_m[tr], B_oof[tr], y_m[tr])
        oof_sg[te], all_wts[te] = gate.predict(Z_m[te], B_oof[te])
    rho_sg = spear(y_m, oof_sg)

    # Best single-block oracle
    best_block_rho = max(spear(y_m, oof_per_block[b]) for b in block_names)

    rho_rows  = [
        {"assay": assay, "method": "ridge_stacking",    "rho": rho_ridge},
        {"assay": assay, "method": "soft_moe_gate",     "rho": rho_sg},
        {"assay": assay, "method": "best_block_oracle", "rho": best_block_rho},
    ]
    gate_rows = [{"assay": assay, "block": b, "mean_weight": float(all_wts[:, i].mean())}
                 for i, b in enumerate(block_names)]

    print(f"  {assay}: ridge={rho_ridge:.3f}  soft_moe={rho_sg:.3f}  oracle={best_block_rho:.3f}")
    return rho_rows, gate_rows


def make_figure(cv_df, gate_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                                 "axes.labelsize": 9, "axes.titlesize": 9, "pdf.fonttype": 42})
    PAL = ["#4C72B0", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.48), constrained_layout=True)

    ax = axes[0]
    methods = ["ridge_stacking", "soft_moe_gate", "best_block_oracle"]
    x = np.arange(len(ASSAYS)); w = 0.25
    for i, meth in enumerate(methods):
        sub = cv_df[cv_df["method"] == meth].set_index("assay")
        vals = [sub.loc[a, "rho"] if a in sub.index else np.nan for a in ASSAYS]
        ax.bar(x + (i-1)*w, vals, w, label=meth.replace("_", " "), color=PAL[i], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(ASSAY5, rotation=35, ha="right", fontsize=7)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Spearman rho"); ax.set_title("(A) Soft MoE gate vs ridge stacking")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)

    ax = axes[1]
    if len(gate_df):
        piv = gate_df.pivot(index="assay", columns="block", values="mean_weight").reindex(ASSAYS)
        im  = ax.imshow(piv.fillna(0).values, aspect="auto", cmap="Blues", vmin=0)
        ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=6)
        ax.set_yticks(np.arange(len(ASSAYS))); ax.set_yticklabels(ASSAY5, fontsize=7)
        ax.set_title("(B) Soft gate weights (mean)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    out = FIGS / "panel_soft_moe.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight", dpi=DPI)
    import matplotlib.pyplot as _p; _p.close(fig)
    print(f"  Figure: {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", default="all")
    parser.add_argument("--split", default="cluster_isotype")
    args = parser.parse_args()

    import gdpa3_eval as GE
    ds, _, B1, _ = GE.build_blocks()
    context_df   = build_gate_context(ds)

    target_assays = ASSAYS if args.assay == "all" else [args.assay]
    all_rho, all_gate = [], []
    for assay in target_assays:
        print(f"\n=== {assay} ===")
        rr, gr = run_soft_moe(assay, ds, B1, context_df)
        all_rho += rr; all_gate += gr

    cv_df   = pd.DataFrame(all_rho)
    gate_df = pd.DataFrame(all_gate)
    cv_df.to_csv(RES / "soft_moe_cv.csv", index=False)
    gate_df.to_csv(RES / "gate_weights.csv", index=False)

    print("\n=== Summary ===")
    if len(cv_df):
        print(cv_df.pivot(index="assay", columns="method", values="rho").round(3))

    make_figure(cv_df, gate_df)
    print(f"\nResults in {RES}")


if __name__ == "__main__":
    main()
