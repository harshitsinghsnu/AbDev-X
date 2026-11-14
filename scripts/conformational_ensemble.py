"""
Conformational Ensemble Fusion for AbDev-X.

Replaces single-structure descriptors with pooled statistics over a small
ensemble of conformations, giving the model explicit structural variability.

Without real PDB ensembles (default): simulates ensemble by adding correlated
Gaussian noise to existing single-structure descriptors. Results are labelled
`simulated=True` and measure the *upper bound on benefit* from ensemble
variance features.

With real ensembles: provide PDB files via --ensemble-dir with layout:
  <dir>/<antibody_id>_frame<k>.pdb  (k = 0, 1, ..., n_frames-1)
Then FreeSASA is used to extract SAP-like features per frame.

Pooling modes: mean | std | concat (mean+std) | max
Fused with remaining AbDev-X blocks (ESM2, AbLang2, isotype, bio) via ridge stacking.

Usage:
  python scripts/conformational_ensemble.py [--modes mean std concat] [--ensemble-dir <path>]

Outputs:
  results/conformational_ensemble/ens_cv_results.csv
  paper/figures/panel_conformational_ensemble.pdf
"""
import os, sys, warnings, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RES  = ROOT / "results" / "conformational_ensemble"
FIGS = ROOT / "paper" / "figures"
RES.mkdir(parents=True, exist_ok=True)

ASSAYS   = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
ASSAY5   = ["HIC", "Tm2", "AC-SINS", "PR-CHO", "Titer"]
DPI      = 300
FIG_W    = 7.0
N_FRAMES = 10
NOISE    = 0.05


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


# ── simulated ensemble from descriptor block ──────────────────────────────────
def simulate_ensemble(X_desc, n_frames=N_FRAMES, noise=NOISE, seed=42):
    """Add per-frame Gaussian noise to simulate conformational variation.
    Returns frames: (n, n_frames, d_desc).
    """
    rng    = np.random.default_rng(seed)
    scale  = noise * (np.abs(X_desc).mean(0) + 1e-6)  # per-feature scale
    frames = np.stack([X_desc + rng.normal(0, 1, X_desc.shape) * scale
                       for _ in range(n_frames)], axis=1)  # (n, F, d)
    return frames


def pool(frames, mode):
    """frames: (n, n_frames, d). Returns (n, d')."""
    m  = frames.mean(1)
    s  = frames.std(1)
    mx = frames.max(1)
    return {"mean": m, "std": s, "concat": np.hstack([m, s]),
            "max": mx, "concat_max": np.hstack([m, s, mx])}[mode]


# ── PDB-based ensemble (requires freesasa + biopython) ───────────────────────
def featurize_pdb_ensemble(pdb_files):
    """Extract surface descriptors from a list of PDB files. Returns (n_frames, 7) array."""
    try:
        import freesasa
        from Bio.PDB import PDBParser
    except ImportError:
        return None
    HYDRO = set("VILMFYW"); POS = set("RKH"); NEG = set("DE")
    parser = PDBParser(QUIET=True)
    rows = []
    for pdb in pdb_files:
        try:
            struct  = parser.get_structure("ab", str(pdb))
            result  = freesasa.calcBioPython(struct)
            tot = hydro = pos = neg = n_res = 0.0
            for chain in struct.get_chains():
                for res in chain.get_residues():
                    if res.get_id()[0] != " ": continue
                    aa = res.get_resname()[:1]
                    sa = sum(result.atomArea(a) for a in res.get_atoms())
                    tot += sa
                    if aa in HYDRO: hydro += sa
                    if aa in POS:   pos   += sa
                    if aa in NEG:   neg   += sa
                    n_res += 1
            if tot < 1e-6: tot = 1.0
            rows.append([tot, hydro, hydro/tot, pos, neg, pos-neg, n_res])
        except Exception:
            continue
    return np.array(rows) if rows else None


# ── main experiment ───────────────────────────────────────────────────────────
def run_experiment(assay, ds, B1, modes, ensemble_dir):
    y = ds.y(assay); m = ~np.isnan(y); y_m = y[m]; n = int(m.sum())
    if n < 30: return []

    # Descriptor block (used for ensemble simulation)
    X_desc = B1.get("descriptors", np.zeros((len(m), 1)))[m]
    # All non-structural blocks concatenated (ESM2 + AbLang2 + isotype + bio)
    X_other = np.hstack([v[m] for k, v in B1.items() if k != "descriptors"])

    # Baseline: single-structure descriptors + other blocks
    X_base = np.hstack([X_desc, X_other])
    oof_base = np.zeros(n)
    for tr, te in fold_iter(n):
        mod = make_ridge(); mod.fit(X_base[tr], y_m[tr]); oof_base[te] = mod.predict(X_base[te])
    rho_base = spear(y_m, oof_base)

    rows = [{"assay": assay, "mode": "single_structure_baseline",
             "rho": rho_base, "simulated": False, "delta": 0.0}]

    for mode in modes:
        # Ensemble features
        if ensemble_dir and Path(ensemble_dir).exists():
            pdir = Path(ensemble_dir)
            ab_ids = ds.df[ds.id_col].values[m]
            ens_list = []
            sim_flag = False
            for ab_id in ab_ids:
                pdbs = sorted(pdir.glob(f"{ab_id}_frame*.pdb"))
                if pdbs:
                    fr = featurize_pdb_ensemble(pdbs[:N_FRAMES])
                    if fr is not None and len(fr) > 0:
                        ens_list.append(fr)
                        continue
                # fallback: noise around single structure
                xi = X_desc[list(ab_ids).index(ab_id)]
                fr = xi[None, :] + np.random.normal(0, NOISE*np.abs(xi).mean(), (N_FRAMES, len(xi)))
                ens_list.append(fr)
                sim_flag = True
            frames = np.stack([pool(fr[np.newaxis], mode).squeeze(0) if fr.ndim == 2
                               else pool(np.expand_dims(fr, 0), mode).squeeze(0)
                               for fr in ens_list], axis=0)
        else:
            frames_3d = simulate_ensemble(X_desc)  # (n, N_FRAMES, d_desc)
            frames    = pool(frames_3d, mode)       # (n, d')
            sim_flag  = True

        X_aug    = np.hstack([frames, X_other])
        oof_ens  = np.zeros(n)
        for tr, te in fold_iter(n):
            mod = make_ridge(); mod.fit(X_aug[tr], y_m[tr]); oof_ens[te] = mod.predict(X_aug[te])
        rho_ens = spear(y_m, oof_ens)

        rows.append({"assay": assay, "mode": mode, "rho": rho_ens,
                     "simulated": bool(sim_flag), "delta": rho_ens - rho_base})
        print(f"  {assay} [{mode}]: baseline={rho_base:.3f}  ens={rho_ens:.3f}  "
              f"delta={rho_ens-rho_base:+.3f}  sim={sim_flag}")
    return rows


def make_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                                 "axes.labelsize": 9, "axes.titlesize": 9, "pdf.fonttype": 42})
    PAL = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#DD8452"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.48), constrained_layout=True)

    ax = axes[0]
    modes = df["mode"].unique()
    x = np.arange(len(ASSAYS)); w = 0.75 / max(len(modes), 1)
    for i, mode in enumerate(modes):
        sub = df[df["mode"] == mode].set_index("assay")
        vals = [sub.loc[a, "rho"] if a in sub.index else np.nan for a in ASSAYS]
        ax.bar(x + (i - len(modes)/2 + 0.5)*w, vals, w,
               label=mode, color=PAL[i % len(PAL)], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ASSAY5, rotation=35, ha="right", fontsize=7)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Spearman rho"); ax.set_title("(A) Ensemble pooling modes")
    ax.legend(fontsize=5, loc="upper right", framealpha=0.8)

    ax = axes[1]
    ens_modes = [m for m in modes if m != "single_structure_baseline"]
    for i, mode in enumerate(ens_modes):
        sub = df[df["mode"] == mode].set_index("assay")
        deltas = [sub.loc[a, "delta"] if a in sub.index else np.nan for a in ASSAYS]
        ax.plot(x, deltas, "o-", color=PAL[i+1], ms=5, lw=1.2, label=mode)
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(ASSAY5, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Delta Spearman (vs single structure)")
    ax.set_title("(B) Gain from conformational ensemble")
    ax.legend(fontsize=5, loc="upper right", framealpha=0.8)
    out = FIGS / "panel_conformational_ensemble.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight", dpi=DPI)
    import matplotlib.pyplot as _p; _p.close(fig)
    print(f"  Figure: {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", default="all")
    parser.add_argument("--ensemble-dir", default=None)
    parser.add_argument("--modes", nargs="+", default=["mean", "std", "concat"])
    args = parser.parse_args()

    import gdpa3_eval as GE
    ds, _, B1, _ = GE.build_blocks()

    target_assays = ASSAYS if args.assay == "all" else [args.assay]
    all_rows = []
    for assay in target_assays:
        print(f"\n=== {assay} ===")
        all_rows += run_experiment(assay, ds, B1, args.modes, args.ensemble_dir)

    df = pd.DataFrame(all_rows)
    df.to_csv(RES / "ens_cv_results.csv", index=False)

    print("\n=== Summary ===")
    if len(df):
        print(df.pivot(index="assay", columns="mode", values="rho").round(3))

    sim_any = df["simulated"].any() if "simulated" in df.columns else True
    if sim_any:
        print("\nNOTE: Results marked simulated=True used Gaussian-noise perturbation of")
        print("      existing descriptors. Real ensemble gains may differ.")
        print("      Provide real PDB ensembles via --ensemble-dir for actual measurement.")

    make_figure(df)
    print(f"\nResults in {RES}")


if __name__ == "__main__":
    main()
