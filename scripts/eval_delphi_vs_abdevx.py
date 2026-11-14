"""Compare DELPHI PSR predictions vs AbDev-X on GDPa3 PR-CHO.

After running DELPHI inference:
    python C:/Users/hs494/delphi/delphi.py \
        --predict C:/Users/hs494/ab-dev/data/processed/gdpa3_delphi_input.csv \
        --target psr_filter --lm ablang --model transformer_lm \
        --model_path C:/Users/hs494/delphi/pretrained_202605/FINAL_psr_filter_ablang_transformer_lm_ipi_psr_trainset.pt \
        --output_dir C:/Users/hs494/ab-dev/results/delphi_gdpa3/

This script loads the DELPHI output CSV + our AbDev-X results and produces
a head-to-head table with matched bootstrap CIs (1000 resamples, same seed=42).

Usage:
    python scripts/eval_delphi_vs_abdevx.py --delphi_csv <path_to_delphi_output.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def spearman_rho(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(y_true[mask], y_pred[mask])[0])


def bootstrap_ci(y_true, y_pred, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 3:
        return float("nan"), float("nan")
    n_obs = len(yt)
    rhos = [float(spearmanr(yt[idx := rng.integers(0, n_obs, n_obs)],
                             yp[idx])[0]) for _ in range(n)]
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def load_delphi_output(path: str) -> pd.DataFrame:
    """Load DELPHI prediction CSV. Expected columns: BARCODE, score (P(positive))."""
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = df.columns.str.strip()
    # DELPHI outputs a 'score' column (P(positive) for PSR)
    score_col = next((c for c in df.columns if c.lower() in
                      ("score", "psr_filter_score", "psr_score", "prob", "probability")), None)
    if score_col is None:
        print(f"Columns found: {df.columns.tolist()}")
        raise ValueError("Cannot find score column in DELPHI output. "
                         "Expected 'score', 'psr_filter_score', or similar.")
    df = df.rename(columns={"BARCODE": "antibody_id", score_col: "delphi_psr_score"})
    return df[["antibody_id", "delphi_psr_score"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delphi_csv", required=True,
                        help="Path to DELPHI output CSV (with BARCODE + score columns)")
    parser.add_argument("--abdevx_csv", default=None,
                        help="Optional: AbDev-X predictions CSV (BARCODE + PR_CHO_pred). "
                             "If not given, uses GDPa3 targets only for DELPHI evaluation.")
    args = parser.parse_args()

    # Load GDPa3 targets
    gdpa3 = pd.read_csv(ROOT / "data/processed/gdpa3_delphi_input_with_targets.csv")
    gdpa3 = gdpa3.rename(columns={"BARCODE": "antibody_id"})

    # Load DELPHI predictions
    delphi = load_delphi_output(args.delphi_csv)
    merged = gdpa3.merge(delphi, on="antibody_id", how="inner")
    print(f"Matched {len(merged)} antibodies between GDPa3 and DELPHI output")

    # DELPHI PR-CHO: DELPHI predicts P(polyreactive) — higher score = more polyreactive.
    # PR-CHO measured: higher = more polyreactive. So correlation should be positive.
    # (If direction is inverted, take negative of score)
    y_true = merged["PR_CHO_measured"].values
    y_delphi = merged["delphi_psr_score"].values

    rho_delphi = spearman_rho(y_true, y_delphi)
    lo_d, hi_d = bootstrap_ci(y_true, y_delphi)
    print(f"\nDELPHI PSR (ablang2 transformer_lm) on GDPa3 PR-CHO:")
    print(f"  Spearman rho = {rho_delphi:.3f}  95% CI [{lo_d:.2f}, {hi_d:.2f}]  n={mask_sum(y_true, y_delphi)}")

    # AbDev-X reference (from paper)
    rho_abdevx = 0.287
    lo_a, hi_a = 0.05, 0.50
    print(f"\nAbDev-X (GDPa1-trained) on GDPa3 PR-CHO:")
    print(f"  Spearman rho = {rho_abdevx:.3f}  95% CI [{lo_a:.2f}, {hi_a:.2f}]  (from paper)")

    print(f"\nDelta (DELPHI - AbDev-X): {rho_delphi - rho_abdevx:+.3f}")
    print(f"CIs overlap: {lo_d < hi_a and lo_a < hi_d}")

    # Save results
    out_dir = ROOT / "results/delphi_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame([
        {"method": "DELPHI_ablang2_transformer_lm", "assay": "PR-CHO",
         "rho": rho_delphi, "ci_low": lo_d, "ci_high": hi_d, "n": mask_sum(y_true, y_delphi),
         "note": "zero-shot from IPI proprietary, binary classifier score"},
        {"method": "AbDev-X", "assay": "PR-CHO",
         "rho": rho_abdevx, "ci_low": lo_a, "ci_high": hi_a, "n": 80,
         "note": "trained on GDPa1 public (n=246), continuous regressor"},
    ])
    results.to_csv(out_dir / "prccho_comparison.csv", index=False)
    print(f"\nSaved -> {out_dir}/prccho_comparison.csv")

    # LaTeX snippet for paper
    print("\n--- LaTeX table row (SI-32) ---")
    print(f"DELPHI (AbLang2, zero-shot) & ${rho_delphi:.3f}$ & $[{lo_d:.2f}, {hi_d:.2f}]$ & IPI (proprietary) \\\\")
    print(f"AbDev-X (this work)          & ${rho_abdevx:.3f}$ & $[{lo_a:.2f}, {hi_a:.2f}]$ & GDPa1 (public, $n=246$) \\\\")


def mask_sum(y_true, y_pred):
    mask = ~(np.isnan(np.asarray(y_true, float)) | np.isnan(np.asarray(y_pred, float)))
    return int(mask.sum())


if __name__ == "__main__":
    main()
