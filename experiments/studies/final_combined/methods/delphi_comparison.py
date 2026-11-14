"""Fair comparison: DELPHI trained on GDPa1 (same cluster_isotype folds) vs AbDev-X.

DELPHI is a binary classifier; we use P(Pass) as a continuous score and compute
Spearman rho against the raw assay values — same approach as Nguyen et al. 2026.

Prerequisites: DELPHI repo at C:/Users/hs494/delphi (already present)

Usage:
  python delphi_comparison.py [--lm ablang] [--assays PR_CHO HIC ...]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1

DELPHI_DIR = ROOT / "baselines" / "delphi"
DELPHI_PY = sys.executable
MODEL_ARCH = "transformer_lm"

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
LABEL_MAP = {
    "HIC":          "hic_rt_avg",
    "Tm2":          "tm2_nanodsf_avg",
    "AC-SINS_pH7.4":"acsins_dLmax_ph7.4_avg",
    "PR_CHO":       "polyreactivity_prscore_cho_avg",
    "Titer":        "titer_avg",
}
LOWER_IS_BETTER = {"HIC", "AC-SINS_pH7.4", "PR_CHO"}

ABDEVX_HELDOUT = {"HIC": 0.527, "Tm2": 0.333, "AC-SINS_pH7.4": 0.300, "PR_CHO": 0.197, "Titer": 0.294}
COMP_HELDOUT   = {"HIC": 0.708, "Tm2": 0.392, "AC-SINS_pH7.4": 0.356, "PR_CHO": 0.337, "Titer": 0.310}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binarize(values: np.ndarray, assay: str) -> np.ndarray:
    median = np.nanmedian(values)
    return (values <= median).astype(int) if assay in LOWER_IS_BETTER \
           else (values >= median).astype(int)


def make_xlsx(df_seq: pd.DataFrame, y: np.ndarray | None, assay: str, path: Path) -> None:
    """Write a DELPHI-format xlsx. y=None for prediction-only files."""
    out = df_seq.copy()
    if y is not None:
        col = _target_col(assay)
        out[col] = binarize(y, assay).astype(float)
        out.loc[np.isnan(y), col] = np.nan
    out.to_excel(path, index=False)


def _target_col(assay: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", assay) + "_filter"


def _score_col(lm: str, db_stem: str) -> str:
    return f"{MODEL_ARCH}_{lm}_{db_stem}_score"


def _find_pred_file(input_path: Path, target_col: str) -> Path | None:
    """Glob for DELPHI's prediction output next to the input file."""
    matches = list(input_path.parent.glob(f"{input_path.stem}_pred_{target_col}_*.xlsx"))
    return matches[0] if matches else None


def _checkpoint(db_stem: str, target_col: str, lm: str) -> Path:
    """Path where DELPHI saves the trained model."""
    return DELPHI_DIR / "build" / "pretrained_models" / \
           f"FINAL_{target_col}_{lm}_{MODEL_ARCH}_{db_stem}.pt"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [DELPHI_PY] + cmd, cwd=str(DELPHI_DIR),
        capture_output=True, encoding="utf-8", errors="replace", env=env
    )


def delphi_train(train_xlsx: Path, target_col: str, lm: str) -> Path | None:
    r = _run(["delphi.py", "--train", "--target", target_col,
               "--lm", lm, "--model", MODEL_ARCH, "--db", str(train_xlsx)])
    if r.returncode != 0:
        print(f"  [DELPHI train fail] {r.stderr[-400:]}")
        return None
    ckpt = _checkpoint(train_xlsx.stem, target_col, lm)
    return ckpt if ckpt.exists() else None


def delphi_predict(predict_xlsx: Path, ckpt: Path,
                   target_col: str, lm: str) -> pd.DataFrame | None:
    r = _run(["delphi.py", "--predict", str(predict_xlsx),
               "--model_path", str(ckpt)])
    if r.returncode != 0:
        print(f"  [DELPHI predict fail] {r.stderr[-400:]}")
        return None
    out = _find_pred_file(predict_xlsx, target_col)
    return pd.read_excel(out) if out else None


# ---------------------------------------------------------------------------
# Sequence DataFrames
# ---------------------------------------------------------------------------

def seq_df(names, heavy, light) -> pd.DataFrame:
    return pd.DataFrame({"BARCODE": list(names),
                          "HSEQ": list(heavy),
                          "LSEQ": list(light)})


# ---------------------------------------------------------------------------
# Cross-validation (cluster_isotype folds)
# ---------------------------------------------------------------------------

def evaluate_cv(seq1: pd.DataFrame, y1: np.ndarray, folds: np.ndarray,
                assay: str, lm: str, tmp: Path) -> float:
    target_col = _target_col(assay)
    all_scores = np.full(len(seq1), np.nan)

    for fold_id in sorted(set(folds.tolist())):
        tr_idx = np.where((folds != fold_id) & ~np.isnan(y1))[0]
        te_idx = np.where(folds == fold_id)[0]
        te_labeled = te_idx[~np.isnan(y1[te_idx])]
        if len(te_labeled) < 3 or len(tr_idx) < 10:
            continue

        db_stem = f"{assay.replace('/','_').replace('.','')}_fold{fold_id}"
        train_xlsx = tmp / f"train_{db_stem}.xlsx"
        test_xlsx  = tmp / f"test_{db_stem}.xlsx"

        make_xlsx(seq1.iloc[tr_idx].reset_index(drop=True), y1[tr_idx], assay, train_xlsx)
        make_xlsx(seq1.iloc[te_idx].reset_index(drop=True), None, assay, test_xlsx)

        ckpt = delphi_train(train_xlsx, target_col, lm)
        if ckpt is None:
            continue
        pred_df = delphi_predict(test_xlsx, ckpt, target_col, lm)
        if pred_df is None:
            continue

        sc = next((c for c in pred_df.columns
                   if pd.api.types.is_numeric_dtype(pred_df[c])
                   and c not in ("BARCODE",)), None)
        if sc is None:
            continue

        barcode_to_score = dict(zip(pred_df["BARCODE"], pred_df[sc]))
        for idx in te_idx:
            b = seq1.iloc[idx]["BARCODE"]
            if b in barcode_to_score:
                all_scores[idx] = barcode_to_score[b]

    mask = ~np.isnan(all_scores) & ~np.isnan(y1)
    return float(spearmanr(y1[mask], all_scores[mask]).statistic) if mask.sum() >= 5 else float("nan")


# ---------------------------------------------------------------------------
# Held-out GDPa3 evaluation
# ---------------------------------------------------------------------------

def evaluate_heldout(seq1: pd.DataFrame, y1: np.ndarray,
                     seq3: pd.DataFrame, y3: np.ndarray,
                     assay: str, lm: str, tmp: Path) -> float:
    target_col = _target_col(assay)
    db_stem = f"{assay.replace('/','_').replace('.','')}_gdpa1"
    train_xlsx = tmp / f"train_{db_stem}.xlsx"
    test_xlsx  = tmp / f"test_gdpa3_{assay.replace('/','_').replace('.','')}.xlsx"

    make_xlsx(seq1, y1, assay, train_xlsx)
    make_xlsx(seq3, None, assay, test_xlsx)

    ckpt = delphi_train(train_xlsx, target_col, lm)
    if ckpt is None:
        return float("nan")
    pred_df = delphi_predict(test_xlsx, ckpt, target_col, lm)
    if pred_df is None:
        return float("nan")

    sc = next((c for c in pred_df.columns
               if pd.api.types.is_numeric_dtype(pred_df[c])
               and c not in ("BARCODE",)), None)
    if sc is None:
        return float("nan")

    ok = ~np.isnan(y3)
    return float(spearmanr(y3[ok], pred_df[sc].values[ok]).statistic)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="ablang")
    ap.add_argument("--assays", nargs="+", default=ASSAYS)
    ap.add_argument("--cv-only", action="store_true")
    args = ap.parse_args()

    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds = ds.folds("cluster_isotype")

    seq1 = seq_df(ds.df["antibody_name"], ds.heavy, ds.light)
    seq3 = seq_df(g3["antibody_name"], g3["vh"], g3["vl"])

    rows = []
    with tempfile.TemporaryDirectory(prefix="delphi_cmp_") as tmp:
        tmp = Path(tmp)
        for a in args.assays:
            y1 = ds.y(a)
            y3 = g3[LABEL_MAP[a]].to_numpy(float)
            print(f"\n{'='*60}\n{a}\n{'='*60}")

            print("  Running CV (cluster_isotype folds)...")
            cv_rho = evaluate_cv(seq1, y1, folds, a, args.lm, tmp)
            print(f"  GDPa1 CV rho: {cv_rho:.3f}")

            heldout_rho = float("nan")
            if not args.cv_only:
                print("  Running held-out GDPa3...")
                heldout_rho = evaluate_heldout(seq1, y1, seq3, y3, a, args.lm, tmp)
                print(f"  GDPa3 heldout rho: {heldout_rho:.3f}")

            rows.append({
                "assay":            a,
                "DELPHI_cv_rho":    round(cv_rho, 3),
                "DELPHI_heldout":   round(heldout_rho, 3),
                "AbDevX_heldout":   ABDEVX_HELDOUT[a],
                "competition_top":  COMP_HELDOUT[a],
            })

    df = pd.DataFrame(rows)
    mean_row = {
        "assay": "MEAN",
        "DELPHI_cv_rho":  round(df["DELPHI_cv_rho"].mean(), 3),
        "DELPHI_heldout": round(df["DELPHI_heldout"].mean(), 3),
        "AbDevX_heldout": round(df["AbDevX_heldout"].mean(), 3),
        "competition_top":round(df["competition_top"].mean(), 3),
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    print(f"\n{'='*70}")
    print("DELPHI (GDPa1-trained, same cluster_isotype splits) vs AbDev-X")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    out = ROOT / "results" / "delphi_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
