"""Process DOTAD 2.0 into AbDev-X canonical format.

Data source:
  DOTAD 2.0 (Li et al. 2024): https://doi.org/10.5281/zenodo.21742780

Strategy:
  - VH/VL sequences come from the benchmark task split files (each has DOTAD_ID + VH + VL)
  - Assay values come from harmonized_antibody_assay_matrix.csv
  - Merge on DOTAD_ID, then map columns to AbDev-X PROPS
  - Keep "Whole mAb" format AND "Unknown" format entries that have paired VH+VL
    (Unknown = format metadata missing from source, not necessarily non-IgG;
     the GDPa3 antibodies — confirmed IgG — appear as "Unknown" in DOTAD)
  - Discard Fab, VHH, Nanobody, bispecific, and any entry missing VL
  - Remove any VH overlapping GDPa3 held-out
  - Dedup by VH (exact)

Column mapping:
  HIC        -> HIC
  AC_SINS    -> AC-SINS_pH7.4  (pH7.4 measurement)
  Tm         -> Tm2
  PR_score   -> PR_CHO         (CHO PSR score)

Output:
  data/processed/dotad_clean.csv    (DOTAD-only, n=?)
  data/processed/mega_dotad.csv     (mega_full + DOTAD)

Usage:
    python scripts/process_dotad.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

OUT_DIR  = ROOT / "data" / "processed"
DOTAD_BASE = ROOT / "data" / "raw" / "DOTAD2" / "DOTAD2_research_archive_v2.0.0_revision_final"
PROPS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]

# assay tasks with benchmark split files containing VH/VL sequences
TASK_DIRS = [
    "hydrophobicity",
    "thermal_stability",
    "self_association",
    "self_interaction",
    "polyreactivity",
    "aggregation",
    "expression",
]
BENCHMARK_DIR = DOTAD_BASE / "analysis_derivatives" / "historical" / "benchmark" / "package" / "DOTAD2_benchmark_package_v1.1"
HARMONIZED    = DOTAD_BASE / "analysis_derivatives" / "historical" / "benchmark" / "data" / "harmonized_antibody_assay_matrix.csv"


def load_sequences() -> pd.DataFrame:
    """Collect all unique (DOTAD_ID, VH, VL, Format) from benchmark task files."""
    frames = []
    for task in TASK_DIRS:
        for split in ("train", "valid", "test"):
            p = BENCHMARK_DIR / task / f"{split}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p, usecols=lambda c: c in
                             ["DOTAD_ID", "Antibody_Name", "VH", "VL", "Format",
                              "hc_subtype", "lc_subtype"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No benchmark task files found under {BENCHMARK_DIR}")
    combined = pd.concat(frames, ignore_index=True)
    # keep one row per DOTAD_ID (sequences are identical across tasks for same ID)
    return combined.drop_duplicates("DOTAD_ID").reset_index(drop=True)


def load_assays() -> pd.DataFrame:
    """Load harmonized assay matrix and rename columns to AbDev-X convention."""
    df = pd.read_csv(HARMONIZED)
    df = df.rename(columns={
        "DOTAD_ID": "DOTAD_ID",
        "HIC":      "HIC",
        "Tm":       "Tm2",
        "AC_SINS":  "AC-SINS_pH7.4",
        "PR_score": "PR_CHO",
    })
    # keep only columns we care about
    keep = ["DOTAD_ID"] + [p for p in PROPS if p in df.columns]
    return df[keep]


def main():
    print("Loading DOTAD 2.0 sequences from benchmark task files...")
    seqs = load_sequences()
    print(f"  {len(seqs)} unique DOTAD antibodies with sequences")
    print(f"  Formats: {dict(seqs['Format'].value_counts())}")

    # keep whole-mAb + Unknown format entries that have a real VL sequence
    # exclude known non-IgG formats explicitly
    exclude_formats = {
        "Fab", "Fab2", "F(ab')2", "scFv", "Nanobody", "VHH",
        "Bispecific mAb", "Bispecific Single Domains (VH-VH')",
        "Bispecific Single Domains (VH-VH'-VH')",
        "Bispecific Single Domains (VH-VH'-VH)",
        "Trispecific Single Domains (VH-VH'-VH''); Bispecific Entry",
        "Mixed Nanobody (VHH-CH2-CH3 dimer)",
        "Single Domain Variable Fragment;H",
        "di-scFv + CH4", "di-scFv+Fc", "Fab + di-Fc",
        "Whole mAb Radiolabelled", "Whole mAb ADC",
    }
    seqs_filtered = seqs[~seqs["Format"].isin(exclude_formats)].copy()
    # require paired VL (length > 10 aa) to ensure IgG-like structure
    seqs_filtered = seqs_filtered[
        seqs_filtered["VL"].notna() & (seqs_filtered["VL"].str.len() > 10)
    ]
    print(f"  After format filter (non-excluded + has VL): {len(seqs_filtered)}")
    print(f"  Format breakdown: {dict(seqs_filtered['Format'].value_counts())}")

    print("\nLoading harmonized assay matrix...")
    assays = load_assays()
    print(f"  {len(assays)} rows, props: {[p for p in PROPS if p in assays.columns]}")

    # merge
    df = seqs_filtered.merge(assays, on="DOTAD_ID", how="left")
    print(f"\nMerged: {len(df)} rows")

    # standardise sequences
    df = df.rename(columns={"VH": "vhn", "VL": "vl"})
    df["vhn"] = df["vhn"].fillna("").astype(str).str.strip()
    df["vl"]  = df["vl"].fillna("").astype(str).str.strip()
    # drop rows with no VH
    df = df[df["vhn"].str.len() > 20].copy()
    print(f"  After removing empty VH: {len(df)}")

    # convert assay columns to numeric
    for p in PROPS:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")
        else:
            df[p] = np.nan

    # dedup by VH (average numeric cols)
    n_before = len(df)
    num_cols = [p for p in PROPS if p in df.columns]
    meta_cols = [c for c in df.columns if c not in num_cols + ["vhn"]]
    agg = {c: "mean" for c in num_cols}
    for c in meta_cols:
        agg[c] = "first"
    df = df.groupby("vhn", as_index=False).agg(agg)
    print(f"  After dedup by VH: {len(df)} (removed {n_before - len(df)} duplicates)")

    # contamination check: remove GDPa3
    g3 = pd.read_csv(OUT_DIR / "gdpa3_table.csv")
    gdpa3_vhn = set(g3["vhn"].dropna().str.strip())
    n_before = len(df)
    df = df[~df["vhn"].isin(gdpa3_vhn)].copy()
    print(f"  Removed {n_before - len(df)} GDPa3 overlapping sequences")

    # finalize
    df["format"] = "IgG"
    df["source"] = "DOTAD"
    df["z_scored"] = False
    dotad_clean = df[["vhn", "vl", "format", "source", "z_scored"] + PROPS]

    # prop coverage
    print("\nProperty coverage in DOTAD clean:")
    for p in PROPS:
        nn = dotad_clean[p].notna().sum()
        print(f"  {p:<18} {nn:>5d} / {len(dotad_clean)} ({nn/max(len(dotad_clean),1):.0%})")

    dotad_clean.to_csv(OUT_DIR / "dotad_clean.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/dotad_clean.csv  ({len(dotad_clean)} antibodies)")

    # check overlap with existing Ginkgo training data
    mega_full = pd.read_csv(OUT_DIR / "mega_full.csv")
    existing_vhn = set(mega_full["vhn"].astype(str).str.strip())
    dotad_new = dotad_clean[~dotad_clean["vhn"].isin(existing_vhn)].copy()
    print(f"DOTAD new (not in mega_full): {len(dotad_new)}")
    print(f"DOTAD already in mega_full:   {len(dotad_clean) - len(dotad_new)}")

    # build mega_dotad
    if "z_scored" not in mega_full.columns:
        mega_full["z_scored"] = False
    mega_dotad = pd.concat([mega_full, dotad_new], ignore_index=True)
    mega_dotad.to_csv(OUT_DIR / "mega_dotad.csv", index=False)
    print(f"\nSaved -> {OUT_DIR}/mega_dotad.csv  ({len(mega_dotad)} total antibodies)")
    print("\nSource breakdown in mega_dotad:")
    print(mega_dotad["source"].value_counts().to_string())
    print("\nNext: python scripts/eval_mega_dataset.py")


if __name__ == "__main__":
    main()
