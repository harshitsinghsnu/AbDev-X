"""Process GDPa2.1 and GDPa5 into evaluation-ready CSVs.

GDPa2.1: 51 VHH-Fc antibodies. Full VHH+Fc protein_sequence → extract VHH domain.
GDPa5:   160 VHH-Fc antibodies. Already has vh_protein_sequence separated.

Both datasets contain only VHH (no VL). We run AbDev-X in VH-only mode:
  vhn = VHH sequence, vl = "" (empty string, zeroed in model).

Usage:
    python scripts/process_new_datasets.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data/raw"
PROC = ROOT / "data/processed"
PROC.mkdir(parents=True, exist_ok=True)

FC_LINKER = re.compile(r"GGGGSEPKSS|GGGGSEPKSC|EPKSCDKTHTC")


def extract_vhh(seq: str) -> str:
    """Extract VHH domain by trimming everything from Fc linker onward."""
    seq = str(seq).strip()
    m = FC_LINKER.search(seq)
    if m:
        return seq[: m.start()]
    # Linker not found: if short (<150 AA) treat as pure VHH; else take first 130 AA
    return seq if len(seq) <= 150 else seq[:130]


# ─────────────────────────────────────────────────────────────────────────────
# GDPa2.1 processing
# ─────────────────────────────────────────────────────────────────────────────

def process_gdpa2(xlsx_path: Path, out_path: Path) -> pd.DataFrame:
    print("\n=== Processing GDPa2.1 ===")
    seqs  = pd.read_excel(xlsx_path, sheet_name="Sequences")
    assay = pd.read_excel(xlsx_path, sheet_name="Assay Data - average")

    # Extract VHH from full VHH+Fc protein_sequence
    seqs["vhn"] = seqs["protein_sequence"].map(extract_vhh)
    seqs["vl"]  = ""  # VHH: no light chain
    vhh_lens = seqs["vhn"].str.len()
    print(f"  VHH extracted: {len(seqs)} sequences, len range [{vhh_lens.min()}, {vhh_lens.max()}]")

    # Rename assay columns to match AbDev-X target names
    col_map = {
        "hic_rt_Avg":               "HIC",
        "acsins_dLmax_pH7.4_Avg":   "AC-SINS",
        "tm2_nanodsf_Avg":          "Tm2",
        "sec_%monomer_Avg":         "SEC_monomer",
        "smac_rt_Avg":              "SMAC",
    }
    assay = assay.rename(columns=col_map)

    df = seqs[["antibody_id", "antibody_name", "vhn", "vl"]].merge(
        assay[["antibody_id"] + [c for c in col_map.values() if c in assay.columns]],
        on="antibody_id", how="left"
    )
    print(f"  Final shape: {df.shape}")
    for col in ["HIC", "AC-SINS", "Tm2"]:
        if col in df:
            print(f"  {col}: {df[col].notna().sum()} non-null")

    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GDPa5 processing
# ─────────────────────────────────────────────────────────────────────────────

def process_gdpa5(xlsx_path: Path, out_path: Path) -> pd.DataFrame:
    print("\n=== Processing GDPa5 ===")
    seqs  = pd.read_excel(xlsx_path, sheet_name="Sequences")
    assay = pd.read_excel(xlsx_path, sheet_name="Assay Data - averaged")

    # VHH sequences already separated
    seqs["vhn"] = seqs["vh_protein_sequence"].astype(str).str.strip()
    seqs["vl"]  = ""  # VHH: no light chain

    col_map = {
        "hic_rt_avg":                    "HIC",
        "acsins_dLmax_ph7.4_avg":        "AC-SINS",
        "tm2_nanodsf_avg":               "Tm2",
        "polyreactivity_prscore_cho_avg":"PR_CHO",
        "sec_%monomer_avg":              "SEC_monomer",
        "smac_rt_avg":                   "SMAC",
    }
    assay_renamed = assay.rename(columns=col_map)

    target_cols = [c for c in col_map.values() if c in assay_renamed.columns]
    df = seqs[["antibody_name", "vhn", "vl"]].merge(
        assay_renamed[["antibody_name", "in_model_cohort"] + target_cols],
        on="antibody_name", how="left"
    )
    print(f"  Final shape: {df.shape}")
    print(f"  in_model_cohort: {df['in_model_cohort'].value_counts().to_dict()}")
    for col in ["HIC", "AC-SINS", "Tm2", "PR_CHO"]:
        if col in df:
            print(f"  {col}: {df[col].notna().sum()} non-null")

    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GDPa4.1 — refresh processed files from Excel (adds BVP score)
# ─────────────────────────────────────────────────────────────────────────────

def process_gdpa4_1(xlsx_path: Path, gdpa1_csv: Path, out_mono: Path, out_bsab: Path):
    print("\n=== Processing GDPa4.1 ===")
    assay = pd.read_excel(xlsx_path, sheet_name="Assay Data - median")
    mab_seqs  = pd.read_excel(xlsx_path, sheet_name="mAb Sequences")
    bsab_seqs = pd.read_excel(xlsx_path, sheet_name="bsAb Sequences")

    # Map GDPa4.1 column names to our target names
    assay_col_map = {
        "hihplc_normretentiontime_median":  "HIC",
        "thermostability_tm2_median":        "Tm2",
        "acsins_deltaLmax_1X PBS_median":    "AC-SINS",
        "pr_score_CHO_median":               "PR_CHO",
        "bvp_score_median":                  "BVP",          # new polyreactivity assay
        "sehplc_pct_mono_median":            "SEC_monomer",
        "smachplc_retentiontime_median":     "SMAC",
    }
    assay = assay.rename(columns=assay_col_map)
    target_cols = [c for c in assay_col_map.values() if c in assay.columns]

    # ── Monospecifics ─────────────────────────────────────────────────────────
    mab_names = set(mab_seqs["antibody_name"])
    mono_assay = assay[assay["antibody_name"].isin(mab_names)].copy()
    mono_df = mab_seqs[["antibody_name","vh_protein_sequence","vl_protein_sequence"]].merge(
        mono_assay[["antibody_name"] + target_cols], on="antibody_name", how="left"
    )
    mono_df = mono_df.rename(columns={"vh_protein_sequence":"vhn","vl_protein_sequence":"vl"})
    print(f"  Monospecifics: {len(mono_df)} rows")
    for c in ["HIC","Tm2","AC-SINS","PR_CHO","BVP"]:
        if c in mono_df: print(f"    {c}: {mono_df[c].notna().sum()}")
    mono_df.to_csv(out_mono, index=False)
    print(f"  Saved mono -> {out_mono}")

    # ── Bispecifics ───────────────────────────────────────────────────────────
    bsab_names = set(bsab_seqs["antibody_name"])
    bsab_assay = assay[assay["antibody_name"].isin(bsab_names)].copy()
    bsab_df = bsab_seqs.merge(
        bsab_assay[["antibody_name"] + target_cols], on="antibody_name", how="left"
    )
    # Rename arm-1 sequences to standard names
    bsab_df = bsab_df.rename(columns={
        "vh_1_protein_sequence": "vhn",
        "vl_1_protein_sequence": "vl",
        "vh_2_protein_sequence": "vhn_2",
        "vl_2_protein_sequence": "vl_2",
    })
    print(f"  Bispecifics: {len(bsab_df)} rows, with arm-1 seqs: {bsab_df['vhn'].notna().sum()}")
    for c in ["HIC","Tm2","AC-SINS","PR_CHO","BVP"]:
        if c in bsab_df: print(f"    {c}: {bsab_df[c].notna().sum()}")
    bsab_df.to_csv(out_bsab, index=False)
    print(f"  Saved bsabs -> {out_bsab}")


if __name__ == "__main__":
    # GDPa4.1
    process_gdpa4_1(
        xlsx_path  = RAW / "GDPa4.1_20260616.xlsx",
        gdpa1_csv  = RAW / "GDPa1_v1.2_20250814.csv",
        out_mono   = PROC / "gdpa4_1_monospecifics.csv",
        out_bsab   = PROC / "gdpa4_1_bispecifics.csv",
    )

    # GDPa2.1
    process_gdpa2(
        xlsx_path = RAW / "GDPa2.1.xlsx",
        out_path  = PROC / "gdpa2_1.csv",
    )

    # GDPa5
    process_gdpa5(
        xlsx_path = RAW / "GDPa5_20260811_full.xlsx",
        out_path  = PROC / "gdpa5.csv",
    )
    print("\nAll datasets processed.")
