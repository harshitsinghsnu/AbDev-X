"""Process GDPa4 into evaluation-ready CSVs.

Two sub-datasets:
  1. gdpa4_monospecifics.csv  — 71 IgG1 parents (N4), sequences from GDPa1 overlap
  2. gdpa4_bispecifics.csv    — 160 bispecifics (N3): two VH/VL arms, assay targets
                                 from tall CSV (measured), parent predictions for comparison

Usage:
    python scripts/process_gdpa4.py

Outputs to data/processed/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_gdpa4_tall(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["usability"] == "Usable"].copy()
    return df


def pivot_assays(tall: pd.DataFrame, assay_map: dict) -> pd.DataFrame:
    """Pivot tall assay table to wide per antibody_name.
    assay_map: {output_col: (assay_name, value_col, condition)}
    """
    records: dict[str, dict] = {}
    for output_col, (assay, vcol, cond) in assay_map.items():
        sub = tall[(tall["assay"] == assay) & (tall["value_col"] == vcol)]
        if cond:
            sub = sub[sub["condition"] == cond]
        for _, row in sub.iterrows():
            ab = row["antibody_name"]
            if ab not in records:
                records[ab] = {"antibody_name": ab}
            # Average over replicates
            if output_col not in records[ab]:
                records[ab][output_col] = []
            records[ab][output_col].append(float(row["value"]))
    # Collapse lists to means
    wide = {}
    for ab, vals in records.items():
        row = {"antibody_name": ab}
        for col in assay_map:
            v = vals.get(col, [])
            row[col] = float(np.mean(v)) if v else float("nan")
        wide[ab] = row
    return pd.DataFrame(list(wide.values()))


def process_monospecifics(tall: pd.DataFrame, gdpa1_path: Path, out_path: Path) -> pd.DataFrame:
    """Extract N4 monospecifics (no '__x__') and join sequences from GDPa1."""
    # GDPa4 measured monospecifics
    mono_names = tall[~tall["antibody_name"].str.contains("__x__")]["antibody_name"].unique()
    print(f"GDPa4 monospecifics: {len(mono_names)}")

    assay_map = {
        "HIC":     ("HPLC-HIC", "hihplc_normretentiontime",  "default"),
        "Tm2":     ("PTS-IF",   "thermostability_tm2",        "Tm2"),
        "AC-SINS": ("AC-SINS",  "acsins_delta_Lmax",          "1X PBS"),
        "PR_CHO":  ("PR",       "pr_score",                   "CHO"),
    }
    wide = pivot_assays(tall[~tall["antibody_name"].str.contains("__x__")], assay_map)
    print(f"Wide monospecific table: {wide.shape}")
    print("Non-null per column:")
    print(wide.count())

    # Join sequences from GDPa1
    gdpa1 = pd.read_csv(gdpa1_path)
    # Find name column
    name_col = next((c for c in gdpa1.columns
                     if c.lower() in ("antibody_name", "antibody name", "name")), None)
    vh_col = next((c for c in gdpa1.columns
                   if "vh_protein" in c.lower() or c.lower() == "vh"), None)
    vl_col = next((c for c in gdpa1.columns
                   if "vl_protein" in c.lower() or c.lower() == "vl"), None)
    print(f"GDPa1 sequence cols: name={name_col}, VH={vh_col}, VL={vl_col}")

    if name_col and vh_col and vl_col:
        seq_df = gdpa1[[name_col, vh_col, vl_col]].drop_duplicates(name_col)
        seq_df = seq_df.rename(columns={name_col: "antibody_name",
                                         vh_col: "vh_protein_sequence",
                                         vl_col: "vl_protein_sequence"})
        wide = wide.merge(seq_df, on="antibody_name", how="left")
        print(f"After sequence join: {wide.shape}  with_seqs={wide['vh_protein_sequence'].notna().sum()}")
    else:
        print("[warn] Could not find sequence columns in GDPa1")

    wide.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return wide


def process_bispecifics(tall: pd.DataFrame, bsab_design_path: Path, out_path: Path) -> pd.DataFrame:
    """Build bispecific eval table: sequences from exported_bsabs, assays from tall CSV."""
    bsabs = pd.read_csv(bsab_design_path)
    print(f"exported_bsabs: {bsabs.shape}")

    # Assay targets for bispecifics from tall CSV
    bsab_names = tall[tall["antibody_name"].str.contains("__x__")]["antibody_name"].unique()
    print(f"GDPa4 bispecifics with assay data: {len(bsab_names)}")

    assay_map = {
        "HIC":     ("HPLC-HIC", "hihplc_normretentiontime", "default"),
        "Tm2":     ("PTS-IF",   "thermostability_tm2",       "Tm2"),
        "AC-SINS": ("AC-SINS",  "acsins_delta_Lmax",         "1X PBS"),
        "PR_CHO":  ("PR",       "pr_score",                  "CHO"),
    }
    bsab_tall = tall[tall["antibody_name"].str.contains("__x__")]
    wide = pivot_assays(bsab_tall, assay_map)

    # Join sequences from bsab design table
    if "bsab_name" in bsabs.columns and "vh_protein_sequence-1" in bsabs.columns:
        bsab_seqs = bsabs[["bsab_name",
                            "vh_protein_sequence-1", "vl_protein_sequence-1",
                            "vh_protein_sequence-2", "vl_protein_sequence-2",
                            "antibody_name-1", "antibody_name-2"]].copy()
        bsab_seqs = bsab_seqs.rename(columns={"bsab_name": "antibody_name"})
        wide = wide.merge(bsab_seqs, on="antibody_name", how="left")
        print(f"After sequence join: {wide.shape}  with_seqs={wide['vh_protein_sequence-1'].notna().sum()}")
    else:
        print("[warn] Could not find bsab_name / sequence cols in exported_bsabs")

    wide.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return wide


def main():
    raw = ROOT / "data/raw"
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)

    tall_path = raw / "GDPa4_N3_N4_Summary_tall.csv"
    gdpa1_path = raw / "GDPa1_v1.2_20250814.csv"
    bsab_design_path = raw / "gdpa4_exported_bsabs.csv"

    print("Loading GDPa4 tall table...")
    tall = load_gdpa4_tall(tall_path)
    print(f"Tall table (usable only): {tall.shape}")
    print("Assays:", sorted(tall["assay"].unique()))

    print("\n=== Processing monospecifics (N4) ===")
    mono = process_monospecifics(tall, gdpa1_path, processed / "gdpa4_monospecifics.csv")

    print("\n=== Processing bispecifics (N3) ===")
    bsabs = process_bispecifics(tall, bsab_design_path, processed / "gdpa4_bispecifics.csv")

    print("\nDone.")
    print(f"  Monospecifics: {len(mono)}, with sequences: {mono.get('vh_protein_sequence', pd.Series(dtype=float)).notna().sum() if 'vh_protein_sequence' in mono.columns else '?'}")
    print(f"  Bispecifics:   {len(bsabs)}, with sequences: {bsabs.get('vh_protein_sequence-1', pd.Series(dtype=float)).notna().sum() if 'vh_protein_sequence-1' in bsabs.columns else '?'}")


if __name__ == "__main__":
    main()
