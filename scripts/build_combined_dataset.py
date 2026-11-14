"""Build combined_train.csv by pooling GDPa1 + GDPa2.1 + GDPa5.

Strategy:
  - GDPa1 (246 IgG): primary training set with all 5 properties
  - GDPa2.1 (51 VHH-Fc): HIC, AC-SINS, Tm2
  - GDPa5 (160 VHH-Fc): HIC, AC-SINS, Tm2, PR_CHO

Output: data/processed/combined_train.csv with columns:
  antibody_id, vhn, vl, format, HIC, Tm2, AC-SINS_pH7.4, PR_CHO, Titer

VHH rows: vl = "" (VL features zeroed by eval_heldout.py)
format column: "IgG" | "VHH-Fc" (can be used as a feature for format-aware models)

Note: GDPa3 (80 IgG) is held-out and NOT included here.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np

PROPS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
OUT = ROOT / "data" / "processed" / "combined_train.csv"


def load_gdpa1() -> pd.DataFrame:
    # Find the GDPa1 CSV
    raw_dir = ROOT / "data" / "raw"
    csvs = sorted(raw_dir.glob("GDPa1*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError("GDPa1 CSV not found in data/raw/")
    df = pd.read_csv(csvs[0])
    print(f"  GDPa1: {len(df)} rows from {csvs[0].name}")

    # Standardize columns
    out = pd.DataFrame()
    out["antibody_id"] = df["antibody_name"] if "antibody_name" in df.columns else df.index.astype(str)
    out["vhn"] = df["vh_protein_sequence"].astype(str).str.strip()
    out["vl"] = df["vl_protein_sequence"].fillna("").astype(str).str.strip()
    out["format"] = "IgG"

    prop_aliases = {
        "HIC":           ["HIC", "HIC_retention_time"],
        "Tm2":           ["Tm2", "Thermostability_Tm2"],
        "AC-SINS_pH7.4": ["AC-SINS_pH7.4", "AC-SINS pH7.4"],
        "PR_CHO":        ["PR_CHO", "Polyreactivity_CHO"],
        "Titer":         ["Titer", "Titer_mg_L"],
    }
    for prop, aliases in prop_aliases.items():
        col = next((c for c in aliases if c in df.columns), None)
        out[prop] = df[col].to_numpy(dtype=float) if col else np.nan
    return out


def load_gdpa2() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "processed" / "gdpa2_1.csv")
    print(f"  GDPa2.1: {len(df)} rows")
    out = pd.DataFrame()
    out["antibody_id"] = df.get("antibody_name", df.get("antibody_id", pd.Series(range(len(df))))).astype(str)
    out["vhn"] = df["vhn"].astype(str).str.strip()
    out["vl"] = ""
    out["format"] = "VHH-Fc"
    out["HIC"] = df["HIC"].to_numpy(dtype=float) if "HIC" in df.columns else np.nan
    out["Tm2"] = df["Tm2"].to_numpy(dtype=float) if "Tm2" in df.columns else np.nan
    out["AC-SINS_pH7.4"] = df["AC-SINS"].to_numpy(dtype=float) if "AC-SINS" in df.columns else np.nan
    out["PR_CHO"] = np.nan
    out["Titer"] = np.nan
    return out


def load_gdpa5() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "processed" / "gdpa5.csv")
    print(f"  GDPa5: {len(df)} rows")
    out = pd.DataFrame()
    out["antibody_id"] = df["antibody_name"].astype(str)
    out["vhn"] = df["vhn"].astype(str).str.strip()
    out["vl"] = ""
    out["format"] = "VHH-Fc"
    out["HIC"] = df["HIC"].to_numpy(dtype=float) if "HIC" in df.columns else np.nan
    out["Tm2"] = df["Tm2"].to_numpy(dtype=float) if "Tm2" in df.columns else np.nan
    out["AC-SINS_pH7.4"] = df["AC-SINS"].to_numpy(dtype=float) if "AC-SINS" in df.columns else np.nan
    out["PR_CHO"] = df["PR_CHO"].to_numpy(dtype=float) if "PR_CHO" in df.columns else np.nan
    out["Titer"] = np.nan
    return out


def main():
    print("Building combined training dataset...")
    parts = [load_gdpa1(), load_gdpa2(), load_gdpa5()]
    combined = pd.concat(parts, ignore_index=True)
    combined["vl"] = combined["vl"].fillna("")

    print(f"\nCombined: {len(combined)} antibodies")
    for prop in PROPS:
        n = combined[prop].notna().sum()
        print(f"  {prop}: {n} labeled ({100*n/len(combined):.0f}%)")
    print(f"  IgG: {(combined['format']=='IgG').sum()}")
    print(f"  VHH-Fc: {(combined['format']=='VHH-Fc').sum()}")

    combined.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
