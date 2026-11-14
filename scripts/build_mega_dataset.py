"""Build a deduplicated mega-dataset from all available training sources.

Sources:
  - GDPa1  (IgG,     n=246): HIC, AC-SINS_pH7.4, Tm2, PR_CHO, Titer  [raw]
  - GDPa2.1 (VHH-Fc, n=51): HIC, AC-SINS, Tm2                         [raw]
  - GDPa5  (VHH-Fc,  n=160): HIC, AC-SINS, Tm2, PR_CHO                [raw]
  - FLAb   (IgG mix, n=340): HIC, AC-SINS, Tm2, PR_CHO, Titer          [z-scored/study]
  - GDPa4 mono (IgG,  n=71): all already in GDPa1 — skipped

Deduplication:
  - Key: VHN sequence (exact match)
  - Within-source: average measurements for identical VHN rows
  - Cross-source: GDPa1 > GDPa2.1/GDPa5 > FLAb (lower-priority rows dropped when VHN matches)
  - GDPa2.1 ∩ GDPa5 (8 VHN): average measurements between the two VHH-Fc sources

Output (data/processed/):
  mega_ginkgo.csv  — GDPa1+GDPa2.1+GDPa5, raw values, deduplicated
  mega_full.csv    — mega_ginkgo + FLAb (z-scored per study per assay)
  mega_dedup_report.txt — overlap / drop counts

Usage:
    python scripts/build_mega_dataset.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

OUT_DIR = ROOT / "data" / "processed"

PROPS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]

# ── helpers ──────────────────────────────────────────────────────────────────

def norm_vhn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def dedup_within(df: pd.DataFrame, key: str = "vhn") -> pd.DataFrame:
    """Average numeric columns for rows with the same key."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    meta_cols = [c for c in df.columns if c not in num_cols and c != key]
    agg = {c: "mean" for c in num_cols}
    for c in meta_cols:
        agg[c] = "first"
    return df.groupby(key, as_index=False).agg(agg)


def standardise_props(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename assay columns to canonical PROPS names; add missing as NaN."""
    df = df.rename(columns=col_map)
    for p in PROPS:
        if p not in df.columns:
            df[p] = np.nan
    return df[["vhn", "vl", "format", "source"] + PROPS]


# ── load GDPa1 ───────────────────────────────────────────────────────────────

def load_gdpa1() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "raw" / "GDPa1_v1.2_20250814.csv")
    df = df.rename(columns={"vh_protein_sequence": "vhn",
                             "vl_protein_sequence": "vl"})
    df["format"] = "IgG"
    df["source"] = "GDPa1"
    return standardise_props(df, {"AC-SINS_pH7.4": "AC-SINS_pH7.4"})


# ── load GDPa2.1 ─────────────────────────────────────────────────────────────

def load_gdpa2() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "processed" / "gdpa2_1.csv")
    df["format"] = "VHH-Fc"
    df["source"] = "GDPa2.1"
    # VHH — no light chain
    if "vl" not in df.columns:
        df["vl"] = ""
    else:
        df["vl"] = df["vl"].fillna("")
    return standardise_props(df, {"AC-SINS": "AC-SINS_pH7.4"})


# ── load GDPa5 ───────────────────────────────────────────────────────────────

def load_gdpa5() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "processed" / "gdpa5.csv")
    df["format"] = "VHH-Fc"
    df["source"] = "GDPa5"
    if "vl" not in df.columns:
        df["vl"] = ""
    else:
        df["vl"] = df["vl"].fillna("")
    return standardise_props(df, {"AC-SINS": "AC-SINS_pH7.4"})


# ── load FLAb (already z-scored per study per assay) ─────────────────────────

def load_flab() -> pd.DataFrame:
    """Load FLAb per-assay z-scored files, merge on (heavy, light)."""
    assay_files = {
        "HIC": "flab_HIC.csv",
        "AC-SINS_pH7.4": "flab_AC-SINS_pH7.4.csv",
        "Tm2": "flab_Tm2.csv",
        "PR_CHO": "flab_PR_CHO.csv",
        "Titer": "flab_Titer.csv",
    }
    seqs = pd.read_csv(OUT_DIR / "flab_unique_seqs.csv")[["key", "heavy", "light"]]
    seqs = seqs.rename(columns={"heavy": "vhn", "light": "vl"})

    prop_dfs: list[pd.DataFrame] = []
    for prop, fname in assay_files.items():
        p = OUT_DIR / fname
        if not p.exists():
            continue
        fp = pd.read_csv(p)[["key", "z"]].rename(columns={"z": prop})
        prop_dfs.append(fp)

    if not prop_dfs:
        return pd.DataFrame(columns=["vhn", "vl", "format", "source"] + PROPS)

    merged = prop_dfs[0]
    for fp in prop_dfs[1:]:
        merged = merged.merge(fp, on="key", how="outer")

    df = seqs.merge(merged, on="key", how="left")
    df["format"] = "IgG"   # FLAb is predominately clinical IgG
    df["source"] = "FLAb"
    df["vl"] = df["vl"].fillna("")
    return standardise_props(df, {})


# ── build merged dataset ──────────────────────────────────────────────────────

def build_mega(include_flab: bool = True) -> tuple[pd.DataFrame, dict]:
    g1 = load_gdpa1()
    g2 = load_gdpa2()
    g5 = load_gdpa5()

    report: dict = {}

    # ── within-source dedup ──────────────────────────────────────────────────
    g1_dedup = dedup_within(g1)
    g2_dedup = dedup_within(g2)
    g5_dedup = dedup_within(g5)

    report["GDPa1_raw"] = len(g1)
    report["GDPa1_dedup"] = len(g1_dedup)
    report["GDPa2.1_raw"] = len(g2)
    report["GDPa2.1_dedup"] = len(g2_dedup)
    report["GDPa5_raw"] = len(g5)
    report["GDPa5_dedup"] = len(g5_dedup)

    # ── cross-source dedup: GDPa2.1 ∩ GDPa5 ─────────────────────────────────
    overlap_g2_g5 = set(g2_dedup["vhn"]) & set(g5_dedup["vhn"])
    report["GDPa2.1_GDPa5_overlap"] = len(overlap_g2_g5)

    if overlap_g2_g5:
        # average overlapping rows from both sources, replace in g2_dedup
        g2_ov = g2_dedup[g2_dedup["vhn"].isin(overlap_g2_g5)].copy()
        g5_ov = g5_dedup[g5_dedup["vhn"].isin(overlap_g2_g5)].copy()
        combined_ov = pd.concat([g2_ov, g5_ov], ignore_index=True)
        averaged = dedup_within(combined_ov)
        averaged["source"] = "GDPa2.1+GDPa5"

        g2_dedup = g2_dedup[~g2_dedup["vhn"].isin(overlap_g2_g5)]
        g5_dedup = g5_dedup[~g5_dedup["vhn"].isin(overlap_g2_g5)]
        vhh_pool = pd.concat([g2_dedup, g5_dedup, averaged], ignore_index=True)
    else:
        vhh_pool = pd.concat([g2_dedup, g5_dedup], ignore_index=True)

    # ── GDPa1 supersedes GDPa4 mono (same VHN) — already handled since G4mono
    # is a strict subset of G1; no action needed.

    # ── merge Ginkgo sources: G1 takes priority over VHH sources ─────────────
    g1_vhn = set(g1_dedup["vhn"])
    vhh_pool_clean = vhh_pool[~vhh_pool["vhn"].isin(g1_vhn)]
    report["VHH_dropped_overlap_with_G1"] = len(vhh_pool) - len(vhh_pool_clean)

    mega_ginkgo = pd.concat([g1_dedup, vhh_pool_clean], ignore_index=True)
    report["mega_ginkgo_total"] = len(mega_ginkgo)

    if not include_flab:
        return mega_ginkgo, report

    # ── FLAb: add sequences not already in Ginkgo pool ───────────────────────
    flab = load_flab()
    flab_dedup = dedup_within(flab)
    report["FLAb_raw"] = len(flab)
    report["FLAb_dedup"] = len(flab_dedup)

    flab_new = flab_dedup[~flab_dedup["vhn"].isin(set(mega_ginkgo["vhn"]))]
    report["FLAb_new_after_dedup"] = len(flab_new)

    # FLAb values are already z-scored per study; flag them
    flab_new = flab_new.copy()
    flab_new["z_scored"] = True
    mega_ginkgo["z_scored"] = False

    mega_full = pd.concat([mega_ginkgo, flab_new], ignore_index=True)
    report["mega_full_total"] = len(mega_full)

    return mega_full, report


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building mega dataset...")

    # Ginkgo-only (raw values)
    mega_ginkgo, report_g = build_mega(include_flab=False)
    mega_ginkgo.to_csv(OUT_DIR / "mega_ginkgo.csv", index=False)
    print(f"  mega_ginkgo: {len(mega_ginkgo)} rows -> data/processed/mega_ginkgo.csv")

    # Full (Ginkgo + FLAb)
    mega_full, report = build_mega(include_flab=True)
    mega_full.to_csv(OUT_DIR / "mega_full.csv", index=False)
    print(f"  mega_full:   {len(mega_full)} rows -> data/processed/mega_full.csv")

    # ── write dedup report ───────────────────────────────────────────────────
    lines = [
        "=== Mega Dataset Deduplication Report ===",
        "",
        "Sources:",
        f"  GDPa1    (IgG):    {report['GDPa1_raw']} raw -> {report['GDPa1_dedup']} unique VHN",
        f"  GDPa2.1  (VHH-Fc): {report['GDPa2.1_raw']} raw -> {report['GDPa2.1_dedup']} unique VHN",
        f"  GDPa5    (VHH-Fc): {report['GDPa5_raw']} raw -> {report['GDPa5_dedup']} unique VHN",
        f"  FLAb     (IgG):    {report['FLAb_raw']} raw -> {report['FLAb_dedup']} unique VHN",
        "",
        "Cross-source deduplication:",
        f"  GDPa2.1 ∩ GDPa5:   {report['GDPa2.1_GDPa5_overlap']} VHN shared (averaged)",
        f"  VHH dropped (in G1): {report['VHH_dropped_overlap_with_G1']}",
        f"  FLAb new (not in Ginkgo): {report['FLAb_new_after_dedup']}",
        "",
        "Final datasets:",
        f"  mega_ginkgo (G1+G2.1+G5):      {report['mega_ginkgo_total']} unique antibodies",
        f"  mega_full  (Ginkgo+FLAb):       {report['mega_full_total']} unique antibodies",
        "",
        "Hold-out: GDPa3 (n=80), 0 VHN overlap with any training source",
    ]
    report_path = OUT_DIR / "mega_dedup_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDedup report -> {report_path}")
    print("\n".join(lines[2:]))

    # ── prop coverage summary ────────────────────────────────────────────────
    print("\nProperty coverage in mega_full:")
    print(f"  {'Prop':<18} {'n_nonNaN':>10} {'frac':>8}")
    for p in PROPS:
        if p in mega_full.columns:
            nn = mega_full[p].notna().sum()
            print(f"  {p:<18} {nn:>10d} {nn/len(mega_full):>8.1%}")

    print("\nFormat breakdown:")
    print(mega_full["format"].value_counts().to_string())
    print("\nSource breakdown:")
    print(mega_full["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
