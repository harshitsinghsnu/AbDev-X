"""Assemble FLAb2 same-assay external data per GDPa1 assay, de-duplicated vs GDPa1 AND GDPa3.

Outputs one tidy table per assay (heavy, light, z-scored value, study) and a union of unique
external sequences for AbLang2 extraction.
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1

FLAB = ROOT / "external/FLAb/data"
# assay -> [(relpath, value_col_or_None)]
MAP = {
    "HIC": [("aggregation/jain2017biophyscial_HICRT.csv", None),
            ("aggregation/jain2024assessment_HIC.csv", "HIC RT (min)"),
            ("aggregation/shanehsazzadeh2023unlocking_HICRRT.csv", None)],
    "AC-SINS_pH7.4": [("aggregation/jain2017biophysical_ACSINS.csv", None),
                       ("aggregation/jain2024assessment_ACSINS.csv", "AC-SINS ∆λmax (nm)"),
                       ("aggregation/shanehsazzadeh2023unlocking_ACSINS.csv", None)],
    "Tm2": [("thermostability/jain2017biophysical_Tm.csv", "Fab Tm by DSF (°C)"),
            ("thermostability/jain2024assessment_Tm.csv", "Fab Tm by DSF (°C)"),
            ("thermostability/hutchinson2023enhancement_top200tm1_igg.csv", "Tm1 (C) Thermostability (IgG)"),
            ("thermostability/tresanco2023nbthermo_tm.csv", "Tm measurement")],
    "Titer": [("expression/jain2017biophysical_HEK.csv", "HEK Titer (mg/L)"),
              ("expression/garbinski2023_exp.csv", "Expression (µg/mL)")],
    "PR_CHO": [("polyreactivity/jain2017biophysical_PSR.csv", None),
                ("polyreactivity/jain2024assessment_PSR.csv", None),
                ("polyreactivity/shanehsazzadeh2023unlocking_polyspecificity.csv", None)],
}


def norm(s): return re.sub(r"[^A-Z]", "", str(s).upper())


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    exclude = set((ds.heavy.map(norm) + "|" + ds.light.map(norm)).tolist())
    exclude |= set((g3["vh"].map(norm) + "|" + g3["vl"].map(norm)).tolist())

    all_seqs = {}
    summary = []
    for assay, studies in MAP.items():
        frames = []
        for rel, valcol in studies:
            p = FLAB / rel
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "heavy" not in df.columns or "light" not in df.columns:
                continue
            if valcol is None:
                cand = [c for c in df.columns if c not in ("heavy", "light", "fitness")]
                valcol = cand[0] if cand else None
            if valcol not in df.columns:
                continue
            sub = df[["heavy", "light", valcol]].dropna()
            sub.columns = ["heavy", "light", "value"]
            sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
            sub = sub.dropna()
            sub["key"] = sub["heavy"].map(norm) + "|" + sub["light"].map(norm)
            sub = sub[~sub["key"].isin(exclude)].drop_duplicates("key")
            if len(sub) < 5:
                continue
            sub["z"] = (sub["value"] - sub["value"].mean()) / (sub["value"].std() + 1e-9)
            sub["study"] = Path(rel).stem
            frames.append(sub[["heavy", "light", "z", "study", "key"]])
        if not frames:
            continue
        cat = pd.concat(frames, ignore_index=True)
        cat.to_csv(ROOT / f"data/processed/flab_{assay.replace('/', '_')}.csv", index=False)
        summary.append({"assay": assay, "n_external": len(cat), "studies": cat["study"].nunique()})
        for _, r in cat.iterrows():
            all_seqs[r["key"]] = (r["heavy"], r["light"])
    uni = pd.DataFrame([{"key": k, "heavy": v[0], "light": v[1]} for k, v in all_seqs.items()])
    uni.to_csv(ROOT / "data/processed/flab_unique_seqs.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nUnique external sequences for embedding: {len(uni)}")


if __name__ == "__main__":
    main()
