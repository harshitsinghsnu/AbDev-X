"""Prepare GDPa3 held-out set: assemble table (VH+VL+subtypes+labels) and extract embeddings.

GDPa3 provides VH + full light chain + labels; we recover the variable VL and the precomputed
structural features by exact VH match to the abdev-benchmark held-out sequences. ESM-2 (mean+CDR)
and AbLang2 embeddings are extracted on GPU and cached, matching the GDPa1 feature construction.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

XLSX = ROOT / "data/raw/GDPa3_20260106_full.xlsx"
BENCH_SEQ = ROOT / "external/abdev-benchmark/data/heldout-set-sequences.csv"
LABELS = {"HIC": "hic_rt_avg", "Tm2": "tm2_nanodsf_avg", "AC-SINS_pH7.4": "acsins_dLmax_ph7.4_avg",
          "PR_CHO": "polyreactivity_prscore_cho_avg", "Titer": "titer_avg"}


def norm(s): return re.sub(r"[^A-Z]", "", str(s).upper())


def build_table():
    xl = pd.ExcelFile(XLSX)
    seq = xl.parse("Sequences"); avg = xl.parse("Assay Data - average")
    bench = pd.read_csv(BENCH_SEQ)
    bench["vhn"] = bench["vh_protein_sequence"].map(norm)
    seq["vhn"] = seq["vh_protein_sequence"].map(norm)
    m = seq.merge(bench[["vhn", "vl_protein_sequence", "antibody_name"]], on="vhn", how="left")
    df = m.merge(avg[["antibody_id"] + list(LABELS.values())], on="antibody_id", how="left")
    df = df.rename(columns={"vh_protein_sequence": "vh", "vl_protein_sequence": "vl"})
    assert df["vl"].notna().all(), "VL join failed for some rows"
    out = ROOT / "data/processed/gdpa3_table.csv"
    df.to_csv(out, index=False)
    print(f"GDPa3 table: {df.shape} -> {out}; all IgG1={all(df.hc_subtype=='IgG1')}", flush=True)
    return df


def main():
    df = build_table()
    import pandas as pd  # noqa
    from abdev.features.plm import esm2_embeddings
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    fv = (df["vh"].astype(str) + df["vl"].astype(str)).tolist()
    esm = esm2_embeddings(fv, cfg["features"]["esm2_model"], pool=("mean", "cdr"),
                          cache_dir=str(ROOT / "data/processed"))
    np.savez_compressed(ROOT / "data/processed/gdpa3_esm.npz", emb=esm)
    print(f"GDPa3 ESM {esm.shape}", flush=True)

    import torch, ablang2
    model = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1,
                               device="cuda" if torch.cuda.is_available() else "cpu")
    pairs = [[str(h), str(l)] for h, l in zip(df["vh"], df["vl"])]
    ab = []
    for i in range(0, len(pairs), 16):
        ab.append(np.asarray(model(pairs[i:i + 16], mode="seqcoding")))
    ab = np.vstack(ab)
    np.savez_compressed(ROOT / "data/processed/gdpa3_ablang2.npz", emb=ab)
    print(f"GDPa3 AbLang2 {ab.shape}", flush=True)


if __name__ == "__main__":
    main()
