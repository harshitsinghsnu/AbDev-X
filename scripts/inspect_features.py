"""Inspect bio block and descriptor columns to plan mechanism splits."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

ds = load_gdpa1(str(ROOT / "configs/default.yaml"))

# --- descriptor columns ---
desc = build_descriptor_matrix(ds.heavy, ds.light)
print(f"\n=== DESCRIPTOR COLUMNS ({len(desc.columns)}) ===")
for c in sorted(desc.columns):
    print(f"  {c}")

# --- bio block columns ---
feat_dir = ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1"
BIO_FILES = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv"]
print("\n=== BIO BLOCK COLUMNS ===")
for f in BIO_FILES:
    p = feat_dir / f
    if not p.exists():
        print(f"  {f}: NOT FOUND"); continue
    df = pd.read_csv(p)
    num_cols = df.select_dtypes("number").columns.tolist()
    print(f"\n  {f} ({len(num_cols)} cols):")
    for c in num_cols:
        print(f"    {c}")
