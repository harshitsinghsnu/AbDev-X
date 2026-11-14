"""Structure-based hydrophobicity features (Spatial Aggregation Propensity) for HIC.

For each antibody PDB (AntiBodyBuilder3), compute solvent-accessible surface area per residue
(freesasa) and weight by residue hydrophobicity -> exposed/hydrophobic SASA descriptors that
physically drive HIC retention. Cached to CSV for GDPa1 and the held-out set.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import freesasa

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "external/abdev-benchmark/data/structures/AntiBodyBuilder3"

# Kyte-Doolittle hydrophobicity (positive = hydrophobic)
KD = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5, "GLU": -3.5,
      "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8,
      "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2}
AROM = {"PHE", "TRP", "TYR"}


def sap_for_pdb(path: Path) -> dict:
    try:
        struct = freesasa.Structure(str(path))
        result = freesasa.calc(struct)
        res_areas = result.residueAreas()
    except Exception:
        return {}
    total_sasa = hydro_sasa = arom_sasa = sap = pos_sap = 0.0
    for chain, residues in res_areas.items():
        for _resnum, area in residues.items():
            aa = area.residueType
            sa = area.total
            total_sasa += sa
            h = KD.get(aa, 0.0)
            sap += sa * h
            if h > 0:
                hydro_sasa += sa
                pos_sap += sa * h
            if aa in AROM:
                arom_sasa += sa
    return {"sap_total": sap, "sap_positive": pos_sap, "hydrophobic_sasa": hydro_sasa,
            "aromatic_sasa": arom_sasa, "total_sasa": total_sasa,
            "hydrophobic_frac_sasa": hydro_sasa / total_sasa if total_sasa else 0.0}


def build(subset: str, names_map: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """names_map: DataFrame with `key` (output id) and `pdb_stem` (file name without .pdb)."""
    d = STRUCT / subset
    rows = []
    for _, r in names_map.iterrows():
        p = d / f"{r['pdb_stem']}.pdb"
        feats = sap_for_pdb(p) if p.exists() else {}
        rows.append({name_col: r["key"], "found": p.exists(), **feats})
    return pd.DataFrame(rows).fillna(0.0)


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from abdev.data.dataset import load_gdpa1
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    # GDPa1 pdb file stems = antibody_name
    g1 = pd.DataFrame({"key": ds.df[ds.id_col].values, "pdb_stem": ds.df[ds.id_col].values})
    df1 = build("GDPa1", g1, "antibody_name")
    df1.to_csv(ROOT / "data/processed/sap_gdpa1.csv", index=False)
    print(f"GDPa1 SAP: {df1['found'].sum()}/{len(df1)} structures found", flush=True)

    # held-out: pdb stems = benchmark antibody_name (from gdpa3 table)
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    g3m = pd.DataFrame({"key": g3["antibody_name"].values, "pdb_stem": g3["antibody_name"].values})
    df3 = build("heldout_test", g3m, "antibody_name")
    df3.to_csv(ROOT / "data/processed/sap_heldout.csv", index=False)
    print(f"Heldout SAP: {df3['found'].sum()}/{len(df3)} structures found", flush=True)


if __name__ == "__main__":
    main()
