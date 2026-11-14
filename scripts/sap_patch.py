"""Spatial Aggregation Propensity (patch-based) features for HIC.

Proper SAP (Chennamsetty et al. 2009): hydrophobicity averaged over a spatial neighborhood, capturing
surface hydrophobic PATCHES (the physical driver of HIC retention & aggregation), not just per-residue
exposure. For each residue we sum (SASA x hydrophobicity) over residues within R Angstrom of its CA,
then aggregate to antibody-level patch descriptors. Cached for GDPa1 + held-out.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import freesasa
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "external/abdev-benchmark/data/structures/AntiBodyBuilder3"
KD = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5, "GLU": -3.5,
      "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8,
      "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2}
PARSER = PDBParser(QUIET=True)


def patch_sap(path: Path, radius=10.0) -> dict:
    try:
        struct = freesasa.Structure(str(path)); res_areas = freesasa.calc(struct).residueAreas()
        model = PARSER.get_structure("x", str(path))[0]
    except Exception:
        return {}
    coords, sasa, hydro = [], [], []
    for chain in model:
        ch = chain.id
        for res in chain:
            if "CA" not in res:
                continue
            resnum = str(res.id[1])
            try:
                sa = res_areas[ch][resnum].total
            except KeyError:
                continue
            coords.append(res["CA"].coord); sasa.append(sa); hydro.append(KD.get(res.resname, 0.0))
    if len(coords) < 5:
        return {}
    coords = np.array(coords); sasa = np.array(sasa); hydro = np.array(hydro)
    tree = cKDTree(coords)
    contrib = sasa * hydro                       # per-residue exposed hydrophobicity
    sap = np.array([contrib[tree.query_ball_point(coords[i], radius)].sum() for i in range(len(coords))])
    pos = sap[sap > 0]
    return {"patch_sap_max": float(sap.max()), "patch_sap_sum_pos": float(pos.sum()),
            "patch_sap_top5": float(np.sort(sap)[-5:].mean()), "patch_sap_n_hot": float((sap > np.percentile(sap, 90)).sum()),
            "patch_sap_mean": float(sap.mean())}


def build(subset, names):
    d = STRUCT / subset
    rows = []
    for nm in names:
        p = d / f"{nm}.pdb"
        rows.append({"antibody_name": nm, **(patch_sap(p) if p.exists() else {})})
    return pd.DataFrame(rows).fillna(0.0)


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from abdev.data.dataset import load_gdpa1
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    build("GDPa1", ds.df[ds.id_col].astype(str).tolist()).to_csv(ROOT / "data/processed/sap_patch_gdpa1.csv", index=False)
    print("GDPa1 patch-SAP done", flush=True)
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    build("heldout_test", g3["antibody_name"].astype(str).tolist()).to_csv(ROOT / "data/processed/sap_patch_heldout.csv", index=False)
    print("Heldout patch-SAP done", flush=True)


if __name__ == "__main__":
    main()
