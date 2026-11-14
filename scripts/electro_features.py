"""Structure-based surface electrostatic features (self-association / polyreactivity drivers).

From each PDB: per-residue SASA (freesasa) weighted by formal charge -> exposed positive/negative
surface, net exposed charge, charge asymmetry, and spatially-patched charge (max +/- patch within
R Angstrom of CA). Cached for GDPa1 + held-out.
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
CHARGE = {"ARG": 1.0, "LYS": 1.0, "HIS": 0.1, "ASP": -1.0, "GLU": -1.0}
PARSER = PDBParser(QUIET=True)


def electro(path: Path, radius=10.0) -> dict:
    try:
        res_areas = freesasa.calc(freesasa.Structure(str(path))).residueAreas()
        model = PARSER.get_structure("x", str(path))[0]
    except Exception:
        return {}
    coords, sasa, q = [], [], []
    for chain in model:
        ch = chain.id
        for res in chain:
            if "CA" not in res:
                continue
            try:
                sa = res_areas[ch][str(res.id[1])].total
            except KeyError:
                continue
            coords.append(res["CA"].coord); sasa.append(sa); q.append(CHARGE.get(res.resname, 0.0))
    if len(coords) < 5:
        return {}
    coords, sasa, q = np.array(coords), np.array(sasa), np.array(q)
    exp_q = sasa * q
    tree = cKDTree(coords)
    patch = np.array([exp_q[tree.query_ball_point(coords[i], radius)].sum() for i in range(len(coords))])
    return {"exp_pos": float(sasa[q > 0].sum()), "exp_neg": float(sasa[q < 0].sum()),
            "net_exp_charge": float(exp_q.sum()), "charge_asym": float(abs(exp_q.sum())),
            "max_pos_patch": float(patch.max()), "max_neg_patch": float(-patch.min()),
            "patch_charge_range": float(patch.max() - patch.min())}


def build(subset, names):
    d = STRUCT / subset
    return pd.DataFrame([{"antibody_name": nm, **(electro(d / f"{nm}.pdb") if (d / f"{nm}.pdb").exists() else {})}
                         for nm in names]).fillna(0.0)


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from abdev.data.dataset import load_gdpa1
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    build("GDPa1", ds.df[ds.id_col].astype(str).tolist()).to_csv(ROOT / "data/processed/electro_gdpa1.csv", index=False)
    print("GDPa1 electro done", flush=True)
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    build("heldout_test", g3["antibody_name"].astype(str).tolist()).to_csv(ROOT / "data/processed/electro_heldout.csv", index=False)
    print("Heldout electro done", flush=True)


if __name__ == "__main__":
    main()
