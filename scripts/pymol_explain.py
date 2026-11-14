"""Structure-mapped explainability: per-residue mechanistic attribution rendered with PyMOL.

For representative antibodies we compute per-residue attribution:
  * HIC (hydrophobicity): exposed hydrophobicity  = SASA_res x max(Kyte-Doolittle, 0)
  * AC-SINS (self-assoc): exposed charge (signed)  = SASA_res x formal_charge
write it into the PDB B-factor column, and ray-trace a surface colored by the attribution
(hot = developability-liability hotspot). Highlights that model-relevant signal localizes to
surface CDR/patch regions, consistent with the SHAP feature attributions.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import freesasa
from Bio.PDB import PDBParser, PDBIO

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "external/abdev-benchmark/data/structures/AntiBodyBuilder3/GDPa1"
OUT = ROOT / "results/explain/structures"; OUT.mkdir(parents=True, exist_ok=True)
KD = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5, "GLU": -3.5,
      "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8,
      "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2}
CHG = {"ARG": 1.0, "LYS": 1.0, "HIS": 0.1, "ASP": -1.0, "GLU": -1.0}
PARSER = PDBParser(QUIET=True)


def write_attribution(pdb, mode):
    res_areas = freesasa.calc(freesasa.Structure(str(pdb))).residueAreas()
    struct = PARSER.get_structure("x", str(pdb))
    vals = []
    for chain in struct[0]:
        for res in chain:
            try:
                sa = res_areas[chain.id][str(res.id[1])].total
            except KeyError:
                sa = 0.0
            if mode == "hic":
                v = sa * max(KD.get(res.resname, 0.0), 0.0)
            else:  # charge (signed)
                v = sa * CHG.get(res.resname, 0.0)
            for atom in res:
                atom.set_bfactor(float(v))
            vals.append(v)
    out = OUT / f"{pdb.stem}_{mode}.pdb"
    io = PDBIO(); io.set_structure(struct); io.save(str(out))
    return out, float(np.max(np.abs(vals)) or 1.0)


def render(pdb_b, mode, vmax, title):
    import pymol2
    png = OUT / f"{pdb_b.stem}.png"
    with pymol2.PyMOL() as p:
        c = p.cmd
        c.load(str(pdb_b), "ab")
        c.hide("everything"); c.show("surface")
        c.bg_color("white"); c.set("ray_opaque_background", 0)
        if mode == "hic":
            c.spectrum("b", "white_orange_red", "ab", minimum=0, maximum=vmax)
        else:
            c.spectrum("b", "blue_white_red", "ab", minimum=-vmax, maximum=vmax)
        c.set("surface_quality", 1); c.orient("ab")
        c.ray(1400, 1050); c.png(str(png), dpi=150)
    return png


def main():
    sys.path.insert(0, str(ROOT / "src"))
    from abdev.data.dataset import load_gdpa1
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    df = ds.df.copy()
    hic_col = ds.target_cols["HIC"]; ac_col = ds.target_cols["AC-SINS_pH7.4"]
    idc = ds.id_col

    picks = []
    # highest-HIC (hydrophobic) antibody with an available structure
    for _, r in df.sort_values(hic_col, ascending=False).iterrows():
        if (STRUCT / f"{r[idc]}.pdb").exists():
            picks.append((r[idc], "hic", f"HIC hotspots: {r[idc]} (HIC RT={r[hic_col]:.1f})")); break
    # highest self-association antibody
    for _, r in df.sort_values(ac_col, ascending=False).iterrows():
        if (STRUCT / f"{r[idc]}.pdb").exists():
            picks.append((r[idc], "charge", f"Charge patches: {r[idc]} (AC-SINS={r[ac_col]:.1f})")); break

    rows = []
    for name, mode, title in picks:
        pdb = STRUCT / f"{name}.pdb"
        pdb_b, vmax = write_attribution(pdb, mode)
        png = render(pdb_b, mode, vmax, title)
        rows.append({"antibody": name, "mode": mode, "png": png.name, "title": title})
        print(f"rendered {png}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)


if __name__ == "__main__":
    main()
