"""Supplementary figure: structure-mapped explainability for ALL five developability properties,
in multiple views (molecular surface + cartoon with CDRs), with a CDR-enrichment ground-truth check.

For one representative antibody we compute per-residue mechanistic attribution for each property and
color both a solvent surface and a cartoon by it. We then quantify what fraction of the top-quartile
attributed residues fall in CDRs -- the literature expectation (TAP/Boughter) is that developability
liabilities concentrate in the CDR loops.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import freesasa
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Polypeptide import three_to_index, index_to_one
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.features.numbering import number_chain

STRUCT = ROOT / "external/abdev-benchmark/data/structures/AntiBodyBuilder3/GDPa1"
OUT = ROOT / "results/explain/supplementary"; OUT.mkdir(parents=True, exist_ok=True)
PARSER = PDBParser(QUIET=True)
KD = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5, "GLU": -3.5,
      "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8,
      "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2}
CHG = {"ARG": 1.0, "LYS": 1.0, "HIS": 0.1, "ASP": -1.0, "GLU": -1.0}

# property -> (attribution mode, colormap, diverging?, panel label, published ground truth)
PROPS = {
    "HIC": ("hydrophobic", "white_orange_red", False, "HIC: exposed hydrophobicity", "surface hydrophobicity (TAP PSH; Raybould 2019)"),
    "Tm2": ("aromatic", "white_yellow_orange", False, "Tm2: exposed aromatic residues", "aromatic/packing relates to melting (weak surface signal)"),
    "Titer": ("chargeabs", "white_purple", False, "Titer: net surface charge magnitude", "charge/pI affects expression and aggregation"),
    "AC-SINS_pH7.4": ("charge", "blue_white_red", True, "AC-SINS: signed exposed charge", "charge asymmetry drives self-association"),
    "PR_CHO": ("poscharge", "white_red", False, "PR-CHO: exposed positive charge", "positive CDR charge (Boughter 2020)"),
}
AROM = {"PHE", "TRP", "TYR"}


def one(resname):
    try:
        return index_to_one(three_to_index(resname))
    except Exception:
        return "X"


def residue_table(pdb):
    res_areas = freesasa.calc(freesasa.Structure(str(pdb))).residueAreas()
    struct = PARSER.get_structure("x", str(pdb))[0]
    rows = []
    for chain in struct[0] if False else struct:
        for res in chain:
            if "CA" not in res:
                continue
            try:
                sa = res_areas[chain.id][str(res.id[1])].total
            except KeyError:
                sa = 0.0
            rows.append(dict(chain=chain.id, resnum=res.id[1], resname=res.resname,
                             aa=one(res.resname), sasa=sa))
    return struct, pd.DataFrame(rows)


def attribution(df, mode):
    if mode == "hydrophobic":
        return df["sasa"] * df["resname"].map(lambda r: max(KD.get(r, 0.0), 0.0))
    if mode == "aromatic":
        return df["sasa"] * df["resname"].map(lambda r: 1.0 if r in AROM else 0.0)
    if mode == "charge":
        return df["sasa"] * df["resname"].map(lambda r: CHG.get(r, 0.0))
    if mode == "chargeabs":
        return df["sasa"] * df["resname"].map(lambda r: abs(CHG.get(r, 0.0)))
    return df["sasa"] * df["resname"].map(lambda r: max(CHG.get(r, 0.0), 0.0))  # poscharge


def cdr_resnums(df):
    """Return set of (chain, resnum) CDR residues + a chain-aware PyMOL selection string."""
    cdr = set(); by_chain = {}
    for cid, sub in df.groupby("chain"):
        seq = "".join(sub["aa"]); reg = number_chain(seq)
        if reg is None:
            continue
        resn = sub["resnum"].tolist()
        for frag in (reg.cdr1, reg.cdr2, reg.cdr3):
            if not frag:
                continue
            i = seq.find(frag)
            if i >= 0:
                for rn in resn[i:i + len(frag)]:
                    cdr.add((cid, rn)); by_chain.setdefault(cid, []).append(rn)
    sel = " or ".join(f"(chain {cid} and resi {'+'.join(map(str, rns))})" for cid, rns in by_chain.items())
    return cdr, sel


def write_bfactor(struct, df, values, path):
    val = dict(zip(zip(df["chain"], df["resnum"]), values))
    for chain in struct:
        for res in chain:
            v = val.get((chain.id, res.id[1]), 0.0)
            for atom in res:
                atom.set_bfactor(float(v))
    io = PDBIO(); io.set_structure(struct); io.save(str(path))


def render(pdb, prop, cmap, diverging, vmax, cdr_sel, view):
    import pymol2
    png = OUT / f"{pdb.stem}_{prop.replace('/', '_')}_{view}.png"
    with pymol2.PyMOL() as p:
        c = p.cmd
        c.load(str(pdb), "ab"); c.bg_color("white"); c.set("ray_opaque_background", 0)
        lo, hi = (-vmax, vmax) if diverging else (0, vmax)
        if view == "surface":
            c.hide("everything"); c.show("surface")
        else:  # cartoon with CDR sticks
            c.hide("everything"); c.show("cartoon")
            if cdr_sel:
                c.select("cdrs", cdr_sel); c.show("sticks", "cdrs"); c.set("stick_radius", 0.3)
        c.spectrum("b", cmap, "ab", minimum=lo, maximum=hi)
        c.orient("ab"); c.set("surface_quality", 1)
        c.ray(1000, 800); c.png(str(png), dpi=140)
    return png


def main():
    from abdev.data.dataset import load_gdpa1
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    names = set(ds.df[ds.id_col].astype(str))
    pick = next((n for n in ["trastuzumab", "adalimumab", "rituximab", "brazikumab"]
                 if n in names and (STRUCT / f"{n}.pdb").exists()), None)
    if pick is None:
        pick = next(n for n in names if (STRUCT / f"{n}.pdb").exists())
    pdb = STRUCT / f"{pick}.pdb"
    print(f"Representative antibody: {pick}", flush=True)

    struct, df = residue_table(pdb)
    cdr, cdr_sel = cdr_resnums(df)

    enrich, panels = [], []
    for prop, (mode, cmap, div, label, gt) in PROPS.items():
        vals = attribution(df, mode).to_numpy(float)
        vmax = float(np.percentile(np.abs(vals), 98)) or 1.0
        bpath = OUT / f"{pick}_{prop.replace('/', '_')}.pdb"
        write_bfactor(struct, df, vals, bpath)
        surf = render(bpath, prop, cmap, div, vmax, cdr_sel, "surface")
        cart = render(bpath, prop, cmap, div, vmax, cdr_sel, "cartoon")
        panels.append((prop, label, gt, surf, cart))
        # ground truth: CDR enrichment of the top-k attribution HOTSPOT residues (chain-aware)
        av = np.abs(vals); k = max(15, int(0.15 * len(df)))
        top_idx = np.argsort(av)[::-1][:k]
        hot_keys = [(df["chain"].iloc[i], int(df["resnum"].iloc[i])) for i in top_idx]
        frac_cdr = float(np.mean([kk in cdr for kk in hot_keys]))
        base = len(cdr) / len(df)
        enrich.append(dict(property=prop, cdr_fraction_of_hotspots=round(float(frac_cdr), 3),
                           cdr_fraction_overall=round(float(base), 3),
                           enrichment=round(float(frac_cdr / (base + 1e-9)), 2)))
    ens = pd.DataFrame(enrich); ens.to_csv(OUT / "cdr_enrichment.csv", index=False)
    print(ens.to_string(index=False), flush=True)

    # assemble grid: rows = properties, cols = [surface, cartoon]
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.4 * n))
    for i, (prop, label, gt, surf, cart) in enumerate(panels):
        for j, (img, vtag) in enumerate([(surf, "surface"), (cart, "cartoon + CDRs")]):
            ax = axes[i, j]; ax.imshow(mpimg.imread(img)); ax.axis("off")
            if j == 0:
                ax.set_title(f"{label}\n(GT: {gt})", fontsize=11, loc="left")
            else:
                e = ens.iloc[i]
                ax.set_title(f"{vtag}  |  CDR hotspot enrichment {e['enrichment']}×", fontsize=11)
    fig.suptitle(f"Supplementary: structure-mapped explainability across properties ({pick})", fontsize=13, y=1.001)
    fig.tight_layout(); fig.savefig(ROOT / "results/figures/figS1_explainability_allprops.png", dpi=150, bbox_inches="tight")
    print("Saved results/figures/figS1_explainability_allprops.png", flush=True)


if __name__ == "__main__":
    main()
