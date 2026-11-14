"""REAL held-out evaluation: train AbDev-X on GDPa1, predict GDPa3, vs competition held-out SOTA.

GDPa3 (80 IgG1s, released with labels) is the actual out-of-distribution test set. We train on all
of GDPa1 and predict GDPa3 once (no folds) -> Spearman per assay, compared to the top competition
held-out test numbers (Ginkgo mAbs 2026, Fig 3b/c). Feature blocks are constructed identically for
both sets; the structural block uses the TAP/MOE/Aggrescan3D/AntiFold features available for BOTH.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix
from abdev.models.fusion import AbDevFusion

ASSAYS = ["HIC", "Tm2", "AC-SINS_pH7.4", "PR_CHO", "Titer"]
# Ginkgo competition top scores on the held-out test set (mAbs 2026, Fig 3b/c).
COMP_HELDOUT = {"HIC": 0.708, "Tm2": 0.392, "AC-SINS_pH7.4": 0.356, "PR_CHO": 0.337, "Titer": 0.310}
GD3 = ROOT / "data/processed/gdpa3_table.csv"
BENCH = ROOT / "external/abdev-benchmark/data/features/processed_features"
BIO_FILES = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv"]  # available for BOTH
ISO_COLS = ["hc_subtype_IgG1", "hc_subtype_IgG2", "hc_subtype_IgG4", "lc_subtype_Kappa", "lc_subtype_Lambda"]


def isotype_block(df):
    d = pd.get_dummies(df[["hc_subtype", "lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS, fill_value=0).to_numpy(np.float32)


def bio_block(feat_dir, names, cols=None):
    base = pd.DataFrame({"antibody_name": list(names)})
    for f in BIO_FILES:
        p = Path(feat_dir) / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        num = d.select_dtypes("number").copy(); num["antibody_name"] = d["antibody_name"].values
        num = num.rename(columns={c: f"{f[:-4]}::{c}" for c in num.columns if c != "antibody_name"})
        base = base.merge(num, on="antibody_name", how="left")
    X = base.drop(columns=["antibody_name"])
    if cols is None:
        cols = list(X.columns)
    X = X.reindex(columns=cols, fill_value=0.0)
    return np.nan_to_num(X.to_numpy(float)), cols


LABEL_MAP = {"HIC": "hic_rt_avg", "Tm2": "tm2_nanodsf_avg", "AC-SINS_pH7.4": "acsins_dLmax_ph7.4_avg",
             "PR_CHO": "polyreactivity_prscore_cho_avg", "Titer": "titer_avg"}


def build_blocks():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(GD3)
    esm1 = next(np.load(p)["emb"] for p in (ROOT / "data/processed").glob("esm2_*.npz")
                if np.load(p)["emb"].shape == (len(ds.df), 2560))
    abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    desc1 = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
    iso1 = isotype_block(ds.df)
    bio1, cols = bio_block(BENCH / "GDPa1", ds.df[ds.id_col])
    B1 = {"esm2": np.nan_to_num(esm1), "descriptors": desc1, "isotype": iso1, "bio": bio1, "ablang2": np.nan_to_num(abl1)}
    esm3 = np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"]
    abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]
    desc3 = np.nan_to_num(build_descriptor_matrix(g3["vh"], g3["vl"]).to_numpy(float))
    iso3 = isotype_block(g3)
    bio3, _ = bio_block(BENCH / "heldout_test", g3["antibody_name"], cols=cols)
    B3 = {"esm2": np.nan_to_num(esm3), "descriptors": desc3, "isotype": iso3, "bio": bio3, "ablang2": np.nan_to_num(abl3)}
    return ds, g3, B1, B3


def main():
    ds, g3, B1, B3 = build_blocks()

    rows = []
    for a in ASSAYS:
        y1 = ds.y(a); m = ~np.isnan(y1)
        B1m = {k: v[m] for k, v in B1.items()}
        model = AbDevFusion().fit(B1m, y1[m])
        pred = model.predict(B3)
        y3 = g3[LABEL_MAP[a]].to_numpy(float)
        ok = ~np.isnan(y3)
        rho = float(spearmanr(y3[ok], pred[ok]).statistic)
        rows.append(dict(assay=a, abdevx_heldout=round(rho, 3), competition_top=COMP_HELDOUT[a],
                         beat=bool(rho > COMP_HELDOUT[a])))
    out = pd.DataFrame(rows)
    out.loc[len(out)] = ["MEAN", round(out["abdevx_heldout"].mean(), 3),
                         round(np.mean(list(COMP_HELDOUT.values())), 3), None]
    print("\n=== GDPa3 HELD-OUT: AbDev-X (train GDPa1) vs competition top held-out ===")
    print(out.to_string(index=False))
    print(f"\nBeats competition held-out on {int(pd.DataFrame(rows)['beat'].sum())}/5 assays.")
    out.to_csv(ROOT / "results/gdpa3_heldout.csv", index=False)


if __name__ == "__main__":
    main()
