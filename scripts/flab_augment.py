"""FLAb2 -> GDPa1 augmentation experiment (large-data training for weak assays).

For a GDPa1 assay, augment the GDPa1 training folds with same-assay FLAb2 studies (per-study
z-scored to absorb protocol scale differences), then predict the GDPa1 held-out fold. FLAb2
antibodies whose VH+VL exactly match any GDPa1 antibody are REMOVED (no leakage). Compares
GDPa1-only vs GDPa1+FLAb2 on the official generalization splits. Sequence-derived features only
(descriptors [+ ESM-mean if cached for FLAb2]) since structural/isotype features need external tools.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings, re
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abdev.data.dataset import load_gdpa1
from abdev.data.splits import fold_iter
from abdev.features.descriptors import build_descriptor_matrix

FLAB = ROOT / "external/FLAb/data"
# GDPa1 assay -> list of (relative_path, value_column) FLAb studies with the SAME assay.
FLAB_MAP = {
    "Titer": [("expression/jain2017biophysical_HEK.csv", "HEK Titer (mg/L)"),
              ("expression/garbinski2023_exp.csv", "Expression (µg/mL)")],
    "Tm2": [("thermostability/jain2017biophysical_Tm.csv", "Fab Tm by DSF (°C)"),
            ("thermostability/jain2024assessment_Tm.csv", "Fab Tm by DSF (°C)"),
            ("thermostability/hutchinson2023enhancement_top200tm1_igg.csv", "Tm1 (C) Thermostability (IgG)")],
    "HIC": [("aggregation/jain2017biophyscial_HICRT.csv", None),
            ("aggregation/jain2024assessment_HIC.csv", "HIC RT (min)")],
    "AC-SINS_pH7.4": [("aggregation/jain2017biophysical_ACSINS.csv", None),
                       ("aggregation/jain2024assessment_ACSINS.csv", "AC-SINS ∆λmax (nm)")],
    "PR_CHO": [("polyreactivity/jain2017biophysical_PSR.csv", None),
                ("polyreactivity/jain2024assessment_PSR.csv", None)],
}
SPLITS = ["cluster", "cluster_isotype"]


def norm_seq(s): return re.sub(r"[^A-Z]", "", str(s).upper())


def load_flab(assay, gdpa_keys):
    frames = []
    for rel, valcol in FLAB_MAP.get(assay, []):
        p = FLAB / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "heavy" not in df.columns or "light" not in df.columns:
            continue
        if valcol is None:  # value = 3rd column (after heavy,light) by convention
            cand = [c for c in df.columns if c not in ("heavy", "light", "fitness")]
            valcol = cand[0] if cand else None
        if valcol not in df.columns:
            continue
        sub = df[["heavy", "light", valcol]].dropna()
        sub.columns = ["heavy", "light", "value"]
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.dropna()
        # per-study z-score (absorb protocol scale)
        sub["z"] = (sub["value"] - sub["value"].mean()) / (sub["value"].std() + 1e-9)
        sub["key"] = sub["heavy"].map(norm_seq) + "|" + sub["light"].map(norm_seq)
        sub = sub[~sub["key"].isin(gdpa_keys)]  # DEDUPE vs GDPa1 (no leakage)
        frames.append(sub[["heavy", "light", "z"]])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def ridge(): return Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))])


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    gkeys = set((ds.heavy.map(norm_seq) + "|" + ds.light.map(norm_seq)).tolist())
    Xg = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))

    rows = []
    for assay in ["Titer", "Tm2", "HIC", "AC-SINS_pH7.4", "PR_CHO"]:
        flab = load_flab(assay, gkeys)
        n_aug = 0 if flab is None else len(flab)
        Xf = zf = None
        if n_aug:
            Xf = np.nan_to_num(build_descriptor_matrix(flab["heavy"], flab["light"]).to_numpy(float))
            zf = flab["z"].to_numpy(float)
        for split in SPLITS:
            y = ds.y(assay)
            for aug in ([False, True] if n_aug else [False]):
                pooled = np.full(len(y), np.nan)
                for _, tr, te in fold_iter(ds, assay, split):
                    ytr = y[tr]; mu, sd = np.nanmean(ytr), np.nanstd(ytr) + 1e-9
                    ytr_z = (ytr - mu) / sd
                    Xtr, Ytr = Xg[tr], ytr_z
                    if aug:
                        Xtr = np.vstack([Xtr, Xf]); Ytr = np.concatenate([Ytr, zf])
                    m = ridge().fit(Xtr, Ytr)
                    pooled[te] = m.predict(Xg[te])
                mm = ~np.isnan(y) & ~np.isnan(pooled)
                rho = float(spearmanr(y[mm], pooled[mm]).statistic)
                rows.append(dict(assay=assay, split=split, aug=aug, n_flab=n_aug, spearman=round(rho, 3)))
        print(f"{assay:14s} n_flab={n_aug}", flush=True)
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["assay", "split"], columns="aug", values="spearman")
    piv.columns = ["GDPa1_only", "GDPa1+FLAb2"]
    piv["delta"] = piv["GDPa1+FLAb2"] - piv["GDPa1_only"]
    print("\n=== FLAb2 augmentation (descriptors, dedup, per-study z) ===")
    print(piv.round(3).to_string())
    out.to_csv(ROOT / "results/flab_augment.csv", index=False)


if __name__ == "__main__":
    main()
