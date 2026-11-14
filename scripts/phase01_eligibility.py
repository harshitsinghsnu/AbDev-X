"""Phase 0.1: Eligibility-constrained block selection + T1–T8 ablation.

For thermal (Tm1, Tm2, Tonset) and SEC, isotype is excluded from block selection.
Runs T1–T8 for all 9 assays:
  T1  ESM-2 only
  T2  AbLang2 only
  T3  Descriptors only
  T4  Bio only
  T5  ESM-2 + AbLang2
  T6  ESM-2 + AbLang2 + Bio + Desc
  T7  Best eligible block (no isotype for thermal/SEC)
  T8  T7 + isotype as explicit covariate

Also runs isotype permutation test (n=200 shuffles) to measure
how much performance depends on isotype labels.
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

ASSAY_MAP = {
    "HIC":     ("HIC",          "hic_rt_avg"),
    "Tm2":     ("Tm2",          "tm2_nanodsf_avg"),
    "Tm1":     ("Tm1",          "tm1_nanodsf_avg"),
    "Tonset":  ("Tonset",       "tonset_nanodsf_avg"),
    "SEC":     ("SEC %Monomer", "sec_%monomer_avg"),
    "AC-SINS": ("AC-SINS_pH7.4","acsins_dLmax_ph7.4_avg"),
    "PR_CHO":  ("PR_CHO",       "polyreactivity_prscore_cho_avg"),
    "SMAC":    ("SMAC",         "smac_rt_avg"),
    "HAC":     ("HAC",          "hac_rt_avg"),
}

# Isotype ineligible for these assays (Phase 0.1 correction)
ISO_INELIGIBLE = {"Tm1", "Tm2", "Tonset", "SEC"}

BIO_FILES = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv"]
ISO_COLS  = ["hc_subtype_IgG1", "hc_subtype_IgG2", "hc_subtype_IgG4",
             "lc_subtype_Kappa", "lc_subtype_Lambda"]
N_PERM = 200


def _iso(df):
    d = pd.get_dummies(df[["hc_subtype", "lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS, fill_value=0).to_numpy(float)


def _bio(feat_dir, names, cols=None):
    base = pd.DataFrame({"antibody_name": list(names)})
    for f in BIO_FILES:
        p = Path(feat_dir) / f
        if not p.exists(): continue
        d = pd.read_csv(p)
        num = d.select_dtypes("number").copy()
        num["antibody_name"] = d["antibody_name"].values
        num = num.rename(columns={c: f"{f[:-4]}::{c}" for c in num.columns
                                  if c != "antibody_name"})
        base = base.merge(num, on="antibody_name", how="left")
    X = base.drop(columns=["antibody_name"])
    if cols is None: cols = list(X.columns)
    return np.nan_to_num(X.reindex(columns=cols, fill_value=0.0).to_numpy(float)), cols


def _head(X):
    steps = [("s", StandardScaler())]
    if X.shape[1] > 100:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)


def ood_cv(X, y, folds, fold_ids):
    m = ~np.isnan(y)
    scores = []
    for f in fold_ids:
        tr = np.where((folds != f) & m)[0]
        te = np.where((folds == f) & m)[0]
        if len(te) < 3 or len(tr) < 10: continue
        pipe = _head(X)
        pipe.fit(X[tr], y[tr])
        scores.append(spearmanr(y[te], pipe.predict(X[te])).statistic)
    return float(np.nanmean(scores)) if scores else -1.0


def held_out(X_tr, y_tr, X_te, y_te):
    m = ~np.isnan(y_tr); ok = ~np.isnan(y_te)
    if m.sum() < 10 or ok.sum() < 5: return np.nan
    pipe = _head(X_tr)
    pipe.fit(X_tr[m], y_tr[m])
    pred = pipe.predict(X_te)
    return float(spearmanr(y_te[ok], pred[ok]).statistic)


def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds    = ds.folds("cluster_isotype")
    fold_ids = sorted(set(folds.tolist()))

    esm_files = list((ROOT / "data/processed").glob("esm2_*.npz"))
    esm1 = None
    for ef in esm_files:
        arr = np.load(ef)["emb"]
        if arr.shape == (len(ds.df), 2560):
            esm1 = arr; break
    abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    esm3 = np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"]
    abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]

    desc1 = np.nan_to_num(build_descriptor_matrix(ds.heavy, ds.light).to_numpy(float))
    desc3 = np.nan_to_num(build_descriptor_matrix(g3["vh"], g3["vl"]).to_numpy(float))
    bio1, bio_cols = _bio(
        ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1",
        ds.df["antibody_name"])
    bio3, _ = _bio(
        ROOT / "external/abdev-benchmark/data/features/processed_features/heldout_test",
        g3["antibody_name"], cols=bio_cols)
    iso1 = _iso(ds.df); iso3 = _iso(g3)
    esm1n = np.nan_to_num(esm1); esm3n = np.nan_to_num(esm3)
    abl1n = np.nan_to_num(abl1); abl3n = np.nan_to_num(abl3)

    single_blocks = {"esm2": (esm1n, esm3n), "ablang2": (abl1n, abl3n),
                     "isotype": (iso1, iso3), "desc": (desc1, desc3), "bio": (bio1, bio3)}

    rows = []
    perm_rows = []

    print("\n" + "="*110)
    print(f"{'Assay':10s}  {'T1-esm2':7s}  {'T2-abl2':7s}  {'T3-desc':7s}  {'T4-bio':7s}  "
          f"{'T5-e+a':7s}  {'T6-all':7s}  {'T7-best':7s}  {'T8-+iso':7s}  "
          f"{'ISO-base':8s}  {'Perm-p':8s}  Note")
    print("="*110)

    for assay, (g1_col, g3_col) in ASSAY_MAP.items():
        y1 = ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df), np.nan)
        y3 = g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(len(g3), np.nan)

        iso_eligible = assay not in ISO_INELIGIBLE

        # T1–T4: single blocks
        t1 = held_out(esm1n, y1, esm3n, y3)
        t2 = held_out(abl1n, y1, abl3n, y3)
        t3 = held_out(desc1, y1, desc3, y3)
        t4 = held_out(bio1,  y1, bio3,  y3)

        # T5: ESM-2 + AbLang2
        X5_tr = np.hstack([esm1n, abl1n]); X5_te = np.hstack([esm3n, abl3n])
        t5 = held_out(X5_tr, y1, X5_te, y3)

        # T6: everything except isotype
        X6_tr = np.hstack([esm1n, abl1n, bio1, desc1])
        X6_te = np.hstack([esm3n, abl3n, bio3, desc3])
        t6 = held_out(X6_tr, y1, X6_te, y3)

        # T7: best eligible block (OOD-CV, isotype excluded for thermal/SEC)
        eligible = {b: (Xtr, Xte) for b, (Xtr, Xte) in single_blocks.items()
                    if b != "isotype" or iso_eligible}
        cv_scores = {b: ood_cv(Xtr, y1, folds, fold_ids) for b, (Xtr, Xte) in eligible.items()}
        best_blk  = max(cv_scores, key=lambda k: cv_scores[k])
        t7 = held_out(eligible[best_blk][0], y1, eligible[best_blk][1], y3)

        # T8: T7 + isotype as appended covariate
        X8_tr = np.hstack([eligible[best_blk][0], iso1])
        X8_te = np.hstack([eligible[best_blk][1], iso3])
        t8 = held_out(X8_tr, y1, X8_te, y3)

        # Isotype-only baseline
        t_iso_base = held_out(iso1, y1, iso3, y3)

        # Isotype permutation test on OOD-CV score of isotype block
        rng = np.random.default_rng(42)
        m1 = ~np.isnan(y1)
        actual_cv_iso = ood_cv(iso1, y1, folds, fold_ids)
        perm_scores = []
        for _ in range(N_PERM):
            iso_shuf = iso1[rng.permutation(len(iso1))]
            perm_scores.append(ood_cv(iso_shuf, y1, folds, fold_ids))
        perm_p = float(np.mean(np.array(perm_scores) >= actual_cv_iso))

        ineligible_note = "ISO-BLOCKED" if not iso_eligible else ""
        def f(v): return f"{v:+7.3f}" if not np.isnan(v) else "    nan"
        print(f"{assay:10s}  {f(t1)}  {f(t2)}  {f(t3)}  {f(t4)}  "
              f"{f(t5)}  {f(t6)}  {f(t7)}[{best_blk[:3]}]  {f(t8)}  "
              f"{t_iso_base:+8.3f}  {perm_p:8.3f}  {ineligible_note}")

        rows.append({"assay": assay, "eligible_best_blk": best_blk,
                     "T1_esm2": round(t1,3) if not np.isnan(t1) else None,
                     "T2_ablang2": round(t2,3) if not np.isnan(t2) else None,
                     "T3_desc": round(t3,3) if not np.isnan(t3) else None,
                     "T4_bio": round(t4,3) if not np.isnan(t4) else None,
                     "T5_esm_abl": round(t5,3) if not np.isnan(t5) else None,
                     "T6_all_no_iso": round(t6,3) if not np.isnan(t6) else None,
                     "T7_best_eligible": round(t7,3) if not np.isnan(t7) else None,
                     "T8_T7_plus_iso": round(t8,3) if not np.isnan(t8) else None,
                     "iso_baseline": round(t_iso_base,3) if not np.isnan(t_iso_base) else None,
                     "perm_p_iso": round(perm_p,3),
                     "iso_ineligible": not iso_eligible})

    print("="*110)
    df = pd.DataFrame(rows)

    # Corrected mean: only genuine signal assays (no isotype confound)
    genuine = df[~df["iso_ineligible"]]
    confounded = df[df["iso_ineligible"]]
    print(f"\n--- Mean held-out rho (genuine assays: HIC, AC-SINS, PR_CHO, SMAC, HAC) ---")
    for col in ["T1_esm2","T2_ablang2","T3_desc","T4_bio","T5_esm_abl","T6_all_no_iso","T7_best_eligible","T8_T7_plus_iso"]:
        print(f"  {col:20s}: genuine={genuine[col].mean():.3f}  confounded={confounded[col].mean():.3f}")

    print(f"\n--- Thermal/SEC with isotype blocked (T7 = best of esm2/abl2/desc/bio) ---")
    for _, r in confounded.iterrows():
        print(f"  {r['assay']:10s}  best_blk={r['eligible_best_blk']:8s}  "
              f"T7={r['T7_best_eligible']}  T8(+iso)={r['T8_T7_plus_iso']}  iso_base={r['iso_baseline']}  "
              f"perm_p={r['perm_p_iso']:.3f}")

    out = ROOT / "results/phase01_eligibility.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
