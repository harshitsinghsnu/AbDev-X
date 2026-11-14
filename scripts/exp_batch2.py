"""Experiments (4) extra generalization splits, (11) TAP-flag baseline, (12) leakage audit."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
from difflib import SequenceMatcher
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE
from abdev.models.fusion import AbDevFusion, DEFAULT_BASES
from abdev.features.numbering import number_chain
ASSAYS = GE.ASSAYS
FIG = ROOT / "results/figures"


def cv_arr(ds, B, y, folds, bases=DEFAULT_BASES):
    per = []
    for f in sorted(set(folds.tolist())):
        tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        Btr = {k: v[tr] for k, v in B.items()}; Bte = {k: v[te] for k, v in B.items()}
        per.append(spearmanr(y[te], AbDevFusion(bases=bases).fit(Btr, y[tr]).predict(Bte)).statistic)
    return float(np.mean(per))


def main():
    ds, g3, B1, B3 = GE.build_blocks()
    B1 = {k: np.nan_to_num(v) for k, v in B1.items()}; B3 = {k: np.nan_to_num(v) for k, v in B3.items()}
    Y3 = {a: g3[GE.LABEL_MAP[a]].to_numpy(float) for a in ASSAYS}
    n = len(ds.df)

    # ---- (4) extra generalization splits ----
    h3len = np.array([len(number_chain(s).cdr3) if number_chain(s) else 0 for s in ds.heavy])
    h3_fold = pd.qcut(h3len + np.random.RandomState(0).normal(0, 1e-6, n), 5, labels=False)  # length-binned
    germ_fold = KMeans(5, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(B1["esm2"]))
    splits = {"random": ds.folds("random"), "cluster": ds.folds("cluster"),
              "cluster_isotype": ds.folds("cluster_isotype"),
              "CDR-H3-length": h3_fold, "germline(ESM-kmeans)": germ_fold}
    rows = []
    for sname, folds in splits.items():
        folds = np.asarray(folds)
        means = [cv_arr(ds, B1, ds.y(a), folds) for a in ASSAYS]
        rows.append(dict(split=sname, **{a: round(m, 3) for a, m in zip(ASSAYS, means)}, MEAN=round(float(np.mean(means)), 3)))
    # held-out MEAN for reference
    ho = []
    for a in ASSAYS:
        y1 = ds.y(a); m = ~np.isnan(y1); ok = ~np.isnan(Y3[a])
        p = AbDevFusion().fit({k: v[m] for k, v in B1.items()}, y1[m]).predict(B3)
        ho.append(spearmanr(Y3[a][ok], p[ok]).statistic)
    sp = pd.DataFrame(rows); sp.to_csv(ROOT / "results/exp_splits.csv", index=False)
    print("=== (4) AbDev-X MEAN CV across split strategies (held-out MEAN = %.3f) ===" % np.mean(ho))
    print(sp[["split", "MEAN"]].to_string(index=False))
    # correlation of split-CV MEAN with held-out MEAN (per assay, over splits) -> which split best predicts held-out
    ho_map = dict(zip(ASSAYS, ho))
    corrs = {}
    for sname, folds in splits.items():
        folds = np.asarray(folds); cvv = [cv_arr(ds, B1, ds.y(a), folds) for a in ASSAYS]
        corrs[sname] = round(float(np.mean([abs(cvv[i] - ho_map[a]) for i, a in enumerate(ASSAYS)])), 3)
    print("Mean |CV - heldout| per split (lower = better held-out proxy):", corrs)

    # ---- (11) TAP-flag baseline ----
    tap = pd.read_csv(ROOT / "external/abdev-benchmark/data/features/processed_features/GDPa1/TAP.csv")
    tapc = [c for c in tap.select_dtypes("number").columns]
    Xtap = np.nan_to_num(pd.DataFrame({"antibody_name": ds.df[ds.id_col].values}).merge(tap, on="antibody_name", how="left")[tapc].to_numpy(float))
    taph = pd.read_csv(ROOT / "external/abdev-benchmark/data/features/processed_features/heldout_test/TAP.csv")
    common = [c for c in tapc if c in taph.columns]
    idx = [tapc.index(c) for c in common]
    Xtap3 = np.nan_to_num(pd.DataFrame({"antibody_name": g3["antibody_name"].values}).merge(taph, on="antibody_name", how="left")[common].to_numpy(float))
    folds = ds.folds("cluster_isotype"); trow = []
    for a in ASSAYS:
        y = ds.y(a); per = []
        for f in sorted(set(folds.tolist())):
            tr = np.where((folds != f) & ~np.isnan(y))[0]; te = np.where((folds == f) & ~np.isnan(y))[0]
            if len(te) < 3:
                continue
            mdl = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))]).fit(Xtap[tr], y[tr])
            per.append(spearmanr(y[te], mdl.predict(Xtap[te])).statistic)
        m = ~np.isnan(y); ok = ~np.isnan(Y3[a])
        mdl = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-2, 4, 12)))]).fit(Xtap[m][:, idx], y[m])
        hov = spearmanr(Y3[a][ok], mdl.predict(Xtap3)[ok]).statistic
        trow.append(dict(assay=a, TAP_CV=round(float(np.mean(per)), 3), TAP_heldout=round(float(hov), 3)))
    tapdf = pd.DataFrame(trow); tapdf.to_csv(ROOT / "results/exp_tap_baseline.csv", index=False)
    print("\n=== (11) TAP-flag ridge baseline ===")
    print(tapdf.to_string(index=False))

    # ---- (12) leakage audit: GDPa3 vs GDPa1 max sequence identity ----
    def ident(a, b): return SequenceMatcher(None, a, b).ratio()
    g1h = ds.heavy.tolist(); g1l = ds.light.tolist()
    maxid = []
    for vh, vl in zip(g3["vh"], g3["vl"]):
        best = max(0.5 * (ident(vh, h) + ident(vl, l)) for h, l in zip(g1h, g1l))
        maxid.append(best)
    maxid = np.array(maxid)
    pd.DataFrame({"gdpa3_max_identity_to_gdpa1": maxid}).to_csv(ROOT / "results/exp_leakage.csv", index=False)
    print("\n=== (12) GDPa3->GDPa1 max VH+VL identity: mean %.3f, max %.3f, >0.95: %d/80 ==="
          % (maxid.mean(), maxid.max(), int((maxid > 0.95).sum())))
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(maxid, bins=20, color=plt.cm.viridis(0.4), edgecolor="k")
    ax.axvline(0.95, color="red", ls="--", label="95% identity"); ax.set_xlabel("Max VH+VL identity of a GDPa3 antibody to any GDPa1 antibody")
    ax.set_ylabel("count"); ax.set_title("Leakage audit: GDPa3 vs GDPa1 sequence identity"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "figS4_leakage.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
