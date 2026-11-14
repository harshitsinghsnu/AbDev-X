"""AbDev-MPO — Multi-Property Optimisation and Developability Profiling.

Stages:
  M1  AW-MoE five-output prediction
  M2  Independent assay-specific baselines
  M3  Uniform weighted multi-property score
  M4  Pareto-based ranking (non-dominated sorting)
  M5  Borda-count ranking
  M6  Geometric desirability score

Direction harmonisation:
  d_p = -1  (lower raw value = more favourable)
  HIC     : higher RT = more hydrophobic exposure   → d = -1
  AC-SINS : higher ΔLmax = more self-association    → d = -1
  PR_CHO  : higher score = more polyreactive         → d = -1
  SMAC    : higher RT = faster clearance surrogate   → d = -1
  HAC     : higher RT = more anti-chimeric response  → d = -1

All percentile transforms and thresholds fitted on GDPa1 labels only.
GDPa3 predictions evaluated against known labels for correlation check only.
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

ASSAYS = {
    "HIC":     ("HIC",           "hic_rt_avg",                     -1),
    "AC-SINS": ("AC-SINS_pH7.4", "acsins_dLmax_ph7.4_avg",        -1),
    "PR_CHO":  ("PR_CHO",        "polyreactivity_prscore_cho_avg", -1),
    "SMAC":    ("SMAC",          "smac_rt_avg",                    -1),
    "HAC":     ("HAC",           "hac_rt_avg",                     -1),
}
ASSAY_NAMES = list(ASSAYS.keys())
P = len(ASSAY_NAMES)

BIO_FILES = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv"]
ISO_COLS  = ["hc_subtype_IgG1","hc_subtype_IgG2","hc_subtype_IgG4",
             "lc_subtype_Kappa","lc_subtype_Lambda"]
N_BOOT = 1000

MGFS_MASKS = {
    "HIC":    {"inc":["hydro","gravy","aliphat","aromat","patch","sasa","SASA"],"exc":["charge","pI","deamid","glyc"]},
    "AC-SINS":{"inc":["charge","pI","ppc","pnc","patch_pos","patch_neg","dipole","electr"],"exc":["hydro","gravy"]},
    "PR_CHO": {"inc":["charge","cdr","CDR","hydro","gravy","patch"],"exc":[]},
    "SMAC":   {"inc":["sasa","SASA","patch","hydro","charge","exposed"],"exc":[]},
    "HAC":    {"inc":["sasa","SASA","patch","hydro","charge","exposed","stab"],"exc":[]},
}
MOE_EXPERTS = {
    "hydro":  ["hydro","gravy","aliphat","aromat","patch_hyd","hyd"],
    "charge": ["charge","pI","ppc","pnc","electr","dipole","patch_pos","patch_neg"],
    "stab":   ["stab","sasa","gyrat","pack","AntiFold","tm","Tm","cavity","burial"],
    "liab":   ["deamid","isomeriz","glyc","oxid","cys","nglyc","asn","asp"],
    "cdr":    ["cdr","CDR","loop","h3","l3","h1","h2","l1","l2"],
    "surface":["sasa","SASA","rSASA","exposed","surface","patch"],
    "global": ["len","mass","count","frac","total","net"],
}
MOE_ELIGIBLE = {
    "HIC":    ["hydro","surface"],
    "AC-SINS":["charge","surface"],
    "PR_CHO": ["charge","cdr","hydro"],
    "SMAC":   ["surface","hydro","charge"],
    "HAC":    ["surface","hydro","charge","stab"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _iso(df):
    d = pd.get_dummies(df[["hc_subtype","lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS, fill_value=0).to_numpy(float)

def _bio(feat_dir, names, cols=None):
    base = pd.DataFrame({"antibody_name":list(names)})
    for f in BIO_FILES:
        p = Path(feat_dir)/f
        if not p.exists(): continue
        d = pd.read_csv(p)
        num = d.select_dtypes("number").copy(); num["antibody_name"] = d["antibody_name"].values
        num = num.rename(columns={c:f"{f[:-4]}::{c}" for c in num.columns if c!="antibody_name"})
        base = base.merge(num, on="antibody_name", how="left")
    X = base.drop(columns=["antibody_name"])
    if cols is None: cols = list(X.columns)
    return np.nan_to_num(X.reindex(columns=cols,fill_value=0).to_numpy(float)), cols

def _head(X):
    steps=[("s",StandardScaler())]
    if X.shape[1]>100: steps.append(("p",PCA(0.9,random_state=0,svd_solver="full")))
    steps.append(("r",RidgeCV(alphas=np.logspace(-2,4,12))))
    return Pipeline(steps)

def _mask_cols(all_cols, inc_kws, exc_kws=()):
    inc=[k.lower() for k in inc_kws]; exc=[k.lower() for k in exc_kws]
    return np.array([(any(k in c.lower() for k in inc) and
                      not any(k in c.lower() for k in exc)) for c in all_cols])

def ood_cv_rho(X, y, folds, fold_ids):
    m=~np.isnan(y); scores=[]
    for f in fold_ids:
        tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
        if len(te)<3 or len(tr)<10: continue
        pipe=_head(X); pipe.fit(X[tr],y[tr])
        scores.append(spearmanr(y[te],pipe.predict(X[te])).statistic)
    return float(np.nanmean(scores)) if scores else -1.0


# ── AW-MoE specialist predictors ─────────────────────────────────────────────

def predict_e1(Xblocks_tr, Xblocks_te, y, folds, fold_ids):
    cv = {b: ood_cv_rho(X, y, folds, fold_ids) for b, X in Xblocks_tr.items()}
    bb = max(cv, key=lambda k: cv[k])
    m = ~np.isnan(y); pipe = _head(Xblocks_tr[bb]); pipe.fit(Xblocks_tr[bb][m], y[m])
    return pipe.predict(Xblocks_te[bb]), bb

def predict_mgfs(Xdb1, Xdb3, all_cols, assay, y):
    cfg = MGFS_MASKS.get(assay,{"inc":[],"exc":[]})
    mask = _mask_cols(all_cols, cfg["inc"], cfg["exc"])
    if mask.sum()<5: mask = np.ones(len(all_cols),bool)
    m=~np.isnan(y); pipe=_head(Xdb1[:,mask]); pipe.fit(Xdb1[m][:,mask], y[m])
    return pipe.predict(Xdb3[:,mask])

def predict_dstack(Xplm1, Xplm3, Xstr1, Xstr3, y, folds, fold_ids):
    m=~np.isnan(y); inner_folds=folds; inner_ids=fold_ids
    oof1=np.zeros(len(y)); oof2=np.zeros(len(y))
    for f in inner_ids:
        tr=np.where((inner_folds!=f)&m)[0]; te=np.where((inner_folds==f)&m)[0]
        if len(te)<3 or len(tr)<8: continue
        p1=_head(Xplm1); p1.fit(Xplm1[tr],y[tr]); oof1[te]=p1.predict(Xplm1[te])
        p2=_head(Xstr1); p2.fit(Xstr1[tr],y[tr]); oof2[te]=p2.predict(Xstr1[te])
    meta=Ridge(alpha=1.0); meta.fit(np.column_stack([oof1,oof2])[m],y[m])
    p1f=_head(Xplm1); p1f.fit(Xplm1[m],y[m]); p2f=_head(Xstr1); p2f.fit(Xstr1[m],y[m])
    return meta.predict(np.column_stack([p1f.predict(Xplm3), p2f.predict(Xstr3)]))

def predict_fmoe(Xdb1, Xdb3, all_cols, assay, y, folds, fold_ids):
    eligible=MOE_ELIGIBLE.get(assay,list(MOE_EXPERTS.keys()))
    m=~np.isnan(y); inner_folds=folds; inner_ids=fold_ids; tau=0.1
    pipes=[]; cvs=[]; masks_used=[]
    for exp in eligible:
        kws=MOE_EXPERTS.get(exp,[]); msk=_mask_cols(all_cols,kws)
        if msk.sum()<3: continue
        Xe=Xdb1[:,msk]; scores=[]
        for f in inner_ids:
            tr=np.where((inner_folds!=f)&m)[0]; te=np.where((inner_folds==f)&m)[0]
            if len(te)<3 or len(tr)<8: continue
            pipe=_head(Xe); pipe.fit(Xe[tr],y[tr])
            scores.append(spearmanr(y[te],pipe.predict(Xe[te])).statistic)
        cv=float(np.nanmean(scores)) if scores else -1.0
        pf=_head(Xe); pf.fit(Xe[m],y[m])
        pipes.append(pf); cvs.append(max(cv,0.0)); masks_used.append(msk)
    if not pipes: return predict_mgfs(Xdb1,Xdb3,all_cols,assay,y)
    ws=np.exp(np.array(cvs)/tau); ws/=ws.sum()
    te_preds=np.column_stack([pipe.predict(Xdb3[:,msk]) for pipe,msk in zip(pipes,masks_used)])
    return te_preds @ ws


# ── percentile transform (fitted on training labels only) ────────────────────

def percentile_rank_transform(y_train, y_test, direction):
    """Map raw predictions to [0,1] percentiles relative to training distribution.
    direction=-1: lower raw = more favourable, so we flip before percentile ranking.
    Returns q in [0,1] where 1 = most favourable predicted position.
    """
    z_train = direction * y_train  # flip so higher = better
    z_test  = direction * y_test
    m = ~np.isnan(z_train)
    ref = np.sort(z_train[m])
    # Assign percentile rank: fraction of training points below this value
    q = np.searchsorted(ref, z_test, side='right') / len(ref)
    return np.clip(q, 0, 1)


# ── Pareto dominance ─────────────────────────────────────────────────────────

def pareto_front(Q):
    """Non-dominated sorting. Returns front assignment (0=best) for each row."""
    n = len(Q); dominated_by = [set() for _ in range(n)]
    dom_count = np.zeros(n, int); fronts = np.full(n, -1, int)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if np.all(Q[j] >= Q[i]) and np.any(Q[j] > Q[i]):
                dominated_by[i].add(j); dom_count[i] += 1
    curr_front = 0; current = [i for i in range(n) if dom_count[i]==0]
    while current:
        for i in current: fronts[i] = curr_front
        next_front = []
        for i in current:
            for j in range(n):
                if fronts[j]==-1 and i in dominated_by[j]:
                    dom_count[j] -= 1
                    if dom_count[j]==0: next_front.append(j)
        current = next_front; curr_front += 1
    return fronts


# ── Borda ranking ─────────────────────────────────────────────────────────────

def borda_score(Q):
    """Mean percentile rank across properties (higher = better overall)."""
    return Q.mean(axis=1)


# ── desirability (geometric mean) ─────────────────────────────────────────────

def desirability_score(Q, thresholds_lo=None):
    """Geometric mean desirability. Each D_p = q_ip (already in [0,1]).
    Optional thresholds_lo: minimum acceptable q per property.
    Below threshold, desirability for that property = 0.
    """
    D = Q.copy()
    if thresholds_lo is not None:
        for p, lo in enumerate(thresholds_lo):
            D[:, p] = np.where(Q[:, p] >= lo, Q[:, p], 0.0)
    # geometric mean: avoid log(0)
    safe = np.where(D < 1e-9, 1e-9, D)
    return np.exp(np.log(safe).mean(axis=1))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds = ds.folds("cluster_isotype"); fold_ids = sorted(set(folds.tolist()))

    esm_files = list((ROOT/"data/processed").glob("esm2_*.npz"))
    esm1 = None
    for ef in esm_files:
        arr = np.load(ef)["emb"]
        if arr.shape==(len(ds.df),2560): esm1=arr; break
    abl1 = np.load(ROOT/"data/processed/ablang2_gdpa1.npz")["emb"]
    esm3 = np.load(ROOT/"data/processed/gdpa3_esm.npz")["emb"]
    abl3 = np.load(ROOT/"data/processed/gdpa3_ablang2.npz")["emb"]

    desc_df1 = build_descriptor_matrix(ds.heavy, ds.light)
    desc_cols = list(desc_df1.columns)
    desc1 = np.nan_to_num(desc_df1.to_numpy(float))
    desc3 = np.nan_to_num(build_descriptor_matrix(g3["vh"],g3["vl"]).to_numpy(float))
    bio1, bio_cols = _bio(ROOT/"external/abdev-benchmark/data/features/processed_features/GDPa1",
                          ds.df["antibody_name"])
    bio3, _ = _bio(ROOT/"external/abdev-benchmark/data/features/processed_features/heldout_test",
                   g3["antibody_name"], cols=bio_cols)
    iso1 = _iso(ds.df); iso3 = _iso(g3)
    esm1n = np.nan_to_num(esm1); esm3n = np.nan_to_num(esm3)
    abl1n = np.nan_to_num(abl1); abl3n = np.nan_to_num(abl3)

    Xdb1 = np.hstack([desc1,bio1]); Xdb3 = np.hstack([desc3,bio3])
    Xplm1 = np.hstack([esm1n,abl1n]); Xplm3 = np.hstack([esm3n,abl3n])
    all_db_cols = desc_cols + list(bio_cols)
    N3 = len(g3)

    # ── M1/M2: AW-MoE and independent predictions ───────────────────────────
    print("\n=== M1/M2: Per-assay predictions (AW-MoE equal ensemble) ===")
    print(f"{'Assay':10s}  {'E1':6s}  {'MGFS':6s}  {'Dstk':6s}  {'FMoE':6s}  {'AW-MoE':6s}  {'True-ρ':6s}  {'dir':4s}")
    print("-"*72)

    pred_matrix = np.full((N3, P), np.nan)   # AW-MoE predictions (raw)
    y3_matrix   = np.full((N3, P), np.nan)   # true GDPa3 labels
    dir_vec     = np.zeros(P)

    for pi, assay in enumerate(ASSAY_NAMES):
        g1_col, g3_col, direction = ASSAYS[assay]
        y1 = ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df),np.nan)
        y3 = g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(N3,np.nan)
        dir_vec[pi] = direction

        blks_tr = {"esm2":esm1n,"ablang2":abl1n,"desc":desc1,"bio":bio1,"isotype":iso1}
        blks_te = {"esm2":esm3n,"ablang2":abl3n,"desc":desc3,"bio":bio3,"isotype":iso3}

        p_e1,  bb  = predict_e1(blks_tr, blks_te, y1, folds, fold_ids)
        p_mgfs     = predict_mgfs(Xdb1, Xdb3, all_db_cols, assay, y1)
        p_dstk     = predict_dstack(Xplm1, Xplm3, Xdb1, Xdb3, y1, folds, fold_ids)
        p_fmoe     = predict_fmoe(Xdb1, Xdb3, all_db_cols, assay, y1, folds, fold_ids)

        aw = np.mean(np.column_stack([p_e1,p_mgfs,p_dstk,p_fmoe]), axis=1)
        pred_matrix[:, pi] = aw
        y3_matrix[:, pi]   = y3

        ok3 = ~np.isnan(y3)
        rho_e1   = spearmanr(y3[ok3], p_e1[ok3]).statistic   if ok3.sum()>=5 else np.nan
        rho_mgfs = spearmanr(y3[ok3], p_mgfs[ok3]).statistic if ok3.sum()>=5 else np.nan
        rho_dstk = spearmanr(y3[ok3], p_dstk[ok3]).statistic if ok3.sum()>=5 else np.nan
        rho_fmoe = spearmanr(y3[ok3], p_fmoe[ok3]).statistic if ok3.sum()>=5 else np.nan
        rho_aw   = spearmanr(y3[ok3], aw[ok3]).statistic     if ok3.sum()>=5 else np.nan

        def f(v): return f"{v:+6.3f}" if not np.isnan(v) else "   nan"
        print(f"{assay:10s}  {f(rho_e1)}  {f(rho_mgfs)}  {f(rho_dstk)}  {f(rho_fmoe)}  {f(rho_aw)}  {f(rho_aw)}  {int(direction):+4d}")

    # ── M2: per-assay Spearman summary ───────────────────────────────────────
    ok3_all = ~np.isnan(y3_matrix)
    aw_rhos = [spearmanr(y3_matrix[ok3_all[:,p],p], pred_matrix[ok3_all[:,p],p]).statistic
               for p in range(P)]
    print(f"\n  AW-MoE mean Spearman (5 genuine): {np.nanmean(aw_rhos):.3f}")

    # ── Direction harmonisation → percentile ranks ────────────────────────────
    # Fit on GDPa1 training labels
    print("\n=== Direction harmonisation & percentile transform ===")
    Q3 = np.full((N3, P), np.nan)
    for pi, assay in enumerate(ASSAY_NAMES):
        g1_col, _, direction = ASSAYS[assay]
        y1 = ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df),np.nan)
        Q3[:, pi] = percentile_rank_transform(y1, pred_matrix[:, pi], direction)
    print(f"  Q matrix shape: {Q3.shape}  (rows=GDPa3 antibodies, cols=5 assays)")
    print(f"  Percentile range: [{Q3.min():.3f}, {Q3.max():.3f}]")
    print(f"  Mean per property: " + "  ".join(f"{ASSAY_NAMES[p]}={Q3[:,p].mean():.2f}" for p in range(P)))

    # ── M3: Uniform weighted score ─────────────────────────────────────────
    score_m3 = Q3.mean(axis=1)

    # ── M4: Pareto fronts ──────────────────────────────────────────────────
    print("\n=== M4: Pareto-based ranking ===")
    fronts = pareto_front(Q3)
    front_counts = pd.Series(fronts).value_counts().sort_index()
    for f_id, cnt in front_counts.items():
        print(f"  Front {f_id}: {cnt:3d} antibodies")

    # ── M5: Borda ranking ─────────────────────────────────────────────────
    score_m5 = borda_score(Q3)

    # ── M6: Geometric desirability (no hard thresholds — use 0.3 as floor) ─
    thresholds = [0.3]*P  # q ≥ 0.3 on every assay = above 30th percentile of training
    score_m6 = desirability_score(Q3, thresholds_lo=thresholds)

    # ── Compile ranked results for GDPa3 ────────────────────────────────────
    print("\n=== M3-M6: Top-15 candidates by AW-MoE Borda score ===")
    results = pd.DataFrame({
        "antibody": g3["antibody_name"].values,
        **{f"q_{assay}": Q3[:, pi] for pi, assay in enumerate(ASSAY_NAMES)},
        "Pareto_front":  fronts,
        "Borda_score":   score_m5,
        "Weighted_score":score_m3,
        "Desirability":  score_m6,
    })
    results = results.sort_values("Borda_score", ascending=False).reset_index(drop=True)
    results["Borda_rank"] = results.index + 1
    results["Pareto_label"] = results["Pareto_front"].apply(lambda x: f"F{x}")

    cols_print = ["antibody","Pareto_label","Borda_score","Weighted_score","Desirability",
                  "q_HIC","q_AC-SINS","q_PR_CHO","q_SMAC","q_HAC"]
    print(results[cols_print].head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Pareto Front 0 (non-dominated) ───────────────────────────────────────
    front0 = results[results["Pareto_front"]==0].sort_values("Borda_score", ascending=False)
    print(f"\n=== Pareto Front 0 (non-dominated) — {len(front0)} candidates ===")
    print(front0[cols_print].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Correlation: multi-property scores vs actual GDPa3 labels ────────────
    print("\n=== Validation: Spearman correlation of ranking scores vs true labels ===")
    for pi, assay in enumerate(ASSAY_NAMES):
        _, g3_col, _ = ASSAYS[assay]
        y3 = g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(N3,np.nan)
        ok = ~np.isnan(y3)
        rho_borda = spearmanr(y3[ok], score_m5[ok]).statistic if ok.sum()>=5 else np.nan
        rho_q     = spearmanr(y3[ok], Q3[ok,pi]).statistic    if ok.sum()>=5 else np.nan
        print(f"  {assay:10s}  q_p vs true: {rho_q:+.3f}   Borda vs true: {rho_borda:+.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== Summary: mean scores across all GDPa3 antibodies ===")
    print(f"  Uniform weighted (M3): {score_m3.mean():.3f}  (std={score_m3.std():.3f})")
    print(f"  Borda score     (M5): {score_m5.mean():.3f}  (std={score_m5.std():.3f})")
    print(f"  Desirability    (M6): {score_m6.mean():.3f}  (std={score_m6.std():.3f})  "
          f"[floor q≥0.30 per assay]")
    print(f"  Pareto Front 0: {(fronts==0).sum():3d} / {N3} candidates ({100*(fronts==0).mean():.1f}%)")
    print(f"  Pareto Front 1: {(fronts==1).sum():3d} / {N3} candidates ({100*(fronts==1).mean():.1f}%)")

    # Save
    out_full   = ROOT/"results/abdev_mpo_full.csv"
    out_pareto = ROOT/"results/abdev_mpo_front0.csv"
    results.to_csv(out_full, index=False)
    front0.to_csv(out_pareto, index=False)
    print(f"\nSaved:\n  {out_full}\n  {out_pareto}")


if __name__ == "__main__":
    main()
