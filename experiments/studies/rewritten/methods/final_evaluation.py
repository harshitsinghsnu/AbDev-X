"""Final evaluation: AW-MoE (final model) + ablation hierarchy.
3 seeds, multiple metrics, error bars.

Final model:  AW-MoE  (equal-weight ensemble of E1, A-MGFS, D-stack, F-MoE)

Ablation hierarchy:
  A0  Random baseline (permuted labels)
  A1  E1 — single best-block (AbDev-X)
  A2  E2 — ESM-2 only
  A3  E3 — AbLang2 only
  A4  E4 — Descriptors only
  A5  E5 — Bio features only
  A6  A-MGFS — assay-specific biological masks
  A7  D-stack — dual-stream OOF stacking
  A8  F-MoE — feature-level expert mixture
  A9  AW-MoE — assay-winner ensemble  ← FINAL MODEL

Metrics per assay:
  Spearman ρ      rank correlation (primary)
  Pearson r        linear correlation
  Kendall τ        concordance
  Top-20% prec     fraction of true top-20% correctly ranked in pred top-20%
  RMSE (norm)      MSE on z-scored labels

Seeds: 0, 1, 2  (controls PCA, fold permutations, XGBoost)
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

GENUINE_ASSAYS = {
    "HIC":     ("HIC",           "hic_rt_avg"),
    "AC-SINS": ("AC-SINS_pH7.4", "acsins_dLmax_ph7.4_avg"),
    "PR_CHO":  ("PR_CHO",        "polyreactivity_prscore_cho_avg"),
    "SMAC":    ("SMAC",          "smac_rt_avg"),
    "HAC":     ("HAC",           "hac_rt_avg"),
}
ASSAY_NAMES = list(GENUINE_ASSAYS.keys()); P = len(ASSAY_NAMES)
BIO_FILES   = ["TAP.csv","MOE_properties.csv","Aggrescan3D.csv","AntiFold.csv"]
ISO_COLS    = ["hc_subtype_IgG1","hc_subtype_IgG2","hc_subtype_IgG4",
               "lc_subtype_Kappa","lc_subtype_Lambda"]
MGFS_MASKS  = {
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
    "HIC":    ["hydro","surface"],"AC-SINS":["charge","surface"],
    "PR_CHO": ["charge","cdr","hydro"],"SMAC":  ["surface","hydro","charge"],
    "HAC":    ["surface","hydro","charge","stab"],
}
SEEDS = [0, 1, 2]
N_BOOT = 1000


# ── helpers ───────────────────────────────────────────────────────────────────

def _iso(df):
    d=pd.get_dummies(df[["hc_subtype","lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS,fill_value=0).to_numpy(float)

def _bio(feat_dir, names, cols=None):
    base=pd.DataFrame({"antibody_name":list(names)})
    for f in BIO_FILES:
        p=Path(feat_dir)/f
        if not p.exists(): continue
        d=pd.read_csv(p); num=d.select_dtypes("number").copy()
        num["antibody_name"]=d["antibody_name"].values
        num=num.rename(columns={c:f"{f[:-4]}::{c}" for c in num.columns if c!="antibody_name"})
        base=base.merge(num,on="antibody_name",how="left")
    X=base.drop(columns=["antibody_name"])
    if cols is None: cols=list(X.columns)
    return np.nan_to_num(X.reindex(columns=cols,fill_value=0).to_numpy(float)),cols

def _mask_cols(all_cols,inc_kws,exc_kws=()):
    inc=[k.lower() for k in inc_kws]; exc=[k.lower() for k in exc_kws]
    return np.array([(any(k in c.lower() for k in inc) and
                      not any(k in c.lower() for k in exc)) for c in all_cols])

def _head(X, seed=0):
    steps=[("s",StandardScaler())]
    if X.shape[1]>100: steps.append(("p",PCA(0.9,random_state=seed,svd_solver="full")))
    steps.append(("r",RidgeCV(alphas=np.logspace(-2,4,12)))); return Pipeline(steps)

def ood_cv_rho(X, y, folds, fold_ids, seed=0):
    m=~np.isnan(y); scores=[]
    for f in fold_ids:
        tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
        if len(te)<3 or len(tr)<10: continue
        pipe=_head(X[tr],seed); pipe.fit(X[tr],y[tr])
        scores.append(spearmanr(y[te],pipe.predict(X[te])).statistic)
    return float(np.nanmean(scores)) if scores else -1.0


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    ok=~np.isnan(y_true)&~np.isnan(y_pred)
    if ok.sum()<5: return {k:np.nan for k in ["spearman","pearson","kendall","top20_prec","rmse_norm"]}
    y,p=y_true[ok],y_pred[ok]
    rho  = float(spearmanr(y,p).statistic)
    r    = float(pearsonr(y,p)[0])
    tau  = float(kendalltau(y,p).statistic)
    # Top-20% precision: fraction of true top-20% found in pred top-20%
    k20=max(1,int(0.2*len(y)))
    true_top=set(np.argsort(y)[:k20])   # lowest = best (all assays: lower=better)
    pred_top=set(np.argsort(p)[:k20])
    top20=len(true_top&pred_top)/k20
    # RMSE on z-scored labels
    y_z=(y-y.mean())/(y.std()+1e-9); p_z=(p-p.mean())/(p.std()+1e-9)
    rmse=float(np.sqrt(np.mean((y_z-p_z)**2)))
    return {"spearman":rho,"pearson":r,"kendall":tau,"top20_prec":top20,"rmse_norm":rmse}


# ── ablation model predictors ─────────────────────────────────────────────────

def pred_single_block(X1, y1, X3, seed=0):
    m=~np.isnan(y1); pipe=_head(X1,seed); pipe.fit(X1[m],y1[m])
    return pipe.predict(X3)

def pred_e1(blks_tr, blks_te, y1, folds, fold_ids, seed=0):
    m=~np.isnan(y1)
    cv={b:ood_cv_rho(X,y1,folds,fold_ids,seed) for b,X in blks_tr.items()}
    bb=max(cv,key=lambda k:cv[k] if not np.isnan(cv[k]) else -9)
    pipe=_head(blks_tr[bb],seed); pipe.fit(blks_tr[bb][m],y1[m])
    return pipe.predict(blks_te[bb]), bb

def pred_mgfs(Xdb1, Xdb3, all_cols, assay, y1, seed=0):
    cfg=MGFS_MASKS.get(assay,{"inc":[],"exc":[]})
    msk=_mask_cols(all_cols,cfg["inc"],cfg["exc"])
    if msk.sum()<5: msk=np.ones(len(all_cols),bool)
    m=~np.isnan(y1); pipe=_head(Xdb1[:,msk],seed); pipe.fit(Xdb1[m][:,msk],y1[m])
    return pipe.predict(Xdb3[:,msk])

def pred_dstack(Xplm1, Xplm3, Xdb1, Xdb3, y1, folds, fold_ids, seed=0):
    m=~np.isnan(y1); oof1=np.zeros(len(y1)); oof2=np.zeros(len(y1))
    for f in fold_ids:
        tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
        if len(te)<3 or len(tr)<10: continue
        p1=_head(Xplm1[tr],seed); p1.fit(Xplm1[tr],y1[tr]); oof1[te]=p1.predict(Xplm1[te])
        p2=_head(Xdb1[tr],seed);  p2.fit(Xdb1[tr],y1[tr]);  oof2[te]=p2.predict(Xdb1[te])
    oof1=np.nan_to_num(oof1); oof2=np.nan_to_num(oof2)
    meta=Ridge(alpha=1.0); meta.fit(np.column_stack([oof1,oof2])[m],y1[m])
    p1f=_head(Xplm1,seed); p1f.fit(Xplm1[m],y1[m])
    p2f=_head(Xdb1,seed);  p2f.fit(Xdb1[m],y1[m])
    return meta.predict(np.column_stack([p1f.predict(Xplm3),p2f.predict(Xdb3)]))

def pred_fmoe(Xdb1, Xdb3, all_cols, assay, y1, folds, fold_ids, seed=0):
    eligible=MOE_ELIGIBLE.get(assay,list(MOE_EXPERTS.keys()))
    m=~np.isnan(y1); pipes=[]; cvs=[]; masks=[]
    for exp in eligible:
        kws=MOE_EXPERTS.get(exp,[]); msk=_mask_cols(all_cols,kws)
        if msk.sum()<3: continue
        Xe=Xdb1[:,msk]; sc=[]
        for f in fold_ids:
            tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
            if len(te)<3 or len(tr)<10: continue
            pf=_head(Xe[tr],seed); pf.fit(Xe[tr],y1[tr]); sc.append(spearmanr(y1[te],pf.predict(Xe[te])).statistic)
        pff=_head(Xe[m],seed); pff.fit(Xe[m],y1[m])
        pipes.append(pff); cvs.append(max(float(np.nanmean(sc)) if sc else -1.0,0)); masks.append(msk)
    if not pipes:
        return pred_mgfs(Xdb1,Xdb3,all_cols,assay,y1,seed)
    ws=np.exp(np.array(cvs)/0.1); ws/=ws.sum()
    return np.column_stack([pf.predict(Xdb3[:,mk]) for pf,mk in zip(pipes,masks)])@ws

def pred_awmoe(blks_tr, blks_te, Xdb1, Xdb3, Xplm1, Xplm3, all_cols, assay, y1, folds, fold_ids, seed=0):
    te_e1,_ = pred_e1(blks_tr,blks_te,y1,folds,fold_ids,seed)
    te_m    = pred_mgfs(Xdb1,Xdb3,all_cols,assay,y1,seed)
    te_d    = pred_dstack(Xplm1,Xplm3,Xdb1,Xdb3,y1,folds,fold_ids,seed)
    te_f    = pred_fmoe(Xdb1,Xdb3,all_cols,assay,y1,folds,fold_ids,seed)
    return np.mean(np.column_stack([te_e1,te_m,te_d,te_f]),axis=1)

def pred_random(y1, y3, rng):
    """Permute training labels — pure noise baseline."""
    m=~np.isnan(y1); y_perm=y1.copy(); y_perm[m]=rng.permutation(y1[m])
    return np.full(len(y3),np.nanmean(y1))  # predict mean always


# ── main evaluation loop ──────────────────────────────────────────────────────

def run_seed(seed, ds, g3, folds, fold_ids,
             esm1n, abl1n, esm3n, abl3n,
             desc1, desc3, bio1, bio3, iso1, iso3,
             Xdb1, Xdb3, Xplm1, Xplm3, all_db_cols):
    rng=np.random.default_rng(seed)
    results={}
    for assay,(g1_col,g3_col) in GENUINE_ASSAYS.items():
        y1=ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df),np.nan)
        y3=g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(len(g3),np.nan)

        blks_tr={"esm2":esm1n,"ablang2":abl1n,"desc":desc1,"bio":bio1,"isotype":iso1}
        blks_te={"esm2":esm3n,"ablang2":abl3n,"desc":desc3,"bio":bio3,"isotype":iso3}

        preds={}
        preds["A0-random"]   = pred_random(y1,y3,rng)
        preds["A1-E1"]       = pred_e1(blks_tr,blks_te,y1,folds,fold_ids,seed)[0]
        preds["A2-ESM2"]     = pred_single_block(esm1n,y1,esm3n,seed)
        preds["A3-AbLang2"]  = pred_single_block(abl1n,y1,abl3n,seed)
        preds["A4-Desc"]     = pred_single_block(desc1,y1,desc3,seed)
        preds["A5-Bio"]      = pred_single_block(bio1,y1,bio3,seed)
        preds["A6-MGFS"]     = pred_mgfs(Xdb1,Xdb3,all_db_cols,assay,y1,seed)
        preds["A7-Dstack"]   = pred_dstack(Xplm1,Xplm3,Xdb1,Xdb3,y1,folds,fold_ids,seed)
        preds["A8-FMoE"]     = pred_fmoe(Xdb1,Xdb3,all_db_cols,assay,y1,folds,fold_ids,seed)
        preds["A9-AW-MoE"]   = pred_awmoe(blks_tr,blks_te,Xdb1,Xdb3,Xplm1,Xplm3,
                                           all_db_cols,assay,y1,folds,fold_ids,seed)
        for model_name,pred in preds.items():
            results[(assay,model_name)]=compute_metrics(y3,pred)
    return results


def main():
    ds=load_gdpa1(str(ROOT/"configs/default.yaml"))
    g3=pd.read_csv(ROOT/"data/processed/gdpa3_table.csv")
    folds=ds.folds("cluster_isotype"); fold_ids=sorted(set(folds.tolist()))

    esm_files=list((ROOT/"data/processed").glob("esm2_*.npz")); esm1=None
    for ef in esm_files:
        arr=np.load(ef)["emb"]
        if arr.shape==(len(ds.df),2560): esm1=arr; break
    abl1=np.load(ROOT/"data/processed/ablang2_gdpa1.npz")["emb"]
    esm3=np.load(ROOT/"data/processed/gdpa3_esm.npz")["emb"]
    abl3=np.load(ROOT/"data/processed/gdpa3_ablang2.npz")["emb"]
    desc_df1=build_descriptor_matrix(ds.heavy,ds.light)
    desc_cols=list(desc_df1.columns); desc1=np.nan_to_num(desc_df1.to_numpy(float))
    desc3=np.nan_to_num(build_descriptor_matrix(g3["vh"],g3["vl"]).to_numpy(float))
    bio1,bio_cols=_bio(ROOT/"external/abdev-benchmark/data/features/processed_features/GDPa1",ds.df["antibody_name"])
    bio3,_=_bio(ROOT/"external/abdev-benchmark/data/features/processed_features/heldout_test",g3["antibody_name"],cols=bio_cols)
    iso1=_iso(ds.df); iso3=_iso(g3)
    esm1n=np.nan_to_num(esm1); esm3n=np.nan_to_num(esm3)
    abl1n=np.nan_to_num(abl1); abl3n=np.nan_to_num(abl3)
    Xdb1=np.hstack([desc1,bio1]); Xdb3=np.hstack([desc3,bio3])
    Xplm1=np.hstack([esm1n,abl1n]); Xplm3=np.hstack([esm3n,abl3n])
    all_db_cols=desc_cols+list(bio_cols)

    # ── 3-seed evaluation ────────────────────────────────────────────────────
    all_results = []
    for seed in SEEDS:
        print(f"\n  Running seed {seed}...", end="", flush=True)
        res=run_seed(seed,ds,g3,folds,fold_ids,
                     esm1n,abl1n,esm3n,abl3n,
                     desc1,desc3,bio1,bio3,iso1,iso3,
                     Xdb1,Xdb3,Xplm1,Xplm3,all_db_cols)
        for (assay,model),metrics in res.items():
            all_results.append({"seed":seed,"assay":assay,"model":model,**metrics})
        print(" done")

    df=pd.DataFrame(all_results)

    MODELS_ORDERED = ["A0-random","A1-E1","A2-ESM2","A3-AbLang2","A4-Desc","A5-Bio",
                      "A6-MGFS","A7-Dstack","A8-FMoE","A9-AW-MoE"]
    METRICS = ["spearman","pearson","kendall","top20_prec","rmse_norm"]

    # ── Primary table: Spearman ρ mean±std across 3 seeds ──────────────────
    print(f"\n{'='*110}")
    print("PRIMARY RESULTS: Spearman ρ  mean ± std (3 seeds)")
    print(f"{'='*110}")
    header = f"{'Model':15s}  {'HIC':12s}  {'AC-SINS':12s}  {'PR_CHO':12s}  {'SMAC':12s}  {'HAC':12s}  {'MEAN':12s}"
    print(header); print("-"*110)

    summary_rows=[]
    for model in MODELS_ORDERED:
        sub=df[df["model"]==model]
        per_assay={}; final_mark=""
        for assay in ASSAY_NAMES:
            vals=sub[sub["assay"]==assay]["spearman"].values
            per_assay[assay]=(float(np.nanmean(vals)),float(np.nanstd(vals)))
        means=[per_assay[a][0] for a in ASSAY_NAMES]
        stds =[per_assay[a][1] for a in ASSAY_NAMES]
        overall_mean=float(np.nanmean(means))
        overall_std =float(np.nanstd(means))
        if model=="A9-AW-MoE": final_mark=" ← FINAL"
        def f(m,s): return f"{m:+.3f}±{s:.3f}" if not np.isnan(m) else "    nan     "
        row_str="  ".join(f(m,s) for m,s in zip(means,stds))
        print(f"{model:15s}  {row_str}  {f(overall_mean,overall_std)}{final_mark}")
        summary_rows.append({"model":model,
                              **{f"{a}_mean":per_assay[a][0] for a in ASSAY_NAMES},
                              **{f"{a}_std":per_assay[a][1] for a in ASSAY_NAMES},
                              "overall_mean":overall_mean,"overall_std":overall_std})
    print(f"{'='*110}")

    # ── All metrics for final model ──────────────────────────────────────────
    print(f"\n{'='*90}")
    print("FINAL MODEL (A9-AW-MoE): All metrics mean ± std")
    print(f"{'='*90}")
    print(f"{'Metric':15s}  {'HIC':12s}  {'AC-SINS':12s}  {'PR_CHO':12s}  {'SMAC':12s}  {'HAC':12s}")
    print("-"*90)
    sub_final=df[df["model"]=="A9-AW-MoE"]
    for metric in METRICS:
        parts=[]
        for assay in ASSAY_NAMES:
            vals=sub_final[sub_final["assay"]==assay][metric].values
            m_=float(np.nanmean(vals)); s_=float(np.nanstd(vals))
            label="↑" if metric in ["spearman","pearson","kendall","top20_prec"] else "↓"
            parts.append(f"{m_:+.3f}±{s_:.3f}")
        print(f"{metric:15s}  {'  '.join(parts)}")
    print(f"{'='*90}")

    # ── Ablation contribution: Δ vs A9-AW-MoE ───────────────────────────────
    print(f"\n{'='*90}")
    print("ABLATION ΔSPEARMAN vs AW-MoE (final model)   [negative = AW-MoE wins]")
    print(f"{'='*90}")
    aw_means={assay: df[(df["model"]=="A9-AW-MoE")&(df["assay"]==assay)]["spearman"].mean()
              for assay in ASSAY_NAMES}
    print(f"{'Model':15s}  {'HIC':7s}  {'AC-SINS':7s}  {'PR_CHO':7s}  {'SMAC':7s}  {'HAC':7s}  {'Mean Δ':8s}")
    print("-"*80)
    for model in MODELS_ORDERED[:-1]:  # exclude AW-MoE itself
        sub=df[df["model"]==model]; deltas=[]
        parts=[]
        for assay in ASSAY_NAMES:
            m_=sub[sub["assay"]==assay]["spearman"].mean()
            delta=m_-aw_means[assay]; deltas.append(delta)
            parts.append(f"{delta:+7.3f}")
        print(f"{model:15s}  {'  '.join(parts)}  {np.nanmean(deltas):+8.3f}")
    print(f"{'='*90}")

    # ── Per-assay best model ────────────────────────────────────────────────
    print(f"\n=== Per-assay winner (by mean Spearman across 3 seeds) ===")
    for assay in ASSAY_NAMES:
        sub=df[df["assay"]==assay].groupby("model")["spearman"].mean()
        best=sub.idxmax(); best_val=sub.max()
        aw_val=sub.get("A9-AW-MoE",np.nan)
        delta=best_val-aw_val
        flag="" if best=="A9-AW-MoE" else f"  (AW-MoE: {aw_val:.3f}, Δ={delta:+.3f})"
        print(f"  {assay:10s}: {best:15s}  ρ={best_val:.3f}{flag}")

    # ── Bootstrap CI for final model ────────────────────────────────────────
    print(f"\n=== Bootstrap 95% CI for AW-MoE (seed=0 predictions) ===")
    from scipy.stats import norm
    rng_ci=np.random.default_rng(99)
    # Re-run seed=0 to get actual predictions
    seed0_res={}
    for assay,(g1_col,g3_col) in GENUINE_ASSAYS.items():
        y1=ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df),np.nan)
        y3=g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(len(g3),np.nan)
        blks_tr={"esm2":esm1n,"ablang2":abl1n,"desc":desc1,"bio":bio1,"isotype":iso1}
        blks_te={"esm2":esm3n,"ablang2":abl3n,"desc":desc3,"bio":bio3,"isotype":iso3}
        pred=pred_awmoe(blks_tr,blks_te,Xdb1,Xdb3,Xplm1,Xplm3,all_db_cols,assay,y1,folds,fold_ids,0)
        ok=~np.isnan(y3); y_,p_=y3[ok],pred[ok]; n_=len(y_)
        if n_<5: continue
        obs=spearmanr(y_,p_).statistic
        boot=np.array([spearmanr(y_[s:=rng_ci.integers(0,n_,n_)],p_[s]).statistic for _ in range(N_BOOT)])
        z0=norm.ppf(np.clip(np.mean(boot<obs),1e-6,1-1e-6))
        jk=np.array([spearmanr(np.delete(y_,i),np.delete(p_,i)).statistic for i in range(n_)])
        jm=jk.mean(); a_=((jm-jk)**3).sum()/(6*(((jm-jk)**2).sum()**1.5)+1e-12)
        def adj(z): return norm.cdf(z0+(z0+z)/(1-a_*(z0+z)))
        ci_lo=float(np.percentile(boot,100*adj(norm.ppf(0.025))))
        ci_hi=float(np.percentile(boot,100*adj(norm.ppf(0.975))))
        print(f"  {assay:10s}  ρ={obs:+.3f}  BCa 95% CI=[{ci_lo:+.3f},{ci_hi:+.3f}]  n={ok.sum()}")

    # Save all results
    df.to_csv(ROOT/"results/final_evaluation_raw.csv",index=False)
    pd.DataFrame(summary_rows).to_csv(ROOT/"results/final_evaluation_summary.csv",index=False)
    print(f"\nSaved results/final_evaluation_raw.csv and results/final_evaluation_summary.csv")


if __name__=="__main__":
    main()
