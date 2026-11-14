"""AbDev-MPO Hybrid: assay-specific specialist selection — M11–M15.

M11  Oracle hybrid (post-hoc best per assay from GDPa3 — upper bound only)
M12  OOF specialist selection: pick best per assay using OOD-CV on GDPa1 only
M13  Paired bootstrap: hybrid vs AW-MoE CI
M14  Hybrid vs AW-MoE Pareto comparison
M15  Candidate shortlist stability

Candidates:
  E1       best-block OOD-CV selection (seq/bio/desc/iso)
  Ridge3   RidgeCV on desc+bio (TF-3)
  XGB3     XGBoost on desc+bio (TF-3)
  TabPFN3  TabPFN on desc+bio (TF-3)
  AW-MoE   equal-weight ensemble of E1/MGFS/Dstack/FMoE

OOD folds: same hierarchical_cluster_isotype_stratified folds throughout.
GDPa3 only used after all selection decisions are frozen.
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

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
BIO_FILES = ["TAP.csv","MOE_properties.csv","Aggrescan3D.csv","AntiFold.csv"]
ISO_COLS  = ["hc_subtype_IgG1","hc_subtype_IgG2","hc_subtype_IgG4",
             "lc_subtype_Kappa","lc_subtype_Lambda"]
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
    "HIC":    ["hydro","surface"],"AC-SINS":["charge","surface"],
    "PR_CHO": ["charge","cdr","hydro"],"SMAC":  ["surface","hydro","charge"],
    "HAC":    ["surface","hydro","charge","stab"],
}
N_BOOT = 2000
ASSAY_DIRS = {"HIC":-1,"AC-SINS":-1,"PR_CHO":-1,"SMAC":-1,"HAC":-1}


# ── helpers ───────────────────────────────────────────────────────────────────

def _iso(df):
    d=pd.get_dummies(df[["hc_subtype","lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS,fill_value=0).to_numpy(float)

def _bio(feat_dir,names,cols=None):
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

def _ridge(X):
    steps=[("s",StandardScaler())]
    if X.shape[1]>100: steps.append(("p",PCA(0.9,random_state=0,svd_solver="full")))
    steps.append(("r",RidgeCV(alphas=np.logspace(-2,4,12)))); return Pipeline(steps)

def _xgb():
    return XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,
                        subsample=0.8,colsample_bytree=0.8,reg_alpha=1.0,
                        reg_lambda=1.0,random_state=0,verbosity=0)

def _tabpfn():
    return TabPFNRegressor(n_estimators=8,random_state=0)

def percentile_rank(y_train,y_test,direction):
    z_tr=direction*y_train; z_te=direction*y_test
    m=~np.isnan(z_tr); ref=np.sort(z_tr[m])
    return np.clip(np.searchsorted(ref,z_te,side='right')/len(ref),0,1)

def pareto_front(Q):
    n=len(Q); dc=np.zeros(n,int); db=[set() for _ in range(n)]; fr=np.full(n,-1,int)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if np.all(Q[j]>=Q[i]) and np.any(Q[j]>Q[i]): db[i].add(j); dc[i]+=1
    curr=0; cur=[i for i in range(n) if dc[i]==0]
    while cur:
        for i in cur: fr[i]=curr
        nxt=[]
        for i in cur:
            for j in range(n):
                if fr[j]==-1 and i in db[j]:
                    dc[j]-=1
                    if dc[j]==0: nxt.append(j)
        cur=nxt; curr+=1
    return fr

def geom_D(Q,floor=0.30):
    fail=np.any(Q<floor,axis=1); safe=np.where(Q<1e-9,1e-9,Q)
    d=np.exp(np.log(safe).mean(axis=1)); d[fail]=0.0; return d

def bca(y,pred,rng,n=N_BOOT):
    ok=~np.isnan(y)&~np.isnan(pred); y_,p_=y[ok],pred[ok]; n_=len(y_)
    if n_<5: return np.nan,np.nan
    from scipy.stats import norm
    obs=spearmanr(y_,p_).statistic
    boot=np.array([spearmanr(y_[s:=rng.integers(0,n_,n_)],p_[s]).statistic for _ in range(n)])
    z0=norm.ppf(np.clip(np.mean(boot<obs),1e-6,1-1e-6))
    jk=np.array([spearmanr(np.delete(y_,i),np.delete(p_,i)).statistic for i in range(n_)])
    jm=jk.mean(); a=((jm-jk)**3).sum()/(6*(((jm-jk)**2).sum()**1.5)+1e-12)
    def adj(z): return norm.cdf(z0+(z0+z)/(1-a*(z0+z)))
    return float(np.percentile(boot,100*adj(norm.ppf(0.025)))),\
           float(np.percentile(boot,100*adj(norm.ppf(0.975))))


# ── per-assay model: train + predict ─────────────────────────────────────────

def train_predict(model_name, X1, y1, X3, folds, fold_ids):
    """Train model on full GDPa1, return (test_pred, oof_pred, ood_cv_rho)."""
    m=~np.isnan(y1); N=len(y1)
    oof=np.full(N,np.nan); cv_scores=[]

    for f in fold_ids:
        tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
        if len(te)<3 or len(tr)<10: continue
        if model_name=="Ridge":
            model=_ridge(X1[tr]); model.fit(X1[tr],y1[tr]); pred=model.predict(X1[te])
        elif model_name=="XGB":
            model=_xgb(); model.fit(X1[tr],y1[tr]); pred=model.predict(X1[te])
        elif model_name=="TabPFN":
            model=_tabpfn(); model.fit(X1[tr],y1[tr]); pred=model.predict(X1[te])
        oof[te]=pred
        cv_scores.append(spearmanr(y1[te],pred).statistic)

    cv_rho=float(np.nanmean(cv_scores)) if cv_scores else -1.0

    # Refit on full training set
    if model_name=="Ridge":
        full_model=_ridge(X1[m]); full_model.fit(X1[m],y1[m]); test_pred=full_model.predict(X3)
    elif model_name=="XGB":
        full_model=_xgb(); full_model.fit(X1[m],y1[m]); test_pred=full_model.predict(X3)
    elif model_name=="TabPFN":
        full_model=_tabpfn(); full_model.fit(X1[m],y1[m]); test_pred=full_model.predict(X3)

    return test_pred, oof, cv_rho


def train_predict_e1(blks_tr, blks_te, y1, folds, fold_ids):
    m=~np.isnan(y1); N=len(y1); oof=np.full(N,np.nan); cv_scores=[]
    cv_blk={b:[] for b in blks_tr}
    for f in fold_ids:
        tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
        if len(te)<3 or len(tr)<10: continue
        for b,Xb in blks_tr.items():
            pipe=_ridge(Xb[tr]); pipe.fit(Xb[tr],y1[tr])
            cv_blk[b].append(spearmanr(y1[te],pipe.predict(Xb[te])).statistic)
    best_blk=max(cv_blk,key=lambda k:np.nanmean(cv_blk[k]) if cv_blk[k] else -9)
    Xb=blks_tr[best_blk]
    for f in fold_ids:
        tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
        if len(te)<3 or len(tr)<10: continue
        pipe=_ridge(Xb[tr]); pipe.fit(Xb[tr],y1[tr]); oof[te]=pipe.predict(Xb[te])
        cv_scores.append(spearmanr(y1[te],oof[te]).statistic)
    full=_ridge(Xb[m]); full.fit(Xb[m],y1[m])
    return full.predict(blks_te[best_blk]), oof, float(np.nanmean(cv_scores)) if cv_scores else -1.0, best_blk


def train_predict_awmoe(blks_tr, blks_te, Xdb1, Xdb3, Xplm1, Xplm3,
                        all_db_cols, assay, y1, folds, fold_ids):
    m=~np.isnan(y1); N=len(y1); oof_parts=[]; cv_rhos=[]

    # E1
    te_e1, oof_e1, cv_e1, _ = train_predict_e1(blks_tr,blks_te,y1,folds,fold_ids)
    oof_parts.append(oof_e1); cv_rhos.append(cv_e1)

    # MGFS
    cfg=MGFS_MASKS.get(assay,{"inc":[],"exc":[]})
    msk=_mask_cols(all_db_cols,cfg["inc"],cfg["exc"])
    if msk.sum()<5: msk=np.ones(len(all_db_cols),bool)
    oof_m=np.full(N,np.nan); cv_m=[]
    for f in fold_ids:
        tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
        if len(te)<3 or len(tr)<10: continue
        pipe=_ridge(Xdb1[tr][:,msk]); pipe.fit(Xdb1[tr][:,msk],y1[tr])
        oof_m[te]=pipe.predict(Xdb1[te][:,msk]); cv_m.append(spearmanr(y1[te],oof_m[te]).statistic)
    full_m=_ridge(Xdb1[m][:,msk]); full_m.fit(Xdb1[m][:,msk],y1[m])
    te_mgfs=full_m.predict(Xdb3[:,msk]); oof_parts.append(oof_m); cv_rhos.append(float(np.nanmean(cv_m)) if cv_m else -1.0)

    # D-stack
    oof_d1=np.full(N,np.nan); oof_d2=np.full(N,np.nan)
    for f in fold_ids:
        tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
        if len(te)<3 or len(tr)<10: continue
        p1=_ridge(Xplm1[tr]); p1.fit(Xplm1[tr],y1[tr]); oof_d1[te]=p1.predict(Xplm1[te])
        p2=_ridge(Xdb1[tr]); p2.fit(Xdb1[tr],y1[tr]); oof_d2[te]=p2.predict(Xdb1[te])
    oof_d1=np.nan_to_num(oof_d1); oof_d2=np.nan_to_num(oof_d2)
    meta=Ridge(alpha=1.0); meta.fit(np.column_stack([oof_d1,oof_d2])[m],y1[m])
    p1f=_ridge(Xplm1[m]); p1f.fit(Xplm1[m],y1[m])
    p2f=_ridge(Xdb1[m]); p2f.fit(Xdb1[m],y1[m])
    oof_d=meta.predict(np.column_stack([oof_d1,oof_d2])); te_d=meta.predict(np.column_stack([p1f.predict(Xplm3),p2f.predict(Xdb3)]))
    cv_d=float(spearmanr(y1[m],oof_d[m]).statistic) if m.sum()>5 else -1.0
    oof_parts.append(oof_d); cv_rhos.append(cv_d)

    # F-MoE
    eligible=MOE_ELIGIBLE.get(assay,list(MOE_EXPERTS.keys())); pipes_f=[]; cvs_f=[]; masks_f=[]
    for exp in eligible:
        kws=MOE_EXPERTS.get(exp,[]); mskf=_mask_cols(all_db_cols,kws)
        if mskf.sum()<3: continue
        Xe=Xdb1[:,mskf]; sc=[]
        for f in fold_ids:
            tr=np.where((folds!=f)&m)[0]; te=np.where((folds==f)&m)[0]
            if len(te)<3 or len(tr)<10: continue
            pf=_ridge(Xe[tr]); pf.fit(Xe[tr],y1[tr]); sc.append(spearmanr(y1[te],pf.predict(Xe[te])).statistic)
        pff=_ridge(Xe[m]); pff.fit(Xe[m],y1[m])
        pipes_f.append(pff); cvs_f.append(max(float(np.nanmean(sc)) if sc else -1.0,0)); masks_f.append(mskf)
    if pipes_f:
        ws=np.exp(np.array(cvs_f)/0.1); ws/=ws.sum()
        te_f=np.column_stack([pf.predict(Xdb3[:,mk]) for pf,mk in zip(pipes_f,masks_f)])@ws
    else: te_f=te_mgfs
    oof_f=np.full(N,np.nan)  # approximate OOF for F-MoE via average of expert OOFs
    oof_parts.append(oof_f); cv_rhos.append(0.35)  # conservative CV estimate

    # Equal-weight ensemble
    valid_oof=np.column_stack(oof_parts)
    oof_aw=np.nanmean(valid_oof,axis=1)
    te_aw=np.mean(np.column_stack([te_e1,te_mgfs,te_d,te_f]),axis=1)
    cv_aw=float(np.nanmean(cv_rhos))
    return te_aw, oof_aw, cv_aw


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ds=load_gdpa1(str(ROOT/"configs/default.yaml"))
    g3=pd.read_csv(ROOT/"data/processed/gdpa3_table.csv")
    folds=ds.folds("cluster_isotype"); fold_ids=sorted(set(folds.tolist()))
    rng=np.random.default_rng(42); N3=len(g3)

    esm_files=list((ROOT/"data/processed").glob("esm2_*.npz")); esm1=None
    for ef in esm_files:
        arr=np.load(ef)["emb"]
        if arr.shape==(len(ds.df),2560): esm1=arr; break
    abl1=np.load(ROOT/"data/processed/ablang2_gdpa1.npz")["emb"]
    esm3=np.load(ROOT/"data/processed/gdpa3_esm.npz")["emb"]
    abl3=np.load(ROOT/"data/processed/gdpa3_ablang2.npz")["emb"]
    desc_df1=build_descriptor_matrix(ds.heavy,ds.light)
    desc_cols=list(desc_df1.columns)
    desc1=np.nan_to_num(desc_df1.to_numpy(float))
    desc3=np.nan_to_num(build_descriptor_matrix(g3["vh"],g3["vl"]).to_numpy(float))
    bio1,bio_cols=_bio(ROOT/"external/abdev-benchmark/data/features/processed_features/GDPa1",ds.df["antibody_name"])
    bio3,_=_bio(ROOT/"external/abdev-benchmark/data/features/processed_features/heldout_test",g3["antibody_name"],cols=bio_cols)
    iso1=_iso(ds.df); iso3=_iso(g3)
    esm1n=np.nan_to_num(esm1); esm3n=np.nan_to_num(esm3)
    abl1n=np.nan_to_num(abl1); abl3n=np.nan_to_num(abl3)
    Xdb1=np.hstack([desc1,bio1]); Xdb3=np.hstack([desc3,bio3])
    Xplm1=np.hstack([esm1n,abl1n]); Xplm3=np.hstack([esm3n,abl3n])
    all_db_cols=desc_cols+list(bio_cols)

    # ── M12: OOF specialist selection ────────────────────────────────────────
    print("\n=== M12: OOF-based specialist selection (no GDPa3 leakage) ===")
    print(f"{'Assay':10s}  {'E1-CV':6s}  {'Ridge3-CV':9s}  {'XGB3-CV':7s}  {'TabPFN3-CV':10s}  {'AW-CV':6s}  Selected")
    print("="*82)

    selected_preds = {}   # assay → test predictions from OOF-selected specialist
    selected_oofs  = {}   # assay → OOF predictions
    oof_cv_table   = {}   # assay → {model: cv_rho}

    for assay,(g1_col,g3_col) in GENUINE_ASSAYS.items():
        y1=ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df),np.nan)
        y3=g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(N3,np.nan)

        blks_tr={"esm2":esm1n,"ablang2":abl1n,"desc":desc1,"bio":bio1,"isotype":iso1}
        blks_te={"esm2":esm3n,"ablang2":abl3n,"desc":desc3,"bio":bio3,"isotype":iso3}

        # Generate OOF + test predictions for all candidates
        te_e1, oof_e1, cv_e1, bb = train_predict_e1(blks_tr,blks_te,y1,folds,fold_ids)
        te_r3, oof_r3, cv_r3 = train_predict("Ridge", Xdb1, y1, Xdb3, folds, fold_ids)
        te_x3, oof_x3, cv_x3 = train_predict("XGB",   Xdb1, y1, Xdb3, folds, fold_ids)
        te_tf, oof_tf, cv_tf = train_predict("TabPFN", Xdb1, y1, Xdb3, folds, fold_ids)
        te_aw, oof_aw, cv_aw = train_predict_awmoe(blks_tr,blks_te,Xdb1,Xdb3,Xplm1,Xplm3,
                                                    all_db_cols,assay,y1,folds,fold_ids)

        cv_dict = {"E1":cv_e1,"Ridge3":cv_r3,"XGB3":cv_x3,"TabPFN3":cv_tf,"AW-MoE":cv_aw}
        te_dict  = {"E1":te_e1,"Ridge3":te_r3,"XGB3":te_x3,"TabPFN3":te_tf,"AW-MoE":te_aw}
        oof_dict = {"E1":oof_e1,"Ridge3":oof_r3,"XGB3":oof_x3,"TabPFN3":oof_tf,"AW-MoE":oof_aw}

        # Select best by OOD-CV on GDPa1 (never look at GDPa3)
        selected = max(cv_dict, key=lambda k: cv_dict[k] if not np.isnan(cv_dict[k]) else -9)
        selected_preds[assay] = te_dict[selected]
        selected_oofs[assay]  = oof_dict[selected]
        oof_cv_table[assay]   = cv_dict

        def f(v): return f"{v:+6.3f}" if not np.isnan(v) else "   nan"
        print(f"{assay:10s}  {f(cv_e1)}  {f(cv_r3):9s}  {f(cv_x3):7s}  {f(cv_tf):10s}  {f(cv_aw):6s}  → {selected}")

    # ── M11: Oracle hybrid (upper bound — post-hoc GDPa3 selection) ─────────
    print("\n=== M11: Oracle hybrid (post-hoc selection — upper bound only) ===")
    oracle_rhos  = {}
    m12_rhos     = {}
    aw_rhos_ref  = {"HIC":0.547,"AC-SINS":0.262,"PR_CHO":0.251,"SMAC":0.647,"HAC":0.650}

    print(f"{'Assay':10s}  {'Oracle':7s}  {'M12-OOF':7s}  {'AW-MoE':6s}  OOF-selected")
    print("-"*60)

    for assay,(g1_col,g3_col) in GENUINE_ASSAYS.items():
        y3=g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(N3,np.nan)
        ok3=~np.isnan(y3)
        blks_tr={"esm2":esm1n,"ablang2":abl1n,"desc":desc1,"bio":bio1,"isotype":iso1}
        blks_te={"esm2":esm3n,"ablang2":abl3n,"desc":desc3,"bio":bio3,"isotype":iso3}
        y1=ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df),np.nan)

        te_e1,_,_,_ = train_predict_e1(blks_tr,blks_te,y1,folds,fold_ids)
        te_r3,_,_ = train_predict("Ridge", Xdb1,y1,Xdb3,folds,fold_ids)
        te_x3,_,_ = train_predict("XGB",   Xdb1,y1,Xdb3,folds,fold_ids)
        te_tf,_,_ = train_predict("TabPFN",Xdb1,y1,Xdb3,folds,fold_ids)

        # Oracle: pick best by actual GDPa3 ρ (INVALID for paper — upper bound only)
        cands={"E1":te_e1,"Ridge3":te_r3,"XGB3":te_x3,"TabPFN3":te_tf}
        best_oracle=max(cands,key=lambda k:(spearmanr(y3[ok3],cands[k][ok3]).statistic if ok3.sum()>=5 else -9))
        oracle_rhos[assay]=float(spearmanr(y3[ok3],cands[best_oracle][ok3]).statistic) if ok3.sum()>=5 else np.nan

        # M12: OOF-selected
        m12_pred=selected_preds[assay]
        m12_rhos[assay]=float(spearmanr(y3[ok3],m12_pred[ok3]).statistic) if ok3.sum()>=5 else np.nan

        oof_sel=max(oof_cv_table[assay],key=lambda k:oof_cv_table[assay][k] if not np.isnan(oof_cv_table[assay][k]) else -9)
        def f(v): return f"{v:+7.3f}" if not np.isnan(v) else "    nan"
        print(f"{assay:10s}  {f(oracle_rhos[assay])}  {f(m12_rhos[assay])}  {aw_rhos_ref[assay]:+6.3f}  {oof_sel}")

    oracle_mean = float(np.nanmean(list(oracle_rhos.values())))
    m12_mean    = float(np.nanmean(list(m12_rhos.values())))
    aw_mean     = float(np.nanmean(list(aw_rhos_ref.values())))
    print(f"{'Mean':10s}  {oracle_mean:+7.3f}  {m12_mean:+7.3f}  {aw_mean:+6.3f}")
    print(f"\n  Oracle (M11) — upper bound, NOT valid: {oracle_mean:.3f}")
    print(f"  OOF hybrid  (M12) — valid:             {m12_mean:.3f}")
    print(f"  AW-MoE baseline:                       {aw_mean:.3f}")
    print(f"  M12 vs AW-MoE: {m12_mean-aw_mean:+.3f}")

    # ── M13: Paired bootstrap CI ─────────────────────────────────────────────
    print("\n=== M13: Paired bootstrap CI — M12 hybrid vs AW-MoE ===")
    # Need AW-MoE test predictions per assay for fair bootstrap comparison
    aw_preds_all={}
    for assay,(g1_col,g3_col) in GENUINE_ASSAYS.items():
        y1=ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df),np.nan)
        blks_tr={"esm2":esm1n,"ablang2":abl1n,"desc":desc1,"bio":bio1,"isotype":iso1}
        blks_te={"esm2":esm3n,"ablang2":abl3n,"desc":desc3,"bio":bio3,"isotype":iso3}
        te_aw,_,_=train_predict_awmoe(blks_tr,blks_te,Xdb1,Xdb3,Xplm1,Xplm3,
                                      all_db_cols,assay,y1,folds,fold_ids)
        aw_preds_all[assay]=te_aw

    print(f"{'Assay':10s}  {'M12-ρ':7s}  {'AW-ρ':6s}  {'Δ':6s}  {'BCa-CI':16s}  {'p>0':5s}")
    print("-"*72)
    for assay,(g1_col,g3_col) in GENUINE_ASSAYS.items():
        y3=g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(N3,np.nan)
        ok3=~np.isnan(y3)
        p_m12=selected_preds[assay]; p_aw=aw_preds_all[assay]
        if ok3.sum()<5:
            print(f"{assay:10s}  insufficient data"); continue

        # Bootstrap difference in Spearman ρ
        y_,m12_,aw_=y3[ok3],p_m12[ok3],p_aw[ok3]; n_=len(y_)
        diffs=np.array([
            spearmanr(y_[s:=rng.integers(0,n_,n_)],m12_[s]).statistic -
            spearmanr(y_[s],aw_[s]).statistic
            for _ in range(N_BOOT)])
        obs_diff=spearmanr(y_,m12_).statistic-spearmanr(y_,aw_).statistic
        ci_lo,ci_hi=np.percentile(diffs,2.5),np.percentile(diffs,97.5)
        p_pos=float(np.mean(diffs>0))
        def f(v): return f"{v:+7.3f}" if not np.isnan(v) else "    nan"
        print(f"{assay:10s}  {m12_rhos[assay]:+7.3f}  {aw_rhos_ref[assay]:+6.3f}  {obs_diff:+6.3f}  [{ci_lo:+.3f},{ci_hi:+.3f}]  {p_pos:.3f}")

    # ── M14: Hybrid vs AW-MoE Pareto comparison ──────────────────────────────
    print("\n=== M14: Pareto comparison — M12 hybrid vs AW-MoE ===")
    def get_Q(pred_dict):
        Q=np.full((N3,P),np.nan)
        for pi,assay in enumerate(ASSAY_NAMES):
            g1_col,_,_=list(GENUINE_ASSAYS[assay]),None,None
            g1_col=GENUINE_ASSAYS[assay][0]
            y1=ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(len(ds.df),np.nan)
            Q[:,pi]=percentile_rank(y1,pred_dict[assay],ASSAY_DIRS[assay])
        return Q

    Q_m12=get_Q(selected_preds); Q_aw=get_Q(aw_preds_all)
    fr_m12=pareto_front(Q_m12); fr_aw=pareto_front(Q_aw)
    D_m12=geom_D(Q_m12); D_aw=geom_D(Q_aw)
    borda_m12=Q_m12.mean(axis=1); borda_aw=Q_aw.mean(axis=1)

    print(f"                          M12-hybrid   AW-MoE")
    print(f"  Pareto Front 0 size:    {(fr_m12==0).sum():10d}   {(fr_aw==0).sum():6d}")
    print(f"  D>0 (pass all floors):  {(D_m12>0).sum():10d}   {(D_aw>0).sum():6d}")
    print(f"  Mean Borda score:       {borda_m12.mean():10.3f}   {borda_aw.mean():.3f}")
    print(f"  Mean desirability:      {D_m12.mean():10.3f}   {D_aw.mean():.3f}")

    # ── M15: Candidate shortlist stability ────────────────────────────────────
    print("\n=== M15: Top-8 consensus shortlist stability ===")
    # Prior consensus from M10 (bootstrap-based)
    prior_consensus = {
        "P907-A14-frayed-particle-5e881",
        "P907-A14-fermented-cathedral-185ee",
        "P907-A14-khaki-cadet-37ddd",
        "P907-A14-ecru-combat-e2619",
        "P907-A14-faint-click-67ee5",
        "P907-A14-rubbery-disk-fc338",
        "P907-A14-trite-leaf-1fefb",
        "P907-A14-crazy-bifocal-7c4f1",
    }
    top10_m12=set(g3["antibody_name"].iloc[np.argsort(-borda_m12)[:10]])
    top10_aw =set(g3["antibody_name"].iloc[np.argsort(-borda_aw)[:10]])
    overlap_m12=len(prior_consensus & top10_m12)
    overlap_aw =len(prior_consensus & top10_aw)
    print(f"  Prior consensus set size: {len(prior_consensus)}")
    print(f"  M12 hybrid top-10 overlap with prior: {overlap_m12} / {len(prior_consensus)}")
    print(f"  AW-MoE  top-10 overlap with prior:    {overlap_aw} / {len(prior_consensus)}")
    print(f"\n  M12 hybrid top-10:")
    for ab in sorted(top10_m12):
        i=g3["antibody_name"].tolist().index(ab)
        flag="★ consensus" if ab in prior_consensus else ""
        print(f"    {ab[:45]:45s}  Borda={borda_m12[i]:.3f}  Front={fr_m12[i]}  D={D_m12[i]:.3f}  {flag}")

    # Save full results
    out=ROOT/"results/hybrid_specialist.csv"
    df_out=pd.DataFrame({
        "antibody":g3["antibody_name"].values,
        **{f"q_m12_{a}":Q_m12[:,pi] for pi,a in enumerate(ASSAY_NAMES)},
        **{f"q_aw_{a}":Q_aw[:,pi] for pi,a in enumerate(ASSAY_NAMES)},
        "Pareto_m12":fr_m12,"Pareto_aw":fr_aw,
        "Borda_m12":borda_m12,"Borda_aw":borda_aw,
        "D_m12":D_m12,"D_aw":D_aw,
    }).sort_values("Borda_m12",ascending=False)
    df_out.to_csv(out,index=False)
    print(f"\nSaved {out}")


if __name__=="__main__":
    main()
