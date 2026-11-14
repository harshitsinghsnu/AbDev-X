"""Reproduce primary fitted functions, save predictions, and exact input-space IG.

Uses the original expert classes without changing research code or result files.
Every fitted expert is affine in its cached inputs. Collapsing preprocessing and
ridge heads gives exact IG and interventional linear SHAP relative to training means.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='1'
import sys,json,hashlib,pickle
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
HERE=Path(__file__).parent; ROOT=HERE.parents[1]; OUT=HERE/'evidence'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import aw_moe as AW
from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix
from sklearn.linear_model import Ridge

ASSAYS={'HIC':('HIC','hic_rt_avg'),'AC-SINS':('AC-SINS_pH7.4','acsins_dLmax_ph7.4_avg'),'PR_CHO':('PR_CHO','polyreactivity_prscore_cho_avg'),'SMAC':('SMAC','smac_rt_avg'),'HAC':('HAC','hac_rt_avg'),'Tm1':('Tm1','tm1_nanodsf_avg'),'Tm2':('Tm2','tm2_nanodsf_avg'),'Tonset':('Tonset','tonset_nanodsf_avg'),'SEC':('SEC %Monomer','sec_%monomer_avg'),'Titer':('Titer','titer_avg')}

def affine(pipe):
    w=np.asarray(pipe.named_steps['r'].coef_).copy(); b=float(pipe.named_steps['r'].intercept_)
    if 'p' in pipe.named_steps:
        p=pipe.named_steps['p']; w=w@p.components_; b-=w@p.mean_
    s=pipe.named_steps['s']; w=w/s.scale_; b-=w@s.mean_
    return w,b

def fit_model(B1,B3,y,folds,assay,cols):
    ids=sorted(set(folds)); bt={k:v for k,v in B1.items() if v.shape[1] and (k!='isotype' or assay not in AW.ISO_INELIGIBLE)}; be={k:B3[k] for k in bt}
    db1=np.hstack([B1['desc'],B1['bio']]); db3=np.hstack([B3['desc'],B3['bio']]); pl1=np.hstack([B1['esm2'],B1['ablang2']]); pl3=np.hstack([B3['esm2'],B3['ablang2']])
    experts=[AW.E1Expert(bt,be,folds,ids,assay),AW.MGFSExpert(db1,db3,cols,assay),AW.DStackExpert(pl1,pl3,db1,db3,folds,ids),AW.FMoEExpert(db1,db3,cols,assay,folds,ids)]
    for e in experts:e.fit_full(y)
    ps=np.column_stack([e.predict() for e in experts]); pred=ps.mean(1)
    weights={k:np.zeros(v.shape[1]) for k,v in B1.items()}; bias=0.
    def add(pipe,blocks,scale=1.,mask=None):
        nonlocal bias
        w,b=affine(pipe);bias+=scale*b
        if mask is not None:
            z=np.zeros(sum(B1[k].shape[1] for k in blocks));z[mask]=w;w=z
        j=0
        for k in blocks:
            n=B1[k].shape[1];weights[k]+=scale*w[j:j+n];j+=n
    e1,e2,e3,e4=experts
    add(e1.pipe,[e1.best_blk],.25)
    cfg=AW.MGFS_MASKS.get(assay,{'inc':[],'exc':[]});m=AW._mask_cols(cols,cfg['inc'],cfg['exc'])
    if m.sum()<5:m=np.ones(len(cols),bool)
    add(e2.pipe,['desc','bio'],.25,m)
    add(e3.p1,['esm2','ablang2'],.25*e3.meta.coef_[0]);add(e3.p2,['desc','bio'],.25*e3.meta.coef_[1]);bias+=.25*e3.meta.intercept_
    for pipe,(_,mask),w in zip(e4.pipes,e4.expert_masks,e4.weights):add(pipe,['desc','bio'],.25*w,mask)
    linear=bias+sum(B3[k]@w for k,w in weights.items())
    assert np.max(np.abs(linear-pred))<1e-5,(assay,np.max(np.abs(linear-pred)))
    return {'experts':experts,'weights':weights,'bias':bias,'pred':pred,'expert_predictions':ps,'best_source':e1.best_blk,'max_linear_error':float(np.max(np.abs(linear-pred)))}

def main():
    ds=load_gdpa1(str(ROOT/'configs/default.yaml'));g3=pd.read_csv(ROOT/'data/processed/gdpa3_table.csv');folds=ds.folds('cluster_isotype')
    esmfile=next(p for p in (ROOT/'data/processed').glob('esm2_*.npz') if np.load(p)['emb'].shape==(len(ds.df),2560))
    d1=build_descriptor_matrix(ds.heavy,ds.light);d3=build_descriptor_matrix(g3.vh,g3.vl)
    bio1,bcols=AW._bio(ROOT/'external/abdev-benchmark/data/features/processed_features/GDPa1',ds.df.antibody_name)
    bio3,_=AW._bio(ROOT/'external/abdev-benchmark/data/features/processed_features/heldout_test',g3.antibody_name,bcols)
    B1={'esm2':np.load(esmfile)['emb'],'ablang2':np.load(ROOT/'data/processed/ablang2_gdpa1.npz')['emb'],'desc':d1.to_numpy(float),'bio':bio1,'isotype':AW._iso(ds.df)}
    B3={'esm2':np.load(ROOT/'data/processed/gdpa3_esm.npz')['emb'],'ablang2':np.load(ROOT/'data/processed/gdpa3_ablang2.npz')['emb'],'desc':d3.to_numpy(float),'bio':bio3,'isotype':AW._iso(g3)}
    B1={k:np.nan_to_num(v) for k,v in B1.items()};B3={k:np.nan_to_num(v) for k,v in B3.items()}
    cols=list(d1.columns)+bcols;models={};rows=[];pdrows=[];igs=[];feature_rows=[]
    recorded=pd.read_csv(ROOT/'results/awmoe_all9.csv').set_index('assay')
    for a,(c1,c3) in ASSAYS.items():
        print('Fitting',a,flush=True); y=ds.df[c1].to_numpy(float);yt=g3[c3].to_numpy(float)
        model=fit_model(B1,B3,y,folds,a,cols);models[a]=model; pred=model['pred'];ok=np.isfinite(yt);rho=float(spearmanr(yt[ok],pred[ok]).statistic)
        lo,hi=AW.bca(yt,pred,np.random.default_rng(42),2000)
        rows.append(dict(assay=a,rho=rho,n=int(ok.sum()),ci_lo=lo,ci_hi=hi,recorded_rho=float(recorded.loc[a,'rho']),difference=rho-recorded.loc[a,'rho'],best_source=model['best_source'],linear_error=model['max_linear_error']))
        for i,name in enumerate(g3.antibody_name):pdrows.append(dict(assay=a,antibody_name=name,observed=yt[i],predicted=pred[i],**dict(zip(['E1','MGFS','DStack','FMoE'],model['expert_predictions'][i]))))
        base=model['bias'];total=np.zeros(len(g3))
        for k,w in model['weights'].items():
            mu=B1[k][np.isfinite(y)].mean(0);phi=(B3[k]-mu)*w;total+=phi.sum(1);base+=mu@w
            for i,name in enumerate(g3.antibody_name):igs.append(dict(assay=a,antibody_name=name,block=k,contribution=phi[i].sum()))
            fcols=list(d1.columns) if k=='desc' else bcols if k=='bio' else AW.ISO_COLS if k=='isotype' else [f'{k}_{j}' for j in range(len(w))]
            for j,f in enumerate(fcols):feature_rows.append(dict(assay=a,block=k,feature=f,coefficient=w[j],mean_abs_ig=np.abs(phi[:,j]).mean(),baseline=mu[j]))
        assert np.max(np.abs(total+base-pred))<1e-7
        model['baseline_prediction']=float(base)
        print(a,'rho',rho,'recorded',recorded.loc[a,'rho'],'source',model['best_source'],flush=True)
        pd.DataFrame(rows).to_csv(OUT/'primary_reproduction.csv',index=False)
    pd.DataFrame(pdrows).to_csv(OUT/'primary_predictions.csv',index=False);pd.DataFrame(igs).to_csv(OUT/'primary_block_IG.csv',index=False);pd.DataFrame(feature_rows).to_csv(OUT/'primary_feature_IG.csv',index=False)
    with open(OUT/'fitted_primary.pkl','wb') as f:pickle.dump({'models':models,'B1':B1,'B3':B3,'descriptor_columns':list(d1.columns),'bio_columns':bcols,'g3':g3,'folds':folds,'train_df':ds.df},f)
    manifest={'primary_code':'scripts/aw_moe.py; scripts/final_evaluation.py','esm_cache':str(esmfile),'dimensions':{k:v.shape[1] for k,v in B1.items()},'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/aw_moe.py',ROOT/'scripts/final_evaluation.py',esmfile,ROOT/'data/processed/gdpa3_esm.npz']},'interpretation':'Exact input-feature IG of primary fitted affine predictor, training-mean baseline. Equivalent to interventional linear SHAP, not conditional/correlation-aware SHAP.'}
    (OUT/'model_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
