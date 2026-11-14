"""Reproduce uncertainty from matched folds or paired antibody observations.
No SD across assays is substituted for SD of a cross-assay mean.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,pickle,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata,bootstrap,norm
from threadpoolctl import threadpool_limits
H=Path(__file__).parent;ROOT=H.parents[1];E=H/'evidence';R=ROOT/'results';V=H.parent/'primary'
os.chdir(ROOT);sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(V)]
import verify_primary as VP
from abdev.data.dataset import load_gdpa1
import exp_crossab as XC,exp_encoder_screen as ES,exp_struct_uncertainty as ST
COMP=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO','Titer'];PRIMARY=['HIC','AC-SINS','PR_CHO','SMAC','HAC']
with open(V/'evidence/fitted_primary.pkl','rb') as f:Z=pickle.load(f)
ds=load_gdpa1(str(ROOT/'configs/default.yaml'));g3=Z['g3'];B1=Z['B1'];B3=Z['B3']
def corr(y,p,axis=-1):
 a=rankdata(y,axis=axis);b=rankdata(p,axis=axis);a=a-a.mean(axis=axis,keepdims=True);b=b-b.mean(axis=axis,keepdims=True)
 return (a*b).sum(axis=axis)/np.sqrt((a*a).sum(axis=axis)*(b*b).sum(axis=axis))
def ci(y,p):
 ok=np.isfinite(y)&np.isfinite(p);y=np.asarray(y)[ok];p=np.asarray(p)[ok];n=len(y);obs=float(spearmanr(y,p).statistic)
 # Same resampling, strict bias correction, and acceleration as aw_moe.bca,
 # with batched ranks to avoid thousands of Python-level Spearman calls.
 ix=np.random.default_rng(42).integers(0,n,size=(2000,n));boot=corr(y[ix],p[ix],axis=1);jki=np.array([np.delete(np.arange(n),i) for i in range(n)]);jk=corr(y[jki],p[jki],axis=1);jm=jk.mean();acc=((jm-jk)**3).sum()/(6*((jm-jk)**2).sum()**1.5+1e-12);z0=norm.ppf(np.clip(np.mean(boot<obs),1e-6,1-1e-6))
 def adj(z):return norm.cdf(z0+(z0+z)/(1-acc*(z0+z)))
 lo,hi=np.percentile(boot,[100*adj(norm.ppf(.025)),100*adj(norm.ppf(.975))]);return obs,float(lo),float(hi)
def mean_ci(ys,ps):
 # Shared antibody resampling preserves the dependence of all assay scores.
 ix=np.random.RandomState(42).randint(0,len(ys),size=(2000,len(ys)));rs=[]
 for j in range(ys.shape[1]):
  y=ys[:,j][ix];p=ps[:,j][ix];ok=np.isfinite(y)&np.isfinite(p);y=np.where(ok,y,np.nan);p=np.where(ok,p,np.nan)
  a=rankdata(y,axis=1,nan_policy='omit');b=rankdata(p,axis=1,nan_policy='omit');a-=np.nanmean(a,axis=1,keepdims=True);b-=np.nanmean(b,axis=1,keepdims=True)
  rs.append(np.nansum(a*b,1)/np.sqrt(np.nansum(a*a,1)*np.nansum(b*b,1)))
 means=np.nanmean(rs,axis=0);return np.percentile(means,[2.5,97.5])
def cvrows(group,model,X,assays=COMP,splits=('cluster_isotype',),head=None):
 rows=[]
 for split in splits:
  fs=ds.folds(split)
  for a in assays:
   y=ds.y(a)
   for f in sorted(set(fs)):
    tr=(fs!=f)&np.isfinite(y);te=(fs==f)&np.isfinite(y)
    pipe=head() if head else ES._head(X.shape[1],high_dim=X.shape[1]>=480)
    pipe.fit(X[tr],y[tr]);pred=pipe.predict(X[te]);rows.append(dict(group=group,model=model,assay=a,split=split,fold=int(f),rho=spearmanr(y[te],pred).statistic))
 return rows
def screens():
 if (E/'screen_cv_folds.csv').exists():return
 rows=[];enc=ES.load_encoders(ds);enc['Descriptors']=B1['desc'];enc['Isotype']=B1['isotype'];teenc=ES.load_encoders_g3();pr=[];predictions={}
 for name,X in enc.items():
  rows+=cvrows('encoder',name,X);print('Encoder',name,flush=True)
  if name in teenc:
   ps=[];ys=[]
   for a in COMP:
    y=ds.y(a);pipe=ES._head(X.shape[1],high_dim=X.shape[1]>=480);ok=np.isfinite(y);pipe.fit(X[ok],y[ok]);p=pipe.predict(teenc[name]);yt=g3[VP.ASSAYS['AC-SINS' if a=='AC-SINS_pH7.4' else a][1]].to_numpy(float);pt,lo,hi=ci(yt,p);pr.append(dict(model=name,assay=a,rho=pt,ci_lo=lo,ci_hi=hi));ps.append(p);ys.append(yt)
   lo,hi=mean_ci(np.column_stack(ys),np.column_stack(ps));pr.append(dict(model=name,assay='MEAN',rho=np.mean([ci(y,p)[0] for y,p in zip(ys,ps)]),ci_lo=lo,ci_hi=hi));predictions[name]=np.column_stack(ps)
  pd.DataFrame(rows).to_csv(E/'screen_cv_progress.csv',index=False)
 cross=XC.build_crossab_features(ROOT/'data/processed/esm_perres_150M.npz',n1=246)
 for name in ['VH+VL_mean','CDR_mean','CrossAbSense']:
  rows+=cvrows('input',name,np.nan_to_num(cross[name]),splits=('cluster_isotype','random'),head=XC._pca_ridge);print('Input',name,flush=True)
 for name,key in [('AbLang2_baseline','AbLang2'),('ESM2_650M_baseline','ESM2_650M')]:rows+=cvrows('input',name,enc[key],splits=('cluster_isotype','random'),head=XC._pca_ridge)
 for name,X in {**ST.load_structural_blocks(ds),'ESM2_650M':enc['ESM2_650M'],'AbLang2':enc['AbLang2'],'Descriptors':B1['desc']}.items():
  rows+=cvrows('structure',name,X,head=lambda X=X:ST._head(high_dim=X.shape[1]>=480));print('Structure',name,flush=True)
 pd.DataFrame(rows).to_csv(E/'screen_cv_folds.csv',index=False);pd.DataFrame(pr).to_csv(E/'encoder_heldout_ci.csv',index=False)
 with open(E/'encoder_predictions.pkl','wb') as f:pickle.dump(predictions,f)
def strategies():
 if (E/'strategy_ci.csv').exists():return
 rows=[];predrows=[];allps={};allys=[]
 for a in PRIMARY:
  y=Z['train_df'][VP.ASSAYS[a][0]].to_numpy(float);yt=g3[VP.ASSAYS[a][1]].to_numpy(float)
  models=Z['models'][a];ps={name:models['expert_predictions'][:,j] for j,name in enumerate(['E1','A-MGFS','D-Stack','F-MoE'])};ps['AbDev-X']=models['pred']
  for name,block in [('ESM-2','esm2'),('AbLang2','ablang2'),('Descriptors','desc'),('Bio','bio')]:
   pipe=VP.AW._head(B1[block]);m=np.isfinite(y);pipe.fit(B1[block][m],y[m]);ps[name]=pipe.predict(B3[block])
  for name,p in ps.items():
   pt,lo,hi=ci(yt,p);rows.append(dict(strategy=name,assay=a,rho=pt,ci_lo=lo,ci_hi=hi));allps.setdefault(name,[]).append(p)
   for i in range(len(p)):predrows.append(dict(strategy=name,assay=a,antibody_name=g3.antibody_name.iloc[i],observed=yt[i],predicted=p[i]))
  allys.append(yt)
 for name,arr in allps.items():
  lo,hi=mean_ci(np.column_stack(allys),np.column_stack(arr));rows.append(dict(strategy=name,assay='MEAN',rho=np.mean([r['rho'] for r in rows if r['strategy']==name]),ci_lo=lo,ci_hi=hi))
 pd.DataFrame(rows).to_csv(E/'strategy_ci.csv',index=False);pd.DataFrame(predrows).to_csv(E/'strategy_predictions.csv',index=False);print('Strategy intervals complete',flush=True)
def proxy_shap():
 if (E/'proxy_shap_feature_ci.csv').exists():return
 from abdev.models.heads import LGBMHead
 import shap
 rows=[]
 for a in ['HIC','Tm2','AC-SINS_pH7.4']:
  y=ds.y(a);ok=np.isfinite(y);X=B1['desc'][ok];head=LGBMHead(random_state=42).fit(X,y[ok]);sv=shap.TreeExplainer(head.model_)(X).values;absval=np.abs(sv);ix=np.random.RandomState(42).randint(0,len(X),size=(2000,len(X)));boot=absval[ix].mean(axis=1);lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
  for j,f in enumerate(Z['descriptor_columns']):rows.append(dict(assay=a,feature=f,mean_abs_shap=absval[:,j].mean(),ci_lo=lo[j],ci_hi=hi[j]))
  pd.DataFrame(sv,columns=Z['descriptor_columns']).to_csv(E/f'proxy_shap_values_{a}.csv',index=False);print('Proxy SHAP',a,flush=True)
 pd.DataFrame(rows).to_csv(E/'proxy_shap_feature_ci.csv',index=False)
def regions():
 if (E/'regional_descriptor_ci.csv').exists():return
 import harn_experiments as HA
 cols=Z['descriptor_columns'];masks={r:HA.get_region_cols(cols,r) for r in HA.REGION_GROUPS};groups={'H1_global':None,'H2_CDR':['H_CDR1','H_CDR2','H_CDR3','L_CDR1','L_CDR2','L_CDR3'],'H3_FR':['H_FR','L_FR','VH_global','VL_global','Fv_global'],'H4_CDR_FR':list(HA.REGION_GROUPS),'H5_chain':list(HA.REGION_GROUPS),'H6_hier':list(HA.REGION_GROUPS)};rows=[]
 for model,regs in groups.items():
  mask=np.ones(len(cols),bool) if regs is None else np.logical_or.reduce([masks[r] for r in regs]);X=B1['desc'][:,mask];XT=B3['desc'][:,mask]
  for a in PRIMARY:
   y=Z['train_df'][VP.ASSAYS[a][0]].to_numpy(float);yt=g3[VP.ASSAYS[a][1]].to_numpy(float);p=HA.held_out_rho_pred(X,y,XT);pt,lo,hi=ci(yt,p);rows.append(dict(model=model,assay=a,rho=pt,ci_lo=lo,ci_hi=hi))
 pd.DataFrame(rows).to_csv(E/'regional_descriptor_ci.csv',index=False);print('Regional intervals complete',flush=True)
if __name__=='__main__':
 with threadpool_limits(1):
  for fn in [screens,strategies,proxy_shap,regions]:print(fn.__name__,flush=True);fn()
