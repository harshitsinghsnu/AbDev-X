import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import inspect,pickle
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from compute_uncertainty import *
def legacy_extensions():
 if (E/'extension_metric_ci.csv').exists():return
 import gdpa3_eval as GE,soft_moe_gating as SM,physics_constrained as PH
 ds0,_,oldb,_=GE.build_blocks();ctx=SM.build_gate_context(ds0)
 src=inspect.getsource(SM.run_soft_moe).replace('return rho_rows, gate_rows','return rho_rows, gate_rows, y_m, oof_ridge, oof_sg, oof_per_block, all_wts, block_names')
 exec(compile(src,'recover_soft_gate_predictions.py','exec'),vars(SM));rows=[];gates=[]
 for a in COMP:
  rr,gr,y,ridge,gate,blockpred,w,blocknames=SM.run_soft_moe(a,ds0,oldb,ctx);best=max(blockpred,key=lambda b:spearmanr(y,blockpred[b]).statistic)
  for name,p in [('ridge_stacking',ridge),('soft_moe_gate',gate),('best_block_oracle',blockpred[best])]:
   pt,lo,hi=ci(y,p);rows.append(dict(experiment='soft',assay=a,method=name,rho=pt,ci_lo=lo,ci_hi=hi))
  ix=np.random.RandomState(42).randint(0,len(w),size=(2000,len(w)));means=w[ix].mean(1);lo,hi=np.percentile(means,[2.5,97.5],axis=0)
  for j,b in enumerate(blocknames):gates.append(dict(assay=a,block=b,mean=w[:,j].mean(),ci_lo=lo[j],ci_hi=hi[j]))
 captured=[]
 def track(y,p):
  val,lo,hi=ci(np.asarray(y),np.asarray(p));captured.append((val,lo,hi));return val
 PH.spear=track
 for name,fn in [('isotonic',PH.exp_isotonic_hic),('rank',PH.exp_rank_regularized),('multitask',PH.exp_multitask_shared)]:
  captured.clear();records=fn(ds0,oldb)
  for r in records:
   for col in ['rho','rho_individual','rho_shared']:
    if col not in r:continue
    val=r[col];matches=[z for z in captured if abs(z[0]-val)<1e-10];lo,hi=matches[0][1:] if matches else (np.nan,np.nan)
    rows.append(dict(experiment=name,assay=r['assay'],method=r.get('method',col),lambda_value=r.get('lambda',np.nan),rho=val,ci_lo=lo,ci_hi=hi))
 pd.DataFrame(rows).to_csv(E/'extension_metric_ci.csv',index=False);pd.DataFrame(gates).to_csv(E/'source_gate_ci.csv',index=False)
def transfer():
 if (E/'transfer_ci.csv').exists():return
 rows=[];d=pd.read_csv(R/'heldout_gdpa5/predictions.csv')
 for a in ['HIC','Tm2','AC-SINS_pH7.4','PR_CHO']:
  for m in ['AbDev-X','ESM2-Ridge','AbLang2-ElasticNet','Descriptor-Ridge']:
   pt,lo,hi=ci(d['true_'+a].to_numpy(float),d[m+'_'+a].to_numpy(float));rows.append(dict(experiment='VHH',assay=a,model=m,rho=pt,ci_lo=lo,ci_hi=hi))
 b=pd.read_csv(R/'heldout_gdpa4_bispecifics/arm_predictions.csv');raw=pd.read_csv(ROOT/'data/processed/gdpa4_bispecifics.csv');b=b.merge(raw,left_on='bsab_name',right_on='antibody_name')
 for a in ['HIC','AC-SINS_pH7.4','PR_CHO']:
  pt,lo,hi=ci(b['AC-SINS' if a=='AC-SINS_pH7.4' else a].to_numpy(float),b['avg_'+a].to_numpy(float));rows.append(dict(experiment='bispecific',assay=a,model='Arm average',rho=pt,ci_lo=lo,ci_hi=hi))
 # The recovered historical scatter baseline; combined-format refit predictions are unavailable.
 d=pd.read_csv(R/'heldout_configs/gdpa3_esm2_3b_eval.yaml/predictions.csv')
 for a in ['HIC','Tm2','PR_CHO','Titer']:
  pt,lo,hi=ci(d['true_'+a].to_numpy(float),d['AbDev-X_'+a].to_numpy(float));rows.append(dict(experiment='combined_baseline',assay=a,model='GDPa1-only',rho=pt,ci_lo=lo,ci_hi=hi))
 pd.DataFrame(rows).to_csv(E/'transfer_ci.csv',index=False)
def expert_refits():
 if (E/'expert_refit_weights.csv').exists():return
 rows=[];fs=Z['folds'];cols=Z['descriptor_columns']+Z['bio_columns']
 for a in PRIMARY:
  y=Z['train_df'][VP.ASSAYS[a][0]].to_numpy(float)
  for f in sorted(set(fs)):
   tr=np.flatnonzero(fs!=f);te=np.flatnonzero(fs==f);db=np.hstack([B1['desc'],B1['bio']]);pl=np.hstack([B1['esm2'],B1['ablang2']]);ids=sorted(set(fs[tr]))
   d=VP.AW.DStackExpert(pl[tr],pl[te],db[tr],db[te],fs[tr],ids);d.fit_full(y[tr])
   for name,w in zip(['PLM','Desc + bio'],d.meta.coef_):rows.append(dict(assay=a,fold=int(f),expert='D-Stack',group=name,weight=float(w)))
   m=VP.AW.FMoEExpert(db[tr],db[te],cols,a,fs[tr],ids);m.fit_full(y[tr])
   for (name,_),w in zip(m.expert_masks,m.weights):rows.append(dict(assay=a,fold=int(f),expert='F-MoE',group=name,weight=float(w)))
  pd.DataFrame(rows).to_csv(E/'expert_refit_weights.csv',index=False);print('Expert refits',a,flush=True)
def other_endpoints():
 if (E/'strategy_extension_ci.csv').exists():return
 rows=[]
 for a in list(VP.ASSAYS)[5:]:
  y=Z['train_df'][VP.ASSAYS[a][0]].to_numpy(float);yt=g3[VP.ASSAYS[a][1]].to_numpy(float);m=Z['models'][a]
  ps={name:m['expert_predictions'][:,j] for j,name in enumerate(['E1','A-MGFS','D-Stack','F-MoE'])};ps['AbDev-X']=m['pred']
  for name,block in [('ESM-2','esm2'),('AbLang2','ablang2'),('Descriptors','desc'),('Bio','bio')]:
   pipe=VP.AW._head(B1[block]);ok=np.isfinite(y);pipe.fit(B1[block][ok],y[ok]);ps[name]=pipe.predict(B3[block])
  for name,p in ps.items():
   pt,lo,hi=ci(yt,p);rows.append(dict(strategy=name,assay=a,rho=pt,ci_lo=lo,ci_hi=hi))
 pd.DataFrame(rows).to_csv(E/'strategy_extension_ci.csv',index=False)
 common=['Tm1','Tm2','Tonset','HIC','SEC','AC-SINS','PR_CHO'];ys=np.column_stack([g3[VP.ASSAYS[a][1]].to_numpy(float) for a in common]);ps=np.column_stack([Z['models'][a]['pred'] for a in common]);lo,hi=mean_ci(ys,ps)
 (E/'shared_seven_mean.json').write_text(json.dumps(dict(assays=common,mean=np.mean([ci(ys[:,j],ps[:,j])[0] for j in range(7)]),ci_lo=lo,ci_hi=hi),indent=2))
def pooling():
 if (E/'pooling_cv_folds.csv').exists():return
 from sklearn.pipeline import Pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import RidgeCV
 z=np.load(ROOT/'data/processed/esm_perres_150M.npz');rows=[]
 for mode in ['mean','cdr','framework']:
  parts=[]
  for ch in ['vh','vl']:
   emb=z[ch+'_e'];mask=z[ch+'_m'];cdr=z[ch+'_c'];w=mask if mode=='mean' else (cdr if mode=='cdr' else mask*(1-cdr));w=w/(w.sum(1,keepdims=True)+1e-9);parts.append(np.einsum('nl,nld->nd',w,emb))
  X=np.nan_to_num(np.hstack(parts));rows+=cvrows('pooling',mode,X,head=lambda:Pipeline([('s',StandardScaler()),('r',RidgeCV(alphas=np.logspace(-2,4,12)))]));print('Pooling',mode,flush=True)
 pd.DataFrame(rows).to_csv(E/'pooling_cv_folds.csv',index=False)
if __name__=='__main__':
 with threadpool_limits(1):
  for fn in [legacy_extensions,transfer,expert_refits,other_endpoints,pooling]:print(fn.__name__,flush=True);fn()
