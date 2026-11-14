"""Matched primary-model CV, learning curves, refitted source removal and paired CI."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,pickle
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,bootstrap
from threadpoolctl import threadpool_limits
import verify_primary as V
H=Path(__file__).parent;O=H/'evidence';PRIMARY=list(V.ASSAYS)[:5]
with open(O/'fitted_primary.pkl','rb') as f:Z=pickle.load(f)
B1=Z['B1'];B3=Z['B3'];folds=Z['folds'];df=Z['train_df'];g3=Z['g3'];cols=Z['descriptor_columns']+Z['bio_columns']
def run():
 rows=[];predrows=[]
 for split in ['random','cluster','cluster_isotype']:
  ds=V.load_gdpa1(str(V.ROOT/'configs/default.yaml'));fs=ds.folds(split)
  for a in PRIMARY:
   y=df[V.ASSAYS[a][0]].to_numpy(float)
   for fold in sorted(set(fs)):
    tr=np.flatnonzero(fs!=fold);te=np.flatnonzero(fs==fold);m=V.fit_model({k:v[tr] for k,v in B1.items()},{k:v[te] for k,v in B1.items()},y[tr],folds[tr],a,cols)
    ok=np.isfinite(y[te]);r=spearmanr(y[te][ok],m['pred'][ok]).statistic;rows.append(dict(split=split,assay=a,fold=int(fold),rho=r))
    if split=='cluster_isotype':
     for j,i in enumerate(te):predrows.append(dict(assay=a,antibody_name=df.antibody_name.iloc[i],observed=y[i],predicted=m['pred'][j],fold=int(fold)))
   print('CV',split,a,flush=True)
   pd.DataFrame(rows).to_csv(O/'primary_cv_folds.csv',index=False)
 pd.DataFrame(predrows).to_csv(O/'primary_cv_predictions.csv',index=False)
 loo=[]
 for a in PRIMARY:
  y=df[V.ASSAYS[a][0]].to_numpy(float);yt=g3[V.ASSAYS[a][1]].to_numpy(float);ok=np.isfinite(yt);full=spearmanr(yt[ok],Z['models'][a]['pred'][ok]).statistic
  for drop in B1:
   b1={k:(v[:,:0] if k==drop else v) for k,v in B1.items()};b3={k:(v[:,:0] if k==drop else v) for k,v in B3.items()};cs=([] if drop=='desc' else Z['descriptor_columns'])+([] if drop=='bio' else Z['bio_columns'])
   m=V.fit_model(b1,b3,y,folds,a,cs);rho=spearmanr(yt[ok],m['pred'][ok]).statistic;loo.append(dict(assay=a,dropped=drop,full_rho=full,reduced_rho=rho,delta=full-rho))
  print('LOO',a,flush=True);pd.DataFrame(loo).to_csv(O/'primary_source_ablation.csv',index=False)
 learning=[]
 for a in PRIMARY:
  y=df[V.ASSAYS[a][0]].to_numpy(float)
  for frac in [.2,.4,.6,.8,1.]:
   for fold in sorted(set(folds)):
    tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);rng=np.random.default_rng(42+int(fold));tr=rng.permutation(tr)[:max(20,int(len(tr)*frac))]
    m=V.fit_model({k:v[tr] for k,v in B1.items()},{k:v[te] for k,v in B1.items()},y[tr],folds[tr],a,cols);ok=np.isfinite(y[te]);rho=spearmanr(y[te][ok],m['pred'][ok]).statistic
    learning.append(dict(assay=a,fraction=frac,fold=int(fold),n_train=int(np.isfinite(y[tr]).sum()),rho=rho))
   print('Learning',a,frac,flush=True);pd.DataFrame(learning).to_csv(O/'primary_learning_folds.csv',index=False)
 sig=[]
 for a in PRIMARY:
  y=df[V.ASSAYS[a][0]].to_numpy(float);yt=g3[V.ASSAYS[a][1]].to_numpy(float);ok=np.isfinite(yt);primary=Z['models'][a]['pred'][ok]
  for block in ['esm2','ablang2']:
   m=V.AW._head(B1[block]);m.fit(B1[block][np.isfinite(y)],y[np.isfinite(y)]);p=m.predict(B3[block])[ok]
   def stat(y,p1,p2):return spearmanr(y,p1).statistic-spearmanr(y,p2).statistic
   bs=bootstrap((yt[ok],primary,p),stat,paired=True,vectorized=False,n_resamples=2000,method='BCa',random_state=42)
   sig.append(dict(assay=a,baseline=block,delta=stat(yt[ok],primary,p),ci_lo=bs.confidence_interval.low,ci_hi=bs.confidence_interval.high))
 pd.DataFrame(sig).to_csv(O/'primary_paired_significance.csv',index=False)
 # Matched single-encoder outer-fold baselines used by the main results figure.
 baselines=[]
 for a in PRIMARY:
  y=df[V.ASSAYS[a][0]].to_numpy(float)
  for block in ['esm2','ablang2']:
   for fold in sorted(set(folds)):
    tr=(folds!=fold)&np.isfinite(y);te=(folds==fold)&np.isfinite(y);pipe=V.AW._head(B1[block][tr]);pipe.fit(B1[block][tr],y[tr]);r=spearmanr(y[te],pipe.predict(B1[block][te])).statistic
    baselines.append(dict(assay=a,baseline=block,fold=int(fold),rho=r))
 pd.DataFrame(baselines).to_csv(O/'primary_baseline_cv_folds.csv',index=False)
 extension=[]
 for a in list(V.ASSAYS)[5:]:
  y=df[V.ASSAYS[a][0]].to_numpy(float)
  for fold in sorted(set(folds)):
   tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);m=V.fit_model({k:v[tr] for k,v in B1.items()},{k:v[te] for k,v in B1.items()},y[tr],folds[tr],a,cols);ok=np.isfinite(y[te]);extension.append(dict(assay=a,fold=int(fold),rho=spearmanr(y[te][ok],m['pred'][ok]).statistic))
 pd.DataFrame(extension).to_csv(O/'extension_cv_folds.csv',index=False)
if __name__=='__main__':
 with threadpool_limits(1):run()
