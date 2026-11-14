"""Numerical refresh for existing panels; retains their original assay axes."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,pickle
import numpy as np,pandas as pd
from scipy.stats import spearmanr,bootstrap
from threadpoolctl import threadpool_limits
H=Path(__file__).parent;ROOT=H.parents[1];VROOT=H.parent/'primary';E=H/'evidence';sys.path.insert(0,str(VROOT))
import verify_primary as V
with open(VROOT/'evidence/fitted_primary.pkl','rb') as f:Z=pickle.load(f)
B1=Z['B1'];B3=Z['B3'];folds=Z['folds'];cols=Z['descriptor_columns']+Z['bio_columns'];df=Z['train_df'];g3=Z['g3']
def main():
 loo=[];learning=[];sig=[]
 for a in ['Tm2','Titer']:
  y=df[V.ASSAYS[a][0]].to_numpy(float);yt=g3[V.ASSAYS[a][1]].to_numpy(float);ok=np.isfinite(yt);pred=Z['models'][a]['pred'];full=spearmanr(yt[ok],pred[ok]).statistic
  for drop in B1:
   b1={k:(v[:,:0] if k==drop else v) for k,v in B1.items()};b3={k:(v[:,:0] if k==drop else v) for k,v in B3.items()};cs=([] if drop=='desc' else Z['descriptor_columns'])+([] if drop=='bio' else Z['bio_columns'])
   m=V.fit_model(b1,b3,y,folds,a,cs);rho=spearmanr(yt[ok],m['pred'][ok]).statistic;loo.append(dict(assay=a,dropped=drop,full_rho=full,reduced_rho=rho,delta=full-rho))
  for fraction in [.2,.4,.6,.8,1.]:
   for fold in sorted(set(folds)):
    tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);tr=np.random.default_rng(42+int(fold)).permutation(tr)[:max(20,int(len(tr)*fraction))]
    m=V.fit_model({k:v[tr] for k,v in B1.items()},{k:v[te] for k,v in B1.items()},y[tr],folds[tr],a,cols);o=np.isfinite(y[te]);r=spearmanr(y[te][o],m['pred'][o]).statistic;learning.append(dict(assay=a,fraction=fraction,fold=int(fold),rho=r))
   print(a,'learning',fraction,flush=True)
  for b in ['esm2','ablang2']:
   pipe=V.AW._head(B1[b]);pipe.fit(B1[b][np.isfinite(y)],y[np.isfinite(y)]);p=pipe.predict(B3[b])[ok]
   def statistic(y,p1,p2):return spearmanr(y,p1).statistic-spearmanr(y,p2).statistic
   bs=bootstrap((yt[ok],pred[ok],p),statistic,paired=True,vectorized=False,n_resamples=2000,method='BCa',random_state=42);sig.append(dict(assay=a,baseline=b,delta=statistic(yt[ok],pred[ok],p),ci_lo=bs.confidence_interval.low,ci_hi=bs.confidence_interval.high))
 for filename,new in [('primary_source_ablation.csv',loo),('primary_learning_folds.csv',learning),('primary_paired_significance.csv',sig)]:
  old=pd.read_csv(VROOT/'evidence'/filename);pd.concat([old,pd.DataFrame(new)],ignore_index=True).to_csv(E/filename,index=False)
 cv=pd.read_csv(VROOT/'evidence/primary_cv_folds.csv');ex=pd.read_csv(VROOT/'evidence/extension_cv_folds.csv');ex['split']='cluster_isotype';cv=pd.concat([cv,ex],ignore_index=True);cv['assay']=cv.assay.replace({'AC-SINS':'AC-SINS_pH7.4'});cv=cv.groupby(['split','assay']).rho.agg(['mean','std']).reset_index();cv['model']='AbDev-X';cv.to_csv(E/'awmoe_cv_errorbars.csv',index=False)
 old=pd.read_csv(ROOT/'results/cv_errorbars.csv');old=old[old.model!='AbDev-X'];pd.concat([old,cv[cv.split=='cluster_isotype']],ignore_index=True).to_csv(E/'cv_errorbars.csv',index=False)
 for f in ['primary_reproduction.csv','primary_component_reproduction.csv']:
  pd.read_csv(VROOT/'evidence'/f).to_csv(E/f,index=False)
 print('Prepared verified numbers on the original panels\' assay axes.',flush=True)
if __name__=='__main__':
 with threadpool_limits(1):main()
