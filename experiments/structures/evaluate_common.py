"""Controlled retrospective downstream comparison; never modifies primary results.

Original four-expert classes and training-only nested selection are reused.
Only bio57 is replaced by the fixed common structural panel. Missing structures
are excluded using one intersection across every named predictor in this run.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,pickle,json,argparse,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata,norm
from threadpoolctl import threadpool_limits
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(P.parent/'primary'),str(ROOT/'scripts'),str(ROOT/'src')]
from verify_primary import fit_model,ASSAYS

def rho_rows(y,p):
 a=rankdata(y,axis=-1);b=rankdata(p,axis=-1);a-=a.mean(-1,keepdims=True);b-=b.mean(-1,keepdims=True)
 return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1))

def interval(y,p,q=None,B=20000,seed=42,alpha=.05):
 """Paired antibody BCa interval for rho or difference of two correlations."""
 ok=np.isfinite(y)&np.isfinite(p)
 if q is not None:ok &=np.isfinite(q)
 y,p=y[ok],p[ok];q=q[ok] if q is not None else None;n=len(y)
 if n<5:return np.nan,np.nan
 def stat(a,b,c=None):return rho_rows(a,b)-(rho_rows(a,c) if c is not None else 0.)
 obs=stat(y,p,q);rng=np.random.default_rng(seed);values=[]
 for start in range(0,B,1000):
  ix=rng.integers(0,n,size=(min(1000,B-start),n));values.extend(stat(y[ix],p[ix],q[ix] if q is not None else None))
 boot=np.asarray(values);boot=boot[np.isfinite(boot)]
 jk=np.array([stat(np.delete(y,i),np.delete(p,i),np.delete(q,i) if q is not None else None) for i in range(n)])
 d=jk.mean()-jk;acc=(d**3).sum()/(6*(d*d).sum()**1.5+1e-15);z0=norm.ppf(np.clip(np.mean(boot<obs),1/(2*B),1-1/(2*B)))
 z=norm.ppf([alpha/2,1-alpha/2]);adjust=norm.cdf(z0+(z0+z)/(1-acc*(z0+z)))
 return tuple(np.quantile(boot,np.clip(adjust,0,1)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--predictors',nargs='+',default=['ABodyBuilder3','IgFold','ESMFold','AlphaFold3']);args=ap.parse_args()
 out=P/('comparison_'+'_'.join(args.predictors));out.mkdir(exist_ok=True)
 with open(P.parent/'primary/evidence/fitted_primary.pkl','rb') as f:cache=pickle.load(f)
 train,test=cache['train_df'],cache['g3'];tables={};featurecols=None
 for predictor in args.predictors:
  st=pd.read_csv(P/f'feature_status_{predictor}.csv')
  if len(st)!=326:raise ValueError(f'{predictor}: feature extraction incomplete ({len(st)}/326)')
  t=pd.read_csv(P/f'common_features_{predictor}.csv').set_index(['split','antibody_name']);cols=list(t.columns)
  if featurecols is None:featurecols=cols
  assert cols==featurecols
  if not np.isfinite(t.to_numpy(float)).all():raise ValueError('nonfinite features')
  tables[predictor]=t
 common=set.intersection(*(set(t.index) for t in tables.values()))
 i1=np.array([i for i,n in enumerate(train.antibody_name) if ('GDPa1',n) in common]);i3=np.array([i for i,n in enumerate(test.antibody_name) if ('heldout_test',n) in common])
 folds=np.asarray(cache['folds'])[i1];cols=cache['descriptor_columns']+featurecols
 predictions=[];scores=[];foldscores=[];saved={}
 for predictor,t in tables.items():
  B1={k:v[i1].copy() for k,v in cache['B1'].items()};B3={k:v[i3].copy() for k,v in cache['B3'].items()}
  B1['bio']=t.loc[[('GDPa1',n) for n in train.antibody_name.iloc[i1]]].to_numpy(float)
  B3['bio']=t.loc[[('heldout_test',n) for n in test.antibody_name.iloc[i3]]].to_numpy(float)
  for assay,(c1,c3) in ASSAYS.items():
   checkpoint=out/f'{predictor}_{assay}.npz';y=train[c1].to_numpy(float)[i1];yt=test[c3].to_numpy(float)[i3]
   if checkpoint.exists():
    z=np.load(checkpoint);pred=z['pred'];oof=z['oof']
   else:
    start=time.time();print('Fitting',predictor,assay,flush=True);oof=np.full(len(i1),np.nan)
    for fold in sorted(set(folds)):
     tr=folds!=fold;va=~tr
     m=fit_model({k:v[tr] for k,v in B1.items()},{k:v[va] for k,v in B1.items()},y[tr],folds[tr],assay,cols)
     oof[va]=m['pred']
    fitted=fit_model(B1,B3,y,folds,assay,cols);pred=fitted['pred']
    np.savez_compressed(checkpoint,pred=pred,oof=oof,train_indices=i1,test_indices=i3)
    print('Finished',predictor,assay,'seconds',round(time.time()-start,1),flush=True)
   saved[predictor,assay]=pred
   fs=[]
   for fold in sorted(set(folds)):
    ok=(folds==fold)&np.isfinite(y);r=float(spearmanr(y[ok],oof[ok]).statistic);fs.append(r)
    foldscores.append(dict(predictor=predictor,assay=assay,fold=int(fold),n=int(ok.sum()),rho=r))
   ok=np.isfinite(yt);lo,hi=interval(yt,pred)
   scores.append(dict(predictor=predictor,assay=assay,cv_mean=np.mean(fs),cv_sd=np.std(fs,ddof=1),heldout_rho=spearmanr(yt[ok],pred[ok]).statistic,ci_lo=lo,ci_hi=hi,n_train=int(np.isfinite(y).sum()),n_test=int(ok.sum())))
   predictions.extend(dict(predictor=predictor,assay=assay,antibody_name=test.antibody_name.iloc[i3[j]],observed=yt[j],predicted=pred[j]) for j in range(len(i3)))
   pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
 pd.DataFrame(predictions).to_csv(out/'heldout_predictions.csv',index=False);pd.DataFrame(foldscores).to_csv(out/'cv_fold_scores.csv',index=False)
 base=args.predictors[0];differences=[];ncontrasts=max(1,(len(args.predictors)-1)*len(ASSAYS))
 for predictor in args.predictors[1:]:
  for assay,(_,c3) in ASSAYS.items():
   y=test[c3].to_numpy(float)[i3];p=saved[predictor,assay];q=saved[base,assay];ok=np.isfinite(y)
   lo,hi=interval(y,p,q);slo,shi=interval(y,p,q,alpha=.05/ncontrasts)
   differences.append(dict(predictor=predictor,reference=base,assay=assay,delta_rho=spearmanr(y[ok],p[ok]).statistic-spearmanr(y[ok],q[ok]).statistic,ci_lo=lo,ci_hi=hi,bonferroni_lo=slo,bonferroni_hi=shi,n=int(ok.sum())))
 pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
 manifest=dict(variant='AbDev-X/common-structure20, not original bio57 primary',predictors=args.predictors,train_n=len(i1),test_n=len(i3),feature_columns=featurecols,common_train_names=train.antibody_name.iloc[i1].tolist(),common_test_names=test.antibody_name.iloc[i3].tolist(),folds=folds.tolist(),bootstrap_replicates=20000,comparison='retrospective; paired antibody BCa; Bonferroni sensitivity across all predictor-assay contrasts; not paired by family',original_primary='unchanged; must be separately labeled because bio feature definitions differ')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 plot(out,pd.DataFrame(scores),args.predictors)
 report=['# Matched structure-predictor comparison','',
 'This is a retrospective AbDev-X/common-structure20 experiment. It is not a replacement of the original primary model results. The other four representation blocks and four-expert training code are held fixed. All methods use the same common-complete-case antibodies and the same outer fold assignments.',
 '',f'Common cases: {len(i1)} GDPa1 training antibodies and {len(i3)} GDPa3 antibodies before assay-specific missing-label exclusions.',
 '', 'CV errors are fold SD; held-out errors are antibody-resampling BCa 95% intervals (20,000 replicates). Paired differences include a Bonferroni interval sensitivity across all predictor-assay contrasts. Bootstrap intervals do not account for training-set uncertainty or antibody family dependence.',
 '', '| Predictor | Assay | CV mean +/- SD | GDPa3 rho [95% CI] | Held-out n |','|---|---|---|---|---|']
 for r in scores:report.append(f"| {r['predictor']} | {r['assay']} | {r['cv_mean']:.3f} +/- {r['cv_sd']:.3f} | {r['heldout_rho']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['n_test']} |")
 report+=['','## Interpretation limits','',
 'AF3 uses seed 42, 10 recycles, five diffusion samples, and official top-ranking-score selection, with empty MSAs and no templates. This is an AF3 single-sequence condition, not the full database-assisted AF3 pipeline. ESMFold uses VH + 25 glycines + VL and the documented chain index offset, with linker atoms excluded. IgFold uses the four-model ensemble without optional refinement. Cached ABodyBuilder3 structures are the baseline; their original inference runtime is unavailable. The same exact-chain check, heavy-atom completion policy, and feature definitions are applied to all sources.',
 '', 'The primary 57-feature block combines TAP, MOE, Aggrescan3D, and AntiFold caches. Its complete regeneration pipeline is unavailable here. Therefore these results assess a common open 20-feature structural panel, not an exact structure-predictor swap within the original bio57 model. No structural RMSD or optimal-predictor claim is supported without experimental reference structures or independent confirmatory evaluation.']
 (out/'RESULTS.md').write_text('\n'.join(report),encoding='utf8')
 print('COMPLETE',out,flush=True)

def plot(out,df,predictors):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,2,figsize=(7.2,4),layout='constrained');assays=list(ASSAYS);colors=['#2474A7','#D58A30','#55A080','#9E65A5','#888888']
 for ax,kind in zip(axs,['cv','heldout']):
  for i,p in enumerate(predictors):
   d=df[df.predictor==p].set_index('assay').loc[assays];yy=np.arange(10)+(i-(len(predictors)-1)/2)*.14
   x=d.cv_mean.to_numpy() if kind=='cv' else d.heldout_rho.to_numpy()
   err=d.cv_sd.to_numpy() if kind=='cv' else np.vstack([np.maximum(0,x-d.ci_lo),np.maximum(0,d.ci_hi-x)])
   ax.errorbar(x,yy,xerr=err,fmt='o',ms=3,capsize=2,lw=.8,label=p,color=colors[i])
  ax.set(yticks=np.arange(10),yticklabels=assays,xlabel='Spearman correlation',title='CV mean and fold SD' if kind=='cv' else 'GDPa3 correlation and BCa 95% CI');ax.invert_yaxis();ax.axvline(0,color='.7',lw=.6)
 axs[0].legend(fontsize=7,loc='lower left');fig.savefig(out/'common_structure_comparison.pdf');fig.savefig(out/'common_structure_comparison.png',dpi=200);plt.close(fig)

if __name__=='__main__':
 os.chdir(ROOT)
 with threadpool_limits(limits=1):main()
