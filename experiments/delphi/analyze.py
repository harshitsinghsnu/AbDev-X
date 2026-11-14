from pathlib import Path
import ast,json
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm,spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;P=O.parent
# Reuse the audited BCa implementation without executing the previous inference script.
tree=ast.parse((P/'frozen_delphi/evaluate.py').read_text(encoding='utf8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['rho_batch','limits']],type_ignores=[]),'bca_functions','exec'),globals())
def auc_batch(y,p):
 r=rankdata(p,axis=-1);n1=y.sum(-1);n0=y.shape[-1]-n1
 with np.errstate(divide='ignore',invalid='ignore'):return ((r*y).sum(-1)-n1*(n1+1)/2)/(n1*n0)
reg=pd.read_csv(O/'regression_predictions.csv');clas=pd.read_csv(O/'classification_predictions.csv');primary=pd.read_csv(P/'primary/evidence/primary_predictions.csv')
results=[];joins=[]
for task,data in [('regression',reg),('classification',clas)]:
 for assay in ['PR_CHO','SEC']:
  dd=data[(data.assay==assay)&(data.split=='GDPa3')];ab=primary[primary.assay==assay][['antibody_name','observed','predicted']] if task=='regression' else dd[dd.model=='AbDev-X binary-target'][['antibody_name','observed','predicted']]
  dp=dd if task=='regression' else dd[dd.model=='DELPHI native classifier']
  for sd in [42,43,44,-1]:
   de=dp[dp.seed==sd][['antibody_name','predicted']] if sd!=-1 else dp.groupby('antibody_name',as_index=False).predicted.mean()
   q=ab.merge(de,on='antibody_name',suffixes=('_abdevx','_delphi'),validate='one_to_one');q=q.dropna();assert len(q)==80
   q['task']=task;q['assay']=assay;q['seed']=sd;joins.append(q)
   y=q.observed.to_numpy(float);aa=q.predicted_abdevx.to_numpy(float);ddp=q.predicted_delphi.to_numpy(float)
   metric=rho_batch if task=='regression' else auc_batch;obs=np.array([metric(y,aa),metric(y,ddp)]);obs=np.r_[obs,obs[0]-obs[1]]
   rng=np.random.default_rng(42);boots=[]
   for j in range(40):
    ii=rng.integers(0,len(y),(500,len(y)));a=metric(y[ii],aa[ii]);d=metric(y[ii],ddp[ii]);boots.append(np.column_stack([a,d,a-d]))
   boot=np.concatenate(boots);jk=[]
   for j in range(len(y)):
    keep=np.arange(len(y))!=j;a=metric(y[keep],aa[keep]);d=metric(y[keep],ddp[keep]);jk.append([a,d,a-d])
   jk=np.array(jk)
   for j,name in enumerate(['AbDev-X','DELPHI','Difference']):
    lo,hi=limits(boot[:,j],obs[j],jk[:,j]);flo,fhi=limits(boot[:,j],obs[j],jk[:,j],.0125)
    results.append(dict(task=task,assay=assay,seed=sd,quantity=name,n=80,estimate=obs[j],ci_lo=lo,ci_hi=hi,family_ci_lo=flo,family_ci_hi=fhi))
pd.DataFrame(results).to_csv(O/'paired_results_all_seeds.csv',index=False);pd.concat(joins).to_csv(O/'paired_external_predictions.csv',index=False)
d=pd.DataFrame(results);main=d[d.seed==42];main.to_csv(O/'primary_seed42_results.csv',index=False)
plt.rcParams.update({'font.family':'Arial','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(183/25.4,5.1),layout='constrained')
for row,task in enumerate(['regression','classification']):
 for i,a in enumerate(['PR_CHO','SEC']):
  for j,(model,c) in enumerate([('AbDev-X','#1A237E'),('DELPHI','#D55E00')]):
   r=main[(main.task==task)&(main.assay==a)&(main.quantity==model)].iloc[0];axs[row,0].errorbar(r.estimate,i+(j-.5)*.16,xerr=[[r.estimate-r.ci_lo],[r.ci_hi-r.estimate]],fmt='o',c=c,capsize=2,label=model if i==0 else None)
  r=main[(main.task==task)&(main.assay==a)&(main.quantity=='Difference')].iloc[0];axs[row,1].errorbar(r.estimate,i,xerr=[[r.estimate-r.ci_lo],[r.ci_hi-r.estimate]],fmt='o',c='#263746',capsize=2)
 for ax in axs[row]:ax.set_yticks([0,1],['PR-CHO','SEC']);ax.set_ylim(1.5,-.5);ax.axvline(0,c='#aaaaaa',lw=.6)
 axs[row,0].legend(loc='lower left',fontsize=6);axs[row,0].set_xlabel('Spearman ρ; BCa 95% CI' if row==0 else 'ROC AUC; BCa 95% CI');axs[row,1].set_xlabel('AbDev-X − DELPHI; paired BCa 95% CI')
 axs[row,0].set_title(('A  Continuous targets: DELPHI regression adaptation' if row==0 else 'C  Common binary labels: native DELPHI classifier'),loc='left',fontsize=7,fontweight='bold');axs[row,1].set_title('B  Regression difference' if row==0 else 'D  Classification difference',loc='left',fontweight='bold',fontsize=7)
axs[1,0].lines[-1].remove()
axs[1,0].axvline(.5,c='#aaaaaa',lw=.6)
axs[1,0].set_xlim(0,1)
for ext in ['pdf','png','svg']:fig.savefig(O/f'same_data_comparison.{ext}',dpi=350)
plt.close(fig)
lines=['Matched-data comparison: same GDPa1 labels and outer folds; identical GDPa3 test antibodies.','Primary DELPHI seed42; seeds43/44 and mean-score ensemble are retained sensitivity analyses. No best-seed selection.',main.to_string(index=False),'No paired 95% interval excludes zero unless explicitly shown by the difference rows. Simultaneous98.75% intervals account for four endpoint/task contrasts.','Regression uses a changed one-output/MSE DELPHI head; classification uses native two-output CE DELPHI and binary-target four-expert AbDev-X.','Cutoffs: PR-CHO PASS<0.27; SEC monomer PASS>=90. SEC criteria do not reproduce unavailable IPI chromatogram calls.','Classification has 12 PR-CHO failures and9 SEC failures in the test set. All uncertainty is conditional on fitted models; three seeds show initialization sensitivity.']
(O/'report.txt').write_text('\n\n'.join(lines),encoding='utf8');print(main.to_string(index=False))
