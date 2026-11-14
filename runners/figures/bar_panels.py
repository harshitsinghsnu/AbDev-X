from pathlib import Path
import re,shutil,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
P=Path(__file__).resolve().parents[2]/'experiments';ROOT=P.parent;R=ROOT/'results';E=P/'evaluation/evidence';OLD=P/'cartoons';O=ROOT/'outputs';F=O/'figures'
F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
COL=['#2474A7','#D58A30','#55A080','#9E65A5','#C15A65','#68858A'];PRIMARY=['HIC','AC-SINS','PR_CHO','SMAC','HAC'];COMP=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO','Titer']
def nice(s):return {'AC-SINS_pH7.4':'AC-SINS','PR_CHO':'PR-CHO','ablang2':'AbLang2','descriptors':'Descriptors','esm2':'ESM-2','isotype':'Subtype','bio':'Bio'}.get(s,s)
def save(fig,name):
 for ext in ['pdf','png','svg']:fig.savefig(F/(name+'.'+ext),dpi=300)
 plt.close(fig)
def bars(ax,values,ys,errs,color,label=None,height=.17,hatch=None):
 ax.barh(ys,values,height,xerr=errs,capsize=2,color=color,alpha=.85,label=label,hatch=hatch,error_kw={'elinewidth':.65,'capthick':.65})
def legend(ax,n=3):ax.legend(fontsize=7,ncol=n,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.01))

# Same-data DELPHI: signed horizontal bars replace floating interval markers.
d=pd.read_csv(P/'delphi/primary_seed42_results.csv')
fig,axs=plt.subplots(2,2,figsize=(7.2,5.7),layout='constrained')
for i,task in enumerate(['regression','classification']):
 for j,model in enumerate(['AbDev-X','DELPHI']):
  q=d[(d.task==task)&(d.quantity==model)].set_index('assay').loc[['PR_CHO','SEC']]
  x=q.estimate.to_numpy();err=np.array([x-q.ci_lo,q.ci_hi-x]);bars(axs[i,0],x,np.arange(2)+(j-.5)*.29,err,COL[j],model,.25)
 q=d[(d.task==task)&(d.quantity=='Difference')].set_index('assay').loc[['PR_CHO','SEC']];x=q.estimate.to_numpy()
 bars(axs[i,1],x,np.arange(2),np.array([x-q.ci_lo,q.ci_hi-x]),'#546B7A',height=.38)
 for ax in axs[i]:ax.set_yticks([0,1],['PR-CHO','SEC']);ax.invert_yaxis();ax.axvline(0,color='.65',lw=.6)
 axs[i,0].legend(fontsize=7,frameon=False,loc='lower left');axs[i,0].set_ylim(1.7,-.6)
 axs[i,0].set_xlabel('Spearman ρ; BCa 95% CI' if i==0 else 'ROC AUC; BCa 95% CI');axs[i,1].set_xlabel('AbDev-X − DELPHI; paired BCa 95% CI')
 axs[i,0].set_title(('A  Continuous-target regression' if i==0 else 'C  Common binary targets'),loc='left',fontweight='bold')
 axs[i,1].set_title('B  Regression difference' if i==0 else 'D  Classification difference',loc='left',fontweight='bold')
 if i==1:axs[i,0].axvline(.5,ls=':',color='.5',lw=.7);axs[i,0].set_xlim(0,1)
save(fig,'same_data_comparison')

# Selection frequencies, stream coefficients, and group weights: grouped bars.
fig,axs=plt.subplots(3,1,figsize=(7.2,8.0),layout='constrained');q=pd.read_csv(R/'reviewer_response/selection_stability.csv');blocks=sorted(q.block.unique());groups=list(q.groupby('assay'));z=norm.ppf(.975);n=2000
for j,(assay,g) in enumerate(groups):
 g=g.set_index('block').reindex(blocks);p=g.selection_fraction.to_numpy();center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
 bars(axs[0],p,np.arange(len(blocks))+(j-(len(groups)-1)/2)*.14,np.maximum(0,[p-(center-half),center+half-p]),COL[j],nice(assay),.125)
axs[0].set_yticks(range(len(blocks)),[nice(b) for b in blocks]);axs[0].set_ylim(-.7,len(blocks)+.2);axs[0].set_xlabel('Selection fraction; Wilson Monte Carlo 95% interval');legend(axs[0],5);axs[0].set_title('A  Source-selection stability',loc='left',fontweight='bold',pad=26)
q=pd.read_csv(E/'expert_refit_weights.csv')
for ax,expert,title in [(axs[1],'D-Stack','B  D-Stack stream coefficients'),(axs[2],'F-MoE','C  F-MoE mechanistic-group weights')]:
 sub=q[q.expert==expert];groups=sorted(sub.group.unique());summary=sub.groupby(['assay','group']).weight.agg(['mean','std']);w=min(.16,.72/len(groups))
 for j,g in enumerate(groups):
  entries=[(k,summary.loc[(a,g)]) for k,a in enumerate(PRIMARY) if (a,g) in summary.index]
  bars(ax,[v['mean'] for _,v in entries],np.array([k for k,_ in entries])+(j-(len(groups)-1)/2)*w,[v['std'] for _,v in entries],COL[j],g,w*.9)
 ax.set_yticks(range(5),[nice(a) for a in PRIMARY]);ax.set_ylim(-.7,5.1);ax.axvline(0,color='.65',lw=.6);ax.set_xlabel('Mean ± SD across five outer-training refits');legend(ax,len(groups));ax.set_title(title,loc='left',fontweight='bold',pad=26)
save(fig,'si03_expert_specialization_unique')

# Routing panel A changes to grouped bars; retain the identities and scopes of B–D.
fig,axs=plt.subplots(2,2,figsize=(7.2,6.5),layout='constrained');d=pd.read_csv(R/'abdevx2_ablation/fold_metrics.csv');d['actual_mean']=d[COMP].mean(axis=1);q=d.groupby(['method','split']).actual_mean.agg(['mean','std']);names=sorted(d.method.unique())
for j,split in enumerate(['random','cluster','cluster_isotype']):
 z=q.xs(split,level='split').reindex(names);bars(axs[0,0],z['mean'],np.arange(len(names))+(j-1)*.23,z['std'],COL[j],split.replace('_',' '),.20)
ax=axs[0,0];ax.set_yticks(range(len(names)),[n.replace('abdevx2_','').replace('_',' ') for n in names]);ax.set_xlabel('Mean CV ρ ± fold SD');legend(ax,3);ax.set_title('A  Historical architecture screen',loc='left',fontweight='bold',pad=28)
z=pd.read_csv(R/'sparse_moe_gdpa3.csv');k=pd.read_csv(R/'kronecker_lstm_gdpa3.csv');ax=axs[0,1];ax.scatter(list(z.MEAN)+[k.MEAN.iloc[1]],range(5),facecolors='none',edgecolors=COL[0],s=22);ax.set_yticks(range(5),['Fixed ridge','Conditional','Top-2','Top-3','KronLSTM']);ax.set_xlabel('GDPa3 mean ρ; uncertainty NR');ax.set_title('B  Archived routing points',loc='left',fontweight='bold')
z=pd.read_csv(R/'hybrid_ensemble.csv').iloc[:3].set_index('model')[COMP];ax=axs[1,0]
for j,(model,row) in enumerate(z.iterrows()):ax.bar(np.arange(5)+(j-1)*.24,row,.22,facecolor='white',edgecolor=COL[j],label=model)
ax.set_xticks(range(5),[nice(a) for a in COMP],rotation=30,ha='right');ax.set_ylabel('GDPa3 ρ; uncertainty NR');ax.legend(fontsize=6,frameon=False);ax.set_title('C  Historical routing trade-offs',loc='left',fontweight='bold')
z=pd.read_csv(E/'pooling_cv_folds.csv').groupby(['model','assay']).rho.agg(['mean','std']);ax=axs[1,1]
for j,model in enumerate(['mean','cdr','framework']):
 q=z.xs(model).reindex(COMP);bars(ax,q['mean'],np.arange(5)+(j-1)*.24,q['std'],COL[j],model,.21)
ax.set_yticks(range(5),[nice(a) for a in COMP]);ax.set_xlabel('CV ρ ± fold SD');ax.legend(fontsize=6.5,frameon=False);ax.set_title('D  Regional pooling',loc='left',fontweight='bold');save(fig,'si04_routing_architecture_unique')

# Additional learners and regional descriptors: outlined NR bars and CI bars.
fig,axs=plt.subplots(2,1,figsize=(7.2,6.5),layout='constrained');z=pd.read_csv(R/'tabular_fm.csv').pivot(index='config',columns='model',values='mean_rho');ax=axs[0]
for j,model in enumerate(z.columns):ax.barh(np.arange(len(z))+(j-1)*.23,z[model],.20,facecolor='white',edgecolor=COL[j],label=model,hatch=['//','..','\\\\'][j])
ax.set_yticks(range(len(z)),[x.split('-',2)[-1] for x in z.index]);ax.set_xlabel('Recorded mean GDPa3 ρ; uncertainty NR');ax.set_ylim(-.7,len(z)+.2);legend(ax,3);ax.set_title('A  Tabular learners: single recorded points',loc='left',fontweight='bold',pad=28)
z=pd.read_csv(E/'regional_descriptor_ci.csv');ax=axs[1];models=['H1_global','H2_CDR','H3_FR','H4_CDR_FR']
for j,a in enumerate(PRIMARY):
 q=z[z.assay==a].set_index('model').loc[models];x=q.rho.to_numpy();bars(ax,x,np.arange(4)+(j-2)*.14,np.array([x-q.ci_lo,q.ci_hi-x]),COL[j],nice(a),.125)
ax.set_yticks(range(4),['Global','CDR only','Framework / global','CDR+FR / chain / hierarchical']);ax.set_xlabel('GDPa3 ρ; BCa 95% CI');ax.set_ylim(-.7,4.3);legend(ax,5);ax.axvline(0,color='.65',lw=.6);ax.set_title('B  Regional descriptors: equivalent views merged',loc='left',fontweight='bold',pad=28);save(fig,'si07_learner_screen_unique')

