from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
H=Path(__file__).parent;ROOT=H.parents[1];E=H/'evidence';F=H/'figures';R=ROOT/'results'
sys.path.insert(0,str(H.parent/'plotting'));import build_readable_figures as b
b.H=H;b.F=F
W=b.W;COMP=b.COMP;PRIMARY=b.PRIMARY;COL=b.COL;BLUE=b.BLUE;GREY=b.GREY
nice=b.nice;ttl=b.ttl;canvas=b.canvas
def rd(n):return pd.read_csv(R/n)
def ed(n):return pd.read_csv(E/n)
def finish(fig,n):
 b.finish(fig,n)
def interval(ax,r,y,orientation='horizontal',c=BLUE,fmt='o',label=None):
 err=np.array([[max(0,r.rho-r.ci_lo)],[max(0,r.ci_hi-r.rho)]])
 if orientation=='horizontal':ax.errorbar(r.rho,y,xerr=err,fmt=fmt,c=c,capsize=1.5,lw=.65,label=label,ms=3)
 else:ax.errorbar(y,r.rho,yerr=err,fmt=fmt,c=c,capsize=1.5,lw=.65,label=label,ms=3)
def foldmean(d,modelcol='model'):
 return d.groupby([modelcol,'fold']).rho.mean().groupby(modelcol).agg(['mean','std'])
def encoders():
 d=ed('screen_cv_folds.csv');fig,axs=canvas(2,3,6.6);names=['VH+VL_mean','CDR_mean','CrossAbSense','AbLang2_baseline','ESM2_650M_baseline'];variants=['VH + VL','CDR only','CrossAbSense','AbLang2','ESM-2']
 for ax,sp,title in [(axs[0,0],'cluster_isotype','A  Input variants: cluster–isotype'),(axs[0,1],'random','B  Input variants: random')]:
  q=d[(d.group=='input')&(d.split==sp)].groupby(['model','assay']).rho.agg(['mean','std'])
  for j,(a,c) in enumerate(zip(COMP,COL)):
   vals=[q.loc[(m,a),'mean'] for m in names];sd=[q.loc[(m,a),'std'] for m in names];ax.errorbar(vals,np.arange(5)+(j-2)*.12,xerr=sd,fmt='o',ms=2.4,c=c,lw=.55,capsize=1,label=nice(a))
  ax.set_yticks(range(5),variants);ax.set(xlabel='CV ρ ± fold SD',xlim=(-.5,.95),ylim=(5.65,-.65));ax.legend(fontsize=5.5,ncol=2,loc='lower left');ttl(ax,title)
 q=d[d.group=='structure'];summ=foldmean(q);summ=summ.drop(index='Descriptors',errors='ignore').sort_values('mean');ax=axs[0,2];ax.barh([nice(x).replace('Combined struct','Combined') for x in summ.index],summ['mean'],xerr=summ['std'],capsize=2,color='#3A7CC1',error_kw={'lw':.65});ax.set_xlabel('Mean CV ρ ± fold SD');ttl(ax,'C  Structural groups and PLMs')
 summ=foldmean(d[d.group=='encoder']).sort_values('mean');ax=axs[1,0];ax.barh([nice(x).replace(' pooled','').replace('ESM2 150M','ESM-2 150M') for x in summ.index],summ['mean'],xerr=summ['std'],capsize=2,color='#1A3A6E',error_kw={'lw':.65});ax.set_xlabel('Mean CV ρ ± fold SD');ttl(ax,'D  Nine-encoder screen')
 q=ed('encoder_heldout_ci.csv');q=q[q.assay=='MEAN'].sort_values('rho');ax=axs[1,1]
 ax.barh([nice(x) for x in q.model],q.rho,color='#3A7CC1')
 for j,r in enumerate(q.itertuples()):interval(ax,r,j,fmt='none',c='#334957')
 ax.set_xlabel('Held-out mean ρ; 95% CI');ttl(ax,'E  Held-out encoder comparison')
 ax=axs[1,2];names=['Aggrescan3D','SAP','Electro','MOE','Combined_struct'];q=d[(d.group=='structure')&d.model.isin(names)].groupby(['model','assay']).rho.agg(['mean','std']);z=q['mean'].unstack().reindex(names)[COMP];im=ax.imshow(z,vmin=-.1,vmax=.7,aspect='auto',cmap='YlGnBu');ax.set_xticks(range(5),[nice(a) for a in COMP],rotation=30,ha='right');ax.set_yticks(range(5),['A3D','SAP','Electro','MOE','Combined'])
 for i,m in enumerate(names):
  for j,a in enumerate(COMP):
   r=q.loc[(m,a)];ax.text(j,i,f'{r["mean"]:.2f}\n±{r["std"]:.2f}',ha='center',va='center',fontsize=5.5,color='white' if im.norm(r['mean'])>.74 else '#151515')
 ttl(ax,'F  Structural CV: mean ± SD');finish(fig,'main04_encoder_structure')
def main_context_and_learning():
 original_rd=b.rd;original_finish=b.finish
 def context_rd(n):
  if n=='all7_assays.csv':
   q=ed('jiang_verified.csv');q['jiang_best']=q.best_point;return q
  return original_rd(n)
 def save(fig,n):
  if n=='main07_baseline_context':
   for ax in fig.axes:
    leg=ax.get_legend()
    if leg:
     for t in leg.get_texts():t.set_text(t.get_text().replace('Jiang: archived estimates','Jiang: best published point'))
  if n=='main05_efficiency':
   ax=fig.axes[1];q=pd.read_csv(b.E/'primary_learning_folds.csv');q.assay=q.assay.replace({'AC-SINS':'AC-SINS_pH7.4'})
   for a,c in zip(COMP,COL):
    z=q[q.assay==a].groupby('fraction').rho.agg(['mean','std']);ax.errorbar(z.index*100,z['mean'],yerr=z['std'],fmt='none',c=c,capsize=1.4,lw=.55)
  original_finish(fig,n)
 b.rd=context_rd;b.finish=save;b.baseline();b.efficiency();b.rd=original_rd;b.finish=original_finish
def detailed_shap():
 old=b.old;old.F=F;old.E=H.parent/'primary/evidence';old.feature_label=b.shortfeature;old.title=ttl;old.heat=lambda ax,df,s,vmin=None,vmax=None,cmap='YlGnBu',digits=2:b.heat(ax,df,s,vmin,vmax,cmap)
 oldrd=old.rd
 def proxy_rd(n):
  if n.startswith('explain/shap_') and 'perantibody' not in n:
   a=n.removeprefix('explain/shap_').removesuffix('.csv');return ed('proxy_shap_feature_ci.csv').query('assay == @a')[['feature','mean_abs_shap']]
  return oldrd(n)
 def save(fig,n):
  q=ed('proxy_shap_feature_ci.csv')
  for ax,a in zip(fig.axes[:3],['HIC','Tm2','AC-SINS_pH7.4']):
   z=q[q.assay==a].nlargest(12,'mean_abs_shap').sort_values('mean_abs_shap');ax.errorbar(z.mean_abs_shap,np.arange(12),xerr=np.array([z.mean_abs_shap-z.ci_lo,z.ci_hi-z.mean_abs_shap]),fmt='none',c='#263746',capsize=1,lw=.55);ax.set_xlabel('Mean |SHAP|; 95% bootstrap CI')
  finish(fig,'si01_detailed_shap')
 old.rd=proxy_rd;old.save=save;old.detailed_shap();old.rd=oldrd
def soft_extensions():
 d=ed('extension_metric_ci.csv');fig=plt.figure(figsize=(W,6.6));fig._manual=True;ax=fig.add_axes([.08,.72,.38,.22]);q=d[d.experiment=='soft'];x=np.arange(5);w=.24
 for j,(m,label,c) in enumerate(zip(['ridge_stacking','soft_moe_gate','best_block_oracle'],['Concatenated Ridge','Soft source gate','Best-block reference'],[BLUE,'#009E73','#E69F00'])):
  z=q[q.method==m].set_index('assay').reindex(COMP);ax.bar(x+(j-1)*w,z.rho,w,color=c,label=label);ax.errorbar(x+(j-1)*w,z.rho,yerr=[z.rho-z.ci_lo,z.ci_hi-z.rho],fmt='none',c='#46545e',capsize=1,lw=.5)
 ax.set_xticks(x,[nice(a) for a in COMP],rotation=28,ha='right');ax.set(ylabel='OOF ρ; conditional 95% CI',ylim=(-.25,.95));ax.legend(fontsize=5.5,loc='upper left');ttl(ax,'A  Random-fold source-fusion diagnostic')
 ax=fig.add_axes([.61,.72,.36,.22]);q=ed('source_gate_ci.csv');blocks=sorted(q.block.unique())
 for j,(a,c) in enumerate(zip(COMP,COL)):
  z=q[q.assay==a].set_index('block').reindex(blocks)
  ax.errorbar(z['mean'],np.arange(len(blocks))+(j-2)*.13,xerr=[z['mean']-z.ci_lo,z.ci_hi-z['mean']],fmt='o',ms=2.4,c=c,lw=.6,label=nice(a))
 ax.set_yticks(range(len(blocks)),[nice(x) for x in blocks]);ax.set(xlabel='Mean weight; conditional 95% CI',xlim=(.05,.40),ylim=(5.7,-.8));ax.legend(fontsize=5.5,ncol=2,loc='lower right');ttl(ax,'B  Soft source-gate weights')
 ax=fig.add_axes([.08,.395,.21,.20]);q=d[d.experiment=='isotonic'].set_index('method').reindex(['baseline_ridge','isotonic_sap']);ax.bar(range(2),q.rho,color=[BLUE,GREY]);ax.errorbar(range(2),q.rho,yerr=[q.rho-q.ci_lo,q.ci_hi-q.rho],fmt='none',c='#46545e',capsize=2,lw=.6);ax.set_xticks(range(2),['Ridge','Isotonic']);ax.set_ylabel('HIC OOF ρ');ttl(ax,'C  Isotonic constraint')
 ax=fig.add_axes([.42,.395,.21,.20]);q=d[d.experiment=='rank']
 for j,(a,z) in enumerate(q.groupby('assay')):ax.errorbar(z.lambda_value,z.rho,yerr=[z.rho-z.ci_lo,z.ci_hi-z.rho],fmt='o-',ms=2.5,lw=.65,c=COL[j],capsize=1,label=nice(a))
 ax.set(xlabel='Rank-loss λ',ylabel='OOF ρ');ax.legend(fontsize=5.5);ttl(ax,'D  Rank regularization')
 ax=fig.add_axes([.76,.395,.21,.20]);q=d[d.experiment=='multitask']
 for a in COMP:
  zz=q[q.assay==a].set_index('method');r=zz.loc['rho_individual'];s=zz.loc['rho_shared'];ax.errorbar(r.rho,s.rho,xerr=[[r.rho-r.ci_lo],[r.ci_hi-r.rho]],yerr=[[s.rho-s.ci_lo],[s.ci_hi-s.rho]],fmt='o',c=BLUE,lw=.5,ms=3)
 ax.plot([-.2,.9],[-.2,.9],':',c=GREY);ax.set(xlabel='Individual OOF ρ',ylabel='Shared-PCA OOF ρ');ttl(ax,'E  Shared-latent diagnostic')
 bs=rd('heldout_gdpa4_bispecifics/arm_predictions.csv');raw=pd.read_csv(ROOT/'data/processed/gdpa4_bispecifics.csv');merged=bs.merge(raw,left_on='bsab_name',right_on='antibody_name');ci=ed('transfer_ci.csv')
 for i,a in enumerate(['HIC','AC-SINS_pH7.4','PR_CHO']):
  ax=fig.add_axes([.08+i*.34,.075,.21,.20]);obs=merged['AC-SINS' if a=='AC-SINS_pH7.4' else a];ax.scatter(obs,merged['avg_'+a],s=4,c=BLUE,alpha=.65,edgecolors='none');ax.set(xlabel='Observed',ylabel='Arm-average prediction');r=ci[(ci.experiment=='bispecific')&(ci.assay==a)].iloc[0];ttl(ax,f'{chr(70+i)}  {nice(a)}: ρ = {r.rho:.2f}');ax.text(.03,.97,f'95% CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}]',transform=ax.transAxes,va='top',fontsize=5.5)
 finish(fig,'si02_soft_moe_extensions')
def headline():
 fig,axs=canvas(1,2,4.4);q=ed('strategy_ci.csv');q=q[q.assay=='MEAN'].sort_values('rho');ax=axs[0];ax.barh(q.strategy,q.rho,color=[BLUE if x=='AbDev-X' else GREY for x in q.strategy])
 for j,r in enumerate(q.itertuples()):interval(ax,r,j,fmt='none',c='#334957')
 ax.set(xlabel='Mean held-out ρ; joint 95% CI',xlim=(-.05,.8));ttl(ax,'A  Matched nine-strategy comparison')
 q=pd.read_csv(H.parent/'recorded_results/evidence/primary_reproduction.csv');ax=axs[1]
 for j,r in enumerate(q.itertuples()):interval(ax,r,j,c=BLUE if r.assay in PRIMARY else GREY)
 ax.set_yticks(range(10),[nice(a) for a in q.assay]);ax.invert_yaxis();ax.axvline(0,c=GREY,ls=':',lw=.6);ax.set_xlabel('Held-out ρ; BCa 95% CI');ttl(ax,'B  Complete ten-assay scope');finish(fig,'main_headline_unique')
def experts():
 fig,axs=canvas(3,1,6.5);q=rd('reviewer_response/selection_stability.csv');blocks=sorted(q.block.unique());zval=norm.ppf(.975);n=2000
 for j,(name,g) in enumerate(q.groupby('assay')):
  g=g.set_index('block').reindex(blocks);p=g.selection_fraction.to_numpy();center=(p+zval*zval/(2*n))/(1+zval*zval/n);half=zval*np.sqrt(p*(1-p)/n+zval*zval/(4*n*n))/(1+zval*zval/n);axs[0].errorbar(p,np.arange(len(blocks))+(j-2)*.13,xerr=np.maximum(0,[p-(center-half),(center+half)-p]),fmt='o',ms=2.5,label=nice(name),lw=.6,capsize=1)
 axs[0].set_yticks(range(len(blocks)),[nice(x) for x in blocks]);axs[0].set_ylim(-.6,5.1);axs[0].set_xlabel('Selection frequency; Monte Carlo 95% interval');axs[0].legend(ncol=5,fontsize=5.5,loc='upper center');ttl(axs[0],'A  Bootstrap source-selection stability')
 q=ed('expert_refit_weights.csv')
 for ax,expert,title in [(axs[1],'D-Stack','B  Primary D-Stack: outer-refit coefficients'),(axs[2],'F-MoE','C  Primary F-MoE: outer-refit group weights')]:
  z=q[q.expert==expert].groupby(['assay','group']).weight.agg(['mean','std']);groups=sorted(q[q.expert==expert].group.unique())
  for j,g in enumerate(groups):
   vals=[];sd=[]
   for a in PRIMARY:
    r=z.loc[(a,g)] if (a,g) in z.index else pd.Series({'mean':np.nan,'std':np.nan});vals.append(r['mean']);sd.append(r['std'])
   ax.errorbar(vals,np.arange(5)+(j-(len(groups)-1)/2)*.13,xerr=sd,fmt='o',ms=2.7,lw=.6,capsize=1,label=g)
  ax.set_yticks(range(5),[nice(a) for a in PRIMARY]);ax.set_ylim(-.6,5.4);ax.set_xlabel('Mean ± SD across five outer-training refits');ax.legend(ncol=len(groups),fontsize=5.5,loc='upper center');ttl(ax,title)
 finish(fig,'si03_expert_specialization_unique')
def routing():
 fig,axs=canvas(2,2,6.5);d=rd('abdevx2_ablation/fold_metrics.csv');d['actual_mean']=d[COMP].mean(axis=1);q=d.groupby(['method','split']).actual_mean.agg(['mean','std']);names=sorted(d.method.unique())
 for j,sp in enumerate(['random','cluster','cluster_isotype']):
  z=q.xs(sp,level='split').reindex(names);axs[0,0].errorbar(z['mean'],np.arange(len(names))+(j-1)*.18,xerr=z['std'],fmt='o',ms=2.5,capsize=1,lw=.55,label=sp.replace('_',' '))
 axs[0,0].set_yticks(range(len(names)),[x.replace('abdevx2_','').replace('_',' ') for x in names]);axs[0,0].set_xlabel('Mean CV ρ ± fold SD');axs[0,0].legend(fontsize=5.5);ttl(axs[0,0],'A  Historical architecture screen')
 z=rd('sparse_moe_gdpa3.csv');k=rd('kronecker_lstm_gdpa3.csv');vals=list(z.MEAN)+[k.MEAN.iloc[1]];ax=axs[0,1];ax.scatter(vals,range(5),facecolors='none',edgecolors=BLUE,s=18);ax.set_yticks(range(5),['Fixed ridge','Conditional','Top-2','Top-3','KronLSTM']);ax.set_xlabel('GDPa3 mean ρ');ax.text(.02,.02,'Open symbols: uncertainty NR',transform=ax.transAxes,fontsize=5.5);ttl(ax,'B  Archived routing points')
 z=rd('hybrid_ensemble.csv').iloc[:3].set_index('model')[COMP]
 for j,(m,r) in enumerate(z.iterrows()):axs[1,0].plot(range(5),r,'o-',label=nice(m),mfc='white',ms=3)
 axs[1,0].set_xticks(range(5),[nice(a) for a in COMP],rotation=28,ha='right');axs[1,0].set_ylabel('CV ρ; uncertainty NR');axs[1,0].legend(fontsize=5.5);ttl(axs[1,0],'C  Historical routing trade-offs')
 z=ed('pooling_cv_folds.csv').groupby(['model','assay']).rho.agg(['mean','std']);ax=axs[1,1]
 for j,m in enumerate(['mean','cdr','framework']):
  q=z.xs(m).reindex(COMP);ax.errorbar(np.arange(5)+(j-1)*.08,q['mean'],yerr=q['std'],fmt='o-',ms=2.5,capsize=1,lw=.6,label=m)
 ax.set_xticks(range(5),[nice(a) for a in COMP],rotation=28,ha='right');ax.set_ylabel('CV ρ ± fold SD');ax.legend(fontsize=5.5);ttl(ax,'D  Regional pooling');finish(fig,'si04_routing_architecture_unique')
def transfer():
 fig,axs=canvas(3,1,6.5);d=ed('transfer_ci.csv');names=['AbDev-X','ESM2-Ridge','AbLang2-ElasticNet','Descriptor-Ridge'];assays=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO'];ax=axs[0]
 for j,m in enumerate(names):
  q=d[(d.experiment=='VHH')&(d.model==m)].set_index('assay').reindex(assays);ax.errorbar(np.arange(4)+(j-1.5)*.12,q.rho,yerr=[q.rho-q.ci_lo,q.ci_hi-q.rho],fmt='o',ms=3,capsize=1,lw=.6,label=nice(m))
 ax.set_xticks(range(4),[nice(a) for a in assays]);ax.set_ylabel('GDPa5 ρ; BCa 95% CI');ax.legend(ncol=4,fontsize=5.5,loc='upper left');ttl(ax,'A  Historical VHH-Fc transfer')
 ax=axs[1];z=rd('combined_training/spearman_summary.csv').set_index('method')[COMP]
 for j,(m,row) in enumerate(z.iterrows()):
  for i,a in enumerate(COMP):
   q=d[(d.experiment=='combined_baseline')&(d.model==m)&(d.assay==a)]
   if len(q):interval(ax,q.iloc[0],i+(j-.5)*.18,c=COL[j])
   else:ax.plot(row[a],i+(j-.5)*.18,'o',mfc='white',mec=COL[j],ms=3)
 ax.set_yticks(range(5),[nice(a) for a in COMP]);ax.set_xlabel('GDPa3 ρ; filled points have recovered BCa CIs');ax.legend(handles=[Line2D([],[],color=COL[0],marker='o',ls='',label='GDPa1-only'),Line2D([],[],color=COL[1],mfc='white',marker='o',ls='',label='Combined training: CI NR')],fontsize=5.5,ncol=2);ttl(ax,'B  Separate combined-format experiment')
 z=rd('ranking_comparison.csv').set_index('model').loc[['Random','ESM2-Ridge','AbDev-X (stacking)'],['P5','P10','P20']];ax=axs[2]
 for m,row in z.iterrows():ax.plot([5,10,20],row,'o-',mfc='white',ms=3,label=nice(m))
 ax.set(xlabel='K',ylabel='Precision at K');ax.legend(fontsize=5.5,ncol=3);ttl(ax,'C  Archived composite ranking: uncertainty NR');finish(fig,'si05_transfer_ranking_unique')
def robustness_and_learners():
 fig,axs=canvas(2,2,6.5);d=rd('exp_batch_correction.csv');ax=axs[0,0]
 for j,(m,z) in enumerate(d.groupby('strategy')):
  z=z.set_index('assay').reindex(COMP);ax.errorbar(np.arange(5)+(j-2)*.05,z.gdpa3,yerr=[z.gdpa3-z.gdpa3_ci95_lo,z.gdpa3_ci95_hi-z.gdpa3],fmt='o-',ms=2.3,capsize=1,lw=.5,label=m)
 ax.set_xticks(range(5),[nice(a) for a in COMP],rotation=28,ha='right');ax.set_ylabel('GDPa3 ρ; recorded 95% CI');ax.legend(fontsize=5.5,ncol=2);ttl(ax,'A  Normalization / augmentation')
 z=rd('exp_struct_noise.csv');ax=axs[0,1]
 for m,q in z[z.assay=='HIC'].groupby('block'):ax.errorbar(q.sigma,q.rho_mean,yerr=q.rho_std,fmt='o-',ms=2.5,capsize=1,lw=.6,label=m.replace('Combined_struct','Combined'))
 ax.set(xlabel='Feature perturbation σ',ylabel='HIC CV ρ ± perturbation SD');ax.legend(fontsize=5.5);ttl(ax,'B  Structural sensitivity')
 z=rd('exp_calibration.csv');ax=axs[1,0];ax.scatter(z.calib_rho_std_vs_error,np.arange(5),facecolors='none',edgecolors=BLUE,s=18);ax.set_yticks(range(5),[nice(a) for a in z.assay]);ax.axvline(0,c=GREY,lw=.6);ax.set_xlabel('Disagreement–error correlation');ttl(ax,'C  Confidence proxy: uncertainty NR')
 z=rd('exp_abstention_errorbars.csv');ax=axs[1,1]
 for a,q in z.groupby('assay'):ax.errorbar(q.abstain_frac,q.rho,yerr=[q.rho-q.ci_lo,q.ci_hi-q.rho],fmt='o-',ms=2.5,capsize=1,lw=.6,label=nice(a))
 ax.set(xlabel='Abstained fraction',ylabel='Retained-set ρ; 95% CI');ax.legend(fontsize=5.5);ttl(ax,'D  Selective prediction');finish(fig,'si06_augmentation_confidence_unique')
 fig,axs=canvas(2,1,5.6);z=rd('tabular_fm.csv').pivot(index='config',columns='model',values='mean_rho');ax=axs[0]
 for j,c in enumerate(z.columns):ax.scatter(z[c],np.arange(len(z))+(j-1)*.15,facecolors='none',edgecolors=COL[j],s=16,label=c)
 ax.set_yticks(range(len(z)),[x.split('-',2)[-1] for x in z.index]);ax.legend(fontsize=5.5,ncol=3);ax.set_xlabel('Recorded mean GDPa3 ρ');ttl(ax,'A  Recorded tabular screen: uncertainty NR')
 z=ed('regional_descriptor_ci.csv');ax=axs[1]
 for j,(m,q) in enumerate(z.groupby('model')):
  q=q.set_index('assay').reindex(PRIMARY);ax.errorbar(np.arange(5)+(j-2.5)*.05,q.rho,yerr=[q.rho-q.ci_lo,q.ci_hi-q.rho],fmt='o-',ms=2.2,capsize=.8,lw=.5,label=m.replace('_',' '))
 ax.set_xticks(range(5),[nice(a) for a in PRIMARY],rotation=28,ha='right');ax.set_ylabel('GDPa3 ρ; BCa 95% CI');ax.legend(fontsize=5.5,ncol=3);ttl(ax,'B  Regional descriptor constructions');finish(fig,'si07_learner_screen_unique')
if __name__=='__main__':
 for fn in [encoders,main_context_and_learning,detailed_shap,soft_extensions,headline,experts,routing,transfer,robustness_and_learners]:print(fn.__name__,flush=True);fn()
