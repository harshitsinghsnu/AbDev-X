"""Curated plots: each retained figure answers a distinct scientific question."""
from pathlib import Path
import shutil,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).parent;ROOT=HERE.parents[2];R=ROOT/'results';F=HERE/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':240})
COMP=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO','Titer'];PRIMARY=['HIC','AC-SINS','PR_CHO','SMAC','HAC']
BLUE='#276b9a';GRAY='#9aaab5'
def rd(n):return pd.read_csv(R/n)
def nice(n):return str(n).replace('AC-SINS_pH7.4','AC-SINS').replace('PR_CHO','PR-CHO')
def title(ax,t):ax.set_title(t,loc='left',fontweight='bold',pad=10)
def save(fig,n):fig.savefig(F/(n+'.pdf'),bbox_inches='tight');fig.savefig(F/(n+'.png'),bbox_inches='tight');plt.close(fig)
def heat(ax,z,t,vmin=0,vmax=1,cmap='YlGnBu'):
 a=z.to_numpy(float);m=ax.imshow(a,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
 ax.set_xticks(range(len(z.columns)),[nice(x) for x in z.columns],rotation=25,ha='right');ax.set_yticks(range(len(z)),[nice(x) for x in z.index]);title(ax,t)
 for i in range(a.shape[0]):
  for j in range(a.shape[1]):
   if np.isfinite(a[i,j]):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if m.norm(a[i,j])>.7 else 'black')
 return m

# Main headline result: no duplicated component matrix or standalone weight heatmap.
fig,axs=plt.subplots(1,2,figsize=(9.0,6.7),layout='constrained')
sm=rd('final_evaluation_summary.csv').set_index('model')
names={'A1-E1':'E1','A2-ESM2':'ESM-2','A3-AbLang2':'AbLang2','A4-Desc':'Descriptors','A5-Bio':'Bio features','A6-MGFS':'A-MGFS','A7-Dstack':'D-Stack','A8-FMoE':'F-MoE','AbDev-X':'AbDev-X'}
vals=sm.loc[list(names),'overall_mean'].sort_values()
axs[0].barh([names[x] for x in vals.index],vals,color=[BLUE if x=='AbDev-X' else '#bccbd5' for x in vals.index])
for i,v in enumerate(vals):axs[0].text(v+.005,i,f'{v:.3f}',va='center',fontsize=9)
axs[0].set(xlim=(0,.56),xlabel='Mean GDPa3 Spearman ρ');title(axs[0],'A  Best aggregate rank performance')
all10=rd('awmoe_all9.csv').set_index('assay');ci=rd('bootstrap_ci_all.csv').set_index('model').loc['AbDev-X']
extra={'Tm1':(-.025,.611),'Tonset':(-.012,.414),'Tm2':(-.063,.371),'SEC':(-.382,.056),'Titer':(-.147,.293)}
order=['HAC','SMAC','HIC','AC-SINS','PR_CHO','Tm1','Tonset','Tm2','SEC','Titer']
for i,a in enumerate(order):
 v=all10.loc[a,'rho'];lo,hi=(ci[a+'_lo'],ci[a+'_hi']) if a in PRIMARY else extra[a]
 axs[1].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=BLUE if a in PRIMARY else GRAY,capsize=3)
axs[1].set_yticks(range(10),[nice(a) for a in order]);axs[1].invert_yaxis();axs[1].axvline(0,color='#777',ls=':');axs[1].set(xlabel='GDPa3 ρ and recorded 95% BCa CI',xlim=(-.47,.92));title(axs[1],'B  Complete ten-assay evaluation')
save(fig,'main_headline_unique')

# Explicitly requested SI composites; preserve their graphics unchanged.
shutil.copy2(HERE.parent/'panel_explainability.pdf',F/'si01_detailed_shap.pdf')
shutil.copy2(HERE.parent/'panel_extensions.pdf',F/'si02_soft_moe_extensions.pdf')

# S3: selection frequency and actual expert-internal gates -- distinct from source coefficients.
j=json.loads((R/'awmoe_explainability.json').read_text())
fig,axs=plt.subplots(3,1,figsize=(8.5,9.2),layout='constrained')
z=rd('reviewer_response/selection_stability.csv').pivot(index='assay',columns='block',values='selection_fraction').fillna(0)
heat(axs[0],z,'A  How stable is the selected source?')
z=pd.DataFrame({a:[j[a]['dstack_plm_weight'],j[a]['dstack_db_weight']] for a in PRIMARY},index=['PLM stream','Descriptor + bio stream']).T
heat(axs[1],z,'B  Two-stream D-Stack internal weights')
z=pd.DataFrame({a:j[a]['fmoe_weights'] for a in PRIMARY}).T.fillna(0)
heat(axs[2],z,'C  F-MoE mechanism-group weights')
save(fig,'si03_expert_specialization_unique')

# S4: architecture alternatives and routing -- no repetition of source-removal curves.
fig,axs=plt.subplots(2,2,figsize=(9.5,8.8),layout='constrained')
z=rd('abdevx2_ablation/metrics.csv').pivot(index='method',columns='split',values='MEAN')
heat(axs[0,0],z,'A  Architecture screen · CV means',vmax=.55)
z=rd('sparse_moe_gdpa3.csv');k=rd('kronecker_lstm_gdpa3.csv');v=list(z.MEAN)+[k.MEAN.iloc[1]];labs=['Fixed ridge','Conditional gate','Sparse top-2','Sparse top-3','KronLSTM']
axs[0,1].barh(labs,v,color=[BLUE]+['#b9c9d3']*4)
for i,x in enumerate(v):axs[0,1].text(x+.004,i,f'{x:.3f}',va='center',fontsize=8)
axs[0,1].set(xlim=(0,.33),xlabel='GDPa3 mean ρ · competition five');title(axs[0,1],'B  Shared-input routing comparison')
z=rd('hybrid_ensemble.csv').iloc[:3].set_index('model')[COMP].T;heat(axs[1,0],z,'C  Mean improvement can hide assay collapse',vmax=.55)
z=rd('exp_pooling.csv').set_index('pooling')[COMP];heat(axs[1,1],z,'D  Regional pooling choices',vmin=-.15,vmax=.6)
save(fig,'si04_routing_architecture_unique')

# S5: VHH, combined training and composite ranking -- no duplicate bispecific plot.
fig,axs=plt.subplots(3,1,figsize=(8.4,9.0),layout='constrained')
z=rd('heldout_gdpa5/spearman_summary.csv').set_index('method');heat(axs[0],z,'A  GDPa5 VHH-Fc transfer',vmin=-.1,vmax=.4)
z=rd('combined_training/spearman_summary.csv').set_index('method')[COMP];heat(axs[1],z,'B  Combined-format training experiment',vmin=-.1,vmax=.4)
z=rd('ranking_comparison.csv').set_index('model').loc[['Random','ESM2-Ridge','AbDev-X (stacking)'],['P5','P10','P20']];heat(axs[2],z,'C  Composite ranking precision',vmax=.4)
save(fig,'si05_transfer_ranking_unique')

# S6: augmentation and confidence diagnostics, excluding physics already in S2.
fig,axs=plt.subplots(2,2,figsize=(9.4,8.7),layout='constrained')
z=rd('exp_batch_correction.csv').pivot(index='strategy',columns='assay',values='gdpa3').reindex(columns=COMP);heat(axs[0,0],z,'A  Study-normalization / augmentation',vmin=-.35,vmax=.45)
z=rd('exp_struct_noise.csv')
for g,v in z[z.assay=='HIC'].groupby('block'):axs[0,1].plot(v.sigma,v.rho_mean,'o-',label=g)
axs[0,1].set(xlabel='Numerical feature perturbation σ',ylabel='HIC CV ρ');axs[0,1].legend(fontsize=7);title(axs[0,1],'B  Structural-feature sensitivity')
z=rd('exp_calibration.csv');axs[1,0].barh([nice(a) for a in z.assay],z.calib_rho_std_vs_error,color=GRAY);axs[1,0].axvline(0,color='#333');axs[1,0].set_xlabel('Correlation: disagreement vs. absolute error');title(axs[1,0],'C  Does disagreement track error?')
z=rd('exp_abstention.csv')
for a,v in z.groupby('assay'):axs[1,1].plot(v.abstain_frac,v.rho_retained,'o-',label=nice(a))
axs[1,1].set(xlabel='Abstained fraction',ylabel='Retained-set CV ρ');axs[1,1].legend(fontsize=7);title(axs[1,1],'D  Selective prediction / abstention')
save(fig,'si06_augmentation_confidence_unique')

# S7: specialized learners and hierarchical descriptors remain distinct experiments.
shutil.copy2(HERE.parent/'rewritten/figures/figS05_extended_learners.pdf',F/'si07_learner_screen_unique.pdf')
print('Built one headline main figure and seven curated supplementary figures.')
