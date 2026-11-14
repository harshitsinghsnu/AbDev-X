"""Rebuild manuscript panels from saved experiments; never refit on held-out data."""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE=Path(__file__).parent
ROOT=HERE.parents[2]
R=ROOT/'results' if (ROOT/'results').exists() else HERE/'evidence'
F=HERE/'figures'; F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':200})
TEAL='#087f8c'; BLUE='#376db4'; GOLD='#d8aa44'; GRAY='#78858d'
COMP=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO','Titer']
PRIMARY=['HIC','AC-SINS','PR_CHO','SMAC','HAC']
def rd(f):return pd.read_csv(R/f)
def nice(s):return str(s).replace('AC-SINS_pH7.4','AC-SINS').replace('PR_CHO','PR-CHO')
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),bbox_inches='tight');plt.close(fig)
def panel(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=10)
def heat(ax,data,title,cmap='YlGnBu',vmin=None,vmax=None,fmt='.2f'):
    ar=np.asarray(data,dtype=float);m=ax.imshow(ar,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(data.columns)),[nice(c) for c in data.columns],rotation=30,ha='right')
    ax.set_yticks(range(len(data.index)),[nice(c) for c in data.index]);panel(ax,title)
    for i in range(ar.shape[0]):
        for j in range(ar.shape[1]):
            if np.isfinite(ar[i,j]):
                norm=m.norm(ar[i,j]); dark=(norm>.68) if cmap!='RdBu_r' else (norm<.15 or norm>.85)
                ax.text(j,i,format(ar[i,j],fmt),ha='center',va='center',fontsize=8,color='white' if dark else 'black')
    return m

# Figure 1. The operational architecture, rather than an unimplemented description.
fig,ax=plt.subplots(figsize=(15.6,8.6));ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
def box(x,y,w,h,s,fc='#ffffff',fs=9):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.10',fc=fc,ec='#555555',lw=.8);ax.add_patch(p)
    ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=fs)
def arr(x,y,u,v):ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':.8,'color':'#444444'})
for x,s in [(.25,'A  Inputs'),(2.5,'B  Feature families'),(6.0,'C  Assay-specific experts'),(12.4,'D  Fusion / outputs')]:ax.text(x,8.6,s,fontsize=12,fontweight='bold')
box(.2,5.8,1.8,1.15,'VH + VL\nsequences', '#eef4f7',11)
box(.2,3.8,1.8,.9,'Predicted Fv\nABodyBuilder3','#eef4f7',10);arr(1.1,5.8,1.1,4.7)
box(.2,2.15,1.8,.9,'HC/LC subtype\nmetadata','#eef4f7',10)
labels=[('ESM-2 650M','2560-dimensional cached embedding'),('AbLang2','480-dimensional cached embedding'),('HC/LC subtype','5 one-hot categories'),('Physicochemical','165 sequence descriptors'),('Structural features','TAP / MOE / A3D / AntiFold'),('Surface biophysics','SAP / charge / exposure families')]
colors=['#efd0d0','#ded0ed','#d0e5eb','#d4edd0','#f5e2a0','#f8edca']
for i,((s,t),c) in enumerate(zip(labels,colors)):
    y=7.2-i*.9;box(2.5,y,3.0,.65,s+'\n'+t,c,8)
arr(2,6.4,2.5,7.5);arr(2,6.4,2.5,6.6);arr(2,6.4,2.5,4.8)
arr(2,4.25,2.5,3.9);arr(2,4.25,2.5,3.0);arr(2,2.6,2.5,5.7)
ax.text(4,1.86,'Six conceptual families; structural and surface\nfeatures are consolidated in the saved bio block.',ha='center',fontsize=8,color='#555555')
rows=[(6.8,'E1','Candidate source models → OOF scoring\n→ best-source selection → refit', '#fbeaea'),
      (5.2,'A-MGFS','Descriptor + bio features → keyword mask\n→ standardization / optional PCA → ridge', '#f0ebf8'),
      (3.6,'D-Stack','PLM stream ────┐\nDescriptor / bio ─┴→ OOF ridge meta-learner', '#eaf3f7'),
      (2.0,'F-MoE','Eligible mechanistic groups → group regressors\n→ OOF-score softmax gates → prediction mix', '#fff5d7')]
for y,s,t,c in rows:
    box(6,y,5.75,1.2,'',c);ax.text(6.2,y+.88,s,fontweight='bold',fontsize=11);ax.text(6.25,y+.43,t,fontsize=9,va='center')
    arr(5.5,y+.6,6,y+.6);arr(11.75,y+.6,12.25,y+.6)
ax.plot([12.25,12.25],[2.6,7.4],color='#555555',lw=.8)
box(12.6,4.65,3.05,2.7,'Constrained four-expert fusion\n\nŷₐ = Σₖ wₐₖ ŷₐₖ\n\nwₐₖ ≥ 0; Σₖ wₐₖ = 1\n\nEvaluated primary solution:\nEqual weights (¼ per expert)','#f1edf7',10)
arr(12.25,5.6,12.6,5.6)
box(12.6,2.05,3.05,1.95,'10 continuous outputs\nHAC · SMAC · HIC · AC-SINS · PR-CHO\nTm1 · Tm2 · Tonset · SEC · Titer\n\nPrimary five-assay evaluation:\nHIC · AC-SINS · PR-CHO · SMAC · HAC','#f7e5bc',9);arr(14.12,4.65,14.12,4.0)
box(.2,.2,15.45,1.05,'GDPa1 (246 antibodies) → cluster–isotype folds → expert OOF predictions → fusion fitting → full-data refit → GDPa3 (80 antibodies)\nAssay-specific labels are masked when missing. Model fitting uses GDPa1; GDPa3 provides retrospective external evaluation.','#f5f7f8',10)
save(fig,'fig01_architecture')

# Figure 2. Primary result, complete assay coverage and component comparison.
summary=rd('final_evaluation_summary.csv').set_index('model')
all10=rd('awmoe_all9.csv')
ci=rd('bootstrap_ci_all.csv').set_index('model').loc['AbDev-X']
fig,axs=plt.subplots(2,2,figsize=(8.8,9.3),layout='constrained')
order=['HAC','SMAC','HIC','AC-SINS','PR_CHO','Tm1','Tonset','Tm2','SEC','Titer']
reported_ci={'Tm1':(-.025,.611),'Tonset':(-.012,.414),'Tm2':(-.063,.371),'SEC':(-.382,.056),'Titer':(-.147,.293)}
for i,a in enumerate(order):
    row=all10.set_index('assay').loc[a];rho=row.rho
    lo,hi=(ci[a+'_lo'],ci[a+'_hi']) if a in PRIMARY else reported_ci[a]
    axs[0,0].errorbar(rho,i,xerr=[[rho-lo],[hi-rho]],fmt='o',color=TEAL if a in PRIMARY else GRAY,capsize=3)
axs[0,0].set_yticks(range(10),[nice(a) for a in order]);axs[0,0].invert_yaxis();axs[0,0].axvline(0,color='gray',ls=':');axs[0,0].set_xlabel('GDPa3 Spearman ρ and reported 95% BCa CI');panel(axs[0,0],'A  Complete ten-assay result')
names={'A1-E1':'E1','A2-ESM2':'ESM-2','A3-AbLang2':'AbLang2','A4-Desc':'Descriptors','A5-Bio':'Bio','A6-MGFS':'A-MGFS','A7-Dstack':'D-Stack','A8-FMoE':'F-MoE','AbDev-X':'AbDev-X'}
v=summary.loc[list(names),'overall_mean'].sort_values();axs[0,1].barh([names[i] for i in v.index],v,color=[TEAL if i=='AbDev-X' else '#b7c8d3' for i in v.index]);axs[0,1].set_xlim(0,.56)
for i,n in enumerate(v):axs[0,1].text(n+.005,i,f'{n:.3f}',va='center',fontsize=8)
axs[0,1].set_xlabel('Mean held-out ρ across the same five assays');panel(axs[0,1],'B  Best recorded aggregate performance')
mat=pd.DataFrame({a:summary.loc[['A1-E1','A6-MGFS','A7-Dstack','A8-FMoE','AbDev-X'],a+'_mean'].values for a in PRIMARY},index=['E1','A-MGFS','D-Stack','F-MoE','AbDev-X'])
heat(axs[1,0],mat,'C  Complementary expert strengths',vmin=0,vmax=.75)
w=rd('aw_moe.csv').set_index('assay').loc[PRIMARY,['w_E1','w_A','w_D','w_F']];w.columns=['E1','A-MGFS','D-Stack','F-MoE']
heat(axs[1,1],w,'D  Recorded top-level fusion weights',vmin=0,vmax=.5)
save(fig,'fig02_primary_results')

# Figure 3. The representation story, with experiment scopes kept separate.
j=json.loads((R/'awmoe_explainability.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(9.5,9.5),layout='constrained')
scores=pd.DataFrame({a:j[a]['e1_block_scores'] for a in PRIMARY}).T
heat(axs[0,0],scores,'A  GDPa1 source selection scores',vmin=0,vmax=.9)
z=rd('all_models_splits.csv'); models=['ESM2-Ridge','AbLang2-ElasticNet','AbDev-X (stacking)'];splits=['random','cluster','cluster_isotype'];xx=np.arange(3)
for i,m in enumerate(models):axs[0,1].plot(xx,z[z.model==m].set_index('split').loc[splits,'MEAN'],'o-',label=m.replace('AbDev-X (stacking)','Legacy source stack'))
axs[0,1].set_xticks(xx,['Random','Cluster','Cluster–isotype']);axs[0,1].set_ylabel('Mean CV ρ · competition five');axs[0,1].legend(fontsize=8);panel(axs[0,1],'B  Validation protocol changes the estimate')
st=rd('exp_struct_comparison.csv').pivot(index='block',columns='assay',values='cv_cluster_isotype').reindex(columns=COMP)
heat(axs[1,0],st,'C  Structural feature screen · GDPa1 CV',vmin=-.15,vmax=.7)
stab=rd('reviewer_response/selection_stability.csv');stab=stab.pivot(index='assay',columns='block',values='selection_fraction').fillna(0)
heat(axs[1,1],stab,'D  Selection frequency under resampling',vmin=0,vmax=1)
save(fig,'fig03_representations')

# Figure 4. Negative ablations explain why the regularized design is retained.
fig,axs=plt.subplots(2,2,figsize=(9.3,9.0),layout='constrained')
ab=rd('exp_ablation.csv').pivot(index='dropped',columns='assay',values='dHeldout').reindex(columns=COMP)
heat(axs[0,0],ab,'A  Legacy stack: full minus source-removed ρ',cmap='RdBu_r',vmin=-.27,vmax=.27)
lc=rd('exp_learning_curves.csv')
for a,g in lc.groupby('assay'):axs[0,1].plot(g.frac,g.rho,'o-',label=nice(a))
axs[0,1].set(xlabel='Training fraction',ylabel='CV Spearman ρ');axs[0,1].legend(fontsize=8,ncol=2);panel(axs[0,1],'B  Learning curves are assay dependent')
v=rd('sparse_moe_gdpa3.csv');v2=rd('kronecker_lstm_gdpa3.csv');labs=['Fixed ridge','Conditional gate','Sparse top-2','Sparse top-3','KronLSTM'];vals=list(v.MEAN)+[v2.MEAN.iloc[1]]
axs[1,0].barh(labs,vals,color=[TEAL]+['#b9c9d3']*4);axs[1,0].set_xlim(0,.32);axs[1,0].set_xlabel('GDPa3 mean ρ · competition five')
for i,x in enumerate(vals):axs[1,0].text(x+.004,i,f'{x:.3f}',va='center',fontsize=8)
panel(axs[1,0],'C  Shared source-prediction routing experiment')
v=rd('hybrid_ensemble.csv');vv=v.iloc[:3].set_index('model')[COMP].T
heat(axs[1,1],vv,'D  Mean gains can conceal an assay collapse',vmin=0,vmax=.55)
save(fig,'fig04_ablations')

# Figure 5. Named features and actual internal expert weights.
fig,axs=plt.subplots(2,3,figsize=(13,7.4),layout='constrained')
for ax,a in zip(axs[0],['HIC','AC-SINS_pH7.4','PR_CHO']):
    z=rd('explain/shap_'+a+'.csv').head(5).iloc[::-1]
    ax.barh(z.feature,z.mean_abs_shap,color=TEAL);ax.set_xlabel('Mean |SHAP| · proxy model units');panel(ax,nice(a)+' · descriptor proxy')
dw=pd.DataFrame({a:[j[a]['dstack_plm_weight'],j[a]['dstack_db_weight']] for a in PRIMARY},index=['PLM','Descriptor + bio']).T
heat(axs[1,0],dw,'D  Saved D-Stack internal weights',vmin=0,vmax=1)
fw=pd.DataFrame({a:j[a]['fmoe_weights'] for a in PRIMARY}).T.fillna(0)
heat(axs[1,1],fw,'E  Saved F-MoE group weights',vmin=0,vmax=1)
ig=rd('exp_ig.csv');axs[1,2].bar([nice(a) for a in ig.assay],ig.enrichment,color='#c4addc');axs[1,2].axhline(1,color=GRAY,ls=':');axs[1,2].tick_params(axis='x',rotation=35);axs[1,2].set_ylabel('CDR attribution / length baseline');panel(axs[1,2],'F  Chain-aware proxy attribution')
save(fig,'fig05_interpretability')

# Figure 6. Format transfer and application limits.
fig,axs=plt.subplots(2,2,figsize=(11.7,8.0),layout='constrained')
bi=rd('heldout_gdpa4_bispecifics/spearman_summary.csv').set_index('method');cols=['HIC','AC-SINS_pH7.4','PR_CHO'];x=np.arange(3)
for i,m in enumerate(bi.index):axs[0,0].bar(x+(i-.5)*.32,bi.loc[m,cols],.32,label=m)
axs[0,0].set_xticks(x,[nice(c) for c in cols]);axs[0,0].legend(fontsize=8);axs[0,0].set_ylabel('GDPa4 Spearman ρ');panel(axs[0,0],'A  Bispecific arm-composition experiment')
vhh=rd('heldout_gdpa5/spearman_summary.csv').set_index('method');heat(axs[0,1],vhh,'B  GDPa5 VHH-Fc · saved transfer run',vmin=-.1,vmax=.4)
rk=rd('ranking_comparison.csv').set_index('model').loc[['Random','ESM2-Ridge','AbDev-X (stacking)'],['P5','P10','P20']]
heat(axs[1,0],rk,'C  Composite ranking · legacy stack',vmin=0,vmax=.4)
comb=rd('combined_training/spearman_summary.csv').set_index('method')[COMP];heat(axs[1,1],comb,'D  Multi-format training · separate experiment',vmin=-.1,vmax=.4)
save(fig,'fig06_transfer')

# Supplementary panels reconstructed from their own experiment tables.
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
enc=rd('exp_encoder_screen.csv');heat(axs[0],enc.pivot(index='encoder',columns='assay',values='cv_cluster_isotype').reindex(columns=COMP),'A  Encoder screen · cluster–isotype CV',vmin=-.25,vmax=.65)
heat(axs[1],enc.pivot(index='encoder',columns='assay',values='gdpa3').reindex(columns=COMP).dropna(how='all'),'B  Available held-out encoder scores',vmin=-.25,vmax=.65);save(fig,'figS01_encoders')
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
ar=rd('abdevx2_ablation/metrics.csv');heat(axs[0],ar.pivot(index='method',columns='split',values='MEAN'),'A  Architecture screen · CV mean',vmin=0,vmax=.55)
pool=rd('exp_pooling.csv').set_index('pooling')[COMP];heat(axs[1],pool,'B  Regional pooling · separate encoder experiment',vmin=-.15,vmax=.6);save(fig,'figS02_architecture_pooling')
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
phy=rd('physics_constrained/physics_rank_reg.csv')
for a,g in phy.groupby('assay'):axs[0].plot(g['lambda'],g.rho,'o-',label=nice(a))
axs[0].legend(fontsize=8);axs[0].set(xlabel='Rank-regularization strength',ylabel='CV Spearman ρ');panel(axs[0],'A  Harder rank constraints can hurt')
mt=rd('physics_constrained/multitask_cv.csv').set_index('assay')[['rho_individual','rho_shared']];heat(axs[1],mt,'B  Independent vs. shared prediction',vmin=0,vmax=.7)
bc=rd('exp_batch_correction.csv').pivot(index='strategy',columns='assay',values='gdpa3').reindex(columns=COMP);heat(axs[2],bc,'C  Study correction · held-out screen',vmin=-.35,vmax=.45);save(fig,'figS03_physics_augmentation')
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
noise=rd('exp_struct_noise.csv')
for b,g in noise[noise.assay=='HIC'].groupby('block'):axs[0].plot(g.sigma,g.rho_mean,'o-',label=b)
axs[0].legend(fontsize=7);axs[0].set(xlabel='Feature perturbation σ',ylabel='HIC CV ρ');panel(axs[0],'A  Numerical structural-feature sensitivity')
cal=rd('exp_calibration.csv');axs[1].barh([nice(a) for a in cal.assay],cal.calib_rho_std_vs_error,color=GRAY);axs[1].axvline(0,color='black',lw=.5);panel(axs[1],'B  Expert disagreement vs. absolute error')
abt=rd('exp_abstention.csv')
for a,g in abt.groupby('assay'):axs[2].plot(g.abstain_frac,g.rho_retained,'o-',label=nice(a))
axs[2].legend(fontsize=7);axs[2].set(xlabel='Abstained fraction',ylabel='Retained-set CV ρ');panel(axs[2],'C  Disagreement-based abstention');save(fig,'figS04_robustness')
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
tf=rd('tabular_fm.csv');heat(axs[0],tf.pivot(index='config',columns='model',values='mean_rho'),'A  Tabular learner screen · five primary assays',vmin=.2,vmax=.5)
harn=rd('harn_experiments.csv').set_index('assay');heat(axs[1],harn[['H1_global','H2_CDR','H3_FR','H4_CDR_FR','H5_chain','H6_hier','H7_PLM']],'B  Regional descriptor experiment',vmin=0,vmax=.7);save(fig,'figS05_extended_learners')

# Preserve existing structural maps as visualizations, without changing their pixels.
for src,dst in [('panel3_v2.pdf','figS06_surface_maps.pdf'),('panel_descriptor_interpretability.pdf','figS07_descriptor_ig.pdf'),('panel_gp_ablation.pdf','figS08_gp_ablation.pdf'),('panel_gp_comparison.pdf','figS09_gp_comparison.pdf')]:
    p=HERE.parent/src
    if p.exists():shutil.copy2(p,F/dst)

# Snapshot the evidence used in the rewrite, including result scopes.
sources=['final_evaluation_summary.csv','bootstrap_ci_all.csv','aw_moe.csv','awmoe_all9.csv','all7_assays.csv','awmoe_cv_errorbars.csv','all_models_splits.csv','exp_ablation.csv','exp_learning_curves.csv','exp_encoder_screen.csv','exp_struct_comparison.csv','delphi_comparison.csv','tabular_fm.csv','abdevfusion_all10_seeds.csv']
E=HERE/'evidence';E.mkdir(exist_ok=True)
for f in sources:
    if (R/f).resolve()!=(E/f).resolve():shutil.copy2(R/f,E/f)
print('Built',len(list(F.glob('*.pdf'))),'PDF panels in',F)
