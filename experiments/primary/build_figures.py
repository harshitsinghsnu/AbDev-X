"""Editable, uniform-width figures from traceable primary and comparison data.
All pages are 183 mm wide; font sizes are specified at final printed size.
"""
from pathlib import Path
import json,pickle,sys,textwrap
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm
H=Path(__file__).parent;ROOT=H.parents[1];R=ROOT/'results';E=H/'evidence';F=H/'figures'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
W=183/25.4;BLUE='#0072B2';ORANGE='#D55E00';GREY='#9ca8ae';COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00']
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5,'lines.linewidth':.85,'lines.markersize':3,'xtick.major.size':2,'ytick.major.size':2,'savefig.dpi':300})
PRIMARY=['HIC','AC-SINS','PR_CHO','SMAC','HAC'];COMP=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO','Titer']
NICE={'esm2':'ESM-2','ablang2':'AbLang2','desc':'Descriptors','descriptors':'Descriptors','bio':'Bio','isotype':'Subtype','AC-SINS_pH7.4':'AC-SINS','PR_CHO':'PR-CHO','Germline/Isotype':'Subtype','BioFeatures(ridge)':'Bio-R','BioFeatures(lgbm)':'Bio-L','Descriptors':'Desc','VH+VL_mean':'VH + VL','CDR_mean':'CDR only','AbLang2_baseline':'AbLang2','ESM2_650M_baseline':'ESM-2','ESM2_650M':'ESM-2 650M','ESM2_150M_pool':'ESM-2 150M','ProGen2_small':'ProGen2','AbDev-X':'Legacy source stack','AbDev-X (stacking)':'Legacy source stack','AbDev-MoE':'Legacy source MoE','abdevx2_full':'Experimental fusion'}
def nice(s):return NICE.get(str(s),str(s).replace('_',' '))
def feature_label(s):
 s=str(s).replace('SAP_','SAP ').replace('ELEC_','Electro ')
 for a,b in [('frac_aromatic','aromatic frac.'),('frac_aliphatic','aliphatic frac.'),('frac_polar','polar frac.'),('frac_pos','positive frac.'),('frac_neg','negative frac.'),('charge_pH','charge'),('hmoment','μH'),('deamidation_NG_NS','NG/NS motifs'),('isomerization_DG_DS','DG/DS motifs'),('aromatic_sasa','aromatic SASA')]:s=s.replace(a,b)
 return s.replace('_',' ')
def rd(n):return pd.read_csv(R/n)
def title(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=8)
def canvas(r,c,height=5.7,**kw):return plt.subplots(r,c,figsize=(W,height),layout='constrained',**kw)
def save(fig,name):
 if not getattr(fig,'_manual',False):fig.set_constrained_layout_pads(w_pad=.08,h_pad=.08,wspace=.08,hspace=.09)
 for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',dpi=300)
 plt.close(fig)
def heat(ax,df,s,vmin=None,vmax=None,cmap='YlGnBu',digits=2):
 z=df.to_numpy(float);diverging=cmap=='RdBu_r' and vmin is not None and vmax is not None and vmin<0<vmax
 kw={'norm':TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)} if diverging else {'vmin':vmin,'vmax':vmax}
 m=ax.imshow(z,aspect='auto',cmap=cmap,interpolation='nearest',**kw)
 ax.set_xticks(range(len(df.columns)),[nice(x) for x in df.columns],rotation=28,ha='right');ax.set_yticks(range(len(df)),[nice(x) for x in df.index]);title(ax,s)
 for i in range(z.shape[0]):
  for j in range(z.shape[1]):
   if np.isfinite(z[i,j]):
    val=0. if abs(z[i,j])<.5*10**(-digits) else z[i,j];norm=m.norm(z[i,j]);dark=norm>.72 or (diverging and norm<.18)
    ax.text(j,i,f'{val:.{digits}f}',ha='center',va='center',fontsize=6.5,color='white' if dark else '#151515')
 return m

def architecture():
 import verify_primary
 with open(E/'fitted_primary.pkl','rb') as f:z=pickle.load(f)
 meta=json.loads((E/'residue_shap_manifest.json').read_text());row=z['g3'].set_index('antibody_name').loc[meta['antibody_name']]
 fig=plt.figure(figsize=(W,6.5));ax=fig.add_axes([.02,.01,.96,.98]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 def box(x,y,w,h,txt,color='#e8f2f8',size=7):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006',facecolor=color,edgecolor='#526675',linewidth=.6));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=size,linespacing=1.5)
 def arrow(x,y,xx,yy):ax.annotate('',(xx,yy),(x,y),arrowprops={'arrowstyle':'-|>','lw':.65,'color':'#526675'})
 ax.text(.01,.985,'a  Paired sequence input and molecular context',fontweight='bold',va='top',fontsize=8)
 for x,lab,seq in [(.015,'Heavy chain (VH)',row.vh),(.395,'Light chain (VL)',row.vl)]:
  ax.text(x,.94,lab,fontweight='bold');ax.text(x,.915,'\n'.join(textwrap.wrap(seq,50)),fontfamily='Courier New',fontsize=6,va='top',linespacing=1.4)
 ax.text(.015,.806,'+ HC/LC subtype metadata',fontsize=7)
 ia=fig.add_axes([.775,.81,.105,.13]);ia.imshow(plt.imread(F/'architecture_asset_190.png'));ia.axis('off')
 ib=fig.add_axes([.88,.81,.105,.13]);ib.imshow(plt.imread(F/'representative_fv.png'));ib.axis('off')
 ax.text(.87,.79,'IgG context     Predicted Fv',ha='center',fontsize=6.5)
 ax.text(.01,.755,'b  Five operational input blocks',fontweight='bold',fontsize=8)
 labels=['ESM-2 650M\nFrozen VH/VL\n2,560 features','AbLang2\nFrozen cache\n480 features','Sequence descriptors\nChain / CDR / FR\n165 features','Structural bio\nTAP / MOE / A3D\nAntiFold · 57 features','Subtype metadata\nIgG1 / 2 / 4\nκ / λ · 5 features']
 for i,t in enumerate(labels):box(.01+i*.2,.63,.18,.095,t,size=6.5)
 for i in range(5):arrow(.1+i*.2,.63,.1+i*.2,.605)
 ax.plot([.1,.9],[.605,.605],color='#526675',lw=.6)
 ax.text(.01,.57,'c  Four experts fitted separately for each assay',fontweight='bold',fontsize=8)
 for x,y,t in [(.015,.37,'E1 · Best-source selector\n5 eligible blocks → OOD-CV scores\nChoose best source → refit ridge head'),(.52,.37,'A-MGFS · Assay-masked descriptors\nDescriptors + bio → keyword mask\nStandardize → optional PCA → RidgeCV'),(.015,.18,'D-Stack · Two-stream ridge stacker\n[ESM-2 + AbLang2] → PLM ridge prediction\n[Descriptors + bio] → second ridge prediction\nInner OOF predictions → ridge meta-head (α = 1)'),(.52,.18,'F-MoE · Mechanistic group mixture\nDescriptors + bio → eligible feature groups\nRidge group predictions → CV-score softmax (τ = 0.1)\nFixed assay-level weights → weighted sum')]:box(x,y,.46,.145,t,size=7)
 ax.text(.5,.535,'Shared head: training-fitted scaling; PCA (90%) if >100 features; RidgeCV',ha='center',fontsize=6.5)
 ax.text(.01,.12,'d  Primary AbDev-X prediction',fontweight='bold',fontsize=8)
 box(.015,.025,.48,.065,'ŷ = (ŷE1 + ŷA-MGFS + ŷD-Stack + ŷF-MoE) / 4','#d9edf0',7)
 box(.53,.025,.455,.065,'Primary evaluation: HIC · AC-SINS · PR-CHO · SMAC · HAC\nAdditional assays evaluated with the same code defaults','#f3f4f5',6.5);arrow(.495,.06,.53,.06)
 save(fig,'architecture_verified')

def main_results():
 p=pd.read_csv(E/'primary_reproduction.csv').set_index('assay');cv=pd.read_csv(E/'primary_cv_folds.csv');fig,axs=canvas(1,2,3.5)
 z=cv[cv.split=='cluster_isotype'].groupby('assay').rho.agg(['mean','std']).reindex(PRIMARY)
 base=pd.read_csv(E/'primary_baseline_cv_folds.csv');b=base[base.baseline=='esm2'].groupby('assay').rho.agg(['mean','std']).reindex(PRIMARY)
 for j,a in enumerate(PRIMARY):
  axs[0].errorbar(z.loc[a,'mean'],j-.1,xerr=z.loc[a,'std'],fmt='o',color=BLUE,capsize=2,label='AbDev-X' if j==0 else None)
  axs[0].errorbar(b.loc[a,'mean'],j+.1,xerr=b.loc[a,'std'],fmt='s',color=GREY,capsize=2,label='ESM-2 Ridge' if j==0 else None)
 axs[0].set_yticks(range(5),[nice(x) for x in PRIMARY]);axs[0].invert_yaxis();axs[0].set(xlabel='Cluster–isotype CV Spearman ρ ± SD',xlim=(-.1,1.02));axs[0].legend(loc='lower left',fontsize=6);title(axs[0],'a  Primary AbDev-X versus ESM-2 Ridge')
 for j,a in enumerate(PRIMARY):
  r=p.loc[a];axs[1].errorbar(r.rho,j,xerr=[[r.rho-r.ci_lo],[r.ci_hi-r.rho]],fmt='o',color=BLUE,capsize=2)
  axs[1].text(.91,j,f'{r.rho:.3f}',ha='right',va='center',fontsize=7)
 axs[1].set_yticks(range(5),[nice(x) for x in PRIMARY]);axs[1].invert_yaxis();axs[1].set(xlabel='GDPa3 Spearman ρ and BCa 95% CI',xlim=(-.1,.95));axs[1].axvline(0,c=GREY,ls=':',lw=.6);title(axs[1],'b  Primary held-out result · mean ρ = 0.472')
 save(fig,'main_results_verified')

def generalization():
 cv=pd.read_csv(E/'primary_cv_folds.csv');p=pd.read_csv(E/'primary_reproduction.csv').set_index('assay');fig,axs=canvas(2,2,5.9)
 z=cv.groupby(['split','assay']).rho.mean().unstack('split').reindex(PRIMARY)[['random','cluster','cluster_isotype']];z.plot.bar(ax=axs[0,0],color=COLORS[:3],width=.75);axs[0,0].set_xticklabels([nice(x) for x in PRIMARY],rotation=25,ha='right');axs[0,0].set(xlabel='',ylabel='Mean CV Spearman ρ');axs[0,0].legend(['Random','Cluster','Cluster–isotype'],loc='upper left',fontsize=6);title(axs[0,0],'a  Primary-model split sensitivity')
 z=cv[cv.split=='cluster_isotype'].groupby('assay').rho.agg(['mean','std'])
 for j,a in enumerate(PRIMARY):
  r=p.loc[a];axs[0,1].errorbar(z.loc[a,'mean'],r.rho,xerr=z.loc[a,'std'],yerr=[[r.rho-r.ci_lo],[r.ci_hi-r.rho]],fmt='o',c=COLORS[j],capsize=2,label=nice(a))
 axs[0,1].plot([0,1],[0,1],':',c=GREY);axs[0,1].set(xlabel='GDPa1 CV ρ ± SD',ylabel='GDPa3 ρ (BCa 95% CI)',xlim=(-.05,1),ylim=(-.05,1));axs[0,1].legend(fontsize=6,loc='upper left');title(axs[0,1],'b  Development-to-held-out calibration')
 z=rd('plm_ablation.csv').set_index('PLM')[COMP];heat(axs[1,0],z,'c  Encoder-only ablation · competition five',vmin=-.1,vmax=.65)
 ig=pd.read_csv(E/'primary_block_IG.csv');ig['abs']=ig.contribution.abs();z=ig.groupby(['assay','block'])['abs'].mean().unstack().reindex(PRIMARY);z=z.div(z.sum(1),axis=0);heat(axs[1,1],z,'d  Primary model · block contribution share',vmin=0,vmax=1)
 save(fig,'main_generalization_verified')

def encoders():
 fig,axs=canvas(3,2,6.5);d=rd('exp_crossab.csv');variants=['VH+VL_mean','CDR_mean','CrossAbSense','AbLang2_baseline','ESM2_650M_baseline']
 for ax,col,t in [(axs[0,0],'cv_cluster_isotype','a  Input variants · cluster–isotype CV'),(axs[0,1],'cv_random','b  Input variants · random CV')]:
  z=d[d.model.isin(variants)].pivot(index='model',columns='assay',values=col).reindex(variants)[COMP];heat(ax,z,t,vmin=-.1,vmax=.7)
 d=rd('exp_struct_comparison.csv');s=d.groupby('block').cv_cluster_isotype.mean().sort_values();axs[1,0].barh([nice(x) for x in s.index],s,color=GREY);axs[1,0].set(xlabel='Mean CV Spearman ρ');title(axs[1,0],'c  Structural groups and sequence references')
 d=rd('exp_encoder_screen.csv');s=d.groupby('encoder').cv_cluster_isotype.mean().sort_values();axs[1,1].barh([nice(x) for x in s.index],s,color=BLUE);axs[1,1].set(xlabel='Mean CV Spearman ρ');title(axs[1,1],'d  Full standalone encoder screen')
 s=d.groupby('encoder').gdpa3.mean().dropna().nlargest(4).sort_values();axs[2,0].barh([nice(x) for x in s.index],s,color=GREY);axs[2,0].set(xlabel='GDPa3 mean Spearman ρ');title(axs[2,0],'e  Held-out encoder-only comparison')
 d=rd('exp_struct_comparison.csv');z=d.pivot(index='block',columns='assay',values='cv_cluster_isotype');z=z.loc[[x for x in z.index if x not in ['ESM2','AbLang2','ESM2_650M','AbLang2_baseline']],COMP];heat(axs[2,1],z,'f  Structural groups · per-assay CV',vmin=-.1,vmax=.7)
 save(fig,'main_encoder_screen')

def ablations():
 fig,axs=canvas(2,2,5.6)
 z=pd.read_csv(E/'primary_source_ablation.csv').pivot(index='dropped',columns='assay',values='delta').reindex(columns=PRIMARY);heat(axs[0,0],z,'a  Refitted primary source removal · Δρ',vmin=-.15,vmax=.15,cmap='RdBu_r')
 d=pd.read_csv(E/'primary_learning_folds.csv')
 for i,(a,z) in enumerate(d.groupby('assay')):
  q=z.groupby('fraction').rho.agg(['mean','std']);axs[0,1].plot(q.index,q['mean'],'o-',label=nice(a),c=COLORS[i])
 axs[0,1].set(xlabel='Fraction of outer-fold training set',ylabel='Cluster–isotype CV Spearman ρ');axs[0,1].legend(loc='lower right',fontsize=6);title(axs[0,1],'b  Primary-model data efficiency')
 d=pd.read_csv(E/'primary_paired_significance.csv')
 for j,a in enumerate(PRIMARY):
  for k,b in enumerate(['esm2','ablang2']):
   r=d[(d.assay==a)&(d.baseline==b)].iloc[0];axs[1,0].errorbar(r.delta,j+(k-.5)*.22,xerr=[[r.delta-r.ci_lo],[r.ci_hi-r.delta]],fmt='o',c=COLORS[k],capsize=2,label=nice(b) if j==0 else None)
 axs[1,0].set_yticks(range(5),[nice(x) for x in PRIMARY]);axs[1,0].invert_yaxis();axs[1,0].axvline(0,c=GREY,ls=':');axs[1,0].set(xlabel='Primary − baseline Δρ · paired BCa 95% CI');axs[1,0].legend(fontsize=6);title(axs[1,0],'c  Matched held-out differences')
 d=pd.read_csv(E/'primary_predictions.csv');out={}
 from scipy.stats import spearmanr
 for a,z in d[d.assay.isin(PRIMARY)].groupby('assay'):
  full=spearmanr(z.observed,z.predicted,nan_policy='omit').statistic;out[a]=[full-spearmanr(z.observed,z[[x for x in ['E1','MGFS','DStack','FMoE'] if x!=drop]].mean(1),nan_policy='omit').statistic for drop in ['E1','MGFS','DStack','FMoE']]
 heat(axs[1,1],pd.DataFrame(out,index=['E1','A-MGFS','D-Stack','F-MoE']).reindex(columns=PRIMARY),'d  Leave-one-expert-out averaging · Δρ',vmin=-.1,vmax=.1,cmap='RdBu_r')
 save(fig,'main_primary_ablations')

def attribution():
 fig=plt.figure(figsize=(W,6.5),layout='constrained');gs=fig.add_gridspec(3,3,height_ratios=[1.05,1,1]);ax=fig.add_subplot(gs[0,:])
 d=pd.read_csv(E/'primary_residue_shap.csv');arrays=[]
 for a in PRIMARY:
  v=d[d.assay==a].shap.to_numpy();arrays.append(v/max(np.quantile(abs(v),.95),1e-12))
 im=ax.imshow(np.clip(arrays,-1,1),aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1);ax.axvline(122.5,c='black',lw=.7);ax.set_yticks(range(5),[nice(a) for a in PRIMARY]);ax.set_xticks([0,49,99,122,172,229],['H1','H50','H100','H123','L50','L107']);title(ax,'a  Primary descriptor pathway · residue SHAP (VH | VL)');fig.colorbar(im,ax=ax,shrink=.8,label='SHAP / 95th percentile |SHAP|',ticks=[-1,0,1])
 for k,a in enumerate(PRIMARY):
  ax=fig.add_subplot(gs[1+k//3,k%3]);ax.imshow(plt.imread(F/f'residue_{a}.png'));ax.axis('off');title(ax,f'{chr(98+k)}  {nice(a)} · conditional residue SHAP')
 ax=fig.add_subplot(gs[2,2]);ax.axis('off');ax.text(.02,.9,'Primary ensemble descriptor pathways\n\nReference: alanine VH + VL\n64 antithetic permutations\nPLM / bio / subtype inputs fixed\n\nRed: increases predicted assay value\nBlue: decreases predicted assay value\nColors capped at the 95th percentile\nwithin each assay; not a causal map.',va='top',fontsize=6.5,linespacing=1.5)
 save(fig,'main_primary_structure_attribution')

def descriptor_primary():
 d=pd.read_csv(E/'primary_feature_IG.csv');d=d[(d.block=='desc')&d.assay.isin(PRIMARY)];fig,axs=canvas(3,2,6.5)
 for ax,a in zip(axs.flat,PRIMARY):
  z=d[d.assay==a].nlargest(8,'mean_abs_ig').sort_values('mean_abs_ig');ax.barh([feature_label(x) for x in z.feature],z.mean_abs_ig,color=BLUE);ax.set(xlabel='Mean |IG| · primary prediction units');title(ax,f'{chr(97+PRIMARY.index(a))}  {nice(a)} · descriptor pathway')
 axs[2,1].axis('off');axs[2,1].text(.02,.9,'Exact IG from the primary fitted ensemble\n\nTraining-label-available feature means\nform the baseline for each assay.\n\nThese values include descriptor paths\nthrough E1, A-MGFS, D-Stack and F-MoE\nwhere those paths are active.',va='top',fontsize=7,linespacing=1.6)
 save(fig,'main_primary_descriptor_ig')

def headline_si():
 fig,axs=canvas(1,2,4.2);d=rd('final_evaluation_summary.csv').set_index('model');labels={'A1-E1':'E1','A2-ESM2':'ESM-2','A3-AbLang2':'AbLang2','A4-Desc':'Descriptors','A5-Bio':'Bio','A6-MGFS':'A-MGFS','A7-Dstack':'D-Stack','A8-FMoE':'F-MoE','AbDev-X':'AbDev-X'};v=d.loc[list(labels),'overall_mean'].sort_values();axs[0].barh([labels[x] for x in v.index],v,color=[BLUE if x=='AbDev-X' else GREY for x in v.index]);axs[0].set(xlim=(0,.55),xlabel='GDPa3 mean Spearman ρ');title(axs[0],'a  Same-five-assay comparison · nine strategies')
 for i,x in enumerate(v):axs[0].text(x+.006,i,f'{x:.3f}',va='center',fontsize=6.5)
 d=pd.read_csv(E/'primary_reproduction.csv')
 for i,r in d.iterrows():axs[1].errorbar(r.rho,i,xerr=[[r.rho-r.ci_lo],[r.ci_hi-r.rho]],fmt='o',c=BLUE if r.assay in PRIMARY else GREY,capsize=2)
 axs[1].set_yticks(range(10),[nice(x) for x in d.assay]);axs[1].invert_yaxis();axs[1].axvline(0,c=GREY,ls=':');axs[1].set(xlabel='GDPa3 ρ · primary code + default extension');title(axs[1],'b  Recomputed ten-assay scope and uncertainty');save(fig,'si_primary_scope')

def legacy_performance():
 # Rebuild exactly the archived panel's data, making model identities explicit.
 fig,axs=canvas(2,2,5.7);d=rd('benchmark_comparison.csv');x=np.arange(5);axs[0,0].bar(x-.17,d.abdevx,.34,label='Historical source stack',color=BLUE);axs[0,0].bar(x+.17,d.best_baseline,.34,label='Stored best baseline',color=GREY);axs[0,0].set_xticks(x,[nice(x) for x in d.assay],rotation=25,ha='right');axs[0,0].set_ylabel('Recorded CV Spearman ρ');axs[0,0].legend(fontsize=6);title(axs[0,0],'a  Archived configuration · local benchmark')
 d=rd('all_models_splits.csv');d=d[(d.split=='cluster_isotype')&(d.model!='AbDev-X')];
 for i,(_,row) in enumerate(d.iterrows()):axs[0,1].scatter(np.arange(5)+(i-len(d)/2)*.04,row[COMP].to_numpy(float),s=9,c=GREY)
 axs[0,1].set_xticks(range(5),[nice(x) for x in COMP],rotation=25,ha='right');axs[0,1].set_ylabel('CV Spearman ρ');title(axs[0,1],'b  Historical comparator CV distribution')
 # Original scatter provenance must not be fabricated; copy its numerical summary.
 axs[1,0].barh(['HIC','Tm2','Titer'],[.296,-.066,.157],color=GREY);axs[1,0].set(xlabel='Archived separate-evaluation Spearman ρ');title(axs[1,0],'c  Archived scatter evaluation · summary')
 axs[1,1].axis('off');axs[1,1].text(.02,.9,'Historical configuration results\n\nThe original selected PDF is preserved\nin the source archive. These values are\nnot the primary four-expert ensemble.\n\nCurrent primary results are reported\nin the main results figure and Table S2.',va='top',fontsize=7,linespacing=1.6);save(fig,'si_archived_performance')

def detailed_shap():
 fig=plt.figure(figsize=(W,6.5));fig._manual=True
 positions=[[.18,.70,.28,.23],[.69,.70,.28,.23],[.18,.365,.28,.23]]
 for i,a in enumerate(['HIC','Tm2','AC-SINS_pH7.4']):
  ax=fig.add_axes(positions[i]);d=rd(f'explain/shap_{a}.csv');z=d.set_index('feature').mean_abs_shap.nlargest(12).sort_values();ax.barh([feature_label(x) for x in z.index],z,color=COLORS[i]);ax.set_xlabel('Mean |SHAP|');title(ax,f'{chr(97+i)}  {nice(a)} · proxy model')
 ax=fig.add_axes([.62,.365,.35,.23]);z=rd('representation_weights.csv').set_index('assay');z.columns=['ESM-2','Desc','Subtype','Bio-R','Bio-L','AbLang2'];heat(ax,z,'d  Historical source-stack coefficients',vmin=-3,vmax=1,cmap='RdBu_r')
 for i,a in enumerate(COMP):
  ax=fig.add_axes([.065+i*.19,.075,.145,.17]);d=rd(f'explain/shap_perantibody_{a.replace("-","")}.csv') if a=='AC-SINS_pH7.4' else rd(f'explain/shap_perantibody_{a}.csv');d=d.select_dtypes('number').drop(columns=['true_label','pred_label','base_value'],errors='ignore');d=d.iloc[np.argsort(d.abs().sum(1))];groups={'Hydro':['gravy','hmoment','aromat','aliphat'],'Charge':['charge','pI','frac_pos','frac_neg'],'Polar':['frac_polar'],'Size':['_len'],'Liability':['deamid','isomer','nglyc','oxid','free_cys']};s=[]
  used=set()
  for cat,keys in groups.items():
   cols=[c for c in d if c not in used and any(k in c for k in keys)];used.update(cols);s.append(d[cols].sum(1).to_numpy())
  pos=np.zeros(len(d));neg=pos.copy()
  for j,v in enumerate(s):ax.bar(np.arange(len(d)),v,bottom=np.where(v>=0,pos,neg),width=1,color=COLORS[j]);pos+=np.maximum(v,0);neg+=np.minimum(v,0)
  ax.set(xticks=[],xlabel=nice(a));ax.tick_params(axis='y',labelsize=6)
 fig.text(.065,.272,'e  Per-antibody signed proxy SHAP: hydro / charge / polar / size / liability',fontsize=7,fontweight='bold')
 save(fig,'si_detailed_shap')

def soft_extensions():
 fig=plt.figure(figsize=(W,6.5));fig._manual=True;ax=fig.add_axes([.08,.72,.38,.22]);z=rd('soft_moe/soft_moe_cv.csv').pivot(index='assay',columns='method',values='rho').reindex(COMP);z.plot.bar(ax=ax,color=COLORS[:3]);ax.set_xticklabels([nice(x) for x in z.index],rotation=25,ha='right');ax.set(xlabel='',ylabel='CV Spearman ρ');ax.legend(['Oracle reference','Ridge stack','Soft gate'],fontsize=6,loc='upper left');title(ax,'a  Historical source-level fusion')
 ax=fig.add_axes([.61,.72,.36,.22]);z=rd('soft_moe/gate_weights.csv').pivot(index='assay',columns='block',values='mean_weight').reindex(COMP);heat(ax,z,'b  Source-gate weights',vmin=0,vmax=.35)
 ax=fig.add_axes([.08,.395,.21,.20]);z=rd('physics_constrained/physics_hic.csv');ax.bar(['Ridge','Isotonic SAP'],z.rho,color=[BLUE,GREY]);ax.set_ylabel('HIC CV ρ');title(ax,'c  Isotonic constraint')
 ax=fig.add_axes([.42,.395,.21,.20]);z=rd('physics_constrained/physics_rank_reg.csv')
 for i,(a,v) in enumerate(z.groupby('assay')):ax.plot(v['lambda'],v.rho,'o-',label=nice(a),c=COLORS[i])
 ax.set(xlabel='Rank-loss λ',ylabel='CV Spearman ρ');ax.legend(fontsize=6);title(ax,'d  Rank regularization')
 ax=fig.add_axes([.76,.395,.21,.20]);z=rd('physics_constrained/multitask_cv.csv');ax.scatter(z.rho_individual,z.rho_shared,s=12,c=BLUE);ax.plot([0,.8],[0,.8],':',c=GREY);ax.set(xlabel='Per-assay CV ρ',ylabel='Shared-task CV ρ');title(ax,'e  Multi-task sharing')
 bs=rd('heldout_gdpa4_bispecifics/arm_predictions.csv');raw=pd.read_csv(ROOT/'data/processed/gdpa4_bispecifics.csv');
 # Prefer merged observed table with explicitly identified key and labels.
 for i,a in enumerate(['HIC','AC-SINS_pH7.4','PR_CHO']):
  ax=fig.add_axes([.08+i*.34,.075,.21,.20]);merged=bs.merge(raw,left_on='bsab_name',right_on='antibody_name');obs=merged['AC-SINS' if a=='AC-SINS_pH7.4' else a];pred=merged['avg_'+a];ax.scatter(obs,pred,s=4,alpha=.65,c=BLUE,edgecolors='none');ax.set(xlabel='Observed assay value',ylabel='Arm-average prediction');title(ax,f'{chr(102+i)}  {nice(a)} · legacy transfer')
 save(fig,'si_soft_extensions')

def descriptor_requested():
 d=rd('explain/abdevx_descriptor_named_ig.csv');cats=['size','hydrophobic','charge','polar'];fig,axs=canvas(2,2,5.7)
 z=d[d.category.isin(cats)].groupby(['assay','category']).ig_share.sum().unstack().reindex(COMP);heat(axs[0,0],z,'a  Descriptor-only model · category IG share',vmin=0,vmax=.5)
 ax=axs[0,1];z.plot.bar(ax=ax,color=COLORS[:4],stacked=True);ax.set_xticklabels([nice(x) for x in z.index],rotation=25,ha='right');ax.set(xlabel='',ylabel='Summed IG share');ax.legend(fontsize=6);title(ax,'b  Four-category descriptor contributions')
 for ax,asub,t in [(axs[1,0],COMP[:3],'c  Top four descriptors · HIC / Tm2 / AC-SINS'),(axs[1,1],COMP[3:],'d  Top four descriptors · PR-CHO / Titer')]:
  q=pd.concat([d[d.assay==a].nsmallest(4,'rank') for a in asub]);ax.barh([nice(a)+' · '+feature_label(f) for a,f in zip(q.assay,q.feature)],q.ig_share,color=GREY);ax.invert_yaxis();ax.set_xlabel('IG share');title(ax,t)
 save(fig,'si_descriptor_requested')

def other_si():
 j=json.loads((R/'awmoe_explainability.json').read_text());fig,axs=canvas(3,1,6.1)
 z=rd('reviewer_response/selection_stability.csv').pivot(index='assay',columns='block',values='selection_fraction').fillna(0);heat(axs[0],z,'a  E1 source-selection stability',vmin=0,vmax=1)
 with open(E/'fitted_primary.pkl','rb') as f:models=pickle.load(f)['models']
 z=pd.DataFrame({a:models[a]['experts'][2].meta.coef_ for a in PRIMARY},index=['PLM stream','Descriptor + bio']).T;heat(axs[1],z,'b  Primary D-Stack · signed ridge coefficients',vmin=-.2,vmax=1.2,cmap='RdBu_r')
 z=pd.DataFrame({a:dict(zip([x[0] for x in models[a]['experts'][3].expert_masks],models[a]['experts'][3].weights)) for a in PRIMARY}).T.fillna(0);heat(axs[2],z,'c  Primary F-MoE · fixed group weights',vmin=0,vmax=1);save(fig,'si_expert_specialization')
 fig,axs=canvas(2,2,5.8)
 z=rd('abdevx2_ablation/metrics.csv').pivot(index='method',columns='split',values='MEAN');heat(axs[0,0],z,'a  Historical architecture screen',vmin=0,vmax=.55)
 z=rd('sparse_moe_gdpa3.csv');k=rd('kronecker_lstm_gdpa3.csv');axs[0,1].barh(['Source stack','Conditional','Top-2','Top-3','KronLSTM'],list(z.MEAN)+[k.MEAN.iloc[1]],color=GREY);axs[0,1].set_xlabel('GDPa3 mean ρ · competition five');title(axs[0,1],'b  Shared-input routing comparison')
 z=rd('hybrid_ensemble.csv').iloc[:3].set_index('model')[COMP].T;heat(axs[1,0],z,'c  Assay-specific routing trade-offs',vmin=-.1,vmax=.55)
 z=rd('exp_ig.csv').set_index('assay');axs[1,1].bar(np.arange(5),z.IG_CDR_hotspot_frac,color=BLUE);axs[1,1].plot(np.arange(5),z.CDR_baseline_frac,'o:',c=GREY);axs[1,1].set_xticks(range(5),[nice(x) for x in z.index],rotation=25,ha='right');axs[1,1].set_ylabel('CDR fraction in top-attributed residues');title(axs[1,1],'d  Separate chain-aware network · IG');save(fig,'si_routing_comparison')
 fig,axs=canvas(3,1,6.1)
 z=rd('heldout_gdpa5/spearman_summary.csv').set_index('method');heat(axs[0],z,'a  Historical source-stack VHH-Fc transfer',vmin=-.1,vmax=.4)
 z=rd('combined_training/spearman_summary.csv').set_index('method')[COMP];heat(axs[1],z,'b  Separate combined-format training study',vmin=-.1,vmax=.4)
 d=pd.read_csv(E/'primary_predictions.csv');rows=[]
 for a,q in d[d.assay.isin(PRIMARY)].groupby('assay'):
  q=q.dropna(subset=['observed','predicted']).reset_index(drop=True)
  for k in [5,10,20]:
   overlap=len(set(q.observed.nsmallest(k).index)&set(q.predicted.nsmallest(k).index));rows.append(dict(assay=a,k=k,precision=overlap/k,overlap=overlap,n=len(q),random_expected=k/len(q)))
 z=pd.DataFrame(rows);z.to_csv(E/'primary_precision_at_k.csv',index=False);heat(axs[2],z.pivot(index='assay',columns='k',values='precision').reindex(PRIMARY),'c  Primary model · per-assay precision at K',vmin=0,vmax=1);save(fig,'si_transfer_ranking')
 fig,axs=canvas(2,2,5.7)
 z=rd('exp_batch_correction.csv').pivot(index='strategy',columns='assay',values='gdpa3').reindex(columns=COMP);heat(axs[0,0],z,'a  Source-stack normalization / augmentation',vmin=-.35,vmax=.45)
 z=rd('exp_struct_noise.csv')
 for i,(g,v) in enumerate(z[z.assay=='HIC'].groupby('block')):axs[0,1].plot(v.sigma,v.rho_mean,'o-',label=g,c=COLORS[i])
 axs[0,1].set(xlabel='Numerical feature perturbation σ',ylabel='HIC CV ρ');axs[0,1].legend(fontsize=6);title(axs[0,1],'b  Structural-feature sensitivity')
 z=rd('exp_calibration.csv');axs[1,0].barh([nice(a) for a in z.assay],z.calib_rho_std_vs_error,color=GREY);axs[1,0].axvline(0,c=GREY);axs[1,0].set_xlabel('Disagreement–error correlation');title(axs[1,0],'c  Legacy source-stack confidence proxy')
 z=rd('exp_abstention.csv')
 for i,(a,v) in enumerate(z.groupby('assay')):axs[1,1].plot(v.abstain_frac,v.rho_retained,'o-',label=nice(a),c=COLORS[i])
 axs[1,1].set(xlabel='Abstained fraction',ylabel='Retained-set CV ρ');axs[1,1].legend(fontsize=6);title(axs[1,1],'d  Selective prediction diagnostic');save(fig,'si_robustness')
 fig,ax=canvas(1,1,3.4)
 z=rd('harn_experiments.csv').set_index('assay').iloc[:,:6];heat(ax,z,'Regional descriptor constructions · GDPa3 Spearman ρ',vmin=0,vmax=.6);save(fig,'si_learner_screen')

if __name__=='__main__':
 for fn in [architecture,main_results,generalization,encoders,ablations,attribution,descriptor_primary,headline_si,legacy_performance,detailed_shap,soft_extensions,descriptor_requested,other_si]:
  print(fn.__name__,flush=True);fn()
