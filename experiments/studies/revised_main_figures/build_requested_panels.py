"""Build the requested main-panel set, keeping each experiment's scope explicit."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE=Path(__file__).parent;ROOT=HERE.parents[2];R=ROOT/'results' if (ROOT/'results').exists() else HERE/'evidence';F=HERE/'figures'
ASSETS=HERE.parent if (HERE.parent/'panel_encoder_struct.pdf').exists() else HERE/'source_assets'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':240})
PRIMARY=['HIC','AC-SINS_pH7.4','PR_CHO','SMAC','HAC']
COMP=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO','Titer']
CHAMP={'HIC':.708,'Tm2':.392,'AC-SINS_pH7.4':.337,'PR_CHO':.356,'Titer':.310}
COLORS=['#3568a8','#dd8452','#55a868','#9066a7','#b19c38']
def rd(n):return pd.read_csv(R/n)
def nice(s):return str(s).replace('AC-SINS_pH7.4','AC-SINS').replace('PR_CHO','PR-CHO')
def ttl(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=10)
def save(fig,n):
 fig.savefig(F/(n+'.pdf'),bbox_inches='tight');fig.savefig(F/(n+'.png'),bbox_inches='tight');plt.close(fig)
def heat(ax,z,s,cmap='RdBu_r',vmin=-.3,vmax=.7):
 ar=z.to_numpy(float);m=ax.imshow(ar,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
 ax.set_xticks(range(len(z.columns)),[nice(c) for c in z.columns],rotation=32,ha='right')
 ax.set_yticks(range(len(z)),[nice(c) for c in z.index]);ttl(ax,s)
 for i in range(ar.shape[0]):
  for j in range(ar.shape[1]):
   if np.isfinite(ar[i,j]):ax.text(j,i,f'{ar[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(m.norm(ar[i,j])-.5)>.37 else 'black')
 return m

# Requested performance overview: actual current ensemble, nine-model means, CV distribution.
cv=rd('awmoe_cv_errorbars.csv');cv=cv[cv.split=='cluster_isotype'].set_index('assay')
eb=rd('cv_errorbars.csv');esm=eb[eb.model=='ESM2-Ridge'].set_index('assay')
fig,axs=plt.subplots(3,1,figsize=(8.6,10.0),layout='constrained')
x=np.arange(5);w=.32
axs[0].bar(x-w/2,cv.loc[COMP,'mean'],w,yerr=cv.loc[COMP,'std'],capsize=3,color=COLORS[0],label='AbDev-X')
axs[0].bar(x+w/2,esm.loc[COMP,'mean'],w,yerr=esm.loc[COMP,'std'],capsize=3,color='#bcc4cc',label='ESM-2 Ridge')
for i,a in enumerate(COMP):
 delta=cv.loc[a,'mean']-esm.loc[a,'mean'];axs[0].text(i,max(cv.loc[a,'mean']+cv.loc[a,'std'],esm.loc[a,'mean']+esm.loc[a,'std'])+.025,f'Δ {delta:+.2f}',ha='center',fontsize=8)
axs[0].set_xticks(x,[nice(a) for a in COMP]);axs[0].set_ylabel('CV Spearman ρ');axs[0].set_ylim(-.3,.95);axs[0].legend(ncol=2,fontsize=8,loc='upper right');ttl(axs[0],'A  GDPa1 cluster–isotype CV · competition-five assays')
sm=rd('final_evaluation_summary.csv').set_index('model')
models=['A2-ESM2','A3-AbLang2','A4-Desc','A5-Bio','A1-E1','A6-MGFS','A7-Dstack','A8-FMoE','AbDev-X']
labels=['ESM-2','AbLang2','Descriptors','Bio','E1','A-MGFS','D-Stack','F-MoE','AbDev-X']
v=sm.loc[models,'overall_mean'].to_numpy();axs[1].bar(np.arange(9),v,color=['#b6bdc5']*8+[COLORS[0]])
for i,z in enumerate(v):axs[1].text(i,z+.007,f'{z:.3f}',ha='center',fontsize=8)
axs[1].set_xticks(range(9),labels,rotation=28,ha='right');axs[1].set_ylim(0,.55);axs[1].set_ylabel('GDPa3 mean ρ');ttl(axs[1],'B  Matched five positive-signal assays · nine strategies')
allcv=rd('all_models_splits.csv');allcv=allcv[allcv.split=='cluster_isotype'];primary=rd('awmoe_cv_splits.csv');pr=primary[primary.split=='cluster_isotype'].iloc[0]
for i,a in enumerate(PRIMARY):
 vals=allcv.loc[allcv.model!='AbDev-X',a].dropna().to_numpy() if a in allcv else np.array([])
 axs[2].scatter(np.full(len(vals),i)+np.linspace(-.1,.1,len(vals)),vals,color='#aab2b9',s=22)
 axs[2].scatter(i,pr[a],marker='D',color=COLORS[0],s=48,zorder=4)
 axs[2].text(i+.08,pr[a],f'{pr[a]:.3f}',fontsize=8,va='center')
axs[2].set_xticks(range(5),[nice(a) for a in PRIMARY]);axs[2].set_ylim(-.15,.98);axs[2].set_ylabel('Pooled OOF Spearman ρ');ttl(axs[2],'C  Recorded CV distribution · diamond = AbDev-X')
axs[2].text(.01,.02,'Comparator CV records are unavailable for SMAC/HAC; no points are imputed.',transform=axs[2].transAxes,fontsize=8,color='#555')
save(fig,'main02_performance')

# Requested representation generalization panel.
fig,axs=plt.subplots(2,2,figsize=(10.3,9.3),layout='constrained')
sp=rd('all_models_splits.csv');models=['AbDev-X (stacking)','ESM2-Ridge','AbLang2-ElasticNet','SaProt-Ridge'];splits=['random','cluster','cluster_isotype']
for i,m in enumerate(models):
 z=sp[sp.model==m].set_index('split').loc[splits,COMP].mean(axis=1)
 axs[0,0].plot(range(3),z,'o-',label=m.replace('AbDev-X (stacking)','Legacy source stack'),color=COLORS[i])
axs[0,0].set_xticks(range(3),['Random IID','Cluster OOD','Cluster–isotype'],rotation=15);axs[0,0].set_ylabel('Mean CV ρ · competition five');axs[0,0].legend(fontsize=7);ttl(axs[0,0],'A  Split sensitivity across four model configurations')
all10=rd('awmoe_all9.csv').set_index('assay');bca=rd('bootstrap_ci_all.csv').set_index('model').loc['AbDev-X']
cis={'Tm2':(-.063,.371),'Titer':(-.147,.293),'Tm1':(-.025,.611),'Tonset':(-.012,.414)}
def point(a):return all10.loc[nice(a).replace('PR-CHO','PR_CHO'),'rho']
def interval(a):
 k=nice(a).replace('PR-CHO','PR_CHO')
 return (float(bca[k+'_lo']),float(bca[k+'_hi'])) if k+'_lo' in bca.index else cis[k]
for a,c in zip(COMP,COLORS):
 y=point(a);lo,hi=interval(a);xx=cv.loc[a,'mean'];sd=cv.loc[a,'std']
 axs[0,1].errorbar(xx,y,xerr=sd,yerr=[[y-lo],[hi-y]],fmt='o',color=c,capsize=2,label=nice(a))
 axs[0,1].plot(xx,CHAMP[a],marker='D',mfc='none',mec=c,ms=5)
axs[0,1].plot([-.25,.9],[-.25,.9],'--',color='#888',lw=.8);axs[0,1].set(xlabel='GDPa1 CV ρ (fold mean ± SD)',ylabel='GDPa3 ρ (reported BCa CI)',xlim=(-.25,.9),ylim=(-.25,.9));axs[0,1].legend(fontsize=7,ncol=2);ttl(axs[0,1],'B  Current ensemble · diamonds = competition winners')
plm=rd('plm_ablation.csv');x=np.arange(len(plm));w=.14
for j,(a,c) in enumerate(zip(COMP,COLORS)):
 axs[1,0].bar(x+(j-2)*w,plm[a],w,yerr=plm[a+'_std'],color=c,label=nice(a),capsize=1,error_kw={'lw':.5})
axs[1,0].set_xticks(x,['AbLang2','ESM-2','AntiBERTy','ProtBert','ProGen2'],rotation=30,ha='right');axs[1,0].set_ylabel('Cluster–isotype CV ρ');axs[1,0].legend(fontsize=7,ncol=2);ttl(axs[1,0],'C  PLM ablation · competition-five assays')
rw=rd('representation_weights.csv').set_index('assay');rw.columns=['ESM2','Desc','Iso','Bio-R','Bio-L','AbL2'];heat(axs[1,1],rw,'D  Legacy source-stack ridge coefficients',vmin=-3,vmax=3)
save(fig,'main03_generalization')

# The existing encoder/structure panel already contains the requested six subpanels.
shutil.copy2(ASSETS/'panel_encoder_struct.pdf',F/'main04_encoder_structure.pdf')

# Requested LOO, learning, archival paired differences, and chain-aware IG.
fig,axs=plt.subplots(2,2,figsize=(10.3,8.9),layout='constrained')
ab=rd('exp_ablation.csv').pivot(index='assay',columns='dropped',values='dHeldout').reindex(index=COMP,columns=['esm2','descriptors','isotype','bio','ablang2']);ab.columns=['ESM2','Desc','Iso','Bio','AbL2'];heat(axs[0,0],ab,'A  Legacy stack · full minus source-removed held-out ρ',vmin=-.27,vmax=.27)
lc=rd('exp_learning_curves.csv')
for a,c in zip(COMP,COLORS):
 g=lc[lc.assay==a];axs[0,1].plot(100*g.frac,g.rho,'o-',label=nice(a),color=c)
axs[0,1].set(xlabel='Training fraction (%)',ylabel='Cluster–isotype CV ρ');axs[0,1].legend(fontsize=7);ttl(axs[0,1],'B  Data efficiency · competition-five assays')
sg=rd('exp_significance.csv')
for i,z in sg.iterrows():axs[1,0].errorbar(z.delta_rho,i,xerr=[[z.delta_rho-z.ci_low],[z.ci_high-z.delta_rho]],fmt='o',color=COLORS[COMP.index(z.assay)],capsize=2)
axs[1,0].set_yticks(range(len(sg)),[nice(z.assay)+' vs '+z.vs for z in sg.itertuples()],fontsize=7);axs[1,0].axvline(0,color='#888',ls='--');axs[1,0].set_xlabel('Archived model − baseline Δρ (reported CI)');ttl(axs[1,0],'C  Archived paired differences · not the primary run')
ig=rd('exp_ig.csv');xx=np.arange(len(ig));axs[1,1].bar(xx,ig.IG_CDR_hotspot_frac,color=COLORS)
axs[1,1].plot(xx,ig.CDR_baseline_frac,'k--',label='CDR-length baseline')
for i,z in enumerate(ig.itertuples()):axs[1,1].text(i,z.IG_CDR_hotspot_frac+.025,f'{z.enrichment:.1f}×',ha='center')
axs[1,1].set_xticks(xx,[nice(a) for a in ig.assay],rotation=25);axs[1,1].set_ylim(0,1);axs[1,1].set_ylabel('Fraction of IG assigned to CDRs');axs[1,1].legend(fontsize=7);ttl(axs[1,1],'D  Chain-aware proxy · CDR enrichment')
save(fig,'main05_efficiency')

# User-requested full SHAP and structural-map panel.
shutil.copy2(ASSETS/'panel3_v2.pdf',F/'main06_shap_structure.pdf')

# Baseline panel: traceable matched/local values plus explicitly contextual records.
fig=plt.figure(figsize=(10.2,10.0),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.05]);a=fig.add_subplot(gs[0,0]);c=fig.add_subplot(gs[0,1]);b=fig.add_subplot(gs[1,:])
lb=rd('full_leaderboard.csv').rename(columns={'Unnamed: 0':'model'}).set_index('model');assays=COMP[:4];x=np.arange(4);legacy=cv.loc[assays,'mean'].to_numpy(float);best=lb.drop(index='AbDevX')[assays].max().to_numpy(float)
a.bar(x-.16,legacy,.32,yerr=cv.loc[assays,'std'],capsize=3,color=COLORS[0],label='AbDev-X (fold mean ± SD)');a.bar(x+.16,best,.32,color='#b4bec6',label='Best local benchmark entry')
for i in range(4):
 if legacy[i]>best[i]:a.text(i,legacy[i]+cv.loc[assays[i],'std']+.025,'★',ha='center')
a.set_xticks(x,[nice(z) for z in assays],rotation=25);a.set_ylim(-.18,.98);a.set_ylabel('GDPa1 CV Spearman ρ');a.legend(fontsize=7,loc='upper right');ttl(a,'A  Local benchmark context · rank stars only')
a.text(.02,.03,'KitAb/Amini: no verified matching numeric table;\nno invented points or error bars are plotted.',transform=a.transAxes,fontsize=7,color='#555')
# Preserve the supplied GP comparison as a contextual subpanel; label its model correctly.
src=fitz.open(ASSETS/'panel_gp_comparison.pdf');p=src[0]
rename_rects=p.search_for('AbDev-X')
for rect in rename_rects:
 p.add_redact_annot(rect,fill=(1,1,1))
p.apply_redactions()
for rect in rename_rects:p.insert_text((rect.x0,rect.y1-2),'D-Stack',fontsize=7.5,color=(0,0,0))
# The graphic remains available as its original PDF in the supplement.
pix=p.get_pixmap(matrix=fitz.Matrix(2.2,2.2));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);c.imshow(im);c.axis('off');ttl(c,'C  Legacy D-Stack configuration vs. GP kernels')
display=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO','Titer','Tm1','Tonset'];xx=np.arange(7)
rho=np.array([point(z) for z in display]);ints=np.array([interval(z) for z in display]);b.bar(xx-.23,rho,.23,color=COLORS[0],label='AbDev-X · recorded BCa CI');b.errorbar(xx-.23,rho,yerr=[rho-ints[:,0],ints[:,1]-rho],fmt='none',ecolor='#333',capsize=3)
j=rd('all7_assays.csv').set_index('assay');jkeys=[nice(z).replace('PR-CHO','PR_CHO') for z in display];jvals=np.array([j.loc[z,'jiang_best'] if z in j.index else np.nan for z in jkeys],float)
b.scatter(xx,jvals,marker='s',facecolors='none',edgecolors=COLORS[2],s=45,label='Jiang · archived/digitized estimates')
de=rd('delphi_comparison.csv').set_index('assay');dvals=[de.loc[z,'DELPHI_heldout'] if z in de.index else np.nan for z in display];b.scatter(xx+.20,dvals,marker='^',color=COLORS[3],s=40,label='DELPHI · stored configuration / sign audit')
for i,z in enumerate(display):
 if z in CHAMP:b.scatter(i,CHAMP[z],marker='D',facecolors='none',edgecolors='black',s=50,label='Competition winner' if i==0 else None)
b.set_xticks(xx,[nice(z) for z in display]);b.axhline(0,color='#888',lw=.7);b.axvline(4.5,color='#aaa',ls=':');b.set_ylim(-.36,.95);b.set_ylabel('GDPa3 Spearman ρ');b.legend(fontsize=7,ncol=2,loc='upper right');ttl(b,'B  Held-out context · seven assays, heterogeneous comparison protocols')
save(fig,'main07_baseline_context')

# Freeze every CSV used by this script for the editable source package.
import ast
for node in ast.walk(ast.parse(Path(__file__).read_text(encoding='utf-8'))):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='rd' and node.args and isinstance(node.args[0],ast.Constant):
  f=node.args[0].value;dst=HERE/'evidence'/f;dst.parent.mkdir(parents=True,exist_ok=True)
  if (R/f).resolve()!=dst.resolve():shutil.copy2(R/f,dst)
print('Built requested main panels, with experiment-scope labels.')
