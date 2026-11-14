from pathlib import Path
import json,shutil,re,runpy
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;O=P/'benchmark_pymol_revision';F=O/'figures';E=O/'evidence';R=P.parents[1]/'results'
shutil.copytree(P/'primary_attribution_revision/figures',F,dirs_exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8,'legend.fontsize':6,'xtick.labelsize':6,'ytick.labelsize':6,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
comp=['HIC','Tm2','AC-SINS','PR_CHO','Titer'];nice=lambda a:a.replace('PR_CHO','PR-CHO')
cv=pd.read_csv(P/'verified_revision/evidence/primary_cv_folds.csv');cv=cv[cv.split=='cluster_isotype'];cv.assay=cv.assay.replace({'AC-SINS_pH7.4':'AC-SINS'})
ext=pd.read_csv(P/'verified_revision/evidence/extension_cv_folds.csv');cv=pd.concat([cv,ext[ext.assay.isin(['Tm2','Titer'])]],ignore_index=True)
summ=cv.groupby('assay').rho.agg(['mean','std'])
gp=pd.read_csv(E/'matched_gp_summary.csv')
fig=plt.figure(figsize=(183/25.4,6.5),layout='constrained');gs=fig.add_gridspec(2,2);a=fig.add_subplot(gs[0,0]);c=fig.add_subplot(gs[0,1]);b=fig.add_subplot(gs[1,:])
lb=pd.read_csv(R/'full_leaderboard.csv').set_index('Unnamed: 0');order=comp[:4];x=np.arange(4)
a.bar(x-.15,summ.loc[order,'mean'],.3,yerr=summ.loc[order,'std'],capsize=2,color='#1A237E',label='AbDev-X ± fold SD')
best=lb.drop(index='AbDevX')[['AC-SINS_pH7.4' if x=='AC-SINS' else x for x in order]].max().to_numpy(float)
a.scatter(x+.15,best,c='#657b83',marker='o',s=17,label='Best stored local point; CI NR')
a.set_xticks(x,[nice(x) for x in order],rotation=28,ha='right');a.set_ylabel('GDPa1 CV Spearman ρ');a.set_ylim(-.25,1);a.set_title('A  Local benchmark comparison',loc='left',fontweight='bold');a.legend(loc='upper left')
for i,model in enumerate(['AbDev-X','GP block-additive','GP stream-additive','GP stream-product']):
 q=summ.loc[comp] if i==0 else gp[gp.model==model].set_index('assay').loc[comp]
 c.bar(np.arange(5)+(i-1.5)*.19,q['mean'],.19,yerr=q['std'],capsize=1.5,label=model,color=['#1A237E','#E65100','#FF8A65','#FFCCBC'][i],error_kw={'lw':.6})
c.set_xticks(range(5),[nice(x) for x in comp],rotation=28,ha='right');c.set_ylabel('CV ρ ± fold SD');c.set_ylim(-.45,1);c.set_title('C  Primary AbDev-X and matched GP kernels',loc='left',fontweight='bold');c.legend(ncol=1,loc='upper right',fontsize=5.5)
order=['HIC','Tm2','AC-SINS','PR_CHO','Titer','Tm1','Tonset'];x=np.arange(7)
p=pd.read_csv(P/'verified_revision/evidence/primary_reproduction.csv').set_index('assay').loc[order]
b.bar(x-.20,p.rho,.26,yerr=[p.rho-p.ci_lo,p.ci_hi-p.rho],capsize=2,color='#1A237E',label='AbDev-X; BCa 95% CI',error_kw={'lw':.7})
j=pd.read_csv(P/'assay_revision/evidence/jiang_verified.csv').set_index('assay');de=pd.read_csv(R/'delphi_comparison.csv').set_index('assay')
for i,aa in enumerate(order):
 if aa in j.index:b.scatter(i+.02,j.loc[aa,'best_point'],marker='s',facecolors='none',edgecolors='#657b83',s=20,label='Jiang best published point; CI NR' if i==0 else None)
 key='AC-SINS_pH7.4' if aa=='AC-SINS' else aa
 if key in de.index:b.scatter(i+.20,de.loc[key,'DELPHI_heldout'],marker='^',facecolors='none',edgecolors='#e15759',s=22,label='DELPHI stored point; CI NR' if i==0 else None)
for i,v in enumerate([.708,.392,.356,.337,.310]):b.scatter(i+.35,v,marker='D',facecolors='none',edgecolors='black',s=16,label='Competition champion; CI NR' if i==0 else None)
b.axhline(0,c='#aaaaaa',lw=.5);b.set_xticks(x,[nice(x) for x in order],rotation=25,ha='right');b.set_ylabel('GDPa3 Spearman ρ');b.set_ylim(-.6,1.02);b.set_title('B  Held-out comparison: intervals where available',loc='left',fontweight='bold');b.legend(ncol=2,loc='upper right',fontsize=6)
for ext in ['pdf','png','svg']:fig.savefig(F/f'main07_baseline_context.{ext}',dpi=350)
plt.close(fig)
# Reuse the established three-row attribution layout, replacing only surface images with PyMOL renders.
source=(P/'update_model_structure_attributions.py').read_text(encoding='utf-8');part=source[source.index('plt.rcParams.update'):source.index("s=(OLD/'AbDevX_evidence_led_narrative_main_SI.tex')")]
import pickle,sys
sys.path[:0]=[str(P.parents[1]/'scripts'),str(P.parents[1]/'src')]
import aw_moe as AW
from scipy.stats import rankdata
with open(P/'verified_revision/evidence/fitted_primary.pkl','rb') as f:z=pickle.load(f)
cols=z['descriptor_columns']+z['bio_columns'];g=z['g3'];V=P/'verified_revision'
part=part.replace("V/'figures'/f'residue_{a}.png'","O/'structures'/f'residue_{a}.png'")
part=part.replace("cmap='RdBu_r'","cmap='bwr'")
exec(part,globals())
s=(P/'primary_attribution_revision/AbDevX_primary_attribution_main_SI.tex').read_text(encoding='utf-8')
s=s.replace(r'\textbf{C}, the archival source-stack/GP graphic, transcribed from its vector bars; it is not a primary-model GP experiment.',r'\textbf{C}, the primary four-expert AbDev-X and three newly specified GP kernels evaluated on the same cached inputs and outer cluster--isotype folds; all bars show fold means $\pm$ sample SD. These kernels are block-additive, stream-additive, and stream-product, as defined in Methods; they are not reproductions of the earlier GP-FA/HA/HP configurations.')
s=s.replace('Published comparator intervals are unavailable; cross-study point comparisons are descriptive.','Published comparator intervals are unavailable: those results are displayed as open point symbols labeled CI NR, rather than bars with invented error bars. All bars in B have BCa 95\% intervals. Cross-study point comparisons are descriptive.')
s=s.replace('primary-code cluster--isotype CV means $\\pm$ fold SD and best stored local entries; stars indicate larger point estimates, not statistical significance.','primary-code cluster--isotype CV means $\\pm$ fold SD and best stored local point estimates (open uncertainty status: CI NR); no significance stars are used.')
s=s.replace('the largest of Jiang','open symbols for the largest of Jiang')
old='The surface is a mesh of the union of atomic van der Waals spheres inflated by a 1.4-\\AA{} probe, sampled on a 0.8-\\AA{} grid; triangle colors inherit the residue score of the nearest atom. All assay views use the same orientation.'
new='Surfaces were ray-traced using PyMOL through pymol2, with a 1.4-\\AA{} solvent probe and its molecular solvent-excluded surface. Residue scores were assigned to the B-factor field, colored by residue using a blue--white--red palette, and saved as PDB and PyMOL session files. All assay views use the same orientation.'
assert old in s;s=s.replace(old,new).replace('mapped to the ABodyBuilder3 Fv of GDPa3 antibody','rendered as PyMOL molecular surfaces and mapped to the ABodyBuilder3 Fv of GDPa3 antibody')
gptext=r'''\subsection{Matched Gaussian-process comparison}
We compared the primary ensemble with three explicitly specified GP kernels on identical outer cluster--isotype folds. Each eligible cached input block was standardized on labeled outer-training antibodies; blocks above 100 dimensions used training-fitted 90\%-variance PCA. Per-block squared Euclidean distances were divided by the transformed dimension. The block-additive kernel averages RBF kernels over blocks. The stream-additive kernel averages two RBF kernels, one on the mean PLM-block distance and one on the mean descriptor/bio/subtype distance. The stream-product kernel applies an RBF kernel to the sum of those two stream distances. Isotype is excluded for ineligible assays. Length scale, signal variance, and noise variance were fitted using outer-training Gaussian marginal likelihood with two deterministic starts; test folds were not used for fitting or kernel selection. Targets were standardized within training folds. Error bars show sample SD across the five folds. These are new matched benchmarks, not reconstructions of the legacy GP-FA/HA/HP experiment.

'''
s=s.replace(r'\subsection{Attribution of the primary fitted predictor}',gptext+r'\subsection{Attribution of the primary fitted predictor}')
summary='The matched GP comparison gives an assay-dependent ordering (Figure~\\ref{fig:baseline_main}C). The GP correlations on AC-SINS and PR-CHO reach approximately $.61$ and $.51$, respectively, showing that the primary ensemble does not uniformly lead these development comparisons. These CV points do not replace the separate GDPa3 evaluation.\n\n'
s=s.replace(r'\subsection{The clearest benefits concern',summary+r'\subsection{The clearest benefits concern')
(O/'AbDevX_benchmarks_PyMOL_main_SI.tex').write_text(s,encoding='utf-8')
print('Figures and manuscript written')
