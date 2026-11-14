from pathlib import Path
import json,shutil,re,runpy
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[2]/'experiments';O=P/'gaussian_process';F=P.parent/'outputs/figures';E=O/'evidence';R=P.parent/'results'
F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8,'legend.fontsize':6,'xtick.labelsize':6,'ytick.labelsize':6,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
comp=['HIC','Tm2','AC-SINS','PR_CHO','Titer'];nice=lambda a:a.replace('PR_CHO','PR-CHO')
cv=pd.read_csv(P/'primary/evidence/primary_cv_folds.csv');cv=cv[cv.split=='cluster_isotype'];cv.assay=cv.assay.replace({'AC-SINS_pH7.4':'AC-SINS'})
ext=pd.read_csv(P/'primary/evidence/extension_cv_folds.csv');cv=pd.concat([cv,ext[ext.assay.isin(['Tm2','Titer'])]],ignore_index=True)
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
p=pd.read_csv(P/'primary/evidence/primary_reproduction.csv').set_index('assay').loc[order]
b.bar(x-.20,p.rho,.26,yerr=[p.rho-p.ci_lo,p.ci_hi-p.rho],capsize=2,color='#1A237E',label='AbDev-X; BCa 95% CI',error_kw={'lw':.7})
j=pd.read_csv(P/'evaluation/evidence/jiang_verified.csv').set_index('assay');de=pd.read_csv(R/'delphi_comparison.csv').set_index('assay')
for i,aa in enumerate(order):
 if aa in j.index:b.scatter(i+.02,j.loc[aa,'best_point'],marker='s',facecolors='none',edgecolors='#657b83',s=20,label='Jiang best published point; CI NR' if i==0 else None)
 key='AC-SINS_pH7.4' if aa=='AC-SINS' else aa
 if key in de.index:b.scatter(i+.20,de.loc[key,'DELPHI_heldout'],marker='^',facecolors='none',edgecolors='#e15759',s=22,label='DELPHI stored point; CI NR' if i==0 else None)
for i,v in enumerate([.708,.392,.356,.337,.310]):b.scatter(i+.35,v,marker='D',facecolors='none',edgecolors='black',s=16,label='Competition champion; CI NR' if i==0 else None)
b.axhline(0,c='#aaaaaa',lw=.5);b.set_xticks(x,[nice(x) for x in order],rotation=25,ha='right');b.set_ylabel('GDPa3 Spearman ρ');b.set_ylim(-.6,1.02);b.set_title('B  Held-out comparison: intervals where available',loc='left',fontweight='bold');b.legend(ncol=2,loc='upper right',fontsize=6)
for ext in ['pdf','png','svg']:fig.savefig(F/f'main07_baseline_context.{ext}',dpi=350)
plt.close(fig)
