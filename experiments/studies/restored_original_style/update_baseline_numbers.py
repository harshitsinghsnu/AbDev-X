"""Update A/B numbers in the selected PDF; retain original layout and C graphic."""
from pathlib import Path
import numpy as np,pandas as pd,fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
H=Path(__file__).parent;ROOT=H.parents[2];F=H/'figures';E=H/'evidence'
src=fitz.open(H/'original_figures/main07_baseline_context.pdf');page=src[0];pw,ph=page.rect.width,page.rect.height
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':6.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
BLUE='#1A237E';GRAY='#90A4AE';RED='#FF5252'
fig=plt.figure(figsize=(pw/72,ph/72));a=fig.add_axes([38.3931/pw,(ph-227.5446)/ph,170.5790/pw,205.2038/ph]);b=fig.add_axes([38.3931/pw,(ph-534.1191)/ph,405.9779/pw,235.9844/ph])
cv=pd.read_csv(E/'awmoe_cv_errorbars.csv');cv=cv[cv.split=='cluster_isotype'].set_index('assay');lb=pd.read_csv(ROOT/'results/full_leaderboard.csv').rename(columns={'Unnamed: 0':'model'}).set_index('model');order=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO'];labels=['HIC','Tm2','AC-SINS','PR_CHO'];x=np.arange(4);w=.24
v=cv.loc[order,'mean'].to_numpy();sd=cv.loc[order,'std'].to_numpy();best=lb.drop(index='AbDevX')[order].max().to_numpy(float)
a.bar(x-w/2,v,w,yerr=sd,capsize=2.5,color=BLUE,error_kw={'lw':.8});a.bar(x+w/2,best,w,color=GRAY)
for i in range(4):
 if v[i]>best[i]:a.text(x[i]-w/2,v[i]+sd[i]+.025,'★',ha='center',fontsize=10)
a.set_xticks(x,labels,rotation=20);a.set_ylim(-.22,.9);a.set_ylabel('Spearman ρ (GDPa1 CV)');a.axhline(0,c='#888',lw=.4);a.text(-.15,1.02,'(A)',transform=a.transAxes,fontweight='bold',fontsize=9)
a.legend([Patch(color=BLUE),Patch(color=GRAY)],['AbDev-X (±1 SD, 5-fold)','Best stored local benchmark'],loc='upper right',frameon=False,fontsize=6)
a.text(.02,.025,'KitAb: matching values unavailable',transform=a.transAxes,fontsize=5.8,color='#555')
p=pd.read_csv(E/'primary_reproduction.csv').set_index('assay');j=pd.read_csv(ROOT/'results/all7_assays.csv').set_index('assay');de=pd.read_csv(ROOT/'results/delphi_comparison.csv').set_index('assay');order=['HIC','Tm2','AC-SINS','PR_CHO','Titer','Tm1','Tonset'];x=np.arange(7);w=.22
v=p.loc[order,'rho'].to_numpy();lo=p.loc[order,'ci_lo'].to_numpy();hi=p.loc[order,'ci_hi'].to_numpy();b.bar(x-w,v,w,yerr=[v-lo,hi-v],color=BLUE,capsize=2.5,error_kw={'lw':.8},label='AbDev-X (BCa 95% CI)')
jj=[j.loc[z,'jiang_best'] if z in j.index else np.nan for z in order];dd=[de.loc['AC-SINS_pH7.4' if z=='AC-SINS' else z,'DELPHI_heldout'] if ('AC-SINS_pH7.4' if z=='AC-SINS' else z) in de.index else np.nan for z in order]
b.bar(x,jj,w,color=GRAY,label='Jiang et al. (archived estimates)');b.bar(x+w,dd,w,color=RED,label='DELPHI (stored configuration)')
champ={'HIC':.708,'Tm2':.392,'AC-SINS':.337,'PR_CHO':.356,'Titer':.310}
for i,z in enumerate(order):
 if z in champ:b.scatter(i,champ[z],marker='D',facecolors='none',edgecolors='#666',s=20,linewidths=.7,label='GDPa competition champion' if i==0 else None)
b.set_xticks(x,['HIC','Tm2','AC-SINS','PR_CHO','Titer','Tm1','Tonset']);b.set_ylim(-.4,.85);b.set_ylabel('Spearman ρ (GDPa3 held-out)');b.axhline(0,c='#aaa',lw=.4);b.axvline(4.5,c='#bbb',ls=':',lw=.5);b.text(-.065,1.03,'(B)',transform=b.transAxes,fontweight='bold',fontsize=9)
b.legend(loc='upper right',ncol=2,frameon=False,fontsize=6);b.text(.55,.018,'competition assays →   ← Jiang-matched only',transform=b.transAxes,fontsize=5.5,color='#999')
fig.savefig(H/'baseline_AB_numeric_overlay.pdf');plt.close(fig)
# The original C panel is copied without redrawing. Only correct its source-model name.
from matplotlib.font_manager import findfont
fontfile=Path(findfont('DejaVu Sans'))
targets=[]
for r in page.search_for('AbDev-X'):
 if r.x0>250 and r.y1<260:
  spans=[s for block in page.get_text('dict')['blocks'] if 'lines'in block for line in block['lines'] for s in line['spans'] if fitz.Rect(s['bbox']).intersects(r)];targets.append((r,spans[0]['size'],spans[0]['origin'][1]));page.add_redact_annot(r,fill=(1,1,1))
page.apply_redactions(images=0,graphics=0)
page.insert_font(fontname='SourceSans',fontfile=str(fontfile))
for r,size,baseline in targets:page.insert_text((r.x0,baseline),'D-Stack',fontname='SourceSans',fontsize=size)
regions=[fitz.Rect(0,0,235,272),fitz.Rect(0,275,pw,ph)]
for rect in regions:page.add_redact_annot(rect,fill=(1,1,1))
page.apply_redactions(images=0,graphics=2)
overlay=fitz.open(H/'baseline_AB_numeric_overlay.pdf');out=fitz.open();q=out.new_page(width=pw,height=ph);q.show_pdf_page(q.rect,src,0)
for rect in regions:q.show_pdf_page(rect,overlay,0,clip=rect)
out.save(F/'main07_baseline_context.pdf')
print('Updated A/B numbers in-place; original bar charts, layout, palette and GP panel retained.')
