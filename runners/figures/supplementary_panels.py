from pathlib import Path
import sys,inspect,re,json,hashlib
import numpy as np,pandas as pd
H=Path(__file__).resolve().parents[2]/'experiments/evaluation';OLD=H;F=H.parents[1]/'outputs/figures'
sys.path.insert(0,str(OLD));import make_revision_figures as m
m.F=F;m.b.F=F
def edit(fn,changes):
 src=inspect.getsource(fn)
 for a,b in changes:
  assert a in src,(fn.__name__,a[:60]);src=src.replace(a,b)
 exec(compile(src,'SI_plot_variety.py','exec'),vars(m))
edit(m.routing,[
 ("axs[1,0].set_ylabel('CV ρ; uncertainty NR')","axs[1,0].set_ylabel('GDPa3 ρ; uncertainty NR')"),
 ("for j,(m,r) in enumerate(z.iterrows()):axs[1,0].plot(range(5),r,'o-',label=nice(m),mfc='white',ms=3)","for j,(model,r) in enumerate(z.iterrows()):axs[1,0].bar(np.arange(5)+(j-1)*.24,r,.22,facecolor='white',edgecolor=COL[j],linewidth=.8,label=nice(model))"),
 ("for j,m in enumerate(['mean','cdr','framework']):\n  q=z.xs(m).reindex(COMP);ax.errorbar(np.arange(5)+(j-1)*.08,q['mean'],yerr=q['std'],fmt='o-',ms=2.5,capsize=1,lw=.6,label=m)\n ax.set_xticks(range(5),[nice(a) for a in COMP],rotation=28,ha='right');ax.set_ylabel('CV ρ ± fold SD');ax.legend(fontsize=5.5);ttl(ax,'D  Regional pooling');finish(fig,'si04_routing_architecture_unique')",
 "for j,model in enumerate(['mean','cdr','framework']):\n  q=z.xs(model).reindex(COMP);ax.barh(np.arange(5)+(j-1)*.24,q['mean'],.22,xerr=q['std'],capsize=1,color=COL[j],error_kw={'lw':.5},label=model)\n ax.set_yticks(range(5),[nice(a) for a in COMP]);ax.set_xlabel('CV ρ ± fold SD');ax.legend(fontsize=5.5,ncol=3,loc='upper center');ax.set_ylim(-.7,5.1);ttl(ax,'D  Regional pooling');finish(fig,'si04_routing_architecture_unique')")])
edit(m.transfer,[
 ("for m,row in z.iterrows():ax.plot([5,10,20],row,'o-',mfc='white',ms=3,label=nice(m))\n ax.set(xlabel='K',ylabel='Precision at K');ax.legend(fontsize=5.5,ncol=3);ttl(ax,'C  Archived composite ranking: uncertainty NR');finish(fig,'si05_transfer_ranking_unique')",
 "for j,(model,row) in enumerate(z.iterrows()):ax.bar(np.arange(3)+(j-1)*.23,row,.21,color=GREY if model=='Random' else COL[j],label=nice(model))\n ax.set_xticks(range(3),['5','10','20']);ax.set(xlabel='K',ylabel='Precision at K',ylim=(0,.35));ax.legend(fontsize=5.5,ncol=3,loc='upper left');ttl(ax,'C  Composite ranking: point estimates');finish(fig,'si05_transfer_ranking_unique')")])
edit(m.robustness_and_learners,[
 ("z=z.set_index('assay').reindex(COMP);ax.errorbar(np.arange(5)+(j-2)*.05,z.gdpa3,yerr=[z.gdpa3-z.gdpa3_ci95_lo,z.gdpa3_ci95_hi-z.gdpa3],fmt='o-',ms=2.3,capsize=1,lw=.5,label=m)\n ax.set_xticks(range(5),[nice(a) for a in COMP],rotation=28,ha='right');ax.set_ylabel('GDPa3 ρ; recorded 95% CI');ax.legend(fontsize=5.5,ncol=2);ttl(ax,'A  Normalization / augmentation')",
 "z=z.set_index('assay').reindex(COMP);ax.errorbar(z.gdpa3,np.arange(5)+(j-2)*.13,xerr=[z.gdpa3-z.gdpa3_ci95_lo,z.gdpa3_ci95_hi-z.gdpa3],fmt='o',ms=2.3,capsize=1,lw=.5,label=m)\n ax.set_yticks(range(5),[nice(a) for a in COMP]);ax.set_xlabel('GDPa3 ρ; recorded 95% CI');ax.set_ylim(-.7,6.1);ax.legend(fontsize=5.5,ncol=2,loc='upper left');ttl(ax,'A  Normalization / augmentation')"),
 ("for j,(m,q) in enumerate(z.groupby('model')):\n  q=q.set_index('assay').reindex(PRIMARY);ax.errorbar(np.arange(5)+(j-2.5)*.05,q.rho,yerr=[q.rho-q.ci_lo,q.ci_hi-q.rho],fmt='o-',ms=2.2,capsize=.8,lw=.5,label=m.replace('_',' '))\n ax.set_xticks(range(5),[nice(a) for a in PRIMARY],rotation=28,ha='right');ax.set_ylabel('GDPa3 ρ; BCa 95% CI');ax.legend(fontsize=5.5,ncol=3);ttl(ax,'B  Regional descriptor constructions');finish(fig,'si07_learner_screen_unique')",
 "models=['H1_global','H2_CDR','H3_FR','H4_CDR_FR']\n for j,a in enumerate(PRIMARY):\n  q=z[z.assay==a].set_index('model').reindex(models);ax.errorbar(q.rho,np.arange(4)+(j-2)*.13,xerr=[q.rho-q.ci_lo,q.ci_hi-q.rho],fmt='o',ms=2.5,capsize=1,lw=.55,label=nice(a))\n ax.set_yticks(range(4),['Global','CDR only','Framework / global','CDR+FR / chain / hierarchical']);ax.set(xlabel='GDPa3 ρ; BCa 95% CI',ylim=(-.6,4.5));ax.legend(fontsize=5.5,ncol=5,loc='upper center');ttl(ax,'B  Regional descriptors: equivalent views merged');finish(fig,'si07_learner_screen_unique')")])
for fn in [m.routing,m.transfer,m.robustness_and_learners]:print(fn.__name__,flush=True);fn()

