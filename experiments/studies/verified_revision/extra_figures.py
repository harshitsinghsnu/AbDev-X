from build_figures import *
from scipy.stats import spearmanr

def context():
 fig,axs=canvas(1,2,3.9);cv=pd.read_csv(E/'primary_cv_folds.csv');cv=cv[cv.split=='cluster_isotype'];cv=pd.concat([cv,pd.read_csv(E/'extension_cv_folds.csv')]);z=cv.groupby('assay').rho.agg(['mean','std']);bench=rd('benchmark_comparison.csv').set_index('assay');order=['HIC','Tm2','AC-SINS','PR_CHO']
 for j,a in enumerate(order):
  axs[0].errorbar(z.loc[a,'mean'],j-.1,xerr=z.loc[a,'std'],fmt='o',c=BLUE,capsize=2,label='Primary code ± fold SD' if j==0 else None)
  key='AC-SINS_pH7.4' if a=='AC-SINS' else a;axs[0].plot(bench.loc[key,'best_baseline'],j+.1,'s',c=GREY,label='Stored best local entry' if j==0 else None)
 axs[0].set_yticks(range(4),[nice(a) for a in order]);axs[0].invert_yaxis();axs[0].set_xlabel('GDPa1 CV Spearman ρ');axs[0].legend(loc='lower left',fontsize=6);title(axs[0],'a  Primary code and local benchmark context')
 p=pd.read_csv(E/'primary_reproduction.csv').set_index('assay');j=rd('all7_assays.csv').set_index('assay');de=rd('delphi_comparison.csv').set_index('assay');order=['HIC','Tm2','AC-SINS','PR_CHO','Titer','Tm1','Tonset']
 for i,a in enumerate(order):
  r=p.loc[a];axs[1].errorbar(r.rho,i,xerr=[[r.rho-r.ci_lo],[r.ci_hi-r.rho]],fmt='o',c=BLUE,capsize=2,label='Primary code · BCa 95% CI' if i==0 else None)
  if a in j.index and pd.notna(j.loc[a,'jiang_best']):axs[1].plot(j.loc[a,'jiang_best'],i+.13,'x',c=ORANGE,label='Archived Jiang estimate' if i==0 else None)
  key='AC-SINS_pH7.4' if a=='AC-SINS' else a
  if key in de.index:axs[1].plot(de.loc[key,'DELPHI_heldout'],i-.13,'+',c=GREY,label='Stored DELPHI configuration' if i==0 else None)
 axs[1].set_yticks(range(len(order)),[nice(a) for a in order]);axs[1].invert_yaxis();axs[1].axvline(0,c=GREY,ls=':');axs[1].set_xlabel('GDPa3 Spearman ρ · heterogeneous protocols');axs[1].legend(loc='lower right',fontsize=6);title(axs[1],'b  External comparison context · seven assays');save(fig,'main_benchmark_context')

def archived():
 fig=plt.figure(figsize=(W,6.2));fig._manual=True
 ax=fig.add_axes([.10,.68,.85,.25]);d=rd('benchmark_comparison.csv');x=np.arange(5);ax.bar(x-.17,d.abdevx,.34,color=BLUE,label='Archived source stack');ax.bar(x+.17,d.best_baseline,.34,color=GREY,label='Stored best baseline');ax.set_xticks(x,[nice(a) for a in d.assay]);ax.set_ylabel('Recorded CV ρ');ax.legend(fontsize=6,loc='lower left');title(ax,'a  Archived local benchmark comparison')
 d=rd('heldout_configs/gdpa3_esm2_3b_eval.yaml/predictions.csv')
 for i,a in enumerate(['HIC','Tm2','Titer']):
  ax=fig.add_axes([.10+i*.315,.365,.23,.20]);y=d['true_'+a];p=d['AbDev-X_'+a];ok=y.notna()&p.notna();ax.scatter(y[ok],p[ok],s=5,c=BLUE,alpha=.7);ax.set(xlabel='Observed',ylabel='Predicted');rho=spearmanr(y[ok],p[ok]).statistic;title(ax,f'b{i+1}  {a} · ρ = {rho:.3f}')
 ax=fig.add_axes([.10,.07,.85,.20]);d=rd('all_models_splits.csv');d=d[(d.split=='cluster_isotype')&(d.model!='AbDev-X')]
 for i,(_,r) in enumerate(d.iterrows()):ax.scatter(np.arange(5)+(i-len(d)/2)*.045,r[COMP].to_numpy(float),s=9,c=GREY)
 ax.set_xticks(range(5),[nice(a) for a in COMP]);ax.set_ylabel('CV Spearman ρ');title(ax,'c  Historical model CV distribution · competition five');save(fig,'si_archived_performance')

def remaining_maps():
 fig,axs=canvas(3,2,6.5);order=['Tm1','Tm2','Tonset','SEC','Titer']
 for i,(ax,a) in enumerate(zip(axs.flat,order)):
  ax.imshow(plt.imread(F/f'residue_{a}.png'));ax.axis('off');title(ax,f'{chr(97+i)}  {a} · default extension attribution')
 axs[2,1].axis('off');axs[2,1].text(.02,.9,'Same representative antibody and\nconditional descriptor-pathway SHAP\nprotocol as the main structure figure.\n\nThe five additional assays use the\nunchanged primary-code defaults.\n\nThese maps do not validate predictive\naccuracy on the weak endpoints.',va='top',fontsize=7,linespacing=1.5);save(fig,'si_extension_structure_maps')

if __name__=='__main__':
 for fn in [context,archived,remaining_maps,main_results,attribution]:print(fn.__name__,flush=True);fn()
