"""Final-size typography; preserve main chart types/palettes and diversify SI."""
from pathlib import Path
import sys,json,pickle,textwrap,os
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import fitz
H=Path(__file__).parent;ROOT=H.parents[2];R=ROOT/'results';F=H/'figures'
os.chdir(ROOT)
E=H.parent/'restored_original_style/evidence';V=H.parent/'verified_revision'
sys.path[:0]=[str(V),str(ROOT/'scripts'),str(ROOT/'src')]
import build_figures as old
W=183/25.4;BLUE='#276b9a';GREY='#9aaab5';COL=['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#bcbd22']
COMP=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO','Titer'];PRIMARY=['HIC','AC-SINS','PR_CHO','SMAC','HAC']
CHAMP={'HIC':.708,'Tm2':.392,'AC-SINS':.356,'PR_CHO':.337,'Titer':.310}
plt.rcParams.update({'font.family':'Arial','font.size':6.5,'axes.titlesize':7,'axes.labelsize':6.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5,'lines.linewidth':.8,'lines.markersize':3,'xtick.major.size':2,'ytick.major.size':2,'savefig.dpi':300})
def rd(n):return pd.read_csv(R/n)
def nice(s):return old.nice(str(s)).replace('Legacy source stack','Source stack').replace('Experimental fusion','Fusion').replace('BioFeatures','Bio')
def ttl(ax,t):ax.set_title(t,loc='left',fontweight='bold',pad=7)
def canvas(r,c,h=6.35):return plt.subplots(r,c,figsize=(W,h),layout='constrained')
def finish(fig,n):
 if not getattr(fig,'_manual',False):fig.set_constrained_layout_pads(w_pad=.065,h_pad=.08,wspace=.10,hspace=.10)
 for ax in fig.axes:
  for t in ax.texts:
   if t.get_fontsize()<5.5:t.set_fontsize(5.5)
 for ext in ['pdf','svg','png']:fig.savefig(F/f'{n}.{ext}',dpi=300)
 plt.close(fig)
def heat(ax,z,t,vmin=0,vmax=1,cmap='YlGnBu'):
 a=z.to_numpy(float);m=ax.imshow(a,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
 ax.set_xticks(range(len(z.columns)),[nice(x) for x in z.columns],rotation=30,ha='right');ax.set_yticks(range(len(z)),[nice(x) for x in z.index]);ttl(ax,t)
 for i in range(a.shape[0]):
  for j in range(a.shape[1]):
   if np.isfinite(a[i,j]):
    norm=m.norm(a[i,j]);dark=norm>.76 or (cmap=='RdBu_r' and norm<.2)
    ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=6,color='white' if dark else '#151515')
 return m
def generalization():
 fig,axs=canvas(2,2);cv=pd.read_csv(E/'awmoe_cv_errorbars.csv');p=pd.read_csv(E/'primary_reproduction.csv').set_index('assay')
 ax=axs[0,0];x=np.arange(5);w=.25
 for i,(s,c,l) in enumerate(zip(['random','cluster','cluster_isotype'],['#1A3A6E','#3A7CC1','#7EC8E3'],['Random','Cluster','Cluster–isotype'])):
  q=cv[(cv.model=='AbDev-X')&(cv.split==s)].set_index('assay').reindex(['HIC','AC-SINS_pH7.4','PR_CHO','SMAC','HAC']);ax.bar(x+(i-1)*w,q['mean'],w,yerr=q['std'],color=c,capsize=1.5,error_kw={'lw':.5},label=l)
 ax.set_xticks(x,[nice(a) for a in PRIMARY],rotation=28,ha='right');ax.set(ylabel='CV Spearman ρ',ylim=(0,1));ax.legend(loc='upper left',fontsize=6);ttl(ax,'A  AbDev-X across CV splits')
 ax=axs[0,1];q=cv[(cv.model=='AbDev-X')&(cv.split=='cluster_isotype')].set_index('assay')
 for a,c in zip(COMP,COL):
  r=p.loc['AC-SINS' if a=='AC-SINS_pH7.4' else a];m=q.loc[a,'mean'];sd=q.loc[a,'std'];ax.errorbar(m,r.rho,xerr=sd,yerr=[[r.rho-r.ci_lo],[r.ci_hi-r.rho]],fmt='o',c=c,capsize=1.5,label=nice(a));ax.plot(m,CHAMP['AC-SINS' if a=='AC-SINS_pH7.4' else a],marker='D',mfc='none',mec=c,ms=4)
 ax.plot([-.4,1],[-.4,1],'k--',alpha=.4,lw=.6);ax.set(xlabel='GDPa1 CV ρ ± SD',ylabel='GDPa3 ρ (BCa 95% CI)',xlim=(-.4,1),ylim=(-.4,1));ax.legend(loc='upper left',ncol=2,fontsize=6);ttl(ax,'B  CV to held-out; diamonds: champions')
 ax=axs[1,0];d=rd('plm_ablation.csv').sort_values('MEAN',ascending=False);x=np.arange(len(d));w=.16
 for i,(a,c) in enumerate(zip(COMP,COL)):ax.bar(x+(i-2)*w,d[a],w,yerr=d[a+'_std'],color=c,capsize=1,error_kw={'lw':.4},label=nice(a))
 ax.set_xticks(x,[x.replace(' (Ab)','').replace(' (small)','').replace(' (650M)','') for x in d.PLM],rotation=30,ha='right');ax.set_ylabel('Cluster–isotype CV ρ');ax.legend(ncol=2,fontsize=6);ttl(ax,'C  Standalone PLM comparison')
 d=rd('representation_weights.csv').set_index('assay').reindex(COMP);d.columns=['ESM2','Desc','Iso','Bio-R','Bio-L','AbL2'];im=heat(axs[1,1],d,'D  Historical stack coefficients',-3,3,'RdBu_r');fig.colorbar(im,ax=axs[1,1],shrink=.7,label='Ridge coefficient');finish(fig,'main03_generalization')
def encoders():
 fig,axs=canvas(2,3);d=rd('exp_crossab.csv');var=['VH+VL_mean','CDR_mean','CrossAbSense','AbLang2_baseline','ESM2_650M_baseline']
 for ax,k,t in [(axs[0,0],'cv_cluster_isotype','A  Cluster–isotype input screen'),(axs[0,1],'cv_random','B  Random input screen')]:
  z=d[d.model.isin(var)].pivot(index='model',columns='assay',values=k).reindex(var)[COMP];heat(ax,z,t,-.1,.7)
 d=rd('exp_struct_comparison.csv');z=d.groupby('block').cv_cluster_isotype.mean().sort_values();ax=axs[0,2];ax.barh([nice(x) for x in z.index],z,color='#3A7CC1');ax.set_xlabel('Mean CV ρ');ttl(ax,'C  Structural feature groups')
 d=rd('exp_encoder_screen.csv');z=d.groupby('encoder').cv_cluster_isotype.mean().sort_values();ax=axs[1,0];ax.barh([nice(x).replace(' pooled','') for x in z.index],z,color='#1A3A6E');ax.set_xlabel('Mean CV ρ');ttl(ax,'D  Full encoder screen')
 z=d.groupby('encoder').gdpa3.mean().dropna().nlargest(4).sort_values();ax=axs[1,1];ax.barh([nice(x) for x in z.index],z,color='#3A7CC1');ax.set_xlabel('GDPa3 mean ρ');ttl(ax,'E  Held-out encoder screen')
 d=rd('exp_struct_comparison.csv');z=d.pivot(index='block',columns='assay',values='cv_cluster_isotype');z=z.loc[[x for x in z.index if x not in ['ESM2','AbLang2','ESM2_650M','AbLang2_baseline']],COMP];heat(axs[1,2],z,'F  Structural CV by assay',-.1,.7);finish(fig,'main04_encoder_structure')
def efficiency():
 fig,axs=canvas(2,2);d=pd.read_csv(E/'exp_ablation.csv');z=d.pivot(index='assay',columns='dropped',values='dHeldout').reindex(index=COMP,columns=['esm2','descriptors','isotype','bio','ablang2']);z.columns=['ESM2','Desc','Iso','Bio','AbL2'];heat(axs[0,0],z,'A  Primary source removal: Δρ',-.27,.27,'RdBu_r')
 d=pd.read_csv(E/'exp_learning_curves.csv')
 for a,c in zip(COMP,COL):
  z=d[d.assay==a];axs[0,1].plot(z.frac*100,z.rho,'o-',c=c,label=nice(a))
 axs[0,1].set(xlabel='Training fraction (%)',ylabel='Cluster–isotype CV ρ');axs[0,1].legend(fontsize=6);ttl(axs[0,1],'B  Data efficiency')
 d=pd.read_csv(E/'exp_significance.csv');ax=axs[1,0]
 for i,r in d.iterrows():ax.errorbar(r.delta_rho,i,xerr=[[r.delta_rho-r.ci_low],[r.ci_high-r.delta_rho]],fmt='o',c=COL[COMP.index(r.assay)],capsize=2)
 ax.set_yticks(range(len(d)),[nice(r.assay)+' / '+r.vs for r in d.itertuples()]);ax.axvline(0,c='#888',ls='--',lw=.6);ax.set_xlabel('AbDev-X − baseline Δρ; BCa 95% CI');ttl(ax,'C  Paired held-out differences')
 d=rd('exp_ig.csv');ax=axs[1,1];x=np.arange(len(d));ax.bar(x,d.IG_CDR_hotspot_frac,color=COL);ax.plot(x,d.CDR_baseline_frac,'k--',lw=.7,label='CDR-length baseline')
 for i,r in enumerate(d.itertuples()):ax.text(i,r.IG_CDR_hotspot_frac+.025,f'{r.enrichment:.1f}×',ha='center',fontsize=6.5)
 ax.set_xticks(x,[nice(a) for a in d.assay],rotation=28,ha='right');ax.set(ylabel='Fraction of IG in CDRs',ylim=(0,1));ax.legend(fontsize=6);ttl(ax,'D  Chain-aware network diagnostic');finish(fig,'main05_efficiency')
def descriptor():
 d=rd('explain/abdevx_descriptor_named_ig.csv');cats=['size','hydrophobic','charge','polar'];colors=['#4c78a8','#f58518','#54a24b','#b279a2'];fig=plt.figure(figsize=(W,6.35),layout='constrained');gs=fig.add_gridspec(2,2,width_ratios=[1,1.7]);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:]);z=d[d.category.isin(cats)].groupby(['assay','category']).ig_share.sum().unstack().reindex(COMP)[cats];heat(a,z,'A  Descriptor IG share',0,.5)
 q=pd.concat([d[d.assay==assay].nsmallest(4,'rank') for assay in COMP]);labels=[]
 for r in q.itertuples():labels.append(shortfeature(r.feature))
 b.bar(np.arange(len(q)),q.ig_share,color=[colors[cats.index(x)] if x in cats else GREY for x in q.category]);b.set_xticks(range(len(q)),labels,rotation=65,ha='right');b.set_ylabel('IG share');ttl(b,'B  Top four descriptors per assay')
 for i,assay in enumerate(COMP):b.text(i*4+1.5,b.get_ylim()[1]*.95,nice(assay),ha='center',fontsize=6);b.axvline(i*4-.5,c='#ddd',lw=.4)
 z.plot.bar(ax=c,stacked=True,color=colors,width=.65);c.set_xticklabels([nice(x) for x in z.index],rotation=28,ha='right');c.set(xlabel='',ylabel='Summed displayed IG share');c.legend(ncol=4,fontsize=6);ttl(c,'C  Descriptor category contributions');finish(fig,'main08_descriptor_interpretability')
def shortfeature(s):
 s=str(s)
 s=s.replace('SAP_hydrophobic_frac_sasa','SAP hydro.%').replace('SAP_sap_total','SAP total')
 for a,b in [('SAP_aromatic_sasa','SAP arom.'),('SAP_hydrophobic_sasa','SAP hydro.'),('SAP_total_sasa','SAP total'),('ELEC_','El. '),('deamidation_NG_NS','NG/NS'),('isomerization_DG_DS','DG/DS'),('frac_aromatic','arom.'),('frac_aliphatic','aliph.'),('frac_polar','polar'),('frac_pos','pos.'),('frac_neg','neg.'),('charge_pH','charge'),('hmoment','μH'),('hydrophobic','hydro.'),('nglyc','N-glyc'),('free_cys','free Cys'),('gravy','KD'),('CDR3','H3' if s.startswith('H_') else 'L3')]:s=s.replace(a,b)
 return s.replace('_',' ').replace('H H3','H3').replace('L L3','L3')
def shap_structure():
 # Retain the exact original ten molecular surface pictures; replace embedded tiny labels.
 source=fitz.open(H.parent/'restored_original_style/figures/main06_shap_structure.pdf');imgs=source[0].get_images(full=True)
 infos=source[0].get_image_info(xrefs=True);infos=sorted(infos,key=lambda x:(round(x['bbox'][1]),x['bbox'][0]));fig=plt.figure(figsize=(W,6.65));fig._manual=True
 from abdev.data.dataset import load_gdpa1
 from abdev.features.descriptors import build_descriptor_matrix
 ds=load_gdpa1(str(ROOT/'configs/default.yaml'));values=build_descriptor_matrix(ds.heavy,ds.light).reset_index(drop=True);ids=ds.df[ds.id_col].astype(str).values
 for file,cols,prefix in [('sap_gdpa1.csv',['sap_total','sap_positive','hydrophobic_sasa','aromatic_sasa','hydrophobic_frac_sasa'],'SAP_'),('electro_gdpa1.csv',['exp_pos','exp_neg','net_exp_charge','charge_asym','max_pos_patch','max_neg_patch'],'ELEC_')]:
  q=pd.read_csv(ROOT/'data/processed'/file).set_index('antibody_name').reindex(ids)[cols].reset_index(drop=True).add_prefix(prefix);values=pd.concat([values,q],axis=1)
 values.index=ids
 for i,a in enumerate(COMP):
  key=a.replace('-','') if a=='AC-SINS_pH7.4' else a;d=rd(f'explain/shap_perantibody_{key}.csv').set_index('antibody_name').select_dtypes('number').drop(columns=['true_label','pred_label','base_value'],errors='ignore');features=d.abs().mean().nlargest(12).index
  ax=fig.add_axes([.092+i*.199,.695,.101,.235]);rng=np.random.default_rng(104+i)
  for j,f in enumerate(reversed(features)):
   v=d[f].to_numpy();rank=values.reindex(d.index)[f].rank(pct=True).fillna(.5).to_numpy();ax.scatter(v,j+rng.uniform(-.22,.22,len(v)),s=1.3,c=rank,cmap='coolwarm',vmin=0,vmax=1,alpha=.65,rasterized=True)
  ax.set_yticks(range(len(features)),[shortfeature(f) for f in reversed(features)],fontsize=5.5);ax.tick_params(axis='x',labelsize=5.5);ax.locator_params(axis='x',nbins=2);ax.axvline(0,c='#aaa',lw=.4);ax.set_xlabel('SHAP',fontsize=6);ttl(ax,nice(a))
 fig.text(.015,.975,'A  Physicochemical proxy SHAP; feature value: low (blue) to high (red)',fontsize=7,fontweight='bold')
 labels=[('SMAC','Hydrophobic exposure'),('HAC','Hydrophobic exposure'),('Tm1','Aromatic exposure'),('Tonset','Aromatic exposure'),('SEC','Aggregation patches'),('HIC','Hydrophobic exposure'),('AC-SINS','Charge field'),('PR-CHO','Descriptor context'),('Tm2','Aromatic exposure'),('Titer','Net surface charge')]
 from PIL import Image
 for k,(assay,field) in enumerate(labels):
  inf=infos[5+k];pix=fitz.Pixmap(source,inf['xref']);arr=np.array(Image.frombytes('RGB' if pix.n==3 else 'RGBA',[pix.width,pix.height],pix.samples).convert('RGB'))
  # These embedded images contain only molecular pixels; keep their complete extent.
  row=k//5;col=k%5;ax=fig.add_axes([.012+col*.199,.365-row*.31,.187,.25]);ax.imshow(arr);ax.axis('off');ax.set_title(assay+'\n'+field,fontsize=6,pad=1)
 fig.text(.015,.64,'B  Additional assay descriptor fields',fontsize=7,fontweight='bold');fig.text(.015,.33,'C  Competition-assay descriptor fields',fontsize=7,fontweight='bold');finish(fig,'main06_shap_structure')
def profiles(ax,z,t,ylabel='Spearman ρ',legend=True):
 for i,(name,row) in enumerate(z.iterrows()):ax.plot(range(len(z.columns)),row,'o-',label=nice(name),color=plt.get_cmap('tab10')(i%10),ms=3,lw=.75)
 ax.set_xticks(range(len(z.columns)),[nice(x) for x in z.columns],rotation=28,ha='right');ax.set_ylabel(ylabel);ttl(ax,t)
 if legend:ax.legend(fontsize=5.5,ncol=2)
def dotcompare(ax,z,t,xlabel='Spearman ρ'):
 for j,c in enumerate(z.columns):ax.scatter(z[c],np.arange(len(z))+(j-(len(z.columns)-1)/2)*.15,s=12,label=nice(c))
 ax.set_yticks(range(len(z)),[nice(x) for x in z.index]);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.legend(fontsize=5.5,ncol=2);ttl(ax,t)
def supplementary():
 # Keep the requested S1/S2 content; render labels in final-size type.
 old.F=F;old.E=V/'evidence';old.feature_label=shortfeature;old.title=ttl;old.save=finish;old.heat=lambda ax,df,s,vmin=None,vmax=None,cmap='YlGnBu',digits=2:heat(ax,df,s,vmin,vmax,cmap)
 old.detailed_shap();(F/'si_detailed_shap.pdf').replace(F/'si01_detailed_shap.pdf')
 old.soft_extensions();(F/'si_soft_extensions.pdf').replace(F/'si02_soft_moe_extensions.pdf')
 old.headline_si();(F/'si_primary_scope.pdf').replace(F/'main_headline_unique.pdf') if (F/'si_primary_scope.pdf').exists() else None
 fig,axs=canvas(3,1,6.4);z=rd('reviewer_response/selection_stability.csv').pivot(index='assay',columns='block',values='selection_fraction').fillna(0);z.plot.bar(ax=axs[0],stacked=True,colormap='tab20',width=.65);axs[0].set_xticklabels([nice(x) for x in z.index],rotation=28,ha='right');axs[0].set(xlabel='',ylabel='Selection frequency');axs[0].legend(ncol=5,fontsize=5.5);ttl(axs[0],'A  Source-selection stability')
 with open(V/'evidence/fitted_primary.pkl','rb') as f:models=pickle.load(f)['models']
 z=pd.DataFrame({a:models[a]['experts'][2].meta.coef_ for a in PRIMARY},index=['PLM','Desc + bio']).T;dotcompare(axs[1],z,'B  Primary D-Stack coefficients','Signed ridge coefficient')
 z=pd.DataFrame({a:dict(zip([x[0] for x in models[a]['experts'][3].expert_masks],models[a]['experts'][3].weights)) for a in PRIMARY}).T.fillna(0);z.plot.bar(ax=axs[2],stacked=True,colormap='tab20',width=.65);axs[2].set_xticklabels([nice(x) for x in z.index],rotation=28,ha='right');axs[2].set(xlabel='',ylabel='Fixed group weight');axs[2].legend(ncol=4,fontsize=5.5);ttl(axs[2],'C  Primary F-MoE mechanistic mixture');finish(fig,'si03_expert_specialization_unique')
 fig,axs=canvas(2,2);z=rd('abdevx2_ablation/metrics.csv').pivot(index='method',columns='split',values='MEAN');z.index=[str(x).replace('abdevx2_','').replace('_',' ') for x in z.index];dotcompare(axs[0,0],z,'A  Architecture screen: CV means')
 z=rd('sparse_moe_gdpa3.csv');k=rd('kronecker_lstm_gdpa3.csv');axs[0,1].barh(['Fixed ridge','Conditional','Top-2','Top-3','KronLSTM'],list(z.MEAN)+[k.MEAN.iloc[1]],color=[BLUE]+['#b9c9d3']*4);axs[0,1].set_xlabel('GDPa3 mean ρ');ttl(axs[0,1],'B  Shared-input routing')
 z=rd('hybrid_ensemble.csv').iloc[:3].set_index('model')[COMP];profiles(axs[1,0],z,'C  Assay-specific routing trade-offs')
 z=rd('exp_pooling.csv').set_index('pooling')[COMP];profiles(axs[1,1],z,'D  Regional pooling');finish(fig,'si04_routing_architecture_unique')
 fig,axs=canvas(3,1);z=rd('heldout_gdpa5/spearman_summary.csv').set_index('method');z.T.plot.bar(ax=axs[0],width=.7,colormap='tab10');axs[0].set_xticklabels([nice(x) for x in z.columns],rotation=28,ha='right');axs[0].set(xlabel='',ylabel='GDPa5 ρ');axs[0].legend(fontsize=5.5,ncol=4);ttl(axs[0],'A  Historical VHH-Fc transfer')
 z=rd('combined_training/spearman_summary.csv').set_index('method')[COMP];dotcompare(axs[1],z.T,'B  Combined-format training')
 z=rd('ranking_comparison.csv').set_index('model').loc[['Random','ESM2-Ridge','AbDev-X (stacking)'],['P5','P10','P20']];profiles(axs[2],z,'C  Historical composite-ranking precision','Precision');finish(fig,'si05_transfer_ranking_unique')
 fig,axs=canvas(2,2);z=rd('exp_batch_correction.csv').pivot(index='strategy',columns='assay',values='gdpa3').reindex(columns=COMP);profiles(axs[0,0],z,'A  Normalization / augmentation')
 z=rd('exp_struct_noise.csv')
 for g,q in z[z.assay=='HIC'].groupby('block'):axs[0,1].plot(q.sigma,q.rho_mean,'o-',label=g)
 axs[0,1].set(xlabel='Feature perturbation σ',ylabel='HIC CV ρ');axs[0,1].legend(fontsize=6);ttl(axs[0,1],'B  Structural sensitivity')
 z=rd('exp_calibration.csv');axs[1,0].barh([nice(a) for a in z.assay],z.calib_rho_std_vs_error,color=GREY);axs[1,0].axvline(0,c='#333',lw=.5);axs[1,0].set_xlabel('Disagreement–error correlation');ttl(axs[1,0],'C  Confidence diagnostic')
 z=rd('exp_abstention.csv')
 for a,q in z.groupby('assay'):axs[1,1].plot(q.abstain_frac,q.rho_retained,'o-',label=nice(a))
 axs[1,1].set(xlabel='Abstained fraction',ylabel='Retained-set CV ρ');axs[1,1].legend(fontsize=6);ttl(axs[1,1],'D  Selective prediction');finish(fig,'si06_augmentation_confidence_unique')
 # Original tabular screen is retained from its recorded CSV; no invented reruns.
 fig,axs=canvas(2,1,5.5);z=rd('tabular_fm.csv').pivot(index='config',columns='model',values='mean_rho');z.index=[x.replace('TF-1-','').replace('TF-2-','').replace('TF-3-','').replace('TF-4-','').replace('TF-5-','') for x in z.index];dotcompare(axs[0],z,'A  Recorded tabular learner screen','Mean GDPa3 ρ')
 z=rd('harn_experiments.csv').set_index('assay').iloc[:,:6];profiles(axs[1],z.T,'B  Regional descriptor constructions');finish(fig,'si07_learner_screen_unique')
def baseline():
 fig=plt.figure(figsize=(W,6.5),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.05]);a=fig.add_subplot(gs[0,0]);c=fig.add_subplot(gs[0,1]);b=fig.add_subplot(gs[1,:]);navy='#1A237E';grey='#90A4AE';red='#FF5252'
 cv=pd.read_csv(E/'awmoe_cv_errorbars.csv');cv=cv[cv.split=='cluster_isotype'].set_index('assay');lb=rd('full_leaderboard.csv').rename(columns={'Unnamed: 0':'model'}).set_index('model');order=COMP[:4];x=np.arange(4);w=.30;v=cv.loc[order,'mean'].to_numpy();sd=cv.loc[order,'std'].to_numpy();best=lb.drop(index='AbDevX')[order].max().to_numpy(float)
 a.bar(x-w/2,v,w,yerr=sd,capsize=2,color=navy,error_kw={'lw':.6},label='AbDev-X ± fold SD');a.bar(x+w/2,best,w,color=grey,label='Best stored local entry')
 for i in range(4):
  if v[i]>best[i]:a.text(i-w/2,v[i]+sd[i]+.025,'*',ha='center',fontsize=8)
 a.set_xticks(x,[nice(z) for z in order],rotation=28,ha='right');a.set(ylabel='GDPa1 CV Spearman ρ',ylim=(-.22,.98));a.legend(fontsize=5.5,loc='upper left');ttl(a,'A  Local benchmark comparison')
 # Transcribe the retained archival C bars from vector geometry, not a new GP run.
 src=fitz.open(H.parent/'restored_original_style/figures/main07_baseline_context.pdf');bars=[]
 for d in src[0].get_drawings():
  r=d['rect']
  if r.x0>275 and r.y0>45 and abs(r.y1-227.5446)<.01 and 4<r.width<6:bars.append((d['fill'],r.x0,(227.5446-r.y0)/(250.2485)))
 groups={}
 for col,x0,vv in bars:groups.setdefault(tuple(col),[]).append((x0,vv))
 colors=[navy,'#E65100','#FF8A65','#FFCCBC'];names=['D-Stack (archival)','GP-FA','GP-HA','GP-HP'];rows=[]
 for i,((col,vals),co,name) in enumerate(zip(groups.items(),colors,names)):
  vals=sorted(vals);vv=[v for _,v in vals];c.bar(np.arange(5)+(i-1.5)*.18,vv,.18,color=co,label=name);rows.extend({'model':name,'assay':aa,'rho_vector_transcription':rr} for aa,rr in zip(COMP,vv))
 pd.DataFrame(rows).to_csv(H/'evidence/archival_gp_vector_transcription.csv',index=False);c.set_xticks(range(5),[nice(z) for z in COMP],rotation=28,ha='right');c.set(ylabel='Cluster–isotype CV ρ',ylim=(0,.9));c.legend(fontsize=5.5,ncol=2,loc='upper right');ttl(c,'C  Archival source stack and GP kernels')
 p=pd.read_csv(E/'primary_reproduction.csv').set_index('assay');j=rd('all7_assays.csv').set_index('assay');de=rd('delphi_comparison.csv').set_index('assay');order=['HIC','Tm2','AC-SINS','PR_CHO','Titer','Tm1','Tonset'];x=np.arange(7);w=.23;v=p.loc[order,'rho'].to_numpy();lo=p.loc[order,'ci_lo'].to_numpy();hi=p.loc[order,'ci_hi'].to_numpy();b.bar(x-w,v,w,yerr=[v-lo,hi-v],capsize=2,color=navy,error_kw={'lw':.6},label='AbDev-X; BCa 95% CI')
 jj=[j.loc[z,'jiang_best'] if z in j.index else np.nan for z in order];dd=[de.loc['AC-SINS_pH7.4' if z=='AC-SINS' else z,'DELPHI_heldout'] if ('AC-SINS_pH7.4' if z=='AC-SINS' else z) in de.index else np.nan for z in order];b.bar(x,jj,w,color=grey,label='Jiang: archived estimates');b.bar(x+w,dd,w,color=red,label='DELPHI: stored configuration')
 for i,assay in enumerate(order):
  if assay in CHAMP:b.scatter(i,CHAMP[assay],marker='D',facecolors='none',edgecolors='#555',s=16,lw=.6,label='Competition champion' if i==0 else None)
 b.set_xticks(x,[nice(z) for z in order],rotation=28,ha='right');b.axhline(0,c='#aaa',lw=.5);b.axvline(4.5,c='#bbb',ls=':',lw=.5);b.set(ylabel='GDPa3 Spearman ρ',ylim=(-.4,.98));b.legend(ncol=2,fontsize=6,loc='upper right');ttl(b,'B  External context across seven assays');finish(fig,'main07_baseline_context')
def historical_font_repair():
 # Keep archived plot geometry and numbers; enlarge text without changing model records.
 src=fitz.open(H.parent/'restored_original_style/figures/main02_performance.pdf');p=src[0];scale=(183/25.4*72)/p.rect.width
 records=[]
 for b in p.get_text('dict')['blocks']:
  for l in b.get('lines',[]):
   for s in l['spans']:
    if s['size']*scale<5.5:
     records.append(s);p.add_redact_annot(s['bbox'],fill=(1,1,1))
 p.apply_redactions(images=0,graphics=0)
 from matplotlib.font_manager import findfont
 p.insert_font(fontname='ArialRepair',fontfile=findfont('Arial'))
 for s in records:p.insert_text(s['origin'],s['text'],fontname='ArialRepair',fontsize=5.5/scale,color=(0,0,0))
 out=fitz.open();q=out.new_page(width=183/25.4*72,height=p.rect.height*scale);q.show_pdf_page(q.rect,src,0);out.save(F/'main02_performance.pdf')
if __name__=='__main__':
 for fn in [generalization,encoders,efficiency,descriptor,shap_structure,supplementary,baseline,historical_font_repair]:print(fn.__name__,flush=True);fn()
