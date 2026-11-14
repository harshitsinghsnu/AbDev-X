"""Refresh numerical inputs using the original figure-generating code and style.

Unchanged figures are copied verbatim. No replacement chart types or panel order.
"""
from pathlib import Path
import shutil,re,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
H=Path(__file__).parent;ROOT=H.parents[2];F=H/'figures';E=H/'evidence';OLD=H/'original_figures';V=H.parent/'verified_revision'
for p in OLD.glob('*.pdf'):shutil.copy2(p,F/p.name)
shutil.copy2(H.parent/'panel_descriptor_interpretability.pdf',F/'main08_descriptor_interpretability.pdf')
points=pd.read_csv(E/'primary_reproduction.csv').set_index('assay')
for n in ['plm_ablation.csv','representation_weights.csv','final_evaluation_summary.csv']:
 shutil.copy2(ROOT/'results'/n,E/n)
points.reset_index().to_csv(E/'awmoe_all9.csv',index=False)
bca=pd.read_csv(ROOT/'results/bootstrap_ci_all.csv');i=bca.index[bca.model=='AbDev-X'][0]
for a in ['HIC','AC-SINS','PR_CHO','SMAC','HAC']:
 bca.loc[i,a+'_lo']=points.loc[a,'ci_lo'];bca.loc[i,a+'_hi']=points.loc[a,'ci_hi']
bca.to_csv(E/'bootstrap_ci_all.csv',index=False)

# Original representation figure: retain every bar/scatter/heatmap panel and color.
src=(ROOT/'scripts/make_combined_panels.py').read_text(encoding='utf-8')
src=src.replace('"AC-SINS_pH7.4": 0.356','"AC-SINS_pH7.4": 0.337').replace('"PR_CHO": 0.337','"PR_CHO": 0.356')
src=src.replace('(-0.063, 0.371)',f'({points.loc["Tm2","ci_lo"]}, {points.loc["Tm2","ci_hi"]})').replace('(-0.147, 0.293)',f'({points.loc["Titer","ci_lo"]}, {points.loc["Titer","ci_hi"]})')
src=src.replace('lo = min(0.0, min(all_vals) - 0.05)','lo = min(0.0, min(all_vals) - 0.05, min(hb["ci_low"]) - 0.03)')
src=src.replace('out_dir = ROOT / "paper" / "paper10assay"','out_dir = OUT')
ns={'__file__':str(ROOT/'scripts/make_combined_panels.py'),'__name__':'original_style_generator'};exec(compile(src,'original_make_combined_panels.py','exec'),ns)
ns.update(RESULTS=E,OUT=F);ns['fig23']();shutil.copy2(F/'panel_gen_rep.pdf',F/'main03_generalization.pdf')

# Original headline figure (now SI): same two plots, palette and typeface.
src=(H.parent/'story_curated/build_unique_panels.py').read_text(encoding='utf-8').split('# Explicitly requested SI composites')[0]
src=src.replace("HERE=Path(__file__).parent;ROOT=HERE.parents[2];R=ROOT/'results';F=HERE/'figures'", "HERE=Path(__file__).parent;ROOT=HERE.parents[2];R=HERE/'evidence';F=HERE/'figures'")
oldextra="extra={'Tm1':(-.025,.611),'Tonset':(-.012,.414),'Tm2':(-.063,.371),'SEC':(-.382,.056),'Titer':(-.147,.293)}"
extra={a:(points.loc[a,'ci_lo'],points.loc[a,'ci_hi']) for a in ['Tm1','Tonset','Tm2','SEC','Titer']};src=src.replace(oldextra,'extra='+repr(extra))
exec(compile(src,'original_headline_panel.py','exec'),{'__file__':str(H/'original_headline_panel.py'),'__name__':'restored_headline'})

# Numerical CSV adapters preserve the original competition-five assay axes.
comp=['HIC','Tm2','AC-SINS_pH7.4','PR_CHO','Titer'];rename={'AC-SINS':'AC-SINS_pH7.4','desc':'descriptors'}
d=pd.read_csv(E/'primary_source_ablation.csv');d['assay']=d.assay.replace(rename);d['dropped']=d.dropped.replace(rename);d=d[d.assay.isin(comp)].rename(columns={'delta':'dHeldout'});d.to_csv(E/'exp_ablation.csv',index=False)
d=pd.read_csv(E/'primary_learning_folds.csv');d['assay']=d.assay.replace(rename);d=d[d.assay.isin(comp)].groupby(['assay','fraction']).rho.mean().reset_index().rename(columns={'fraction':'frac'});d.to_csv(E/'exp_learning_curves.csv',index=False)
d=pd.read_csv(E/'primary_paired_significance.csv');d['assay']=d.assay.replace(rename);d=d[d.assay.isin(comp)].rename(columns={'delta':'delta_rho','ci_lo':'ci_low','ci_hi':'ci_high','baseline':'vs'});d['vs']=d.vs.replace({'esm2':'ESM2','ablang2':'AbLang2'});d.to_csv(E/'exp_significance.csv',index=False)
shutil.copy2(ROOT/'results/exp_ig.csv',E/'exp_ig.csv')
src=(H.parent/'revised_main_figures/build_requested_panels.py').read_text(encoding='utf-8');head=src[:src.index('# Requested performance overview')]
body=src[src.index('# Requested LOO, learning, archival paired differences'):src.index('# User-requested full SHAP')]
head=head.replace("R=ROOT/'results' if (ROOT/'results').exists() else HERE/'evidence'","R=HERE/'evidence'")
body=body.replace('A  Legacy stack · full minus source-removed held-out ρ','A  AbDev-X · full minus source-removed held-out ρ').replace('Archived model − baseline Δρ (reported CI)','AbDev-X − baseline Δρ (paired BCa 95% CI)').replace('C  Archived paired differences · not the primary run','C  Paired held-out differences · primary-code model')
exec(compile(head+body,'original_efficiency_panel.py','exec'),{'__file__':str(H/'original_efficiency_panel.py'),'__name__':'restored_ablation'})
print('Original representation, headline and ablation layouts refreshed with verified values.')
