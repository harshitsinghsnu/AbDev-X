"""Replace descriptor illustrations with verified primary-model attributions.
No fitting, model selection, research result files, or unrelated figures change.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import sys, json, pickle, shutil, re, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

P=Path(__file__).resolve().parent; R=P.parents[1]
OLD=P/'narrative_literature_revision'; V=P/'verified_revision'; OUT=P/'primary_attribution_revision'
E=OUT/'evidence'; F=OUT/'figures'
E.mkdir(parents=True,exist_ok=True);shutil.copytree(OLD/'figures',F,dirs_exist_ok=True)
sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import aw_moe as AW
from abdev.features.descriptors import build_descriptor_matrix
with open(V/'evidence'/'fitted_primary.pkl','rb') as f:z=pickle.load(f)
manifest=json.loads((V/'evidence'/'model_manifest.json').read_text())
for file,h in manifest['sha256'].items():
    assert hashlib.sha256((R/file).read_bytes()).hexdigest()==h,file
meta=json.loads((V/'evidence'/'residue_shap_manifest.json').read_text())
res=pd.read_csv(V/'evidence'/'primary_residue_shap.csv',dtype={'pdb_residue':str})
g=z['g3'];i=g.index[g.antibody_name==meta['antibody_name']][0];row=g.loc[i]
desccols=z['descriptor_columns'];biocols=z['bio_columns'];cols=desccols+biocols
original=build_descriptor_matrix(pd.Series([row.vh]),pd.Series([row.vl])).iloc[0].to_numpy(float)
alanine=build_descriptor_matrix(pd.Series(['A'*len(row.vh)]),pd.Series(['A'*len(row.vl)])).iloc[0].to_numpy(float)
assert np.allclose(original,z['B3']['desc'][i])
pdb=PDBParser(QUIET=True).get_structure('rep',meta['structure'])
for chain,sequence in [('H',row.vh),('L',row.vl)]:
    assert ''.join(seq1(r.resname) for r in pdb[0][chain] if r.id[0]==' ')==sequence

def direct(m,b,assay):
    e1,e2,e3,e4=m['experts']; db=np.r_[b['desc'],b['bio']][None,:]
    cfg=AW.MGFS_MASKS.get(assay,{'inc':[],'exc':[]}); mask=AW._mask_cols(cols,cfg['inc'],cfg['exc'])
    if mask.sum()<5: mask=np.ones(len(cols),bool)
    p1=e1.pipe.predict(b[e1.best_blk][None,:])[0]
    p2=e2.pipe.predict(db[:,mask])[0]
    streams=[[e3.p1.predict(np.r_[b['esm2'],b['ablang2']][None,:])[0],e3.p2.predict(db)[0]]]
    p3=e3.meta.predict(streams)[0]
    p4=sum(w*p.predict(db[:,mask])[0] for p,(_,mask),w in zip(e4.pipes,e4.expert_masks,e4.weights))
    return np.mean([p1,p2,p3,p4])

errors=[]; sums=[]
for a,m in z['models'].items():
    predicted=m['bias']+sum(z['B3'][k]@w for k,w in m['weights'].items())
    errors.append(float(np.max(abs(predicted-m['pred']))))
    b={k:v[i].copy() for k,v in z['B3'].items()}
    actual=direct(m,b,a);b['desc']=alanine; reference=direct(m,b,a)
    delta=actual-reference; value=res.loc[res.assay==a,'shap'].sum()
    assert abs(delta-value)<1e-5,(a,delta,value)
    sums.append({'assay':a,'prediction':actual,'descriptor_reference_prediction':reference,'descriptor_prediction_change':delta,'residue_sum':value,'sum_error':value-delta})
assert max(errors)<1e-5
pd.DataFrame(sums).to_csv(E/'prediction_attribution_completeness.csv',index=False)
for f in ['primary_residue_shap.csv','residue_shap_manifest.json','surface_render_manifest.json','model_manifest.json']:
    shutil.copy2(V/'evidence'/f,E/f)

plt.rcParams.update({'font.family':'Arial','font.size':6,'axes.titlesize':7,'axes.labelsize':6,'xtick.labelsize':5.5,'ytick.labelsize':5.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5,'pdf.fonttype':42,'svg.fonttype':'none','xtick.major.size':2,'ytick.major.size':2})
COMP=['HIC','Tm2','AC-SINS','PR_CHO','Titer']; EXTRA=['SMAC','HAC','Tm1','Tonset','SEC']
nice=lambda a: {'PR_CHO':'PR-CHO'}.get(a,a)
def short(s):
    s=s.replace('MOE_properties::','MOE ').replace('TAP::','TAP ').replace('Aggrescan3D::','A3D ').replace('AntiFold::','AF ')
    for x,y in [('frac_aromatic','arom.'),('frac_aliphatic','aliph.'),('frac_polar','polar'),('frac_pos','pos.'),('frac_neg','neg.'),('deamidation_NG_NS','NG/NS'),('isomerization_DG_DS','DG/DS'),('hmoment','μH'),('aromatic_sasa','arom.SASA'),('charge_pH','charge')]:s=s.replace(x,y)
    return s.replace('_',' ').replace('aggrescan 90 score','score 90').replace('hyd strength cdr','CDR hyd.')

fig=plt.figure(figsize=(183/25.4,169/25.4))
fig.text(.015,.982,'A  AbDev-X named-feature attributions',fontsize=7,fontweight='bold',va='top')
fig.text(.015,.958,'Actual four-expert predictor; training-mean reference. Feature value: low (blue) to high (red).',fontsize=6,va='top')
tables=[]
for j,a in enumerate(COMP):
    m=z['models'][a];w=np.r_[m['weights']['desc'],m['weights']['bio']]
    x=np.hstack([z['B3']['desc'],z['B3']['bio']]);train=np.hstack([z['B1']['desc'],z['B1']['bio']])
    ycol=AW.GENUINE_ASSAYS.get(a,(a,''))[0];mu=train[np.isfinite(z['train_df'][ycol].to_numpy(float))].mean(0)
    phi=(x-mu)*w; selected=np.argsort(np.mean(abs(phi),axis=0))[-12:]
    ax=fig.add_axes([.105+j*.198,.585,.088,.325])
    for t,c in enumerate(selected):
        values=phi[:,c];span=max(np.ptp(values),1e-9);bins=np.floor((values-values.min())/span*15).astype(int);offset=np.zeros(len(values))
        for b in np.unique(bins):
            ids=np.flatnonzero(bins==b); levels=np.arange(len(ids))-(len(ids)-1)/2
            offset[ids]=levels*.035
        offset=np.clip(offset,-.30,.30)
        ax.scatter(values,t+offset,c=rankdata(x[:,c])/len(x),cmap='coolwarm',vmin=0,vmax=1,s=2.1,alpha=.72,linewidths=0)
        tables.append({'assay':a,'feature':cols[c],'label':short(cols[c]),'mean_abs_attribution':np.mean(abs(values))})
    ax.set_yticks(range(12),[short(cols[c]) for c in selected]);ax.tick_params(axis='y',length=0,pad=2)
    ax.axvline(0,c='#aaaaaa',lw=.5,zorder=-1);ax.set_ylim(-.6,11.6);ax.set_title(nice(a),fontweight='bold',pad=5);ax.set_xlabel('Contribution',labelpad=2)
    ax.locator_params(axis='x',nbins=2)
pd.DataFrame(tables).to_csv(E/'figure_named_feature_labels.csv',index=False)

for label,assays,y in [('B',EXTRA,.305),('C',COMP,.065)]:
    fig.text(.015,y+.238,f'{label}  AbDev-X residue contributions through sequence descriptors',fontsize=7,fontweight='bold',va='top')
    for j,a in enumerate(assays):
        ax=fig.add_axes([.015+j*.198,y,.19,.205]);ax.imshow(plt.imread(V/'figures'/f'residue_{a}.png'));ax.axis('off');ax.set_title(nice(a),fontsize=7,fontweight='bold',pad=0)
cb=fig.add_axes([.36,.039,.26,.009]);fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-1,1),cmap='RdBu_r'),cax=cb,orientation='horizontal',ticks=[-1,0,1]);cb.tick_params(labelsize=5.5,pad=1)
fig.text(.015,.009,'Blue: lowers predicted assay value; red: raises it. Residue scores scaled within assay; PLM, bio and subtype fixed.',fontsize=6)
for ext in ['pdf','png','svg']:fig.savefig(F/f'main06_primary_attribution.{ext}',dpi=350)
plt.close(fig)

s=(OLD/'AbDevX_evidence_led_narrative_main_SI.tex').read_text(encoding='utf-8')
a=s.index(r'\subsection{Physicochemical analyses give molecular context to the assay-dependent signal}')
b=s.index(r'A complementary descriptor-only Integrated Gradients analysis',a)
replacement=r'''\subsection{Primary-model attributions connect prediction contributions to molecular positions}
We next asked which named features and residue positions contribute to the actual AbDev-X prediction, rather than to a separate explanatory proxy. With source selection, masks, and weights fixed after training, the four-expert predictor is affine in its cached input features. We propagated its fitted scaling, PCA, regression, stacking, and averaging operations to obtain an exact input-feature decomposition~\citep{shap,ig}. Figure~\ref{fig:shap_main}A displays the twelve named descriptor/bio features with the largest mean absolute contributions for each competition assay. These contributions use the actual primary predictor and a training-mean reference; omitted PLM dimensions and metadata remain part of the full decomposition.

To map model-dependent contributions onto a predicted antibody structure, we treated individual VH and VL residues as players in a sequence-descriptor perturbation game. At each step we restored a residue from an alanine reference, recomputed the original descriptors, and evaluated their contribution through all fitted expert routes that use descriptors. We used the alphabetically first GDPa3 antibody with an available predicted structure, selected independently of labels or prediction magnitude. The signed scores were aligned to its exact heavy- and light-chain PDB residues and displayed on the same Fv surface for all ten assays (B--C). Positive values raise the predicted assay readout and negative values lower it relative to this reference; their biological desirability depends on the endpoint.

These surfaces now visualize model-derived residue contributions, rather than prespecified hydrophobicity or charge fields. Their sums recover the descriptor-pathway prediction change within $10^{-5}$ prediction units for every assay. The attribution is conditional on fixed PLM embeddings, bio features, and subtype metadata: it explains the sequence-descriptor pathway of the actual ensemble, not every effect of changing a residue on its complete molecular inputs. The alanine reference and sampled permutations also affect the estimates. Thus, highlighted positions identify model-dependent hypotheses for investigation, without establishing mutation effects or validating prediction accuracy on the weaker endpoints.

\figpanel{main06_primary_attribution.pdf}{\textbf{Primary AbDev-X feature attribution and structure-mapped residue contributions across ten assays.} \textbf{A}, exact input-feature contributions of the fitted four-expert predictor relative to assay-specific training means. The twelve highest mean-absolute contributions among named sequence descriptors and bio features are shown for each competition assay; each point is one of the 80 GDPa3 antibodies, and color denotes the feature-value percentile. PLM dimensions and metadata are omitted from the display, not from the prediction or complete input decomposition. \textbf{B--C}, permutation Shapley estimates for the sequence-descriptor pathway, mapped to the ABodyBuilder3 Fv of GDPa3 antibody P907-A14-arid-pond-618a6 (VH 123 residues, VL 107 residues). Missing residues in each coalition are replaced by alanine; descriptors are recomputed and their effects propagated through the fitted primary ensemble. Sixty-four permutations (32 antithetic pairs) are averaged. PLM, bio, and subtype inputs remain fixed. Red raises and blue lowers the predicted assay value; each assay uses its own symmetric scale capped at the 95th percentile of absolute residue scores. Color magnitudes cannot be compared across assays. The scores sum to the descriptor-pathway prediction difference, not the full prediction difference under complete molecular recomputation. B shows SMAC, HAC, Tm1, Tonset, and SEC; C shows HIC, Tm2, AC-SINS, PR-CHO, and Titer. These are reference-dependent model attributions, not causal hotspots or evidence of accuracy on weak endpoints.}{fig:shap_main}{}
\FloatBarrier
'''
s=s[:a]+replacement+s[b:]
old='Interpretation and transfer impose further requirements. Proxy SHAP, descriptor IG, and surface maps help formulate hypotheses, but direct molecular explanations require attributions of the actual ensemble and experimental perturbation.'
new='Interpretation and transfer impose further requirements. Exact primary-model input contributions and structure-mapped residue scores now connect the fitted prediction to named features and the sequence-descriptor pathway. The residue analysis holds other inputs fixed, so complete sequence-level attribution would additionally require propagating changes through both PLMs and the structure-informed bio features. Proxy SHAP and the separate descriptor-only IG remain supplementary or complementary analyses. Experimental perturbation is needed to test whether any highlighted position is a useful engineering target.'
assert old in s;s=s.replace(old,new)
a=s.index('The displayed SHAP beeswarms and per-antibody category plots use separate physicochemical LightGBM proxy models')
b=s.index(r'\subsection{Representation and learner screens}',a)
s=s[:a]+r'''\subsection{Attribution of the primary fitted predictor}
The fitted primary model is affine in its cached representation inputs because each retained head consists of standardization, optional linear PCA, and ridge regression; source selections, keyword masks, and group weights are fixed after fitting. We propagated the linear maps and intercepts through D-Stack and the final equal-weight average to obtain $f_a(x)=b_a+\sum_j w_{aj}x_j$. The collapsed function reproduced the four-expert predictions to within $10^{-5}$ on GDPa3. Exact feature IG relative to the assay-labeled GDPa1 training means is $\phi_{aj}=w_{aj}(x_j-\mu_{aj})$~\citep{ig}; it also equals interventional linear SHAP relative to that background~\citep{shap}, not correlation-dependent conditional SHAP. The complete contribution sum plus the reference prediction equals the primary prediction. Panel A displays only named descriptor and bio features, selected by mean absolute contribution across all 80 external antibodies, without using their labels.

For residue attribution, we defined $g_a(S)=f_a(x_{\mathrm{PLM}},d(s_S),x_{\mathrm{bio}},x_{\mathrm{subtype}})$: $s_S$ retains the observed residues in coalition $S$ and uses alanine at all other VH/VL positions, while $d$ is the original descriptor pipeline, including its numbering fallback. PLM, bio, and subtype inputs are fixed to those of the selected external antibody. The average marginal change over 64 residue permutations (32 random orders and their reversals, seed 42) estimates the Shapley value for each position. This is a fixed-other-input pathway game, not conditional-expectation SHAP. Scores sum to $g_a(\mathrm{all})-g_a(\varnothing)$; we verified this identity by evaluating the original fitted expert heads, with error below $10^{-5}$ for every assay. Completeness does not establish that individual scores have converged with respect to permutation sampling or reference choice.

The representative antibody is the alphabetically first GDPa3 identifier with an available predicted Fv structure; selection does not depend on labels or scores. Both chain sequences were checked for exact agreement with PDB residues before mapping. The surface is a mesh of the union of atomic van der Waals spheres inflated by a 1.4-\AA{} probe, sampled on a 0.8-\AA{} grid; triangle colors inherit the residue score of the nearest atom. All assay views use the same orientation. Colors are clipped symmetrically at the assay-specific 95th percentile of absolute scores; the scores are in prediction units before this display normalization. Full residue tables, reference definitions, and completeness checks accompany the analysis. The alanine background can be far from observed antibodies, and neither structural relaxation nor mutation effects are modeled. The five additional endpoints retain the unchanged primary-code defaults.

The supplementary SHAP importance and category plots retain separate physicochemical LightGBM proxy models~\citep{lightgbm,shap}. Metadata, including observed labels, is excluded from feature-importance summaries. The descriptor-only IG panel~\citep{ig} uses a standardized ridge head without PCA and a zero raw-feature reference on GDPa1; its saved table contains the top 15 features per assay. A separate chain-aware network supplies the CDR-enrichment diagnostic. Historical source-stack coefficients, GP comparisons, and transfer experiments retain their distinct model identities.

'''+s[b:]
s=s.replace('Main-text beeswarms and SI per-antibody stacks retain their separate expanded-proxy sources.','Main-text attribution distributions use exact primary-model coefficients; SI per-antibody stacks retain their separate expanded-proxy source.')
dest=OUT/'AbDevX_primary_attribution_main_SI.tex';dest.write_text(s,encoding='utf-8')
audit={'primary_prediction_reconstruction_max_error':max(errors),'model_code_and_cache_hashes_match':True,'residue_completeness_max_error':max(abs(r['sum_error']) for r in sums),'representative':meta['antibody_name'],'all_ten_structures_are_descriptor_pathway_attributions':True,'other_inputs_fixed':['ESM-2','AbLang2','bio','subtype'],'not_complete_end_to_end_sequence_attribution':True}
(E/'attribution_validation.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2));print(dest)
