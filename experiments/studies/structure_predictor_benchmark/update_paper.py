"""Integrate completed predictor benchmark without changing prior figure assets."""
from pathlib import Path
import shutil,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;OUT=P.parent/'structure_comparison_revision';OUT.mkdir(exist_ok=True)
DATA=P/'comparison_ABodyBuilder3_IgFold_ESMFold_AlphaFold3'
shutil.copytree(P/'figures',OUT/'figures',dirs_exist_ok=True)
scores=pd.read_csv(DATA/'scores.csv');diff=pd.read_csv(DATA/'paired_differences.csv')
predictors=['ABodyBuilder3','IgFold','ESMFold','AlphaFold3'];assays=scores.assay.unique().tolist();colors=['#2474A7','#D58A30','#55A080','#9E65A5']
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(7.2,4.7));fig.subplots_adjust(left=.11,right=.99,top=.91,bottom=.18,wspace=.31)
for ax,kind,letter in zip(axs,['cv','heldout'],['A','B']):
 for i,p in enumerate(predictors):
  d=scores[scores.predictor==p].set_index('assay').loc[assays];y=np.arange(10)+(i-1.5)*.16
  x=d.cv_mean.to_numpy() if kind=='cv' else d.heldout_rho.to_numpy()
  err=d.cv_sd.to_numpy() if kind=='cv' else np.vstack([x-d.ci_lo,d.ci_hi-x])
  ax.errorbar(x,y,xerr=err,fmt='o',ms=3,capsize=2,lw=.75,label='AlphaFold 3' if p=='AlphaFold3' else p,color=colors[i])
 ax.set(yticks=np.arange(10),yticklabels=[a.replace('PR_CHO','PR-CHO') for a in assays],xlabel=r'Spearman $\rho$',title='Cluster–isotype CV' if kind=='cv' else 'GDPa3 held-out')
 ax.invert_yaxis();ax.axvline(0,color='.7',lw=.6);ax.text(-.22,1.05,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.53,.015),frameon=False,fontsize=8,columnspacing=1.2,handletextpad=.3)
fig.savefig(OUT/'figures/si_structure_predictors.pdf');fig.savefig(OUT/'figures/si_structure_predictors.png',dpi=200);plt.close(fig)

text=(P/'AbDevX_structure_scope_main_SI.tex').read_text(encoding='utf8')
results=r'''\subsection{Alternative structures do not establish a consistent downstream advantage}
The structural-feature screen shows that surface information can be useful, but it does not establish which structure predictor supplies the most useful inputs. We therefore compared ABodyBuilder3, IgFold, ESMFold, and AlphaFold~3 using identical paired sequences and a common open structural-feature panel~\citep{abb3,igfold,lin2023esm2,alphafold3}. All three newly run alternatives produced structures for the 326 antibodies. Two supplied ABodyBuilder3 training structures were unavailable, leaving a common set of 244 GDPa1 and 80 GDPa3 antibodies before assay-specific missing-label exclusions. We replaced the cached 57-feature bio block with the same 20 structural features for each predictor, held the other four representation blocks fixed, and refitted the original four-expert procedure on identical rows and folds. This controlled variant is termed AbDev-X/common-structure20; it is separate from the primary ensemble reported above.

Across HIC, AC-SINS, PR-CHO, SMAC, and HAC, the descriptive mean GDPa3 correlation was $.443$ for ABodyBuilder3, $.442$ for ESMFold, $.436$ for IgFold, and $.418$ for AlphaFold~3 (Supplementary Figure~\ref{fig:structure_predictors} and Table~\ref{tab:structure_predictors}). ABodyBuilder3 had the largest HIC and AC-SINS points, ESMFold the largest SMAC and HAC points, and AlphaFold~3 the largest PR-CHO point. The ordering also depended on assay scope: across all ten endpoints, IgFold had the largest mean ($.245$), followed by ESMFold ($.240$), ABodyBuilder3 ($.232$), and AlphaFold~3 ($.222$). These means summarize assay-level correlations; they do not imply validated performance on the weak endpoints.

Paired antibody-bootstrap comparisons did not establish a consistent advantage from switching predictors. The nominal 95\% intervals favored ABodyBuilder3 over AlphaFold~3 for HIC and IgFold over ABodyBuilder3 for Titer, but every interval included zero after adjustment across the 30 alternative-versus-ABodyBuilder3 assay comparisons (Supplementary Table~\ref{tab:structure_differences}). We therefore retained ABodyBuilder3 as the structural source used in this work. This is a decision supported by the absence of an established advantage from the tested alternatives, rather than evidence of equivalence or universal optimality. The comparison is retrospective, AlphaFold~3 was run without MSAs or templates, and the common feature panel does not regenerate the full primary bio block.

'''
anchor=r'\subsection{Source removal connects the ensemble gains to complementary information}'
assert text.count(anchor)==1;text=text.replace(anchor,results+anchor)
start=text.index('Structure-predictor choice remains an untested source of variation.')
end=text.index('\n\n',start)
text=text[:start]+r'''The controlled structural-source comparison refines the role of predicted structure. ABodyBuilder3 and ESMFold gave nearly identical five-positive-signal mean correlations, whereas IgFold led the ten-assay average; no alternative-versus-ABodyBuilder3 assay contrast remained resolved after multiplicity adjustment (Supplementary Figure~\ref{fig:structure_predictors}). This supports retaining ABodyBuilder3 for the current workflow without claiming that it is the optimal structure predictor. The experiment replaces the original bio block with 20 common open features; the primary 57-feature cache combines TAP, MOE, Aggrescan3D, and AntiFold outputs with tool-specific provenance. An exact predictor swap in that pipeline would require regenerating all of those features. Moreover, the tested AlphaFold~3 single-sequence condition does not evaluate its full database-assisted pipeline, and a structural-accuracy comparison would require matched experimental structures. Failure to resolve a difference in this small external panel is not evidence that the structural sources are interchangeable.'''+text[end:]
old='The analysis uses deterministic supplied structures. No matched structure-predictor screen was performed, and no optimal-predictor claim is made. Feature-noise and simulated-ensemble experiments are reported separately and do not establish propagation of physical conformational uncertainty.'
new='The primary analysis retains its supplied cached features. A separate matched structural-source experiment uses a common open 20-feature panel as described below; it does not regenerate the primary 57-feature block. Feature-noise and simulated-ensemble experiments are reported separately and do not establish propagation of physical conformational uncertainty.'
assert old in text;text=text.replace(old,new)
methods=r'''
\paragraph{Controlled structure-predictor comparison.}
Paired VH/VL sequences were modeled using IgFold 1.0.1 with AntiBERTy 1.0.0 (four-model ensemble; optional refinement and renumbering disabled), ESMFold v1 (Transformers 4.51; four recycles; trunk chunk size 32), and official AlphaFold~3 v3.0.1 source (commit \texttt{231efc9}) with JAX 0.4.34. The supplied ABodyBuilder3 predictions formed the reference~\citep{abb3,igfold,lin2023esm2,alphafold3}. ESMFold encoded VH, a 25-glycine linker, and VL with a 512-position light-chain offset; linker atoms were excluded from every output and feature calculation. Its sequence language model used half precision and folding components single precision. AlphaFold~3 used explicit paired chains, empty paired and unpaired MSAs, no templates, seed 42, 10 recycles, and five diffusion samples; the official highest-ranking-score structure was retained without using assay labels. Inference used an NVIDIA RTX A4000 with 16~GB GPU memory. All 326 predictions completed for each newly run alternative; the ABodyBuilder3 cache contained 324 structures, including all 80 GDPa3 antibodies. Original cached ABodyBuilder3 inference times were unavailable, so no matched speed ranking is inferred.

Chains were identified by exact sequence matching and normalized to H/L numbering. Hydrogens and terminal OXT atoms were excluded. PDBFixer 1.12 with OpenMM 8.6.1 added missing heavy atoms using seed 42 without relaxing already complete coordinates; IgFold backbone/CB outputs therefore required side-chain completion. The common panel contained 20 quantities: total, hydrophobic, and aromatic solvent-accessible surface area (SASA); hydrophobic SASA fraction; signed and positive SASA-weighted Kyte--Doolittle exposure; signed, positive, and negative SASA-weighted charge; maximum and top-five-mean hydrophobic patch scores; maximum, minimum, and absolute-maximum charge patch scores; C$_\alpha$ radius of gyration; VH--VL C$_\alpha$ contact counts below 8 and 10~\AA; minimum interchain C$_\alpha$ distance; chain-centroid distance; and SASA buried upon chain association. SASA used FreeSASA~\citep{freesasa}; patches summed residue SASA-weighted properties within 10~\AA\ C$_\alpha$ neighborhoods. The fixed charge proxy assigned Arg/Lys $+1$, Asp/Glu $-1$, and His $+0.1$; it is not a titration calculation. The definitions and missing-atom policy were identical across sources.

The common-complete-case intersection comprised 244 GDPa1 and 80 GDPa3 antibodies. Missing structures were excluded rather than zero-filled. The original ESM-2, AbLang2, sequence-descriptor, and subtype blocks were unchanged, and bio57 was replaced by the source-specific common 20-feature matrix. All ten assay models reused the original E1, A-MGFS, D-Stack, and F-MoE classes and fixed final expert weights of $.25$. For each outer cluster--isotype fold, preprocessing, regularization, source selection, stream stacking, and group weighting were refitted using only the remaining training folds. Full-data refits used the same common training rows. Assay-specific label availability was preserved. CV is reported as the mean $\pm$ sample SD across five folds; GDPa3 correlations and alternative-minus-ABodyBuilder3 differences use BCa intervals from 20,000 paired antibody resamples. Bonferroni-adjusted BCa intervals use $\alpha=.05/30$ across three alternatives and ten assays. These intervals are conditional on the fitted models and do not account for training-set uncertainty or antibody-family dependence. The experiment is retrospective because GDPa3 outcomes had previously been examined; no predictor was selected as a newly optimized primary model from these results.

'''
anchor=r'\subsection{Four assay-aware prediction strategies}'
assert text.count(anchor)==1;text=text.replace(anchor,methods+anchor)
refs=r'''\bibitem{igfold} Ruffolo JA, Chu LS, Mahajan SP, Gray JJ. Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies. Nat Commun. 2023;14:2389. \url{https://doi.org/10.1038/s41467-023-38063-x}.
\bibitem{alphafold3} Abramson J, Adler J, Dunger J, Evans R, Green T, Pritzel A, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature. 2024;630:493--500. \url{https://doi.org/10.1038/s41586-024-07487-w}.
'''
assert r'\bibitem{igfold}' not in text;text=text.replace(r'\end{thebibliography}',refs+r'\end{thebibliography}')

si=r'''
\clearpage
\subsection{Matched structure-predictor comparison}
This experiment asks whether the structural information supplied to the same assay-aware predictor changes when the underlying Fv prediction method changes. It does not rerun the original pipeline combining TAP, MOE, Aggrescan3D, and AntiFold. All four sources instead supply the identical common 20-feature panel, and every source is trained and evaluated on the same common-complete-case antibodies. Consequently, its numerical results should be compared within this experiment rather than substituted into the primary bio57 performance figures.

The source ordering changes both by endpoint and by evaluation stage (Figure~\ref{fig:structure_predictors}). Across the five positive-signal assays, mean CV correlations were $.557$ for ABodyBuilder3, $.559$ for IgFold, $.567$ for ESMFold, and $.554$ for AlphaFold~3. On GDPa3, ABodyBuilder3 and ESMFold were closely matched in the corresponding mean ($.443$ and $.442$), with IgFold at $.436$ and AlphaFold~3 at $.418$. For example, ABodyBuilder3 gave HIC $\rho=.518$, compared with $.475$, $.512$, and $.455$ for IgFold, ESMFold, and AlphaFold~3, respectively. Conversely, ESMFold gave the largest HAC point ($.625$), and AlphaFold~3 the largest PR-CHO point ($.253$). Thus, the experiment supplies no stable all-assay source ordering.

The ten-assay means further illustrate why assay scope must accompany aggregate results: IgFold reached $.245$, ESMFold $.240$, ABodyBuilder3 $.232$, and AlphaFold~3 $.222$. These are descriptive unweighted averages of assay correlations, not cross-assay confidence estimates. Tm2 and SEC remained negative for all four sources on GDPa3, and several other endpoint intervals included zero (Table~\ref{tab:structure_predictors}). A larger ten-assay mean therefore does not establish useful prediction for the weak endpoints.

Paired differences localize the apparent source effects (Table~\ref{tab:structure_differences}). Relative to ABodyBuilder3, AlphaFold~3 reduced HIC correlation by $.064$ (nominal BCa 95\% CI $[-.174,-.005]$), while IgFold increased Titer correlation by $.081$ ($[.019,.170]$). Neither remained resolved after Bonferroni adjustment over the 30 comparisons, and no other adjusted interval excluded zero. These findings support retaining the available ABodyBuilder3 workflow; they do not establish equivalence, its optimality, or the absence of a useful alternative under another feature definition. In particular, AF3 was evaluated in a single-sequence condition and IgFold required missing-side-chain reconstruction. No matched experimental structures were available to determine structural accuracy independently of downstream prediction.

\widepanel{si_structure_predictors.pdf}{\textbf{Controlled comparison of structural sources in AbDev-X/common-structure20.} \textbf{A}, mean cluster--isotype CV Spearman correlation $\pm$ sample SD across five outer folds. \textbf{B}, GDPa3 correlation with antibody-bootstrap BCa 95\% intervals (20,000 resamples). Colors identify ABodyBuilder3, IgFold, ESMFold, and AlphaFold~3. All models use the same 244 common GDPa1 antibodies and 80 GDPa3 antibodies before assay-specific label exclusions, identical fold assignments, the same 20 structural feature definitions, the unchanged remaining four representation blocks, and the original four-expert training procedure. GDPa3 sample sizes are HIC 79, HAC 61, Tm2 74, and 80 for the remaining endpoints. AF3 uses empty MSAs and no templates. This comparison is retrospective and concerns a common-feature variant, not regeneration of the primary 57-feature bio block. No adjusted paired difference from ABodyBuilder3 excluded zero across the 30 assay comparisons.}{fig:structure_predictors}

\small
\setlength{\tabcolsep}{5pt}
\begin{longtable}{llrrr}
\caption{Matched structural-source results. CV entries are mean $\pm$ sample fold SD; GDPa3 entries are correlation [BCa 95\% CI]. Per-assay common label counts are identical across sources. The two missing ABodyBuilder3 training structures are excluded from all sources.}\label{tab:structure_predictors}\\
\toprule Source & Assay & CV $\rho$ & GDPa3 $\rho$ [95\% CI] & $n_{\rm train}/n_{\rm test}$\\\midrule\endfirsthead
\multicolumn{5}{l}{\textit{Table \thetable\ continued}}\\\toprule Source & Assay & CV $\rho$ & GDPa3 $\rho$ [95\% CI] & $n_{\rm train}/n_{\rm test}$\\\midrule\endhead
\bottomrule\endfoot
'''
for r in scores.to_dict('records'):
 name='AlphaFold 3' if r['predictor']=='AlphaFold3' else r['predictor'];a=r['assay'].replace('_','-')
 si+=f"{name} & {a} & ${r['cv_mean']:.3f}\\pm{r['cv_sd']:.3f}$ & ${r['heldout_rho']:.3f}$ [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] & {r['n_train']}/{r['n_test']}\\\\\n"
si+=r'''\end{longtable}
\begin{longtable}{llrrr}
\caption{Paired GDPa3 differences from ABodyBuilder3. Positive $\Delta\rho$ favors the alternative. BCa intervals use 20,000 paired antibody resamples. The last column adjusts across all 30 contrasts using Bonferroni $\alpha=.05/30$; every adjusted interval includes zero. These are conditional intervals, not equivalence tests.}\label{tab:structure_differences}\\
\toprule Alternative & Assay & $\Delta\rho$ & Nominal 95\% CI & Adjusted CI\\\midrule\endfirsthead
\multicolumn{5}{l}{\textit{Table \thetable\ continued}}\\\toprule Alternative & Assay & $\Delta\rho$ & Nominal 95\% CI & Adjusted CI\\\midrule\endhead
\bottomrule\endfoot
'''
for r in diff.to_dict('records'):
 name='AlphaFold 3' if r['predictor']=='AlphaFold3' else r['predictor'];a=r['assay'].replace('_','-')
 si+=f"{name} & {a} & ${r['delta_rho']:+.3f}$ & [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] & [{r['bonferroni_lo']:.3f}, {r['bonferroni_hi']:.3f}]\\\\\n"
si+=r'''\end{longtable}
\normalsize
'''
text=text.replace(r'\end{document}',si+'\n'+r'\end{document}')
(OUT/'AbDevX_structure_comparison_main_SI.tex').write_text(text,encoding='utf8')
shutil.copytree(DATA,OUT/'structure_comparison_data',dirs_exist_ok=True)
print(OUT)
