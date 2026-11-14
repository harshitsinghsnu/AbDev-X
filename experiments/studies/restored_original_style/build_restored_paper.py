"""Restore the selected figure designs; keep verified numbers and requested placement."""
from pathlib import Path
import re,shutil,json,hashlib
import fitz
H=Path(__file__).parent;OLD=H.parent/'story_curated';VER=H.parent/'verified_revision';F=H/'figures'
original=(OLD/'AbDevX_curated_main_SI.tex').read_text(encoding='utf-8');verified=(VER/'AbDevX_verified_main_SI.tex').read_text(encoding='utf-8')
# Direct export of the original PowerPoint; crop only blank outside margins.
doc=fitz.open(H/'architecture_export.pdf');p=doc[0];boxes=[]
for b in p.get_text('dict')['blocks']:
 if b['type']==1:boxes.append(fitz.Rect(b['bbox']))
 elif b['type']==0:
  for line in b['lines']:
   for s in line['spans']:
    if s['text'].strip():boxes.append(fitz.Rect(s['bbox']))
for item in p.get_drawings():
 if (item.get('fill') and min(item['fill'])<.98) or (item.get('color') and min(item['color'])<.98):boxes.append(item['rect'])
box=boxes[0]
for b in boxes[1:]:box|=b
header=p.search_for('(a) Input')[0]
p.set_cropbox(fitz.Rect(box.x0-12,header.y0-12,box.x1+12,box.y1+12)&p.rect);doc.save(F/'fig01_architecture_user.pdf')

pre=verified[:verified.index(r'\begin{abstract}')].replace('top=2.25cm,bottom=2.25cm','top=1.35cm,bottom=1.35cm')
abstract=original[original.index(r'\begin{abstract}'):original.index(r'\section{Background}')].replace('$.298$','$.249$').replace('$0.298$','$0.249$')
abstract=abstract.replace('The recorded constrained fusion weights were uniform, showing that the measured benefit comes from combining assay-aware experts rather than demonstrated specialization of the final gate.', 'The evaluated final model averages four experts. Recomputed paired intervals support improvements over both PLM baselines on HIC, SMAC, and HAC.')
intro=original[original.index(r'\section{Background}'):original.index(r'\section{Methods}')]
methods=verified[verified.index(r'\section{Methods}'):verified.index(r'\subsection{Direct primary-model attribution and structure mapping}')]
methods=methods.replace('Figure~\\ref{fig:architecture} separates molecular inputs, feature families, expert predictors, and final fusion.', 'Figure~\\ref{fig:architecture} separates molecular inputs, feature families, expert predictors, and final fusion. The structural and surface feature boxes depict conceptual families consolidated into one operational bio block.')
methods+=r'''
\subsection{Interpretation and comparison analyses}
The displayed SHAP beeswarms and per-antibody category plots use separate physicochemical LightGBM proxy models. The descriptor-only IG panel uses a standardized ridge head without PCA and a zero raw-feature reference on GDPa1; its saved table contains the top 15 features per assay. Structure maps project selected descriptor fields onto a representative ABodyBuilder3 prediction. These analyses provide molecular context, but are not presented as complete spatial attributions of the four-expert primary model. A separate chain-aware network supplies the CDR-enrichment diagnostic. Historical source-stack coefficients, GP comparisons, and transfer experiments retain their distinct model identities.
'''
def panel(file,caption,label):return '\\figpanel{'+file+'}{'+caption+'}{'+label+'}{}\n\\FloatBarrier\n'
def original_panel(file):
 return next(line for line in original.splitlines() if line.startswith(('\\figpanel{'+file+'}','\\widepanel{'+file+'}')))

results=r'''
\section{Results}
\subsection{Complementary experts give AbDev-X the strongest aggregate held-out result}
The principal finding is that combining assay-aware experts improves aggregate external ranking over the component strategies and frozen sequence representations. Across HIC, AC-SINS, PR-CHO, SMAC, and HAC, \AbDev{} reaches mean GDPa3 Spearman $\rho=.472$, compared with $.452$ for D-Stack, $.444$ for E1, $.307$ for ESM-2 Ridge, and $.273$ for AbLang2 Ridge. All 45 assay-by-strategy scores in this controlled nine-strategy comparison reproduce from the primary expert code (Supplementary Figure~\ref{fig:primary}A; Table~\ref{tab:components}). The five-assay grouping is descriptive of the observed results, rather than a prospectively registered subset.

Figure~\ref{fig:architecture} shows the molecular inputs and four-expert construction. E1 selects a useful source, A-MGFS applies assay-related keyword masks, D-Stack combines PLM and descriptor/bio streams, and F-MoE mixes eligible mechanistic-group predictors. Final inference averages their four outputs. The structural and surface representation boxes are conceptual families within the same operational bio block. The measured gain therefore concerns complementary assay-aware experts, without evidence that a learned antibody-dependent final gate is required.
'''
results+=panel('fig01_architecture_user.pdf',r'\textbf{AbDev-X architecture.} Paired VH/VL inputs feed frozen PLM embeddings, subtype information, sequence descriptors, and structural/surface feature families. The latter two families are consolidated into one operational bio block, giving five sources for E1. Four experts perform source selection, assay-specific feature masking, two-stream stacking, and mechanistic group mixing. The primary final prediction averages their outputs with weight $.25$ each. The intact IgG is illustrative (PDB 1IGT); the predicted brazikumab Fv corresponds to the example sequence. Additional assay outputs use the same expert code with its default masks and eligibility rules.','fig:architecture')
results+=r'''
The reproduced primary held-out correlations are HIC $.547$, AC-SINS $.262$, PR-CHO $.251$, SMAC $.647$, and HAC $.650$. Their recomputed BCa intervals exclude zero, although the AC-SINS lower limit is close to zero. Sample sizes range from 61 to 80. Extending the unchanged expert classes to Tm1, Tm2, Tonset, SEC, and Titer gives $.200$, $-.107$, $.132$, $-.229$, and $.133$, respectively; all five intervals include zero. The mean across all ten assays is $.249$ (Supplementary Figure~\ref{fig:primary}B; Table~\ref{tab:allten}). This separates reliable positive-signal evidence from the broader code-default extension.

\subsection{Local and external benchmark context defines the scope of the gain}
Figure~\ref{fig:baseline_main} compares the primary-code ensemble with stored local benchmarks and provides the external comparison context. The primary GDPa1 points are recomputed cluster--isotype fold means; the GDPa3 points and intervals use the same reproduced predictor. Matching KitAb/Amini numerical values are unavailable, so they are not inferred. Archived Jiang and DELPHI results have different protocol provenance and do not constitute matched superiority tests. The GP subpanel concerns a historical source-stack configuration, not the primary four-expert ensemble.

The model's strongest comparative result is its reproducible mean across the five primary assays. Competition champions are per-assay specialist submissions; their point estimates exceed AbDev-X on the competition-five endpoints. Interval overlap is not a test of equivalence or noninferiority. The historical baseline/scatter/CV-distribution panel is retained separately in Supplementary Figure~\ref{fig:performance_main}, with its original configuration identity.
'''
results+=panel('main07_baseline_context.pdf',r'\textbf{Local benchmarks, external comparison context, and GP kernel variants.} \textbf{A}, primary-code GDPa1 cluster--isotype CV means $\pm$ fold SD versus the best stored local entries on four assays; stars indicate higher point estimates, not significance. No matching KitAb/Amini points are inferred. \textbf{B}, reproduced GDPa3 results and BCa 95\% intervals on seven assays, with archived Jiang estimates, stored DELPHI configuration scores, and competition-winner diamonds. Tm2, Titer, Tm1, and Tonset use the code-default extension. \textbf{C}, the original historical source-stack/GP comparison, with its model label clarified as D-Stack. Heterogeneous comparison protocols do not establish universal superiority.','fig:baseline_main')
results+=r'''
\subsection{Representation and split sensitivity explain the assay-aware design}
The aggregate finding motivates the next question: which information helps each assay, and how stable is that evidence under distribution shift? The primary split-sensitivity comparison shows a larger random-to-cluster--isotype reduction for AC-SINS ($.586$ to $.505$) and PR-CHO ($.586$ to $.431$) than for HIC ($.693$ to $.686$; Figure~\ref{fig:generalization_main}A). Cluster-only and cluster--isotype estimates also differ. We therefore declare cluster--isotype as the primary development protocol rather than selecting the most favorable split.

The CV-to-held-out plot shows that development performance does not transfer equally across assays (Figure~\ref{fig:generalization_main}B). The standalone PLM screen and historical source-stack coefficient map add complementary representation context (C--D). Those coefficients belong to a different stacking model: the large Tm2 descriptor coefficient and subtype offset do not describe the weights of the current four-expert average.
'''
results+=panel('main03_generalization.pdf',r'\textbf{Representation generalization and PLM comparison across CV split strategies.} \textbf{A}, primary-model GDPa1 fold means $\pm$ SD under random, cluster, and cluster--isotype splits on HIC, AC-SINS, PR-CHO, SMAC, and HAC. \textbf{B}, primary-code competition-five CV versus GDPa3 correlations with recomputed BCa intervals; diamonds show competition winners and the diagonal indicates equality. Tm2/Titer are default-extension outputs. \textbf{C}, the standalone five-encoder competition-five ablation. \textbf{D}, historical six-source ridge coefficients, including Tm2 Desc $-2.95$/Iso $+.92$ and AC-SINS AbLang2 $+.53$. C--D retain their original comparison-model identities.','fig:generalization_main')
results+=r'''
The broader encoder and structural-feature screen supports the same design rationale (Figure~\ref{fig:encoder_main}). Chain/CDR pooling, CrossAbSense, antibody-specific encoders, and structural-source combinations have different strengths by assay. Combined structural features reach HIC $\rho\simeq .66$ in that standalone screen, whereas Tm2 remains structurally insensitive. These are controlled representation alternatives, not full-ensemble predictions. They support combining sources flexibly without implying that every larger encoder or added structural group improves prediction.
'''
line=original_panel('main04_encoder_structure.pdf');results+=line+'\n\\FloatBarrier\n'
results+=r'''
\subsection{Matched source ablations and data efficiency explain complementarity}
The source-removal experiment refits the primary expert procedure after removing each representation, while preserving the competition-five assay axes in Figure~\ref{fig:efficiency_main}A. A negative full-minus-reduced difference means removal improves held-out ranking. ESM-2 removal improves PR-CHO by approximately $.054$, whereas AbLang2 removal reduces AC-SINS correlation by $.112$. Dependence is therefore assay-specific; no source is universally beneficial.

Learning curves generally benefit from additional training data, with intermediate decreases on some assays (Figure~\ref{fig:efficiency_main}B). These observations do not support a blanket claim of monotonic improvement or absence of saturation. Paired bootstrap comparisons now use the primary predictions and matched PLM baselines (C). On the five primary assays, intervals support improvement over both PLMs for HIC, SMAC, and HAC, while AC-SINS and PR-CHO intervals include zero (Table~\ref{tab:paired}). The CDR enrichment panel (D) remains a separate chain-aware-network analysis and is not used as a direct attribution of the primary ensemble.
'''
results+=panel('main05_efficiency.pdf',r'\textbf{Leave-one-source-out ablation, data efficiency, held-out differences, and CDR enrichment.} \textbf{A}, refitted primary-code full-minus-source-removed GDPa3 $\Delta\rho$ on the competition-five assays. \textbf{B}, primary-code cluster--isotype learning curves from 20\% to 100\% of outer-fold training data. \textbf{C}, paired primary-minus-PLM correlation differences and BCa 95\% intervals (2,000 resamples; exploratory, no multiplicity adjustment). Tm2 and Titer use the unchanged code defaults. \textbf{D}, the original separate chain-aware-network CDR IG diagnostic versus its CDR-length reference.','fig:efficiency_main')
results+=r'''
\subsection{Physicochemical explanations provide molecular context at distinct analysis levels}
The physicochemical LightGBM SHAP analysis identifies hydrophobic, charge-related, regional, and liability features associated with assay predictions (Figure~\ref{fig:shap_main}A). Structure maps visualize selected descriptor fields on a representative predicted Fv (B--C). These explanatory levels are distinct from the four-expert primary prediction function. In particular, a colored surface does not become a primary-model spatial attribution merely because its descriptor is used by an expert. The PR-CHO distributed-signal annotation is illustrative, not evidence of a validated molecular mechanism.

The descriptor-only IG panel makes the regional and category contributions explicit (Figure~\ref{fig:descriptor}). Its source table retains the top 15 features per competition assay, so displayed category shares do not partition every descriptor contribution. Detailed proxy SHAP and per-antibody category variation are provided in Supplementary Figure~\ref{fig:detailed_shap}. Primary expert-internal and historical routing analyses are reported separately with their actual model identities.
'''
results+=original_panel('main06_shap_structure.pdf')+'\n\\FloatBarrier\n'
results+=panel('main08_descriptor_interpretability.pdf',r'\textbf{Descriptor interpretability across the five competition assays.} \textbf{A}, four-category IG shares from the saved descriptor-only top-15 feature list. \textbf{B}, top-four descriptor contributions per assay, colored by category. \textbf{C}, summed displayed category shares. This standardized ridge analysis uses a zero raw-feature reference on GDPa1, without PCA; it is separate from the four-expert primary ensemble.','fig:descriptor')

discussion=verified[verified.index(r'\section{Discussion}'):verified.index(r'\section{Methods}')]
a=discussion.index('Direct attribution also changes');b=discussion.index('The historical format-transfer',a)
discussion=discussion[:a]+r'''The explanation panels address the physicochemical context of prediction. Proxy SHAP, descriptor-only IG, and surface descriptor maps should be interpreted at their stated analysis levels. Direct attribution of the primary predictor requires propagating the actual fitted expert functions; full sequence-to-structure mutation attribution additionally requires recomputing embeddings and structural features under a justified reference. The present displayed surface fields are therefore not evidence for a causal mutational mechanism.

'''+discussion[b:]
discussion=discussion.replace(r'\ref{fig:soft}',r'\ref{fig:soft_extensions}').replace(r'\ref{fig:transfer}',r'\ref{fig:transfer_unique}').replace(r'\ref{fig:learners}',r'\ref{fig:learner_unique}')
discussion=discussion.replace('Direct primary-model IG and conditional residue SHAP connect predictions to explicit molecular features and highlighted regions of a matching predicted Fv.','The complementary physicochemical analyses provide molecular context while retaining their distinct explanatory scope.')
tail=verified[verified.index(r'\section*{Abbreviations}'):verified.index(r'\clearpage',verified.index(r'\section*{Abbreviations}'))]

supp=r'''
\clearpage\appendix
\setcounter{figure}{0}\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{table}{0}\renewcommand{\thetable}{S\arabic{table}}
\section{Supplementary Information}
\subsection{Detailed SHAP attribution and source-stack coefficients}
'''
supp+=original_panel('si01_detailed_shap.pdf')+'\n'
supp+=r'\subsection{Soft gating, physical constraints, and bispecific transfer}'+'\n'+original_panel('si02_soft_moe_extensions.pdf')+'\n'
supp+=r'\subsection{Principal finding and complete assay scope}'+'\n'
supp+=panel('main_headline_unique.pdf',r'\textbf{The principal AbDev-X finding and its assay-level scope.} \textbf{A}, the nine-strategy GDPa3 comparison across HIC, AC-SINS, PR-CHO, SMAC, and HAC; the primary ensemble has mean $.472$. \textbf{B}, reproduced primary scores and the unchanged-code default extension to all ten assays, with recomputed BCa 95\% intervals. Blue marks the primary five-assay comparison and grey the other outputs. Sample sizes differ by assay. Intervals concern assay correlations, not individual predictions or the cross-assay mean. The ten-assay mean is $.249$.','fig:primary')
supp+=r'\subsection{Historical baseline, scatter, and CV-distribution comparison}'+'\n'+original_panel('main02_performance.pdf')+'\n'
for subsection,file in [
 ('Source-selection stability and expert-internal comparisons','si03_expert_specialization_unique.pdf'),
 ('Alternative routing and regional pooling','si04_routing_architecture_unique.pdf'),
 ('Format transfer and composite ranking','si05_transfer_ranking_unique.pdf'),
 ('Robustness and confidence diagnostics','si06_augmentation_confidence_unique.pdf'),
 ('Additional learner and regional descriptor studies','si07_learner_screen_unique.pdf')]:
 supp+='\\subsection{'+subsection+'}\n'+original_panel(file)+'\n'
supp+=r'''
\subsection{Numerical primary-model results}
The following tables report the reproduced component comparison, the ten-assay code-default extension, and matched paired PLM differences. Additional supplementary source-stack, proxy, routing, and transfer panels retain their separate experimental identities; their results are not substituted into the primary comparison.
'''
for label in ['tab:components','tab:allten','tab:paired']:
 i=verified.index('\\label{'+label+'}');a=verified.rfind(r'\begin{table}',0,i);b=verified.index(r'\end{table}',i)+len(r'\end{table}');supp+=verified[a:b]+'\n\\FloatBarrier\n'
out=pre+abstract+intro+results+discussion+methods+tail+supp+r'\end{document}'
# Map the original captions' references to the requested SI placements.
out=out.replace('Figure~\\ref{fig:primary}', 'Figure~\\ref{fig:primary}')
(H/'AbDevX_original_style_main_SI.tex').write_text(out,encoding='utf-8')
print('Restored 7 main and 9 SI figures. All numerical tables are supplementary.')
