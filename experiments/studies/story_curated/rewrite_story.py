from pathlib import Path
import re

HERE=Path(__file__).parent
base=(HERE/'AbDevX_combined_main_SI.tex').read_text(encoding='utf-8')
head=base[:base.index('\\section{Results}')]
# One physical figure width throughout: portrait A4, text width, no height cap.
head=head.replace('\\begin{landscape}\n','').replace('\\end{landscape}\n','')
head=head.replace(r'\newcommand{\figpanel}[4]{\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{#1}\caption{#2}\label{#3}\end{figure}#4}',r'\newcommand{\figpanel}[4]{\begin{figure}[!htbp]\centering\includegraphics[width=\textwidth]{#1}\caption{#2}\label{#3}\end{figure}#4}')
head=head.replace(r'\newcommand{\widepanel}[3]{\FloatBarrier\begin{figure}[p]\centering\includegraphics[width=\linewidth]{#1}\caption{#2}\label{#3}\end{figure}\FloatBarrier}',r'\newcommand{\widepanel}[3]{\FloatBarrier\begin{figure}[p]\centering\includegraphics[width=\textwidth]{#1}\caption{#2}\label{#3}\end{figure}\FloatBarrier}')
head=head.replace('width=\\linewidth','width=\\textwidth')
head=head.replace(r'\newcommand{\widepanel}[3]{\FloatBarrier\begin{figure}[p]\centering\includegraphics[width=\textwidth]{#1}\caption{#2}\label{#3}\end{figure}\FloatBarrier}',r'\newcommand{\widepanel}[3]{\begin{figure}[!htbp]\centering\includegraphics[width=\textwidth]{#1}\caption{#2}\label{#3}\end{figure}\FloatBarrier}')

def figure_line(filename):
 return next(s for s in base.splitlines() if s.startswith(('\\figpanel{'+filename+'}','\\widepanel{'+filename+'}')))
perf=figure_line('main02_performance.pdf')
gen=figure_line('main03_generalization.pdf')
bench=figure_line('main07_baseline_context.pdf')
enc=figure_line('main04_encoder_structure.pdf')
loo=figure_line('main05_efficiency.pdf')
shap=figure_line('main06_shap_structure.pdf')
perf=perf.replace('and Supplementary Figure~\\ref{fig:primary}','and Figure~\\ref{fig:primary}')
# Previously supplementary figures are now placed beside the corresponding main result.
enc=enc.replace('\\widepanel{','\\figpanel{',1)+'{}'
shap=shap.replace('\\widepanel{','\\figpanel{',1)+'{}'

def table(label):
 i=base.index('\\label{'+label+'}')
 a=base.rfind('\\begin{table}',0,i);b=base.index('\\end{table}',i)+len('\\end{table}')
 return base[a:b]
components=table('tab:components');allten=table('tab:allten')

results=r'''\section{Results}
\subsection{Complementary experts give AbDev-X the strongest aggregate held-out result}
The central question is whether an assay-aware combination provides a better
external ranking than its representations or individual experts. On the same five
positive-signal GDPa3 assays, it does: \AbDev{} achieves mean Spearman
$\rho=.472$, the highest value among the nine evaluated strategies
(Figure~\ref{fig:primary}A). D-Stack is the strongest individual expert at $.452$,
followed by E1 at $.444$, while ESM-2 and AbLang2 reach $.307$ and $.273$.
The ensemble gains $.020$ over D-Stack and $.028$ over E1. These are descriptive
held-out differences; they do not establish significance for every pairwise comparison.

\figpanel{main_headline_unique.pdf}{\textbf{The principal AbDev-X finding and its assay-level scope.} \textbf{A}, mean GDPa3 Spearman correlation across the same five positive-signal assays for nine prediction strategies; AbDev-X has the highest recorded mean ($.472$). \textbf{B}, the complete ten-assay evaluation with recorded BCa 95\% intervals. Blue marks the five positive-signal assays; grey marks the remaining endpoints. These five assays are a descriptive grouping, and sample sizes differ by assay. Confidence intervals concern the assay-level correlation, not individual antibody predictions or the cross-assay mean.}{fig:primary}{}

The components contribute different strengths (Table~\ref{tab:components}).
E1 is strongest among the four experts on AC-SINS, F-MoE on PR-CHO, D-Stack on
SMAC, and A-MGFS on HAC. The ensemble has the highest point estimates in this
comparison on HIC and HAC, but does not win every individual assay. Its advantage
is the aggregate balance obtained from complementary predictions. Importantly,
the saved final weights are equal: the result demonstrates useful combination of
assay-aware experts rather than an independently established gain from a
nonuniform top-level router.

COMPONENT_TABLE
\FloatBarrier

The ten-assay evaluation defines where this advantage applies
(Figure~\ref{fig:primary}B; Table~\ref{tab:allten}). HAC reaches $.650$
[$.428,.796$], SMAC $.647$ [$.500,.750$], and HIC $.547$ [$.362,.698$].
AC-SINS ($.262$) and PR-CHO ($.251$) have smaller positive estimates with recorded
intervals excluding zero. The thermal endpoints have intervals crossing zero,
SEC has a negative estimate ($-.171$), and Titer is near zero ($.075$).
The all-ten mean is consequently $.298$. This boundary is part of the main finding:
the present representations support several chromatographic and interaction-related
rankings more strongly than thermal, SEC, or expression predictions.

ALLTEN_TABLE
\FloatBarrier

\subsection{Benchmark context separates aggregate gains from configuration-specific scores}
The headline result comes from a matched five-assay comparison. Broader benchmark
panels address a different question: how the approach relates to other development
configurations and published specialist results. Figure~\ref{fig:performance_main}
retains the previously selected baseline/scatter/CV-distribution graphic.
Its scatter plots and reference-model CV values belong to that recorded
configuration; the nine-model $.472$ result is Figure~\ref{fig:primary}A.
Keeping the configuration distinction prevents improvements observed in different
experiments from being combined into one apparent model result.

PERFORMANCE_FIGURE

Published competition winners provide a demanding external reference.
Their point estimates are $.708$ for HIC, $.392$ for Tm2, $.337$ for AC-SINS,
$.356$ for PR-CHO, and $.310$ for Titer~\citep{ginkgo2026}. These per-assay winners
come from different submissions, and exceed the corresponding complete-panel
\AbDev{} estimates. Without paired predictions, an overlapping interval is not
an equivalence or noninferiority test. The method's strongest comparative position
is its aggregate performance in the controlled component comparison, rather than
a claim to outperform every assay specialist.

Figure~\ref{fig:baseline_main} preserves the requested local/literature/GP panel
as comparison context. Its archived KitAb/Amini and digitized Jiang values have
incomplete matching provenance; the local DELPHI configuration uses median-binarized
targets and pass-probability scores rather than an identical regression objective.
Score direction for lower-is-better assays also requires verification. These records
are not used to establish the primary model's superiority. Jiang et al. themselves
study assay-level descriptors and embeddings~\citep{jiang2026}; the contribution
of \AbDev{} is the particular complementary expert construction and its evaluated
ensemble result, not the invention of assay-level prediction.

BASELINE_FIGURE
\FloatBarrier

\subsection{Representation ablations explain why a single source is insufficient}
The main result motivates the next question: what information do the experts
combine? Representation and split analyses show that both the assay and the
evaluation protocol matter (Figure~\ref{fig:generalization_main}). In the
source-selection record, bio features are selected for HIC, SMAC, and HAC,
AbLang2 for AC-SINS, and ESM-2 for PR-CHO. This pattern is consistent with
structural exposure information helping chromatographic readouts and language-model
features providing complementary sequence information. It is an association-level
interpretation, not proof that an individual representation captures a causal mechanism.

GENERALIZATION_FIGURE

The six-panel encoder/structural screen tests this explanation more directly
(Figure~\ref{fig:encoder_main}). Combined structural features reach HIC CV
correlation of approximately $.66$, while Tm2 remains weak across structural groups.
The encoder ranking varies across assays, and tested cross-attention or region-only
inputs do not consistently improve on the stronger frozen encoders. Adding more
features or replacing an encoder is therefore an assay-dependent choice.

ENCODER_FIGURE

Selection stability adds a second reason to combine experts rather than rely only
on a winner. In the recorded resampling analysis, AC-SINS selects AbLang2 in
98.8\% of resamples, whereas HIC divides between bio features (58.4\%) and ESM-2
(41.6\%; Supplementary Figure~\ref{fig:expert_specialization}A). Globally corrected
source-comparison tests do not establish a uniquely superior source for every assay.
The ensemble can retain useful alternatives when source preferences are uncertain.

Validation difficulty also affects the estimate. In the separate legacy source-stack
experiment, competition-five mean CV correlation falls from $.528$ on random splits
to $.462$ on cluster--isotype splits. These values are distinct from the current
ensemble's external $.472$. They show why development performance should not be
treated as a substitute for ranking new antibodies.
\FloatBarrier

\subsection{Source-removal and routing ablations favor selective information and restrained fusion}
Source-removal experiments test whether these representation differences translate
into useful predictive information (Figure~\ref{fig:efficiency_main}A). In the
legacy stack, removing bio features reduces HIC held-out correlation by $.134$,
while removing AbLang2 reduces AC-SINS by $.107$. Other removals improve a readout:
the full-minus-removed difference for ESM-2 on PR-CHO is $-.060$. Complementarity
therefore concerns the quality and relevance of information, not simply its quantity.

EFFICIENCY_FIGURE

Learning curves indicate that more matched data can help, with important assay-specific
exceptions (Figure~\ref{fig:efficiency_main}B). HIC rises from $.396$ at 20\% training
fraction to $.669$ at full data and AC-SINS from $.339$ to $.560$. Tm2 peaks at
60\% and then declines; PR-CHO and Titer also contain local reversals. These curves
do not establish universal monotonic improvement or an asymptotic performance ceiling.
The paired-difference record in panel C belongs to another configuration and is
not used as a significance test of the primary $.472$ ensemble.

More flexible routing is not consistently better. In the shared-input experiment,
fixed ridge fusion reaches competition-five mean GDPa3 correlation $.251$, compared
with $.216$ for sparse top-2 routing, $.214$ for sparse top-3, and $.240$ for KronLSTM
(Supplementary Figure~\ref{fig:routing_unique}B). A conditional gate improves four
assays but reduces AC-SINS from $.249$ to $.010$, leaving its mean at $.249$.
Even a hybrid mean gain to $.275$ retains an AC-SINS decrease. These experiments
explain why per-assay behavior, rather than a mean alone, guides architecture choice.

The separate source-level Soft MoE experiment reaches the same practical issue:
gate weights remain close to $.2$ for most assays, with more differentiation for Titer
(Supplementary Figure~\ref{fig:soft_extensions}A--B). This is distinct from F-MoE's
internal mechanistic groups and the final four-expert ensemble. Together these
ablations support restrained fusion at small sample size, without showing that
all forms of learned routing must fail.

The broader architecture, pooling, and tabular screens provide additional alternatives
without a universal replacement (Supplementary Figures~\ref{fig:routing_unique}
and~\ref{fig:learner_unique}). The strongest recorded tabular mean is $.456$ for
TabPFN on descriptor-plus-bio features. Some configurations win individual endpoints,
but the primary ensemble retains the stronger recorded aggregate on its evaluated
five-assay set.
\FloatBarrier

\subsection{Expert specialization and molecular proxies connect the gain to interpretable information}
The architecture's specialization occurs within its experts even though the final
primary weights are equal. The recorded D-Stack analysis emphasizes descriptor/bio
information for HIC, SMAC, and HAC, while the PLM stream receives $.976$ for AC-SINS
and $.624$ for PR-CHO (Supplementary Figure~\ref{fig:expert_specialization}B).
F-MoE's HIC groups divide between hydrophobicity ($.431$) and surface ($.569$),
while the PR-CHO record emphasizes charge ($.710$; panel C). These are internal
expert quantities, not interchangeable with the legacy source-stack coefficients
or final ensemble weights.

The SHAP and structural panel makes the molecular association visible
(Figure~\ref{fig:shap_main}). Aromatic and hydrophobic descriptors are prominent
for HIC, charge and pI for AC-SINS, and deamidation-related and charge descriptors
for PR-CHO in the respective LightGBM proxies. Surface maps illustrate exposure
and charge-related fields on a predicted structure. They do not independently
validate the predictor or establish that a colored patch causes the assay outcome.

SHAP_FIGURE

Supplementary Figure~\ref{fig:detailed_shap} adds a different level of evidence:
the top-12 mean absolute SHAP summaries and the per-antibody category contributions.
This allows a common assay-level trend to coexist with different dominant terms
for individual antibodies. The source-stack coefficient heatmap supplies historical
context but is not a feature attribution of the four-expert ensemble.

The chain-aware proxy shows CDR enrichment of approximately 2.56--3.54 times the
sequence-length baseline (Figure~\ref{fig:efficiency_main}D). A shallow pooled-embedding
proxy has much weaker enrichment. Their difference concerns explanation-model
structure and positional information; neither provides a complete or causal
residue-level explanation of \AbDev{}.
\FloatBarrier

\subsection{Transfer and decision-use experiments define the remaining deployment boundary}
Exploratory format transfer tests whether the useful information extends beyond the
evaluated IgG panel. On GDPa4, averaging two parental-arm predictions gives recorded
correlations of $.920$ for HIC, $.870$ for AC-SINS, and approximately $.670$ for
PR-CHO (Supplementary Figure~\ref{fig:soft_extensions}D). Shared parental antibodies,
including parents represented in GDPa1, make these rows statistically dependent.
The high correlations support this panel's arm-average approximation but do not
establish parent-disjoint bispecific generalization.

VHH-Fc transfer is less stable. An earlier GDPa2.1 run reports HIC $.605$ and
AC-SINS $.520$, whereas the recorded GDPa5 run has HIC $.162$, AC-SINS $.121$,
PR-CHO $.326$, and Tm2 $.018$ (Supplementary Figure~\ref{fig:transfer_unique}A).
Combined IgG/VHH-Fc training changes a reconstruction's mean from $.189$ to $.198$
but reduces AC-SINS from $.321$ to $.219$ (panel B). More data from another format
are not automatically equivalent to more matched IgG examples.

Candidate selection is another separate validation task. A legacy stack's composite
ranking reaches correlation $.200$ and P@5/P@10/P@20 of $.20$/$ .20$/$ .25$;
P@20 equals the recorded random expectation (Supplementary
Figure~\ref{fig:transfer_unique}C). These results support, at most, narrow exploratory
liability-ranking uses rather than a validated multi-property lead-selection score.

Finally, disagreement is not a generally calibrated confidence estimate. Its recorded
correlation with absolute error ranges from $-.329$ to $.169$, and abstention can
reduce retained-set rank correlation (Supplementary Figure~\ref{fig:confidence_unique}).
The distinct concepts of prediction uncertainty, metric confidence intervals,
and OOD sensitivity should remain separate. These limitations motivate prospective
evaluation on new matched panels, rather than adding architectural complexity alone.
'''
results=results.replace('COMPONENT_TABLE',components).replace('ALLTEN_TABLE',allten)
for key,value in [('PERFORMANCE_FIGURE',perf),('BASELINE_FIGURE',bench),('GENERALIZATION_FIGURE',gen),('ENCODER_FIGURE',enc),('EFFICIENCY_FIGURE',loo),('SHAP_FIGURE',shap)]:results=results.replace(key,value)

# Retain the detailed Discussion, declarations, and verified bibliography.
discussion=base[base.index('\\section{Discussion}'):base.index('\\appendix')]
discussion=discussion.removesuffix('\\clearpage\n')
discussion=discussion.replace('\\subsection{What the principal result means}','\\subsection{A small-data advantage from complementary molecular views}')
discussion=discussion.replace('\\subsection{Why the ablations strengthen the contribution}','\\subsection{The ablations explain the architecture rather than merely rank it}')
discussion=discussion.replace('\\subsection{What is and is not established about superiority}','\\subsection{Scope of the comparative advantage}')

supp=r'''
\clearpage
\appendix
\setcounter{figure}{0}\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{table}{0}\renewcommand{\thetable}{S\arabic{table}}
\section{Supplementary Information}
The supplementary analyses examine descriptor-level interpretation, fusion alternatives,
and transfer beyond the primary IgG evaluation. Figures S1--S2 present explanatory
and extension experiments; Figures S3--S7 examine source-selection stability,
alternative routing, format transfer, robustness, and learner choice.

\subsection{Detailed SHAP feature attribution and per-antibody variation}
Supplementary Figure~\ref{fig:detailed_shap} summarizes the descriptor LightGBM
proxy at both population and antibody levels. For HIC, aromatic and aliphatic
composition features lead the mean absolute SHAP ranking. The Tm2 proxy emphasizes
regional hydrophobic-moment and composition features; these associations should
not be interpreted as a demonstration of a germline-mediated causal mechanism.
Fv charge and pI dominate the AC-SINS proxy.

The historical source-stack weights provide context for representation mixing:
Tm2 has a large negative descriptor coefficient ($-2.95$) and positive subtype
coefficient ($+.92$), while AC-SINS has AbLang2 $+.53$. The per-antibody category
plot shows signed descriptor contributions sorted by magnitude. Different antibodies
can have different dominant category contributions despite a common assay model.
The five endpoints here are the original Ginkgo competition assays, not the
HAC/SMAC positive-signal subset. The source-weight heatmap provides context for
interpreting population-level feature rankings alongside individual contributions.

\widepanel{si01_detailed_shap.pdf}{\textbf{Detailed SHAP feature attribution, block meta-weights, and per-antibody category contributions.} \textbf{A--C}, top-12 mean absolute SHAP for the HIC, Tm2, and AC-SINS physicochemical proxy models. \textbf{D}, six-source ridge coefficients over five competition assays, including Tm2 Desc $-2.95$/Iso $+.92$ and AC-SINS AbL2 $+.53$. \textbf{E}, signed per-antibody category contributions for hydrophobic, charge, polar, size, and liability descriptors. These proxy summaries and historical source-stack coefficients are not direct attributions of the full ensemble.}{fig:detailed_shap}

\subsection{Soft routing, physics-informed constraints, and bispecific composition}
Supplementary Figure~\ref{fig:soft_extensions} combines three extension questions.
The source-level soft gate does not consistently improve on ridge stacking, and
its weights are near $.2$ for most source/assay pairs. Titer is more differentiated,
with Bio $.301$ and AbLang2 $.264$. The plotted results do not support treating this
source router as a broadly successful specialization mechanism.

Physics-inspired constraints can also remove useful flexibility. The isotonic
SAP predictor has HIC correlation approximately $-.020$, compared with $.634$ for
the baseline ridge predictor. Increasing rank-regularization strength reduces
AC-SINS and PR-CHO correlations in this record; shared-task prediction has mixed
effects. These findings concern the tested formulations rather than all possible
physical priors.

The GDPa4 arm-average panel reports monospecific reference correlations of $.950$
for HIC, $.905$ for AC-SINS, and $.864$ for PR-CHO, and corresponding bispecific
values of $.920$, $.870$, and approximately $.670$. Shared parents limit interpretation
as an independent held-out test of novel parental sequences.

\widepanel{si02_soft_moe_extensions.pdf}{\textbf{Soft MoE gating, physics-informed constraints, and bispecific arm-average transfer.} \textbf{A}, ridge, soft-gate, and best-block reference CV results. \textbf{B}, source-gate weights, mostly near uniform. \textbf{C}, isotonic SAP, rank-regularization, and shared-task experiments; isotonic HIC is $-.020$. \textbf{D}, arm-average GDPa4 scatter plots: bispecific HIC $.920$, AC-SINS $.870$, and PR-CHO approximately $.670$. These extension experiments have separate model configurations; parent sharing limits bispecific generalization claims.}{fig:soft_extensions}

\subsection{Source-selection stability and expert-internal specialization}
Figure~\ref{fig:expert_specialization} asks whether selected sources and internal
expert weights are stable or mechanistically differentiated. Panel A is a
resampling frequency, panel B describes the two-stream D-Stack analysis, and panel C
describes F-MoE mechanistic groups. These quantities differ from both the six-block
legacy heatmap and the equal final four-expert weights. Thus this figure adds
information that is not conveyed by the main source-coefficient heatmap.

\widepanel{si03_expert_specialization_unique.pdf}{\textbf{Stability and internal specialization.} \textbf{A}, source-selection frequencies in the recorded resampling study. \textbf{B}, PLM versus descriptor/bio stream weights in the saved two-stream analysis. \textbf{C}, F-MoE mechanism-group weights. Absent/ineligible groups are displayed as zero. These expert-internal quantities should not be interpreted as antibody-dependent final ensemble gates.}{fig:expert_specialization}

\subsection{Architecture alternatives and regional pooling}
Figure~\ref{fig:routing_unique} complements the main source-removal experiment by
testing different fusion architectures. It compares the seven-architecture CV
screen, a controlled shared-input routing experiment, conditional-gate trade-offs,
and regional pooling. Source-removal and CDR-enrichment analyses are presented in Figure~\ref{fig:efficiency_main}.
The architecture experiment's fixed-ridge reference and the primary four-expert
ensemble have different configurations and should not share a model name.

\widepanel{si04_routing_architecture_unique.pdf}{\textbf{Architecture and routing alternatives.} \textbf{A}, mean CV correlations for seven architectures under three splits. \textbf{B}, shared-input GDPa3 routing comparison. \textbf{C}, conditional/hybrid fusion exposes an AC-SINS failure that a mean can obscure. \textbf{D}, mean/CDR/framework pooling on competition-five assays. These are distinct ablation experiments; none replaces the primary ensemble's matched five-assay evaluation.}{fig:routing_unique}

\subsection{Format transfer, combined training, and candidate ranking}
Figure~\ref{fig:transfer_unique} adds VHH-Fc and decision-use evidence not shown in
S2's bispecific panel. GDPa5 transfer does not reproduce the strongest GDPa2.1
surface-assay results. Combining format-specific datasets gives only a modest
aggregate change and an AC-SINS decrease in its reconstruction. Composite ranking
does not establish validated multi-property candidate selection. These results
define separate validation requirements for deploying a developability predictor.

\widepanel{si05_transfer_ranking_unique.pdf}{\textbf{Transfer and ranking limits beyond the bispecific experiment.} \textbf{A}, GDPa5 VHH-Fc correlations for four predictors. \textbf{B}, a separate GDPa1-only versus combined-format training reconstruction. \textbf{C}, composite precision at 5, 10, and 20 for random, ESM-2, and legacy stacking. Bispecific transfer is evaluated separately in Figure~\ref{fig:soft_extensions}D.}{fig:transfer_unique}

\subsection{External-data normalization, structural sensitivity, and confidence}
Figure~\ref{fig:confidence_unique} groups four distinct robustness diagnostics.
Normalization and augmentation can improve an assay in a reconstruction but do not
provide a universal gain. Numerical structural-feature perturbation measures
sensitivity, not physical conformational uncertainty. Expert disagreement correlates
weakly or negatively with error in several assays, and abstention can worsen
retained-set ranking. These analyses must not be converted into calibrated
per-antibody prediction intervals without additional validation.

\widepanel{si06_augmentation_confidence_unique.pdf}{\textbf{Data and prediction robustness diagnostics.} \textbf{A}, external-study normalization/augmentation strategies. \textbf{B}, HIC structural-feature perturbation. \textbf{C}, disagreement versus absolute-error correlation. \textbf{D}, retained-set correlation after abstention. Physics-informed constraints are evaluated separately in Figure~\ref{fig:soft_extensions}C.}{fig:confidence_unique}

\subsection{Tabular learners and hierarchical regional descriptors}
Figure~\ref{fig:learner_unique} compares learner families and explicit regional
descriptor constructions. The strongest tabular mean is $.456$, while particular
settings improve individual assays. Global, CDR-only, framework-only, and hierarchical
descriptors have different assay-specific strengths. This figure concerns different
regressors and feature constructions from the main frozen-encoder screen.

\widepanel{si07_learner_screen_unique.pdf}{\textbf{Additional learners and regional descriptor constructions.} \textbf{A}, ridge, XGBoost, and TabPFN over five feature configurations, summarized on the five positive-signal assays. \textbf{B}, regional/hierarchical descriptor results by assay. These are exploratory configuration screens rather than newly locked external tests.}{fig:learner_unique}

\subsection{Comparison provenance and figure guide}
The archived performance, generalization, and baseline panels summarize experiments
with distinct configurations. Their captions identify the corresponding evaluation scope. The archived DELPHI script needs score-direction checking;
the KitAb/Amini values lack verified matching provenance; and digitized Jiang
values are contextual, not a matched reproduction. The GP extension uses a reduced
feature reconstruction with incomplete raw-table provenance. It therefore does not
support a new superiority or uncertainty claim for the primary ensemble.

\begin{longtable}{>{\raggedright\arraybackslash}p{.14\linewidth}>{\raggedright\arraybackslash}p{.37\linewidth}>{\raggedright\arraybackslash}p{.40\linewidth}}
\caption{Curated figure map: the question answered by each nonredundant figure.}\\
\toprule Figure & Scientific question & Distinct content\\\midrule\endfirsthead
\toprule Figure & Scientific question & Distinct content\\\midrule\endhead
\ref{fig:architecture} & What enters the predictor? & Updated paired-sequence architecture and predicted Fv\\
\ref{fig:primary} & What is the principal external result? & True nine-strategy mean and complete ten-assay uncertainty\\
\ref{fig:performance_main} & What does the selected historical configuration show? & Benchmark, scatter, and CV-distribution source panel\\
\ref{fig:baseline_main} & How does broader comparison context differ? & Literature/local/GP source panel with stated scope\\
\ref{fig:generalization_main} & Do sources and splits matter? & Split sensitivity, PLM ablation, historical source coefficients\\
\ref{fig:encoder_main} & Which encoders/structural groups help? & Six-panel sequence/structural screen\\
\ref{fig:efficiency_main} & Is information useful, and is more data helpful? & Source removal, learning, historical paired record, CDR IG\\
\ref{fig:shap_main} & What molecular information is associated with predictions? & SHAP beeswarms and structure maps\\
\ref{fig:detailed_shap} & How do aggregate and individual contributions differ? & Top-12 proxy SHAP and per-antibody category contributions\\
\ref{fig:soft_extensions} & Do soft routing, constraints, and arm composition help? & Source soft gate, physical constraints, bispecific prediction\\
\ref{fig:expert_specialization} & Where does expert specialization occur? & Selection stability and two expert-internal weight systems\\
\ref{fig:routing_unique} & Can flexible routing/regional pooling replace the design? & Architecture screen and routing failure modes\\
\ref{fig:transfer_unique} & Does it support other formats and candidate selection? & VHH transfer, combined training, composite precision\\
\ref{fig:confidence_unique} & Are data shifts and confidence scores robust? & Normalization, perturbation, calibration, abstention\\
\ref{fig:learner_unique} & Are alternative learners or descriptors better? & Tabular and hierarchical-descriptor screens\\\bottomrule
\end{longtable}
\end{document}
'''
out=head+results+'\n'+discussion+supp
# Normalize the compact precision notation without changing any score.
out=out.replace('$.20$/$ .20$/$ .25$',r'$.20$, $.20$, and $.25$')
(HERE/'AbDevX_curated_main_SI.tex').write_text(out,encoding='utf-8')
print('Created narrative revision: headline result first; 8 main and 7 unique SI figures.')
