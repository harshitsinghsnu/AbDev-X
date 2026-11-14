from pathlib import Path
import re

HERE=Path(__file__).parent
s=(HERE/'AbDevX_rewritten.tex').read_text(encoding='utf-8')

def take_line(filename):
 global s
 match=next(x for x in s.splitlines() if x.startswith(('\\figpanel{'+filename+'}','\\widepanel{'+filename+'}')))
 s=s.replace(match+'\n','')
 return match

# Move the earlier detailed/auxiliary figures into the supplement without losing references.
detail=[]
for f in ['fig02_primary_results.pdf','fig03_representations.pdf','fig04_ablations.pdf','fig05_interpretability.pdf','fig06_transfer.pdf']:
 detail.append(take_line(f))
take_line('figS01_encoders.pdf')
take_line('figS06_surface_maps.pdf')
s=s.replace('fig:sencoders','fig:encoder_main').replace('fig:smaps','fig:shap_main')

s=s.replace('fig01_architecture.pdf','fig01_architecture_user.pdf')
start=s.index('\\caption{\\textbf{Operational architecture of \\AbDev{}.}')
end=s.index('\n\\label{fig:architecture}',start)
s=s[:start]+r'''\caption{\textbf{Updated architecture supplied by the authors.} \textbf{A}, paired VH/VL sequence input and the sequence-derived Fv structure. The full brazikumab VH and VL examples correspond to the ABodyBuilder3 prediction shown below them; the intact IgG image is illustrative (experimental PDB 1IGT). Heavy and light chains are blue and green. \textbf{B}, six conceptual representation families. \textbf{C}, four expert strategies followed by constrained weighted fusion. \textbf{D}, ten continuous assay outputs. The drawing preserves the supplied PowerPoint. Operational details are specified in Methods: structural/surface families are consolidated in the saved bio block, the primary A-MGFS implementation uses keyword masks, and recorded top-level primary fusion weights are uniform.}'''+s[end:]

perf=r'''\figpanel{main02_performance.pdf}{\textbf{AbDev-X versus ESM-2 Ridge and the model comparison record.} \textbf{A}, GDPa1 cluster--isotype CV correlation (fold mean $\pm$ SD) for the current ensemble and ESM-2 Ridge on the five competition assays; annotations show signed differences in fold means. \textbf{B}, GDPa3 mean correlation for nine strategies on the same five positive-signal assays, with AbDev-X highlighted ($\rho=.472$). No confidence interval for the cross-assay mean is inferred by averaging marginal interval endpoints. \textbf{C}, recorded pooled-OOF CV correlations for other configurations (grey) and AbDev-X (diamonds), with values .665, .510, .463, .560, and .878 for HIC, AC-SINS, PR-CHO, SMAC, and HAC. Comparator SMAC/HAC scores are unavailable in this table and are not imputed. Fold means in A and pooled-OOF correlations in C are different summaries.}{fig:performance_main}{}'''
anchor='The per-assay results explain why the ensemble helps.'
pos=s.index('\n\n',s.index(anchor))
s=s[:pos]+'\n\n'+perf+s[pos:]
s=s.replace('(Figure~\\ref{fig:primary}B; Table~\\ref{tab:components})','(Figure~\\ref{fig:performance_main}B; Table~\\ref{tab:components})')

gen=r'''\figpanel{main03_generalization.pdf}{\textbf{Representation generalization and PLM comparison across CV strategies.} \textbf{A}, mean GDPa1 CV Spearman $\rho$ under random IID, cluster OOD, and cluster--isotype splits for the legacy source-stacking configuration, ESM-2 Ridge, AbLang2-ElasticNet, and SaProt-Ridge. These four-model records cover the five competition assays, not the HAC/SMAC positive-signal subset. \textbf{B}, current AbDev-X fold-mean CV $\rho\pm$SD versus GDPa3 $\rho$ with its recorded BCa interval; open diamonds mark published assay-specific competition winners, and the dashed diagonal is equality of the two correlations. \textbf{C}, five-encoder PLM ablation on the competition-five assays. \textbf{D}, signed ridge coefficients for ESM2, Desc, Iso, Bio-R, Bio-L, and AbL2 in the legacy source stack. Tm2 has Desc $-2.95$ and Iso $+.92$; AC-SINS has AbL2 $+.53$. These internal legacy coefficients are not the final four-expert fusion weights.}{fig:generalization_main}{}'''
enc=r'''\widepanel{main04_encoder_structure.pdf}{\textbf{Encoder screen and structural feature ablation.} \textbf{A--B}, sequence-input variants (VH+VL, CDR, CrossAbSense, AbLang2, ESM-2) on cluster--isotype and random splits. \textbf{C}, mean CV correlations for A3D, SAP, electrostatics, MOE, and their combination, with ESM-2 and AbLang2 references. \textbf{D}, nine-source encoder screen on cluster--isotype CV. \textbf{E}, GDPa3 results for four evaluated encoders. \textbf{F}, structural-source CV correlations by assay: combined features reach HIC $\rho\simeq .66$, while Tm2 shows weak structural signal. These screens use HIC, Tm2, AC-SINS, PR-CHO, and Titer. Error bars and averages belong to the source experiment and are not the primary ensemble's five-positive-signal evaluation.}{fig:encoder_main}'''
needle='The structural screen supports a similarly assay-dependent conclusion.'
pos=s.index(needle)
s=s[:pos]+gen+'\n\n'+enc+'\n\n'+s[pos:]
s=s.replace('The structural screen supports a similarly assay-dependent conclusion.','The structural screen supports a similarly assay-dependent conclusion (Figure~\\ref{fig:encoder_main}C,F).')
s=s.replace('The full structural encoder','The full structural encoder')
s=s.replace('On the competition-five subset, the legacy source stack falls from mean CV correlation $.528$ under random splits to $.462$ under cluster--isotype splits (Figure~\\ref{fig:representations}B).','On the competition-five subset, the legacy source stack falls from mean CV correlation $.528$ under random splits to $.462$ under cluster--isotype splits (Figure~\\ref{fig:generalization_main}A).')

eff=r'''\figpanel{main05_efficiency.pdf}{\textbf{Leave-one-source-out ablation, data efficiency, paired-comparison records, and CDR enrichment.} \textbf{A}, legacy source-stack held-out differences, defined as full-model minus source-removed $\rho$ for ESM2, Desc, Iso, Bio, and AbL2; a negative value means removal improves ranking (ESM2/PR-CHO: $-.060$). \textbf{B}, cluster--isotype learning curves from 20\% to 100\% training data on the competition-five assays; curves are not uniformly monotonic. \textbf{C}, archived paired differences and reported intervals versus ESM2/AbLang2. These point differences do not match the current primary ensemble and do not establish its pairwise significance; intervals are displayed as historical exploratory results without a new multiplicity claim. \textbf{D}, chain-aware proxy IG fraction in CDRs versus the CDR-length baseline, with enrichment of approximately $3.5\times$, $3.2\times$, $2.6\times$, $2.8\times$, and $2.6\times$.}{fig:efficiency_main}{}'''
needle='In the shared-input architecture experiment,'
pos=s.index(needle)
s=s[:pos]+eff+'\n\n'+s[pos:]
s=s.replace('Leave-one-source-out experiments on the legacy stack identify contributions that differ by assay (Figure~\\ref{fig:ablations}A).','Leave-one-source-out experiments on the legacy stack identify contributions that differ by assay (Figure~\\ref{fig:efficiency_main}A).')
s=s.replace('The learning curves reinforce the small-data interpretation without supporting universal monotonicity (Figure~\\ref{fig:ablations}B).','The learning curves reinforce the small-data interpretation without supporting universal monotonicity (Figure~\\ref{fig:efficiency_main}B).')

shap=r'''\widepanel{main06_shap_structure.pdf}{\textbf{Physicochemical proxy SHAP and structural descriptor maps across ten assays.} \textbf{A}, LightGBM proxy SHAP beeswarms for the five Ginkgo competition assays (HIC, AC-SINS, PR-CHO, Tm2, Titer); points represent antibodies and colors denote feature values. \textbf{B}, representative structure maps for SMAC, HAC, Tm1, Tonset, and SEC. \textbf{C}, maps for the original five assays. Fields visualize hydrophobic exposure, charge-related descriptors, aromatic exposure, or aggregation-related patches, as labeled. These maps project selected descriptors onto an ABodyBuilder3 prediction; they are not spatial attributions from E1 or the full ensemble. The PR-CHO ``globally distributed'' label is an illustrative interpretation, not evidence establishing a molecular mechanism. The SHAP models, structure visualizations, and primary prediction ensemble are distinct analysis levels.}{fig:shap_main}'''
needle='The chain-aware proxy assigns CDR regions'
pos=s.index(needle)
s=s[:pos]+shap+'\n\n'+s[pos:]
s=s.replace('The named-feature proxies are consistent with complementary molecular signals.','The named-feature proxies are consistent with complementary molecular signals (Figure~\\ref{fig:shap_main}A).')
s=s.replace('Supplementary Figure~\\ref{fig:shap_main}','Figure~\\ref{fig:shap_main}')
s=s.replace('Supplementary Figure~\\ref{fig:encoder_main}','Figure~\\ref{fig:encoder_main}')

base=r'''\figpanel{main07_baseline_context.pdf}{\textbf{Local benchmarks, external comparison context, and GP kernel variants.} \textbf{A}, GDPa1 cluster--isotype CV fold means $\pm$SD for the current AbDev-X ensemble versus the best stored local benchmark entry on HIC, Tm2, AC-SINS, and PR-CHO; stars indicate higher point estimates, not statistical significance. A verified matching KitAb/Amini table is unavailable, so numerical entries and error bars are not invented. \textbf{B}, current AbDev-X GDPa3 results and recorded BCa intervals on seven assays, contextual archived/digitized Jiang estimates, stored DELPHI configuration scores, and competition-winner diamonds. Sample sizes are assay dependent (HIC 79, Tm2 74, otherwise 80 here). Literature estimates and the DELPHI sign-audit record are not matched-protocol superiority tests. \textbf{C}, the supplied legacy source-stack versus GP-FA/GP-HA/GP-HP CV graphic, retained as an archival comparison with separate feature/run provenance. Neither B nor C establishes universal superiority of the primary model.}{fig:baseline_main}{}'''
needle='\\subsection{Format transfer exposes the boundary of the learned information}'
pos=s.index(needle)
s=s[:pos]+base+'\n\\FloatBarrier\n\n'+s[pos:]
s=s.replace('Published methods and competition winners provide useful context but require separate interpretation.','Published methods and competition winners provide useful context but require separate interpretation (Figure~\\ref{fig:baseline_main}).')

# Relocate the previous main figures as grouped supplementary detail.
intro=r'''\subsection{Additional component, representation, architecture, and transfer detail}
The following five figures retain the detailed numerical audit underlying the main
panels. They provide the full ten-assay interval plot, explicit top-level weights,
source-selection stability, controlled routing comparisons, internal expert weights,
and exploratory transfer/ranking results. Their model variants and assay subsets
remain labeled separately from the primary result.
'''
anchor='\\subsection{Encoder diversity and sequence-input ablations}'
detailblock=intro+'\n\n'.join(detail)+'\n\\FloatBarrier\n\n'
s=s.replace(anchor,detailblock+anchor,1)
# Main structural maps are no longer duplicated in the supplement.
start=s.index('\\subsection{Structural visualization and proxy explanations}')
end=s.index('Descriptor-level IG provides',start)
s=s[:start]+r'''\subsection{Additional descriptor-level proxy explanations}
The complete surface-map and SHAP panel is in the main manuscript
(Figure~\ref{fig:shap_main}). The supplementary analysis below concerns a separate
descriptor-level IG proxy.

'''+s[end:]

# Replace the old fixed-number provenance table with references to the new panel set.
start=s.index('\\subsection{Result provenance and panel placement}')
s=s[:start]+r'''\subsection{Result provenance and main/supplementary placement}
\begin{longtable}{>{\raggedright\arraybackslash}p{.15\linewidth}>{\raggedright\arraybackslash}p{.34\linewidth}>{\raggedright\arraybackslash}p{.42\linewidth}}
\caption{Main figure placement and provenance. Main figures follow the corresponding Results subsections; all additional quantitative panels are supplementary.}\\
\toprule Figure & Scientific role & Source\\\midrule\endfirsthead
\toprule Figure & Scientific role & Source\\\midrule\endhead
\ref{fig:architecture} & Updated sequence-input architecture & Author-supplied PowerPoint exported directly with PowerPoint\\
\ref{fig:performance_main} & Current CV baseline comparison, nine-model mean, CV distribution & CV error bars, final evaluation summary, all-model/ensemble CV tables\\
\ref{fig:generalization_main} & Split sensitivity, CV/external comparison, PLM screen, ridge weights & Split tables, BCa table, PLM-ablation CSV, representation weights CSV\\
\ref{fig:encoder_main} & Six-panel encoder and structural screen & Supplied \path{panel_encoder_struct.pdf}\\
\ref{fig:efficiency_main} & Source removal, learning, archived paired differences, CDR enrichment & Ablation, learning-curve, archived significance and IG tables\\
\ref{fig:shap_main} & SHAP and structural maps across ten assays & Supplied \path{panel3_v2.pdf}\\
\ref{fig:baseline_main} & Baseline/literature/GP context & Local leaderboard, all-ten summary, archived Jiang/DELPHI entries, supplied GP graphic\\
\ref{fig:primary}--\ref{fig:transfer} & Additional component and mechanistic detail & Previous revision's verified detailed panels\\
Other supplementary figures & Architecture/pooling, constraints, augmentation, uncertainty, learner, IG, and GP audits & Experiment-specific CSVs and retained source assets\\\bottomrule
\end{longtable}
\end{document}
'''
s=s.replace('Fifteen','Nineteen')
(HERE/'AbDevX_main_figures.tex').write_text(s,encoding='utf-8')
print('Updated main figure placement: architecture + six requested panels; remaining panels supplementary.')
