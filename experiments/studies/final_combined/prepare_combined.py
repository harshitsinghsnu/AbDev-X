from pathlib import Path
import shutil,re,json,hashlib

HERE=Path(__file__).parent
s=(HERE/'AbDevX_main_figures.tex').read_text(encoding='utf-8')
mapping={
 'main02_performance.pdf':Path(r'C:\Users\hs494\ab-dev\paper\figures_backup_20261103\panel_performance_comparison.pdf'),
 'main03_generalization.pdf':Path(r'C:\Users\hs494\ab-dev\paper\paper10assay\panel_gen_rep.pdf'),
 'main07_baseline_context.pdf':Path(r'C:\Users\hs494\ab-dev\paper\paper10assay\panel_baseline_comparison.pdf')
}
manifest=[]
for name,p in mapping.items():
 dst=HERE/'figures'/name;shutil.copy2(p,dst)
 manifest.append({'source':str(p),'manuscript_asset':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'exact_copy':dst.read_bytes()==p.read_bytes()})
(HERE/'selected_figure_sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def take_line(filename):
 global s
 line=next(z for z in s.splitlines() if z.startswith(('\\figpanel{'+filename+'}','\\widepanel{'+filename+'}')))
 s=s.replace(line+'\n','');return line
relocated=[take_line(f) for f in ['main04_encoder_structure.pdf','main05_efficiency.pdf','main06_shap_structure.pdf']]

perf=r'''\figpanel{main02_performance.pdf}{\textbf{Baseline comparison and model CV distribution.} \textbf{A}, the source panel compares its AbDev-X/source-stacking configuration with the labeled ``Best baseline'' on five competition assays, with displayed error bars and signed margins; the source does not identify this comparator exclusively as ESM-2 Ridge. \textbf{B}, observed-versus-predicted scatter plots for a separate evaluation: HIC $\rho=.296$, Tm2 $-.066$, and Titer $.157$. This supplied panel contains scatter plots, not the current nine-model mean comparison. \textbf{C}, source-panel cluster--isotype CV distributions for HIC, Tm2, AC-SINS, PR-CHO, and Titer, with its reference model shown by diamonds. These are historical configuration results, distinct from the current primary ensemble. The matched nine-model comparison with mean $.472$ is reported in Table~\ref{tab:components} and Supplementary Figure~\ref{fig:primary}.}{fig:performance_main}{}'''
gen=r'''\figpanel{main03_generalization.pdf}{\textbf{Representation generalization and PLM comparison across CV split strategies.} \textbf{A}, per-assay GDPa1 CV Spearman $\rho$ (fold mean $\pm$SD) under random IID, cluster OOD, and cluster--isotype OOD for HIC, AC-SINS, PR-CHO, SMAC, and HAC. The supplied panel compares one recorded ensemble configuration across split strategies; it is not a four-model cross-assay-mean plot. \textbf{B}, recorded CV versus held-out correlation with uncertainty, champion diamonds, and an equality diagonal on the original competition-five assays. \textbf{C}, AbLang2, ESM-2, AntiBERTy, ProtBert, and ProGen2 ablation on those competition-five assays. \textbf{D}, signed source-stack ridge coefficients for ESM2, Desc, Iso, Bio-R, Bio-L, and AbL2: Tm2 has Desc $-2.95$ and Iso $+.92$; AC-SINS has AbL2 $+.53$. These are internal legacy coefficients, not the primary four-expert final weights.}{fig:generalization_main}{}'''
base=r'''\figpanel{main07_baseline_context.pdf}{\textbf{Local benchmarks, external comparison context, and GP kernel variants.} \textbf{A}, source-panel GDPa1 CV comparisons labeled AbDev-X, best in-competition, and KitAb (Amini 2026); stars denote point-estimate rankings, not statistical significance. These archived configuration values are not a newly verified matched comparison, and KitAb provenance remains unresolved. \textbf{B}, source-panel GDPa3 estimates and displayed intervals with archived Jiang/DELPHI values and competition diamonds, over five competition assays plus Tm1 and Tonset. These heterogeneous protocol records do not establish matched-model superiority; the current primary estimates are in Table~\ref{tab:allten}. \textbf{C}, the source-stack/GP-FA/GP-HA/GP-HP CV comparison from the supplied graphic. Its model labels and run provenance differ from the current four-expert ensemble. Interval overlap, historical rank stars, and the archived GP comparison are not used as proof of universal superiority.}{fig:baseline_main}{}'''
new={'main02_performance.pdf':perf,'main03_generalization.pdf':gen,'main07_baseline_context.pdf':base}
for f,line in new.items():
 old=next(z for z in s.splitlines() if z.startswith('\\figpanel{'+f+'}'));s=s.replace(old,line)

# Every graphic occupies the full available text width. Portrait placement avoids
# a height cap silently shrinking dense figures below page width.
old=next(z for z in s.splitlines() if z.startswith('\\newcommand{\\widepanel}'))
s=s.replace(old,r'\newcommand{\widepanel}[3]{\FloatBarrier\begin{figure}[p]\centering\includegraphics[width=\linewidth]{#1}\caption{#2}\label{#3}\end{figure}\FloatBarrier}')
s=s.replace('width=\\linewidth,height=.77\\textheight,keepaspectratio','width=\\linewidth')

anchor='\\subsection{Additional component, representation, architecture, and transfer detail}'
intro=r'''\subsection{Additional encoder, structural, and attribution panels}
The encoder/structural screen, source-removal/data-efficiency panel, and complete
SHAP/structure-map panel are supplementary in this version. They complement the
three exact source PDFs selected for the main manuscript.
'''
s=s.replace(anchor,intro+'\n\n'.join(relocated)+'\n\\FloatBarrier\n\n'+anchor,1)

# Bring narrative references into agreement with the exact selected figures.
s=s.replace('(Figure~\\ref{fig:performance_main}B; Table~\\ref{tab:components})','(Table~\\ref{tab:components}; Supplementary Figure~\\ref{fig:primary}B)')
s=s.replace('The structural screen supports a similarly assay-dependent conclusion (Figure~\\ref{fig:encoder_main}C,F).','The structural screen supports a similarly assay-dependent conclusion (Supplementary Figure~\\ref{fig:encoder_main}C,F).')
s=s.replace('(Figure~\\ref{fig:generalization_main}A). This is a development result for that legacy stack,','(Supplementary Figure~\\ref{fig:representations}B). This is a development result for that legacy stack,')
s=s.replace('Published methods and competition winners provide useful context but require separate interpretation (Figure~\\ref{fig:baseline_main}).','Published methods and competition winners provide useful context but require separate interpretation. Figure~\\ref{fig:baseline_main} preserves the selected archival comparison panel; its configuration-specific values are not substituted for the current primary results.')
s=s.replace('The complete surface-map and SHAP panel is in the main manuscript','The complete surface-map and SHAP panel is in the Supplementary Information')
s=s.replace('Source-based panels and detailed reanalyses have different model scopes.','Source-based panels and detailed reanalyses have different model scopes.')
# Source-panel CV distribution differs from the current primary CV table.
needle='The per-assay results explain why the ensemble helps.'
pos=s.index('\n\n',s.index(needle))
s=s[:pos]+r'''

Figure~\ref{fig:performance_main} preserves the specifically selected archived
baseline/scatter/distribution panel. Its scatter correlations and reference-model
CV values belong to that source experiment. The nine-model comparison in
Table~\ref{tab:components}, rather than the scatter panel, supports the current
$.472$ aggregate result.
'''+s[pos:]

# Replace the old placement table so its main/SI designation remains accurate.
start=s.index('\\subsection{Result provenance and main/supplementary placement}')
s=s[:start]+r'''\subsection{Exact figure sources and manuscript placement}
\begin{longtable}{>{\raggedright\arraybackslash}p{.13\linewidth}>{\raggedright\arraybackslash}p{.38\linewidth}>{\raggedright\arraybackslash}p{.40\linewidth}}
\caption{Main-figure source manifest. The three selected PDFs are exact byte-for-byte copies, embedded at page text width without changing their graphics. Other figure panels are supplementary.}\\
\toprule Figure & Source / role & Scope\\\midrule\endfirsthead
\toprule Figure & Source / role & Scope\\\midrule\endhead
\ref{fig:architecture} & Updated sequence-input PowerPoint & Direct PDF export; blank outside margins cropped only\\
\ref{fig:performance_main} & \path{figures_backup_20261103/panel_performance_comparison.pdf} & Archived baseline/scatter/CV-distribution configuration\\
\ref{fig:generalization_main} & \path{paper10assay/panel_gen_rep.pdf} & Split sensitivity, external context, PLM ablation, source-stack coefficients\\
\ref{fig:baseline_main} & \path{paper10assay/panel_baseline_comparison.pdf} & Archived benchmark/literature/GP comparisons\\
\ref{fig:encoder_main} & Encoder/structural screen & Supplementary six-panel screen\\
\ref{fig:efficiency_main} & Source removal / learning / paired record / CDR IG & Supplementary; model and assay subsets remain labeled\\
\ref{fig:shap_main} & Complete SHAP and structural maps & Supplementary proxy and visualization analyses\\
\ref{fig:primary}--\ref{fig:transfer} & Detailed component, representation, routing, interpretation, and transfer figures & Supplementary verified reanalyses\\
Other supplementary figures & Architecture, constraints, augmentation, uncertainty, tabular learners, descriptor IG, and GP audits & Experiment-specific records\\\bottomrule
\end{longtable}
\end{document}
'''
(HERE/'AbDevX_combined_main_SI.tex').write_text(s,encoding='utf-8')
print('Prepared exact three selected panels; architecture + three main figures; all remaining panels in SI.')
