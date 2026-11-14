from pathlib import Path
import re,json,html
import fitz
H=Path(__file__).parent
base=(H/'manuscript_base.tex').read_text(encoding='utf-8');story=json.loads((H/'narrative.json').read_text(encoding='utf-8'))
# Keep selected panels, their placement, and their factual scope.
panels={}
for line in base.splitlines():
 m=re.match(r'\\(?:figpanel|widepanel)\{([^}]+)\}',line)
 if m:panels[m[1]]=line
pre=base[:base.index(r'\begin{abstract}')].replace(r'\documentclass[11pt,a4paper]{article}',r'\documentclass[11pt,a4paper]{article}'+'\n'+r'\pdfminorversion=7')
abstract=r'''\begin{abstract}
\noindent\textbf{Background.} Antibody developability assays measure distinct molecular properties, making it difficult to choose a universally useful representation from a small labeled panel. We developed AbDev-X to combine complementary sequence and physicochemical information through assay-aware experts.

\noindent\textbf{Results.} AbDev-X averages four experts for source selection, descriptor masking, two-stream stacking, and mechanistic feature-group prediction. Development used GDPa1 (246 antibodies) with cluster--isotype cross-validation; GDPa3 supplied an external 80-antibody panel with assay-dependent label availability. Across HIC, AC-SINS, PR-CHO, SMAC, and HAC, the ensemble achieved the highest mean held-out Spearman correlation among nine evaluated strategies: $.472$, versus $.452$ for the strongest individual expert, $.307$ for ESM-2 Ridge, and $.273$ for AbLang2 Ridge. This five-assay grouping is descriptive of the observed results. Matched source ablations revealed assay-specific contributions, and exploratory paired BCa intervals supported improvements over both PLM baselines on HIC, SMAC, and HAC. Applying the unchanged code to ten endpoints gave mean correlation $.249$, with weak or uncertain thermal, SEC, and expression results. Representation screens and physicochemical explanations provided context for the observed complementarity.

\noindent\textbf{Conclusions.} Assay-aware expert construction improves aggregate antibody ranking within the evaluated five-assay comparison. Its value comes from combining complementary predictors through a simple final average. The full endpoint evaluation defines clear limits and motivates prospective, assay- and format-matched validation.
\end{abstract}
\textbf{Keywords:} antibody developability; expert ensemble; protein language models; molecular descriptors; external validation.
\clearpage
'''
text=pre+abstract+'\n\\section{Background}\n'+'\n\n'.join(story['background'])+'\n\n\\section{Results}\n'
for section in story['results']:
 text+='\\subsection{'+section['title']+'}\n'+'\n\n'.join(section['text'])+'\n\n'+panels[section['figure']]+'\n\\FloatBarrier\n'
 if section.get('after'):text+='\n\n'.join(section['after'])+'\n\n'
 if section.get('extra_figure'):text+=panels[section['extra_figure']]+'\n\\FloatBarrier\n'
text+='\\section{Discussion}\n'+'\n\n'.join(story['discussion'])+'\n\n\\section{Conclusions}\n'+story['conclusions']+'\n'
methods=base[base.index(r'\section{Methods}'):base.index(r'\section*{Abbreviations}')]
methods=methods.replace('the legacy seven-source-prediction stacking experiment','the historical source-prediction stacking experiment')
methods=methods.replace('We use these supplied assignments rather than asserting that the evaluated code constructs new CD-HIT clusters at 70\\% identity.','We use these supplied assignments directly.')
methods=methods.replace('It should be described as HC/LC subtype encoding. Although germline-aware modeling is a useful broader design idea, explicit IGHV/IGKV/IGLV gene one-hot vectors are not required by the saved primary expert code.','These encode HC/LC subtype, rather than explicit IGHV/IGKV/IGLV gene identities.')
methods=methods.replace('The descriptor pipeline calculates 165 named features','The descriptor pipeline calculates 165 named features')
methods=methods.replace('These descriptors provide explicit molecular summaries that complement the frozen embeddings.','Hydrophobicity and helical hydrophobic moments follow established sequence scales~\\citep{kyte,eisenberg}. These descriptors provide explicit molecular summaries that complement the frozen embeddings.')
methods=methods.replace('available TAP, MOE-properties, Aggrescan3D, and AntiFold feature files.','available TAP, MOE-properties, Aggrescan3D, and AntiFold feature files~\\citep{raybould2019tap,moe,aggrescan3d,antifold}. These are precomputed benchmark features; no new MOE calculation or software version is inferred.')
methods=methods.replace('A ridge meta-learner combines their OOF predictions:', 'A ridge meta-learner combines their OOF predictions, following the stacking principle~\\citep{stacking}:')
methods=methods.replace('The primary four-expert implementation uses standardized ridge pipelines,','The primary four-expert implementation uses standardized ridge regression~\\citep{ridge} in scikit-learn pipelines~\\citep{sklearn},')
methods=methods.replace('Tree-based learners, including LightGBM and XGBoost, occur in the broader source and learner screens.','Tree-based learners, including LightGBM and XGBoost, occur in the broader source and learner screens~\\citep{lightgbm,xgboost}; the tabular comparison additionally includes TabPFN~\\citep{tabpfn}.')
methods=methods.replace('For each assay, we recompute BCa 95\\% confidence intervals','For each assay, we compute bias-corrected and accelerated (BCa) 95\\% confidence intervals~\\citep{bca}')
methods=methods.replace('The displayed SHAP beeswarms and per-antibody category plots use separate physicochemical LightGBM proxy models.','The displayed SHAP beeswarms and per-antibody category plots use separate physicochemical LightGBM proxy models~\\citep{lightgbm,shap}. SHAP metadata columns, including the observed assay label, are excluded from feature-importance summaries. Beeswarms show the 12 highest mean-absolute-SHAP features per assay, with points colored by the corresponding feature-value percentile.')
methods=methods.replace('The descriptor-only IG panel uses','The descriptor-only IG panel~\\citep{ig} uses')
methods+='\n'+r'''\subsection{Representation and learner screens}
Standalone encoder comparisons include AntiBERTy~\citep{antiberty}, ProtBert from ProtTrans~\citep{prottrans}, ProGen2~\citep{progen2}, and IgBERT~\citep{igbert}, alongside ESM-2 and AbLang2. The input screen compares whole-variable-region, CDR-only, and CrossAbSense variants using their stored experiment tables. These analyses retain their original configurations; they are not substituted for predictions of the primary ensemble. The intact-antibody illustration uses the experimentally determined IgG structure 1IGT~\citep{harris}; the paired-chain Fv example uses an ABodyBuilder3 prediction. The E1 diagram shows five source heads, the A-MGFS diagram shows masking and preprocessing, and the four displayed F-MoE groups are illustrative of the assay-dependent eligible set.
'''
text+=methods
tail=base[base.index(r'\section*{Abbreviations}'):base.index(r'{\small',base.index(r'\section*{Declarations}'))]
tail=tail.replace(r'(\url{https://datapoints.ginkgo.bio/dataset-access})',r'~\citep{dataset}')
tail=tail.replace('The accompanying manuscript package contains the result-table snapshots and figure-generation code used for this revision; it does not constitute a new public release of the full training pipeline.','The accompanying revision directory contains the figure-generation scripts and verification records.')
supp=base[base.index(r'\clearpage\appendix'):base.rindex(r'\end{document}')]
supp=supp.replace('\\textbf{C}, isotonic SAP, rank-regularization, and shared-task experiments;', '\\textbf{C--E}, isotonic SAP, rank-regularization, and shared-task experiments;').replace('\\textbf{D}, arm-average GDPa4 scatter plots:', '\\textbf{F--H}, arm-average GDPa4 scatter plots:')
supp=supp.replace('Figure~\\ref{fig:soft_extensions}C.', 'Figure~\\ref{fig:soft_extensions}C--E.')
supp=supp.replace('\\textbf{A}, source-selection frequencies','\\textbf{A}, source-selection frequencies')
supp=supp.replace('\\textbf{B}, D-Stack internal PLM-versus-descriptor/bio weights.','\\textbf{B}, signed ridge coefficients of the primary D-Stack PLM and descriptor/bio streams.')
supp=supp.replace('\\textbf{B}, PLM versus descriptor/bio stream weights in the saved two-stream analysis.', '\\textbf{B}, signed ridge coefficients of the primary D-Stack PLM and descriptor/bio streams.')
supp=supp.replace('\\textbf{C}, F-MoE mechanism-group weights.', '\\textbf{C}, fixed primary F-MoE mechanism-group weights, shown as stacked proportions.')
supp=supp.replace('Bispecific transfer is evaluated separately in Figure~\\ref{fig:soft_extensions}D.', 'Bispecific transfer is evaluated separately in Figure~\\ref{fig:soft_extensions}F--H.')
supp=supp.replace('These are exploratory configuration screens rather than newly locked external tests.', 'The tabular screen is a recorded configuration table, without recovered run-level uncertainty; it is not a new primary-model test. Regional results are a separate exploratory screen.')
# Captions refer to the actual regenerated panels.
text=text.replace('The PR-CHO ``globally distributed\'\' label is an illustrative interpretation, not evidence establishing a molecular mechanism.', 'PR-CHO shows descriptor context without a validated localized mechanistic hotspot.')
text=text.replace('points represent antibodies and colors denote feature values.', 'points represent antibodies; 12 features per assay are shown, with color denoting feature-value percentiles (blue, low; red, high).')
text=text.replace('(HIC, AC-SINS, PR-CHO, Tm2, Titer); points', '(HIC, Tm2, AC-SINS, PR-CHO, Titer); points')
text=text.replace('\\textbf{C}, the original historical source-stack/GP comparison, with its model label clarified as D-Stack.', '\\textbf{C}, historical source-stack/GP scores transcribed from the supplied vector graphic, retaining its model identity and bar design; these are not new primary-model GP fits.')
text=text.replace('The intact IgG is illustrative (PDB 1IGT);', 'VH/VL sequence excerpts are shown; the intact IgG is illustrative (PDB 1IGT);')
text=text.replace('Four experts perform source selection, assay-specific feature masking, two-stream stacking, and mechanistic group mixing.', 'Four experts perform source selection, keyword masking followed by scaling/optional PCA, two-stream stacking, and mechanistic group mixing. E1 has five operational source heads; the displayed F-MoE groups are illustrative.')
# Build verified Vancouver references in first-citation order, including SI first uses.
records=json.loads((H/'references/doi_metadata.json').read_text(encoding='utf-8'))
journals={'Proceedings of the National Academy of Sciences':'Proc Natl Acad Sci U S A','Journal of Chemical Information and Modeling':'J Chem Inf Model','F1000Research':'F1000Res','IEEE Transactions on Pattern Analysis and Machine Intelligence':'IEEE Trans Pattern Anal Mach Intell','Cell Systems':'Cell Syst','PLOS Computational Biology':'PLoS Comput Biol','Bioinformatics Advances':'Bioinform Adv','Nucleic Acids Research':'Nucleic Acids Res','Neural Networks':'Neural Netw','Journal of the American Statistical Association':'J Am Stat Assoc','Journal of Molecular Biology':'J Mol Biol'}
def esc(s):
 s=html.unescape(re.sub('<[^>]+>','',s));s=' '.join(s.split());s=s.rstrip(' ,')
 for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_')]:s=s.replace(a,b)
 return s
def initials(g):return ''.join(''.join(re.findall('[A-Z]',x)) or x[0].upper() for x in re.split('[ .-]+',g) if x)
def entry(k):
 r=records[k];au=r.get('author',[]);names=[esc(a.get('family',''))+' '+initials(a.get('given','')) for a in au[:6]];authors=', '.join(names)+(', et al' if len(au)>6 else '')+'. '
 title=esc(r['title'][0]);journal=' '.join(r.get('container-title',[''])[0].split());year=r.get('published-print',r['published'])['date-parts'][0][0]
 if k=='antifold':year=2025
 if k=='prottrans':year=2022
 if k=='xgboost':return authors+'XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York: ACM; 2016. p. 785--794. '+r'\url{https://doi.org/'+r['DOI']+'}.'
 out=authors+title+'. '+journals.get(journal,journal)+'. '+str(year)+';'+r.get('volume','')
 pages=r.get('page',r.get('article-number',''))
 if pages:out+=':'+pages.replace('-','--')
 return out+'. '+r'\url{https://doi.org/'+r['DOI']+'}.'
extras={
'shap':r'Lundberg SM, Lee SI. A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems. 2017;30. \url{https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html}.',
'lightgbm':r'Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, et al. LightGBM: a highly efficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems. 2017;30. \url{https://proceedings.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html}.',
'ig':r'Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. In: Proceedings of the 34th International Conference on Machine Learning. Proceedings of Machine Learning Research. 2017;70:3319--3328. \url{https://proceedings.mlr.press/v70/sundararajan17a.html}.',
'antiberty':r'Ruffolo JA, Gray JJ, Sulam J. Deciphering antibody affinity maturation with language models and weakly supervised learning. arXiv. 2021;2112.07782. \url{https://doi.org/10.48550/arXiv.2112.07782}.',
'sklearn':r'Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825--2830. \url{https://jmlr.org/papers/v12/pedregosa11a.html}.',
'delphi':r'Institute for Protein Innovation. DELPHI: software for antibody developability prediction. \url{https://github.com/proteininnovation/delphi}. Accessed 11 Nov 2026.',
'moe':r'Chemical Computing Group. Molecular Operating Environment (MOE). \url{https://www.chemcomp.com/en/Products.htm}. Accessed 11 Nov 2026.',
'dataset':r'Ginkgo Datapoints. Antibody developability datasets. \url{https://datapoints.ginkgo.bio/dataset-access}. Accessed 11 Nov 2026.'}
order=[]
for keys in re.findall(r'\\cite[pt]?\{([^}]+)\}',text+tail+supp):
 for k in keys.split(','):
  if k not in order:order.append(k)
bib='{\\small\n\\begin{thebibliography}{99}\n'
audit=[]
for i,k in enumerate(order,1):
 value=extras[k] if k in extras else entry(k);bib+='\\bibitem{'+k+'} '+value+'\n';audit.append({'number':i,'key':k,'reference':value,'source':'https://doi.org/'+records[k]['DOI'] if k in records else re.search(r'\\url\{([^}]+)',value)[1],'verification':'DOI-deposited bibliographic metadata' if k in records else 'Original proceedings, author manuscript, or official software/data website'})
bib+='\\end{thebibliography}\n}\n'
text+=tail+bib+supp+'\\end{document}\n'
(H/'AbDevX_publication_main_SI.tex').write_text(text,encoding='utf-8')
(H/'references/reference_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
# Crop original native architecture to the panel content, omitting the deck title.
doc=fitz.open(H/'architecture_export.pdf');p=doc[0];rects=[]
for b in p.get_text('dict')['blocks']:
 if b['type']==1:rects.append(fitz.Rect(b['bbox']))
 else:
  for l in b.get('lines',[]):
   for s in l['spans']:
    if s['text'].strip():rects.append(fitz.Rect(s['bbox']))
for d in p.get_drawings():
 if (d.get('fill') and min(d['fill'])<.98) or (d.get('color') and min(d['color'])<.98):rects.append(d['rect'])
box=rects[0]
for r in rects[1:]:box|=r
header=p.search_for('(a) Input')[0];p.set_cropbox(fitz.Rect(box.x0-12,header.y0-12,box.x1+12,box.y1+12)&p.rect)
out=fitz.open();q=out.new_page(width=183/25.4*72,height=p.rect.height*183/25.4*72/p.rect.width);q.show_pdf_page(q.rect,doc,0);out.save(H/'figures/fig01_architecture_user.pdf')
assert 'greedy selection' not in q.get_text().lower()
assert 'six source' not in q.get_text().lower()
print('Built scientific narrative,',len(order),'verified references, and the code-matched Figure 1.')
