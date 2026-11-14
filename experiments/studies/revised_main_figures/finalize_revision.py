from pathlib import Path
import re,json,hashlib,zipfile,shutil
import fitz

HERE=Path(__file__).parent
stem='AbDevX_main_figures'
doc=fitz.open(HERE/(stem+'.pdf'))
tex=(HERE/(stem+'.tex')).read_text(encoding='utf-8')
aux=(HERE/(stem+'.aux')).read_text(encoding='utf-8',errors='replace')
log=(HERE/(stem+'.log')).read_text(encoding='utf-8',errors='replace')
texts=[p.get_text() for p in doc]
refs=re.findall(r'\\ref\{([^}]+)\}',tex)
labels=re.findall(r'\\newlabel\{([^}]+)\}',aux)
figlabels=re.findall(r'\\newlabel\{(fig:[^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)
citations=[c for group in re.findall(r'\\citep?\{([^}]+)\}',tex) for c in group.split(',')]
bib=re.findall(r'\\bibitem\{([^}]+)\}',tex)
sources=re.findall(r'\\(?:figpanel|widepanel)\{([^}]+\.pdf)\}',tex)
sources+=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+\.pdf)\}',tex)
report={
 'combined_pages':len(doc),
 'main_figures':sum(not n.startswith('S') for _,n,_ in figlabels),
 'supplementary_figures':sum(n.startswith('S') for _,n,_ in figlabels),
 'undefined_references':sorted(set(refs)-set(labels)),
 'undefined_citations':sorted(set(citations)-set(bib)),
 'unresolved_pdf_markers':sum(t.count('??') for t in texts),
 'missing_figure_files':[f for f in sources if not (HERE/'figures'/f).exists()],
 'overfull_boxes':len(re.findall(r'Overfull \\[hv]box',log)),
 'ignored_latex_errors':log.count('ignored error'),
 'final_reference_rerun_warning':'Label(s) may have changed' in log,
 'architecture_source_sha256':hashlib.sha256((HERE/'architecture_source.pptx').read_bytes()).hexdigest(),
 'figure_pages':[{'label':a,'figure':n,'page':int(p)} for a,n,p in figlabels]
}
for key in ['undefined_references','undefined_citations','unresolved_pdf_markers','missing_figure_files','overfull_boxes','ignored_latex_errors']:
 assert not report[key],(key,report[key])
assert report['main_figures']==7 and report['supplementary_figures']==12,report
split=next(i for i,t in enumerate(texts) if 'Supplementary results and experiment provenance' in ' '.join(t.split()))
report['main_manuscript_pages']=split;report['supplementary_pages']=len(doc)-split
for name,first,last in [('AbDevX_main_manuscript.pdf',0,split-1),('AbDevX_supplementary.pdf',split,len(doc)-1)]:
 d=fitz.open();d.insert_pdf(doc,from_page=first,to_page=last)
 target=HERE/name
 if target.exists():
  tmp=HERE/(name+'.new.pdf');d.save(tmp);d.close();tmp.replace(target)
 else:d.save(target);d.close()
(HERE/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(HERE/'figure_placement.csv').write_text('figure,label,combined_pdf_page\n'+'\n'.join(f'{n},{a},{p}' for a,n,p in figlabels)+'\n',encoding='utf-8')

notes='''Revision dated 11 November 2026

ACTIVE MANUSCRIPT: AbDevX_main_figures.tex
COMBINED PDF: AbDevX_main_figures.pdf
SEPARATE FILES: AbDevX_main_manuscript.pdf and AbDevX_supplementary.pdf

MAIN FIGURE SET
1. Exact updated architecture supplied in AbDevX_architecture_sequence_input.pptx.
   Exported by PowerPoint; only blank outside margins cropped. Editable source
   and original uncropped PDF export are included. The source file was not edited.
2. Performance: current ensemble vs ESM-2, nine model means, CV distribution.
3. Generalization: four models / three splits, CV vs GDPa3, PLM screen, ridge weights.
4. Encoder and structural-feature screen: six supplied subpanels.
5. LOO ablation, learning curves, archival paired differences, CDR IG enrichment.
6. Full SHAP beeswarms and structure maps across ten assays.
7. Current CV vs local benchmark, held-out literature/configuration context, GP kernels.

SUPPLEMENTARY SET
Twelve figures retain detailed ten-assay results, component matrices, true fusion
weights, source stability, controlled routing, expert-internal weights, transfer,
regional/architecture screens, augmentation/constraints, uncertainty/robustness,
tabular learners, descriptor IG, and the full exploratory GP panels.

CAPTION AND PROVENANCE CORRECTIONS
- Split comparisons across D-Stack / ESM-2 / AbLang2 / SaProt have available results
  for the original competition-five set, not HAC/SMAC high-signal five.
- Tm2 and Titer are not members of the positive-signal five.
- The CV fold mean and pooled-OOF correlation are different summaries; both are labeled.
- Learning curves have reversals, particularly Tm2, and are not called universally monotonic.
- Older paired-difference intervals do not match the current primary run; the panel
  is labeled archival and is not used as proof of current-model significance.
- The cross-assay mean is not given an invalid CI made by averaging per-assay bounds.
- Competition winner mapping follows the published assay identities.
- KitAb/Amini lacks a verified matching numerical source in the available records.
  It is acknowledged in the baseline panel without fabricated values or error bars.
- Jiang points are explicitly archived/digitized estimates. DELPHI points are an
  audited local configuration with a score-direction concern, not a platform-wide claim.
- GP subpanel is the supplied archival CV configuration; its original 'AbDev-X'
  legend was relabeled D-Stack to distinguish it from the primary four-expert model.
- The primary final fusion weights remain equal in the saved results; this evidence
  and the true component scores are retained in the supplementary detail.
- Structure maps and SHAP are distinguished from direct ensemble attributions.

The original 10 November editorial audit is retained as historical background.
Its initial 6-main/9-supplement figure count is superseded by this revision's 7/12.

BUILD
Run pdflatex AbDevX_main_figures.tex twice (or until references stabilize).
Figures are prebuilt, and bibliography entries are inline. No BibTeX run is required.
build_requested_panels.py rebuilds the requested quantitative panels using the
project's results folder; evidence snapshots are included. The unchanged encoder,
SHAP, and GP source assets are included in figures/ or the archived source directory.
No models were retrained and no new inferential results were invented.
'''
(HERE/'REVISION_NOTES.txt').write_text(notes,encoding='utf-8')

# Preserve exact source assets required by the panel script, so the package is portable.
A=HERE/'source_assets';A.mkdir(exist_ok=True)
for f in ['panel_encoder_struct.pdf','panel3_v2.pdf','panel_gp_comparison.pdf']:
 shutil.copy2(HERE.parent/f,A/f)

bundle=HERE.parent/'AbDevX_main_figures_source_package.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for p in HERE.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(HERE)
  if p.suffix in ['.aux','.out','.log'] or p.name.startswith(('main_check_','final_check_','main_layout_contact')):continue
  if p.name in ['package_and_check.py']:continue
  z.write(p,rel)
print(json.dumps({k:v for k,v in report.items() if k!='figure_pages'},indent=2))
print(bundle)
