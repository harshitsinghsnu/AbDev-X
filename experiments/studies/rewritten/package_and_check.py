"""Validate the compiled artifact and package its editable sources and evidence."""
from pathlib import Path
import ast, json, re, shutil, zipfile
import fitz

HERE=Path(__file__).parent
ROOT=HERE.parents[2]
R=ROOT/'results'
E=HERE/'evidence';E.mkdir(exist_ok=True)
tree=ast.parse((HERE/'build_figures.py').read_text(encoding='utf-8'))
files=set()
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='rd' and node.args and isinstance(node.args[0],ast.Constant):
        files.add(node.args[0].value)
files.add('awmoe_explainability.json')
files.update('explain/shap_'+a+'.csv' for a in ['HIC','AC-SINS_pH7.4','PR_CHO'])
for f in files:
    dst=E/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/f,dst)
M=HERE/'methods';M.mkdir(exist_ok=True)
for f in ['aw_moe.py','final_evaluation.py','compute_awmoe_cv.py','delphi_comparison.py']:
    shutil.copy2(ROOT/'scripts'/f,M/f)
shutil.copy2(ROOT/'configs/default.yaml',M/'default.yaml')

pdf=fitz.open(HERE/'AbDevX_rewritten.pdf')
tex=(HERE/'AbDevX_rewritten.tex').read_text(encoding='utf-8')
log=(HERE/'AbDevX_rewritten.log').read_text(encoding='utf-8',errors='replace')
texts=[p.get_text() for p in pdf]
alltext='\n'.join(texts)
labels=re.findall(r'\\label\{([^}]+)\}',tex)
labels+=re.findall(r'\\newlabel\{([^}]+)\}',(HERE/'AbDevX_rewritten.aux').read_text(encoding='utf-8',errors='replace'))
refs=re.findall(r'\\ref\{([^}]+)\}',tex)
citations=[z for group in re.findall(r'\\citep?\{([^}]+)\}',tex) for z in group.split(',')]
bib=re.findall(r'\\bibitem\{([^}]+)\}',tex)
figs=re.findall(r'\{(fig(?:\d|S)\w*\.pdf)\}',tex)
bad_bounds=[]
for i,p in enumerate(pdf):
    if p.rotation:continue
    for b in p.get_text('blocks'):
        if b[0]<-1 or b[1]<-1 or b[2]>p.rect.width+1 or b[3]>p.rect.height+1:bad_bounds.append(i+1)
report={
    'pdf_pages':len(pdf),'approx_text_words_including_plots':len(alltext.split()),
    'main_figure_count':6,'supplementary_figure_count':9,
    'figure_assets_present':all((HERE/'figures'/f).exists() for f in figs),
    'undefined_references':sorted(set(refs)-set(labels)),
    'undefined_citations':sorted(set(citations)-set(bib)),
    'unresolved_pdf_markers':alltext.count('??'),
    'overfull_boxes':len(re.findall(r'Overfull \\[hv]box',log)),
    'latex_ignored_errors':len(re.findall('ignored error',log)),
    'out_of_page_text_blocks':sorted(set(bad_bounds)),
    'near_blank_pages':[i+1 for i,t in enumerate(texts) if len(t.strip())<100],
    'evidence_files':len(list(E.rglob('*.*'))),
    'scope':'Retrospective manuscript rewrite; no model training or newly calculated inferential results.'
}
assert not report['undefined_references'],report
assert not report['undefined_citations'],report
assert report['figure_assets_present'],report
assert not report['unresolved_pdf_markers'],report
assert not report['overfull_boxes'],report
assert not report['latex_ignored_errors'],report
assert not report['out_of_page_text_blocks'],report
(HERE/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

# Convenience extracts retain the compiled document's original page numbering.
supp_start=next(i for i,t in enumerate(texts) if 'Supplementary results and experiment provenance' in ' '.join(t.split()))
main=fitz.open();main.insert_pdf(pdf,to_page=supp_start-1);main.save(HERE/'AbDevX_main_manuscript.pdf')
supp=fitz.open();supp.insert_pdf(pdf,from_page=supp_start);supp.save(HERE/'AbDevX_supplementary.pdf')
bundle=HERE.parent/'AbDevX_rewritten_source_package.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for p in HERE.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(HERE)
        if p.suffix in ['.aux','.out','.log'] or p.name.startswith(('check_page_','review_','layout_contact')):continue
        z.write(p,rel)
print(json.dumps(report,indent=2))
print('Source bundle:',bundle)
