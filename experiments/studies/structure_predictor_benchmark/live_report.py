"""Keep a local human-readable report current while the authorized queue runs."""
import csv,html,json,time,os
from pathlib import Path
P=Path(__file__).resolve().parent
def readcsv(path):
 try:
  with open(path,newline='',encoding='utf8') as f:return list(csv.DictReader(f))
 except (OSError,csv.Error):return []
def esc(value):return html.escape(str(value))
def number(value):
 try:return f'{float(value):.3f}'
 except (ValueError,TypeError):return 'unavailable'
def build():
 try:state=json.loads((P/'execution_status.json').read_text())
 except (OSError,ValueError):state={}
 terminal=state.get('stages',{}).get('queue',{}).get('status')
 content=['<h1>AbDev-X structure-predictor benchmark</h1>',f'<p>Updated {esc(time.strftime("%Y-%m-%d %H:%M:%S"))} local time. Reload to see current progress.</p>',
 '<p><strong>Scope:</strong> matched AbDev-X/common-structure20 variant. This replaces the original cached bio57 block with 20 identically extracted open structural features. The other four blocks and original four-expert training procedure are held fixed. Original primary-model results are unchanged.</p>',
 '<h2>Execution progress</h2><table><tr><th>Predictor</th><th>Structures</th><th>Successful feature rows</th></tr>']
 for predictor,log in [('ABodyBuilder3',None),('IgFold','igfold_run.csv'),('ESMFold','esmfold_run.csv'),('AlphaFold3',None),('MOE',None)]:
  rows=readcsv(P/f'feature_status_{predictor}.csv');features=sum(r.get('status')=='success' for r in rows)
  if log:structures=str(sum(r.get('status')=='success' for r in readcsv(P/log)))+' / 326'
  elif predictor=='AlphaFold3':structures=str(len(list((P/'af3_results').glob('*/*_model.cif'))))+' / 326'
  else:structures='Supplied cache'
  content.append(f'<tr><td>{predictor}</td><td>{structures}</td><td>{features} / 326 ({len(rows)} processed)</td></tr>')
 content.append('</table>')
 for stage,st in state.get('stages',{}).items():content.append(f'<p>{esc(stage)}: <strong>{esc(st.get("status"))}</strong></p>')
 dirs=sorted(P.glob('comparison_*'),key=lambda p:p.stat().st_mtime,reverse=True)
 finished=[p for p in dirs if (p/'RESULTS.md').exists() and (p/'manifest.json').exists()]
 if finished:
  out=finished[0];manifest=json.loads((out/'manifest.json').read_text());scores=readcsv(out/'scores.csv')
  content += ['<h2>Completed matched evaluation</h2>',f'<p>Common complete cases before missing-label exclusion: {manifest["train_n"]} GDPa1 and {manifest["test_n"]} GDPa3 antibodies. Comparison is retrospective; GDPa3 has previously been inspected in this project.</p>',
   f'<p><a href="{out.name}/RESULTS.md">Detailed report</a> | <a href="{out.name}/scores.csv">Scores CSV</a> | <a href="{out.name}/paired_differences.csv">Paired differences CSV</a> | <a href="{out.name}/common_structure_comparison.pdf">Vector PDF figure</a></p>',
   f'<img alt="Matched structural-source comparison with uncertainty" src="{out.name}/common_structure_comparison.png">',
   '<table><tr><th>Predictor</th><th>Assay</th><th>CV mean ± fold SD</th><th>GDPa3 ρ [BCa 95% CI]</th><th>Held-out n</th></tr>']
  for r in scores:content.append(f'<tr><td>{esc(r["predictor"])}</td><td>{esc(r["assay"])}</td><td>{number(r["cv_mean"])} ± {number(r["cv_sd"])}</td><td>{number(r["heldout_rho"])} [{number(r["ci_lo"])}, {number(r["ci_hi"])}]</td><td>{esc(r["n_test"])}</td></tr>')
  content.append('</table><p>Paired intervals, including Bonferroni sensitivity across predictor–assay comparisons, are in the linked difference table. Higher point estimates alone do not establish superiority.</p>')
 else:content.append('<h2>Evaluation pending</h2><p>There are no completed matched downstream results yet. Structure generation counts are not predictive-performance results.</p>')
 content.append('<h2>Conditions and limits</h2><p>AF3: single-sequence paired-chain condition, empty MSAs/no templates, seed 42, 10 recycles, five diffusion samples, official top-ranked structure. ESMFold: paired-chain encoding with 25-glycine linker removed from outputs, four recycles. IgFold: four-model ensemble without optional refinement. All sources use exact chain checks and the same missing-heavy-atom completion and feature definitions. Two supplied ABodyBuilder3 training structures are absent. MOE is an additional cached source.</p><p>The common structural panel does not reproduce the original TAP/MOE/Aggrescan3D/AntiFold bio57 pipeline. No claim that a predictor is optimal for the original primary model is justified by this experiment alone. Experimental reference structures are unavailable for an accuracy benchmark. Confidence intervals concern assay-level correlations and do not include training-set uncertainty.</p>')
 refresh='' if terminal in ['finished','incomplete'] else '<meta http-equiv="refresh" content="60">'
 document='<!doctype html><html lang="en"><meta charset="utf-8">'+refresh+'<title>AbDev-X benchmark results</title><style>body{font:16px/1.55 Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#183044}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{text-align:left;padding:9px;border-bottom:1px solid #cdd8df}th{background:#edf3f7}img{max-width:100%;height:auto}a{color:#12639b}</style><body>'+''.join(content)+'</body></html>'
 temp=P/'results_live.tmp';temp.write_text(document,encoding='utf8');os.replace(temp,P/'results_live.html')
 return terminal
if __name__=='__main__':
 while True:
  try:
   if build() in ['finished','incomplete']:break
  except Exception as e:
   with open(P/'live_report_errors.log','a',encoding='utf8') as f:f.write(repr(e)+'\n')
  time.sleep(60)
