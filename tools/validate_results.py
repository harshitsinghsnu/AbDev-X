"""Recompute reported metrics from retained individual predictions and folds."""
from pathlib import Path
import json,sys,hashlib,ast
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'experiments/studies';checks=[]
def close(name,a,b,tol=1e-9):
 if not np.isfinite(a) or not np.isfinite(b) or abs(a-b)>tol:raise AssertionError((name,a,b))
 checks.append(dict(check=name,recomputed=float(a),reference=float(b),absolute_error=abs(float(a-b))))
def main():
 primary=pd.read_csv(P/'verified_revision/evidence/primary_predictions.csv');gold=pd.read_csv(ROOT/'reference/primary_metrics.csv').set_index('assay')
 for assay,d in primary.groupby('assay'):
  ok=np.isfinite(d.observed)&np.isfinite(d.predicted);close('primary '+assay,spearmanr(d.loc[ok,'observed'],d.loc[ok,'predicted']).statistic,gold.loc[assay,'rho'])
 current=pd.read_csv(P/'verified_revision/evidence/primary_reproduction.csv').set_index('assay')
 for assay in gold.index:
  for column in ['ci_lo','ci_hi']:close('primary '+assay+' '+column,current.loc[assay,column],gold.loc[assay,column])
 d=pd.read_csv(P/'same_data_delphi/paired_external_predictions.csv');gold=pd.read_csv(ROOT/'reference/same_data_delphi_metrics.csv')
 for (task,assay),q in d[d.seed==42].groupby(['task','assay']):
  metric=roc_auc_score if task=='classification' else lambda y,p:spearmanr(y,p).statistic
  a=metric(q.observed,q.predicted_abdevx);b=metric(q.observed,q.predicted_delphi)
  for quantity,value in [('AbDev-X',a),('DELPHI',b),('Difference',a-b)]:close('matched DELPHI '+task+' '+assay+' '+quantity,value,gold[(gold.task==task)&(gold.assay==assay)&(gold.quantity==quantity)].estimate.iloc[0])
 folder=P/'structure_predictor_benchmark/comparison_ABodyBuilder3_IgFold_ESMFold_AlphaFold3';pred=pd.read_csv(folder/'heldout_predictions.csv');gold=pd.read_csv(ROOT/'reference/structure_metrics.csv').set_index(['predictor','assay'])
 for key,q in pred.groupby(['predictor','assay']):
  ok=np.isfinite(q.observed)&np.isfinite(q.predicted);close('structure '+' '.join(key),spearmanr(q.loc[ok,'observed'],q.loc[ok,'predicted']).statistic,gold.loc[key,'heldout_rho'])
 folds=pd.read_csv(folder/'cv_fold_scores.csv')
 for key,q in folds.groupby(['predictor','assay']):
  close('structure CV mean '+' '.join(key),q.rho.mean(),gold.loc[key,'cv_mean']);close('structure CV SD '+' '.join(key),q.rho.std(ddof=1),gold.loc[key,'cv_sd'])
 # Affine input attribution and conditional residue-pathway completeness.
 refs=P/'primary_attribution_revision/evidence/prediction_attribution_completeness.csv'
 if refs.exists():
  q=pd.read_csv(refs)
  for row in q.itertuples():close('residue completeness '+row.assay,row.residue_sum,row.descriptor_prediction_change,1e-5)
 source=json.loads((ROOT/'reference/source_manifest.json').read_text())
 for item in source['files']:
  f=ROOT/item['path'];ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f))
  if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:raise AssertionError('Source changed since manifest: '+item['path'])
 out=ROOT/'outputs/validation';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(checks).to_csv(out/'recomputed_metrics.csv',index=False);(out/'summary.json').write_text(json.dumps(dict(status='passed',numeric_checks=len(checks),python_sources_parsed=len(source['files']),limitations='Historical summary-only results are not promoted to regenerated training experiments; see docs/RESULT_PROVENANCE.md.'),indent=2));print('PASSED',len(checks),'numeric checks and',len(source['files']),'source syntax checks')
if __name__=='__main__':main()
