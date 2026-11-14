"""Read-only live counts for the persistent benchmark queue."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parent
for predictor,log in [('IgFold','igfold_run.csv'),('ESMFold','esmfold_run.csv')]:
 p=P/log
 try:
  d=pd.read_csv(p);print(predictor,'successful predictions',int((d.status=='success').sum()),'/ 326')
 except Exception as e:print(predictor,repr(e))
print('AlphaFold3 top-ranked CIF outputs',len(list((P/'af3_results').glob('*/*_model.cif'))),'/ 326')
for predictor in ['ABodyBuilder3','MOE','IgFold','ESMFold','AlphaFold3']:
 p=P/f'feature_status_{predictor}.csv'
 if p.exists():
  try:
   d=pd.read_csv(p);print(predictor,'features',int((d.status=='success').sum()),'success;',len(d),'processed')
  except Exception as e:print(predictor,repr(e))
if (P/'execution_status.json').exists():
 d=json.loads((P/'execution_status.json').read_text());print('Worker PID',d['pid'],'updated',d['updated'])
 for stage,item in d['stages'].items():print(stage,item['status'])
