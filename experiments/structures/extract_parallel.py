import os
os.environ.setdefault('OPENMM_CPU_THREADS','1')
from pathlib import Path
import argparse,time,json,concurrent.futures
import pandas as pd
from structure_features import prepare,features
P=Path(__file__).resolve().parent
def one(job):
 predictor,row=job;name=row['antibody_name'];split=row['split'];src=P/'structures'/predictor/split/(name+'.pdb');t=time.time()
 if predictor in ['ABodyBuilder3','MOE']:
  src=P.parents[1]/'external/abdev-benchmark/data/structures'/('AntiBodyBuilder3' if predictor=='ABodyBuilder3' else 'MOE_structures')/split/(name+'.pdb')
 out=P/'standardized_v3'/predictor/split/(name+'.pdb')
 try:
  if not src.exists():raise FileNotFoundError(str(src))
  prepare(src,out,row['VH'],row['VL']);vals=features(out,row['VH'],row['VL'])
  return dict(predictor=predictor,antibody_name=name,split=split,status='success',seconds=time.time()-t),dict(antibody_name=name,split=split,**vals)
 except Exception as e:return dict(predictor=predictor,antibody_name=name,split=split,status='failed',seconds=time.time()-t,error=repr(e)),None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('predictor');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();rows=[];status=[];jobs=[(args.predictor,r) for r in pd.read_csv(P/'paired_sequences_and_folds.csv').to_dict('records')]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  for st,vals in pool.map(one,jobs,chunksize=1):
   status.append(st)
   if vals:rows.append(vals)
   pd.DataFrame(status).to_csv(P/f'feature_status_{args.predictor}.csv',index=False);pd.DataFrame(rows).to_csv(P/f'common_features_{args.predictor}.csv',index=False)
   if len(status)%10==0:print(args.predictor,len(status),'/',len(jobs),'success',len(rows),flush=True)
 print('COMPLETE',args.predictor,len(rows),'of',len(jobs),flush=True)
if __name__=='__main__':main()
