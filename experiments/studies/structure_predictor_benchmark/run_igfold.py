import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE');os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from pathlib import Path
import sys,time,json,argparse
import numpy as np,pandas as pd,torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from igfold import IgFoldRunner
sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(2)
P=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
d=pd.read_csv(P/'paired_sequences_and_folds.csv');d=d.head(args.limit) if args.limit else d
log=P/('igfold_pilot.csv' if args.limit else 'igfold_run.csv')
prior={(r['split'],r['antibody_name']):r for r in pd.read_csv(log).to_dict('records')} if log.exists() else {}
runner=IgFoldRunner(num_models=4,device='cuda',verbose=True);records=[]
for row in d.itertuples():
 out=P/'structures'/'IgFold'/row.split/(row.antibody_name+'.pdb');out.parent.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 try:
  cached=prior.get((row.split,row.antibody_name))
  if out.exists() and cached and cached['status']=='success':records.append(cached);continue
  if not out.exists():runner.fold(str(out),sequences={'H':row.VH,'L':row.VL},do_refine=False,do_renum=False)
  model=PDBParser(QUIET=True).get_structure('ab',str(out))[0]
  found={c.id:''.join(seq1(r.resname) for r in c if r.id[0]==' ') for c in model}
  assert found.get('H')==row.VH and found.get('L')==row.VL,found
  records.append(dict(predictor='IgFold',split=row.split,antibody_name=row.antibody_name,status='success',seconds=time.perf_counter()-start,peak_gpu_GB=torch.cuda.max_memory_allocated()/2**30,atoms=len(list(model.get_atoms()))))
 except Exception as e:
  records.append(dict(predictor='IgFold',split=row.split,antibody_name=row.antibody_name,status='failed',error=repr(e),seconds=time.perf_counter()-start))
 pd.DataFrame(records).to_csv(log,index=False)
 print(len(records),'/',len(d),row.antibody_name,records[-1]['status'],round(records[-1]['seconds'],2),flush=True)
 if records[-1].get('error') and any(x in records[-1]['error'] for x in ['illegal memory access','device-side assert']):
  print('CUDA context failed; stop and resume in a fresh process instead of counting subsequent inputs as prediction failures.',flush=True);raise SystemExit(1)
