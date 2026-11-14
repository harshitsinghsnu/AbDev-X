"""Run the organized, manuscript-specific workflows from any working directory."""
from pathlib import Path
import argparse,os,subprocess,sys,json,time
ROOT=Path(__file__).resolve().parent;P='experiments/studies/'
STAGES={
 'primary':[P+'verified_revision/verify_primary.py'],
 'cross_validation':['scripts/compute_awmoe_cv.py'],
 'ablations':[P+'verified_revision/primary_ablations.py'],
 'uncertainty':[P+'assay_revision/compute_uncertainty.py',P+'assay_revision/extend_uncertainty.py'],
 'attributions':[P+'verified_revision/primary_residue_shap.py'],
 'encoders':['scripts/exp_crossab.py','scripts/exp_encoder_screen.py','scripts/plm_ablation.py'],
 'delphi_regression':[P+'same_data_delphi/train_regression.py'],
 'delphi_classification':[P+'same_data_delphi/train_classification.py'],
 'delphi_statistics':[P+'same_data_delphi/analyze.py'],
 'gp_kernels':[P+'run_matched_gp.py'],
 'structure_comparison':[P+'structure_predictor_benchmark/evaluate_common.py'],
 'figures':['workflows/figures/regenerate.py'],
 'verify':['tools/validate_results.py'],
}
ALL=['primary','cross_validation','ablations','encoders','uncertainty','attributions','delphi_regression','delphi_classification','delphi_statistics','gp_kernels','structure_comparison','figures','verify']
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['list','run']);ap.add_argument('stage',nargs='?',choices=list(STAGES)+['all']);ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
 if args.action=='list':
  for name,files in STAGES.items():print(name+': '+', '.join(files))
  return
 if not args.stage:ap.error('run requires a stage')
 stages=ALL if args.stage=='all' else [args.stage];env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/P/'verified_revision'),env.get('PYTHONPATH','')]);env['PYTHONUTF8']='1';env['PYTHONUNBUFFERED']='1';env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
 logdir=ROOT/'outputs/run_logs';logdir.mkdir(parents=True,exist_ok=True);manifest=[]
 for stage in stages:
  for script in STAGES[stage]:
   command=[sys.executable,str(ROOT/script)];print('RUN',stage,script,flush=True)
   if args.dry_run:continue
   start=time.time();log=logdir/(stage+'_'+Path(script).stem+'.log')
   with log.open('w',encoding='utf8') as output:ret=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
   item=dict(stage=stage,script=script,exit_code=ret.returncode,seconds=time.time()-start,log=log.relative_to(ROOT).as_posix());manifest.append(item);(logdir/'latest_run.json').write_text(json.dumps(manifest,indent=2))
   if ret.returncode:print('FAILED. See',log);raise SystemExit(ret.returncode)
   print('DONE',stage,round(item['seconds'],1),'seconds',flush=True)
if __name__=='__main__':main()
