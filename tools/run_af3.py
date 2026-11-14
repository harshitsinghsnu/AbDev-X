"""Run the stated AF3 single-sequence condition in an installed Linux AF3 env."""
from pathlib import Path
import argparse,os,subprocess,sys
R=Path(__file__).resolve().parents[1];P=R/'experiments/structures'
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',required=True,type=Path);ap.add_argument('--weights',required=True,type=Path);args=ap.parse_args()
runner=args.source.resolve()/'run_alphafold.py'
if not runner.exists():raise SystemExit('Official AF3 run_alphafold.py not found in --source')
if not (P/'af3_inputs/000.json').exists():raise SystemExit('Run tools/prepare_structure_inputs.py first')
env=dict(os.environ);env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');env.setdefault('XLA_FLAGS','--xla_gpu_enable_triton_gemm=false');env['OMP_NUM_THREADS']='2'
subprocess.run([sys.executable,str(runner),'--input_dir='+str(P/'af3_inputs'),'--model_dir='+str(args.weights.resolve()),'--output_dir='+str(P/'af3_results'),'--run_data_pipeline=false','--flash_attention_implementation=xla','--buckets=256,384','--num_diffusion_samples=5','--num_recycles=10','--jax_compilation_cache_dir='+str(P/'af3_compile_cache')],cwd=R,env=env,check=True)
