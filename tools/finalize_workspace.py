"""Create organized workflow shortcuts and refresh the source manifest."""
from pathlib import Path
import json,ast,hashlib,re,subprocess
R=Path(__file__).resolve().parents[1]
groups={
 'primary':['primary','cross_validation','ablations'],
 'representations':['encoders','uncertainty'],
 'delphi':['delphi_regression','delphi_classification','delphi_statistics'],
 'attribution':['attributions'],
 'gp_kernels':['gp_kernels'],
 'structure_predictors':['structure_comparison'],
}
for name,stages in groups.items():
 folder=R/'workflows'/name;folder.mkdir(exist_ok=True)
 code='from pathlib import Path\nimport subprocess,sys\nR=Path(__file__).resolve().parents[2]\n'
 code+='for stage in '+repr(stages)+':\n subprocess.run([sys.executable,str(R/"reproduce.py"),"run",stage],cwd=R,check=True)\n'
 (folder/'run.py').write_text(code)
 (folder/'README.md').write_text('# '+name.replace('_',' ').title()+'\n\nRuns these manuscript stages in order: '+', '.join('`'+s+'`' for s in stages)+'.\n\nExecute `python workflows/'+name+'/run.py` from the repository.\nSee `python reproduce.py list` and `docs/RESULT_PROVENANCE.md` for source paths and scope.\n')
# Keep generated logs and local fitted models out of commits.
ignore=R/'.gitignore';text=ignore.read_text()
for rule in ['outputs/','*.egg-info/','.pytest_cache/','experiments/**/af3_compile_cache/']:
 if rule not in text.splitlines():text+='\n'+rule+'\n'
ignore.write_text(text)
# Old standalone DELPHI entry points receive the same relative path fix as the
# current matched-data comparison. The primary scientific code is untouched.
for f in [R/'scripts/delphi_comparison.py',*list((R/'experiments/studies').glob('*/methods/delphi_comparison.py'))]:
 s=f.read_text(encoding='utf8');s=s.replace('Path(r"C:\\Users\\hs494\\delphi")','ROOT / "external" / "delphi"');s=re.sub(r'DELPHI_PY\s*=\s*r"C:[^"]+"','DELPHI_PY = sys.executable',s);f.write_text(s,encoding='utf8')
manifest=R/'reference/source_manifest.json';data=json.loads(manifest.read_text());files=[]
candidates=[f for f in R.rglob('*.py') if not set(f.parts)&{'__pycache__','.venv','.git'}]
names=[f.relative_to(R).as_posix() for f in candidates]
check=subprocess.run(['git','check-ignore','--stdin','-z'],input=('\0'.join(names)+'\0').encode(),cwd=R,stdout=subprocess.PIPE)
ignored={n.decode() for n in check.stdout.split(b'\0') if n}
for f in candidates:
 if f.relative_to(R).as_posix() in ignored:continue
 ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f));files.append(dict(path=f.relative_to(R).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
data['files']=files;data['additional_portability_changes']='DELPHI host paths use repository-relative baselines/delphi and sys.executable; organized workflow shortcuts added; original host-specific GPU queue is retired in favor of portable predictor entry points.';manifest.write_text(json.dumps(data,indent=2));print('Source files:',len(files))
