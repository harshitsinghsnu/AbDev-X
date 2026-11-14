"""Refresh code hashes after an intentional source edit; no model runs."""
from pathlib import Path
import ast,hashlib,json,subprocess
R=Path(__file__).resolve().parents[1];candidates=[p for p in R.rglob('*.py') if not set(p.parts)&{'.git','.venv','__pycache__'}];names=[p.relative_to(R).as_posix() for p in candidates]
result=subprocess.run(['git','check-ignore','--stdin','-z'],cwd=R,input=('\0'.join(names)+'\0').encode(),stdout=subprocess.PIPE)
ignored={name.decode() for name in result.stdout.split(b'\0') if name};entries=[]
for path in candidates:
 name=path.relative_to(R).as_posix()
 if name in ignored:continue
 ast.parse(path.read_text(encoding='utf-8-sig'),filename=name);entries.append(dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(R/'checks/source_manifest.json').write_text(json.dumps(dict(files=entries,scope='Executable experiment, model and evaluation code. Manuscript preparation files are excluded.'),indent=2))
print('Code files checked:',len(entries))
