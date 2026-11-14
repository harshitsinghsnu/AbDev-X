"""Audit the Git-visible release without staging, committing, or pushing files."""
from pathlib import Path
import subprocess,json,re
R=Path(__file__).resolve().parents[1]
raw=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=R)
files=sorted(set(n.decode('utf8') for n in raw.split(b'\0') if n and (R/n.decode('utf8')).is_file()));oversized=[];credential_paths=[];unexpected=[]
pattern=re.compile(r'(?:hf_[A-Za-z0-9]{25,}|ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
for name in files:
 path=R/name
 if path.stat().st_size>=95*1024**2:oversized.append(name)
 if name.startswith(('data/raw/','data/processed/','external/','outputs/')) or path.suffix in ['.pkl','.npz','.pt','.pth','.safetensors','.bin']:unexpected.append(name)
 if path.suffix in ['.py','.md','.tex','.bib','.txt','.yaml','.yml','.json','.toml']:
  if pattern.search(path.read_text(encoding='utf8',errors='replace')):credential_paths.append(name)
out=R/'outputs/validation';out.mkdir(parents=True,exist_ok=True)
report=dict(git_visible_files=len(files),git_visible_MB=round(sum((R/n).stat().st_size for n in files)/1024**2,2),oversized_files=oversized,possible_credential_files=credential_paths,unexpected_private_input_files=unexpected,remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=R,text=True).strip(),pushed=False)
(out/'release_audit.json').write_text(json.dumps(report,indent=2))
if oversized or credential_paths or unexpected:raise SystemExit('Release audit failed; see outputs/validation/release_audit.json')
print(json.dumps(report,indent=2))
