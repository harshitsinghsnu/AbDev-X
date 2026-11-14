from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1];data=json.loads((R/'checks/input_manifest.json').read_text());missing=[];changed=[]
for record in data['files']:
 path=R/record['path']
 if not path.is_file():missing.append(record['path'])
 elif hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:changed.append(record['path'])
report=dict(expected=len(data['files']),missing=missing,changed=changed);out=R/'outputs/validation';out.mkdir(parents=True,exist_ok=True);(out/'input_check.json').write_text(json.dumps(report,indent=2))
print('Input files:',len(data['files']),'Missing:',len(missing),'Changed:',len(changed))
if missing or changed:raise SystemExit(1)
