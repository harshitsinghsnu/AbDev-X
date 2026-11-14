"""Restore only the declared inputs from an authorized existing artifact tree."""
from pathlib import Path
import argparse,json,hashlib,shutil
R=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--from',dest='source',type=Path,required=True);ap.add_argument('--replace',action='store_true');args=ap.parse_args();source=args.source.resolve();manifest=json.loads((R/'checks/input_manifest.json').read_text());records=manifest['files'];artifacts=R/'checks/artifact_manifest.json';records+=json.loads(artifacts.read_text())['files'] if artifacts.exists() else [];copied=0;missing=[]
for record in records:
 rel=Path(record['path']);src=(source/rel).resolve();dest=(R/rel).resolve()
 if not src.is_relative_to(source) or not dest.is_relative_to(R):raise ValueError('Unexpected path in input manifest')
 if not src.is_file():missing.append(rel.as_posix());continue
 if hashlib.sha256(src.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Source checksum mismatch: '+rel.as_posix())
 if dest.exists():
  if hashlib.sha256(dest.read_bytes()).hexdigest()==record['sha256']:continue
  if not args.replace:raise SystemExit('Destination differs; use --replace explicitly: '+str(dest))
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);copied+=1
print('Restored',copied,'files; unavailable source inputs',len(missing))
if missing:
 out=R/'outputs/validation';out.mkdir(parents=True,exist_ok=True);(out/'restore_missing.json').write_text(json.dumps(missing,indent=2));raise SystemExit(1)
