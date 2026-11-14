"""Vendor the exact comparator implementation, preserving its MIT license."""
from pathlib import Path
import shutil
R=Path(__file__).resolve().parents[1];source=R/'external/delphi';destination=R/'baselines/delphi'
if not (source/'LICENSE').exists():raise SystemExit('Obtain the official DELPHI code in external/delphi first.')
if 'MIT License' not in (source/'LICENSE').read_text():raise SystemExit('Unexpected upstream license; review before vendoring.')
shutil.copytree(source,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
changed=[]
for path in [*(R/'scripts').glob('*.py'),*(R/'experiments/studies').rglob('*.py')]:
 old=path.read_text(encoding='utf8');new=old.replace("D=R/'external/delphi'","D=R/'baselines/delphi'").replace('ROOT / "external" / "delphi"','ROOT / "baselines" / "delphi"')
 if new!=old:path.write_text(new,encoding='utf8');changed.append(path.relative_to(R).as_posix())
(destination/'VENDORED_NOTICE.md').write_text('''# DELPHI comparator implementation

This is the exact model/configuration code used for the matched comparison.
Upstream: https://github.com/proteininnovation/delphi .
Copyright: Institute for Protein Innovation. The original MIT LICENSE is retained.
No pretrained weights or proprietary IPI training data are included.
The AbDev-X adapters are in experiments/studies/same_data_delphi/.
''')
print('Vendored comparator; repository-relative path changes:',len(changed))
