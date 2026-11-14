from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parents[2]
for stage in ['delphi_regression', 'delphi_classification', 'delphi_statistics']:
 subprocess.run([sys.executable,str(R/"reproduce.py"),"run",stage],cwd=R,check=True)
