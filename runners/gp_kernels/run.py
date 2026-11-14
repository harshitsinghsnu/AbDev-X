from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parents[2]
for stage in ['gp_kernels']:
 subprocess.run([sys.executable,str(R/"reproduce.py"),"run",stage],cwd=R,check=True)
