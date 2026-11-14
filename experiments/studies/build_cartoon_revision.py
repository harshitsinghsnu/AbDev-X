from pathlib import Path
import json,pickle,sys,shutil,zipfile
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
P=Path(__file__).resolve().parent;OLD=P/'benchmark_pymol_revision';O=P/'cartoon_revision';F=O/'figures';E=O/'evidence'
O.mkdir(exist_ok=True);E.mkdir(exist_ok=True);shutil.copytree(OLD/'figures',F,dirs_exist_ok=True)
renderer=(P/'render_pymol_primary.py').read_text(encoding='utf8')
renderer=renderer.replace("O=P/'benchmark_pymol_revision'","O=P/'cartoon_revision'")
renderer=renderer.replace("c.show('surface','fv')","c.dss('fv');c.show('cartoon','fv');c.set('cartoon_fancy_helices',1);c.set('cartoon_flat_sheets',1);c.set('cartoon_smooth_loops',1);c.set('cartoon_sampling',14);c.set('cartoon_loop_radius',0.28);c.set('cartoon_discrete_colors',1)")
renderer=renderer.replace("'surface':'PyMOL molecular solvent-excluded surface'","'representation':'PyMOL cartoon, secondary structure assigned with DSS'")
exec(compile(renderer,str(P/'render_pymol_primary.py'),'exec'),{'__file__':str(P/'render_pymol_primary.py'),'__name__':'__main__'})
sys.path[:0]=[str(P.parents[1]/'scripts'),str(P.parents[1]/'src')]
import aw_moe as AW
with open(P/'verified_revision/evidence/fitted_primary.pkl','rb') as f:z=pickle.load(f)
cols=z['descriptor_columns']+z['bio_columns'];g=z['g3'];V=P/'verified_revision'
source=(P/'update_model_structure_attributions.py').read_text(encoding='utf8');part=source[source.index('plt.rcParams.update'):source.index("s=(OLD/'AbDevX_evidence_led_narrative_main_SI.tex')")]
part=part.replace("V/'figures'/f'residue_{a}.png'","O/'structures'/f'residue_{a}.png'").replace("cmap='RdBu_r'","cmap='bwr'")
exec(part,globals())
s=(OLD/'AbDevX_benchmarks_PyMOL_main_SI.tex').read_text(encoding='utf8')
s=s.replace('structure-mapped residue contributions across ten assays','cartoon-mapped residue contributions across ten assays')
s=s.replace('rendered as PyMOL molecular surfaces','rendered as PyMOL cartoons')
s=s.replace('displayed on the same Fv surface for all ten assays','displayed on the same Fv cartoon for all ten assays')
s=s.replace('These surfaces now visualize model-derived residue contributions','These cartoons visualize model-derived residue contributions')
s=s.replace('Surfaces were ray-traced using PyMOL through pymol2, with a 1.4-\\AA{} solvent probe and its molecular solvent-excluded surface.','Cartoons were ray-traced using PyMOL through pymol2 after secondary-structure assignment with DSS. Beta strands, helices, and loops are shown with residue-level attribution colors.')
(O/'AbDevX_PyMOL_cartoon_main_SI.tex').write_text(s,encoding='utf8')
with zipfile.ZipFile(P/'AbDevX_PyMOL_cartoon_structures.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for f in (O/'structures').iterdir():archive.write(f,f.name)
print('Cartoon figure and manuscript ready',flush=True)
