"""Extract numerical plotting modules from the original analysis scripts."""
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'experiments/studies';W=R/'workflows/figures';W.mkdir(exist_ok=True)
# This portion only plots the four requested bar-style panels.
s=(P/'revise_requested_bars.py').read_text(encoding='utf8').split("text=(OLD/'AbDevX_BMC_clean_layout_main_SI.tex')")[0]
s=s.replace('P=Path(__file__).resolve().parent;',"P=Path(__file__).resolve().parents[2]/'experiments/studies';",1)
s=s.replace("O=P/'final_bmc_revision';F=O/'figures'","O=ROOT/'outputs';F=O/'figures'")
s=s.replace("shutil.copytree(OLD/'figures',F,dirs_exist_ok=True)","F.mkdir(parents=True,exist_ok=True)")
(W/'bar_panels.py').write_text(s,encoding='utf8')
# Preserve the primary-model attribution calculation and established layout.
source=(P/'update_model_structure_attributions.py').read_text(encoding='utf8')
part=source[source.index('plt.rcParams.update'):source.index("s=(OLD/'AbDevX_evidence_led_narrative_main_SI.tex')")]
part=part.replace("V/'figures'/f'residue_{a}.png'","crop/f'residue_{a}.png'").replace("cmap='RdBu_r'","cmap='bwr'")
prefix='''from pathlib import Path
import sys,pickle
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from PIL import Image
R=Path(__file__).resolve().parents[2];P=R/'experiments/studies';V=P/'verified_revision'
F=R/'outputs/figures';E=R/'outputs/attributions';F.mkdir(parents=True,exist_ok=True);E.mkdir(parents=True,exist_ok=True)
crop=R/'outputs/structure_crops';crop.mkdir(parents=True,exist_ok=True)
for a in ['HIC','Tm2','AC-SINS','PR_CHO','Titer','SMAC','HAC','Tm1','Tonset','SEC']:
 path=P/'layout_chimerax_revision/structures'/('residue_'+a+'.png')
 if not path.exists():raise FileNotFoundError('Render the cartoons with ChimeraX first: '+str(path))
 image=Image.open(path);box=image.getchannel('A').getbbox();image.crop(box).save(crop/path.name)
sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import aw_moe as AW
with open(V/'evidence/fitted_primary.pkl','rb') as f:z=pickle.load(f)
cols=z['descriptor_columns']+z['bio_columns'];g=z['g3']
'''
(W/'attribution_panels.py').write_text(prefix+part,encoding='utf8')
# The baseline/GP script's first portion produces only a numerical panel.
s=(P/'finish_benchmark_pymol.py').read_text(encoding='utf8').split('# Reuse the established three-row attribution layout')[0]
s=s.replace('P=Path(__file__).resolve().parent;',"P=Path(__file__).resolve().parents[2]/'experiments/studies';",1)
s=s.replace("F=O/'figures'","F=P.parents[1]/'outputs/figures'")
s=s.replace("shutil.copytree(P/'primary_attribution_revision/figures',F,dirs_exist_ok=True)","F.mkdir(parents=True,exist_ok=True)")
(W/'baseline_panel.py').write_text(s,encoding='utf8')
# This code ends before the historical manuscript-discussion editing section.
s=(P/'si_narrative_revision/revise_si.py').read_text(encoding='utf8').split('DISCUSSIONS={')[0]
s=s.replace('H=Path(__file__).parent;',"H=Path(__file__).resolve().parents[2]/'experiments/studies/si_narrative_revision';",1)
s=s.replace("F=H/'figures'","F=H.parents[2]/'outputs/figures'")
(W/'supplementary_panels.py').write_text(s,encoding='utf8')
print('Prepared numerical plot modules; no manuscript files are required.')
