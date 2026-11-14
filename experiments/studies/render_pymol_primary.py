from pathlib import Path
import json,sys,shutil
import numpy as np,pandas as pd
from PIL import Image
import pymol2

P=Path(__file__).resolve().parent;OLD=P/'primary_attribution_revision';O=P/'benchmark_pymol_revision';F=O/'structures';F.mkdir(parents=True,exist_ok=True)
meta=json.loads((OLD/'evidence/residue_shap_manifest.json').read_text());scores=pd.read_csv(OLD/'evidence/primary_residue_shap.csv',dtype={'pdb_residue':str})
rows=[]
with pymol2.PyMOL() as app:
    c=app.cmd;c.load(meta['structure'],'fv');c.remove('hydro');c.hide('everything');c.show('surface','fv');c.bg_color('white')
    for k,v in {'surface_quality':2,'solvent_radius':1.4,'surface_solvent':0,'orthoscopic':1,'ray_opaque_background':0,'ray_shadows':0,'specular':.18,'ambient':.5,'direct':.55,'antialias':2,'max_threads':4}.items():c.set(k,v)
    c.orient('fv');c.turn('y',25);c.turn('x',-10);c.zoom('fv',buffer=2);view=c.get_view()
    for a,d in scores.groupby('assay',sort=False):
        scale=max(float(np.quantile(abs(d.shap),.95)),1e-12);lookup={(r.chain,str(r.pdb_residue)):float(np.clip(r.shap/scale,-1,1)) for r in d.itertuples()}
        residues={(at.chain,at.resi) for at in c.get_model('fv').atom};assert residues==set(lookup),(a,residues-set(lookup),set(lookup)-residues)
        c.alter('fv','b=lookup[(chain,resi)]',space={'lookup':lookup});c.spectrum('b','blue_white_red','fv',minimum=-1,maximum=1,byres=1);c.set_view(view)
        c.png(str(F/f'residue_{a}.png'),width=1500,height=1300,dpi=350,ray=1,quiet=1)
        c.save(str(F/f'residue_{a}.pdb'),'fv');c.save(str(F/f'residue_{a}.pse'))
        im=Image.open(F/f'residue_{a}.png').convert('RGBA');bb=im.getchannel('A').getbbox();im.crop(bb).save(F/f'residue_{a}.png')
        rows.append({'assay':a,'pdb_residues':len(residues),'display_scale':scale,'png':str(F/f'residue_{a}.png')});print('PyMOL rendered',a,flush=True)
    version=c.get_version()[0]
(F/'render_manifest.json').write_text(json.dumps({'renderer':'PyMOL via pymol2','version':version,'surface':'PyMOL molecular solvent-excluded surface','solvent_radius_A':1.4,'source_structure':meta['structure'],'source_residue_scores':str(OLD/'evidence/primary_residue_shap.csv'),'view':list(view),'palette':'blue_white_red','scores_unchanged':True,'renders':rows},indent=2))
