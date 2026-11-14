from pathlib import Path
import csv,json,numpy as np,traceback
from chimerax.core.commands import run
P=Path(__file__).resolve().parent
OUT=P/'layout_chimerax_revision';D=OUT/'structures';D.mkdir(parents=True,exist_ok=True)
try:
 meta=json.loads((P/'verified_revision/evidence/residue_shap_manifest.json').read_text())
 with open(P/'verified_revision/evidence/primary_residue_shap.csv') as f:rows=list(csv.DictReader(f))
 model=run(session,'open "'+meta['structure'].replace('\\','/')+'"')[0]
 for command in ['hide atoms','cartoon','dssp','set bgColor white','lighting soft','graphics silhouettes true','camera ortho','view','turn y 25','turn x -10','view']:
  run(session,command)
 output=[]
 for assay in dict.fromkeys(r['assay'] for r in rows):
  selected=[r for r in rows if r['assay']==assay];scale=max(float(np.quantile([abs(float(r['shap'])) for r in selected],.95)),1e-12)
  lookup={(r['chain'],r['pdb_residue']):float(r['shap']) for r in selected}
  assert len(model.residues)==len(lookup)
  for residue in model.residues:
   key=(residue.chain_id,str(residue.number)+residue.insertion_code.strip());value=np.clip(lookup[key]/scale,-1,1)
   rgb=[1,1-value,1-value] if value>=0 else [1+value,1+value,1]
   color=np.array([round(255*x) for x in rgb]+[255],dtype=np.uint8)
   residue.ribbon_color=color;residue.atoms.colors=np.tile(color,(len(residue.atoms),1))
  session.main_view.redraw_needed=True
  path=D/f'residue_{assay}.png'
  run(session,f'save "{path.as_posix()}" width 1500 height 1300 supersample 3 transparentBackground true')
  run(session,f'save "{(D/("residue_"+assay+".cxs")).as_posix()}"')
  output.append(dict(assay=assay,scale=scale,path=str(path),residues=len(lookup)))
 (D/'render_manifest.json').write_text(json.dumps(dict(renderer='UCSF ChimeraX 1.11.1',representation='cartoon',source_structure=meta['structure'],source_scores=str(P/'verified_revision/evidence/primary_residue_shap.csv'),scores_unchanged=True,palette='blue-white-red; per-assay symmetric 95th percentile clipping',renders=output),indent=2))
except Exception:
 (OUT/'render_error.txt').write_text(traceback.format_exc())
run(session,'exit')
