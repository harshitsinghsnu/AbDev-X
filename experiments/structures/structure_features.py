"""Fixed common structural panel for a matched predictor comparison.
This replaces the cached bio block in an explicitly named AbDev-X/common-structure
variant. It is not the original 57-feature TAP/MOE/A3D/AntiFold block.
"""
import os
os.environ.setdefault('OPENMM_CPU_THREADS','1')
from pathlib import Path
import argparse,json,time,io
import numpy as np,pandas as pd,freesasa
freesasa.setVerbosity(freesasa.nowarnings)
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser,PDBIO,Structure,Model,Chain
import copy
from Bio.SeqUtils import seq1
from pdbfixer import PDBFixer
from openmm import Platform
from openmm.app import PDBFile
P=Path(__file__).resolve().parent
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-.8,'T':-.7,'W':-.9,'Y':-1.3,'V':4.2}
CHARGE={'R':1.,'K':1.,'H':.1,'D':-1.,'E':-1.}
def prepare(path,out,h,l):
 if out.exists():return
 out.parent.mkdir(parents=True,exist_ok=True)
 raw=PDBParser(QUIET=True).get_structure('raw',str(path))[0]
 # Map chain identifiers by exact sequence, rather than assuming the source IDs.
 normalized=Structure.Structure('fv');nm=Model.Model(0);normalized.add(nm)
 used=set()
 for cid,sequence in [('H',h),('L',l)]:
  matches=[c for c in raw if c.id not in used and ''.join(seq1(r.resname) for r in c if r.id[0]==' ')==sequence]
  if len(matches)!=1:raise ValueError('No unique exact chain match for '+cid)
  used.add(matches[0].id);chain=Chain.Chain(cid);nm.add(chain)
  for i,r in enumerate([r for r in matches[0] if r.id[0]==' '],1):
   rr=copy.deepcopy(r);rr.detach_parent();rr.id=(' ',i,' ')
   for atom in list(rr):
    if atom.element=='H' or atom.name=='OXT':rr.detach_child(atom.id)
   chain.add(rr)
 buffer=io.StringIO();writer=PDBIO();writer.set_structure(normalized);writer.save(buffer);buffer.seek(0)
 fixer=PDBFixer(pdbfile=buffer,platform=Platform.getPlatformByName('CPU'))
 fixer.findMissingResidues();fixer.missingResidues={};fixer.findNonstandardResidues()
 if fixer.nonstandardResidues:raise ValueError('Nonstandard residues require explicit mapping')
 fixer.findMissingAtoms();fixer.missingTerminals={}
 if fixer.missingAtoms:fixer.addMissingAtoms(seed=42)
 with open(out,'w') as f:PDBFile.writeFile(fixer.topology,fixer.positions,f,keepIds=True)
def features(path,h,l):
 model=PDBParser(QUIET=True).get_structure('m',str(path))[0]
 sequences={c.id:''.join(seq1(r.resname) for r in c if r.id[0]==' ') for c in model}
 assert sequences.get('H')==h and sequences.get('L')==l,sequences
 result=freesasa.calc(freesasa.Structure(str(path))).residueAreas();ca=[];sa=[];hyd=[];q=[];chains=[];ar=[];residues=[]
 for c in model:
  for r in c:
   if r.id[0]!=' ' or 'CA' not in r:continue
   aa=seq1(r.resname);key=str(r.id[1])+r.id[2].strip();a=result[c.id][key].total
   ca.append(r['CA'].coord);sa.append(a);hyd.append(KD[aa]);q.append(CHARGE.get(aa,0));chains.append(c.id);ar.append(aa in 'FWY');residues.append(r)
 ca=np.asarray(ca);sa=np.asarray(sa);hyd=np.asarray(hyd);q=np.asarray(q);chains=np.asarray(chains);ar=np.asarray(ar);dist=cdist(ca,ca);neighborhood=dist<=10
 hp=neighborhood@(sa*np.maximum(hyd,0));cp=neighborhood@(sa*q)
 heavy=ca[chains=='H'];light=ca[chains=='L'];inter=cdist(heavy,light)
 out={'surface_total_sasa':sa.sum(),'hydrophobic_exposed_sasa':sa[hyd>0].sum(),'aromatic_exposed_sasa':sa[ar].sum(),'hydrophobic_sasa_fraction':sa[hyd>0].sum()/sa.sum(),'hydrophobic_exposure_signed':sa@hyd,'hydrophobic_exposure_positive':sa@np.maximum(hyd,0),'charge_exposure_signed':sa@q,'charge_exposure_positive':sa@np.maximum(q,0),'charge_exposure_negative':sa@np.minimum(q,0),'hydro_patch_max':hp.max(),'hydro_patch_top5_mean':np.sort(hp)[-5:].mean(),'charge_patch_max':cp.max(),'charge_patch_min':cp.min(),'charge_patch_abs_max':abs(cp).max(),'surface_gyration_CA':np.sqrt(np.mean(np.sum((ca-ca.mean(0))**2,axis=1))),'surface_interface_contacts8':float(np.sum(inter<8)),'surface_interface_contacts10':float(np.sum(inter<10)),'surface_interface_min_distance':inter.min(),'surface_chain_centroid_distance':float(np.linalg.norm(heavy.mean(0)-light.mean(0)))}
 # SASA lost on chain association, calculated with the same atom definitions.
 isolated=0.
 for chain in ['H','L']:
  s=freesasa.Structure()
  for r in residues:
   if r.parent.id!=chain:continue
   for a in r:
    if a.element=='H':continue
    s.addAtom(a.name,r.resname,str(r.id[1])+r.id[2].strip(),chain,*map(float,a.coord))
  isolated+=freesasa.calc(s).totalArea()
 out['surface_interface_buried_sasa']=max(0.,isolated-sa.sum())
 assert np.isfinite(list(out.values())).all()
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('predictor',choices=['ABodyBuilder3','MOE','IgFold','ESMFold','AlphaFold3']);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args();manifest=pd.read_csv(P/'paired_sequences_and_folds.csv');rows=[];status=[]
 if args.limit:manifest=manifest.head(args.limit)
 for row in manifest.itertuples():
  source=P/'structures'/args.predictor/row.split/(row.antibody_name+'.pdb')
  if args.predictor in ['ABodyBuilder3','MOE']:
   folder='AntiBodyBuilder3' if args.predictor=='ABodyBuilder3' else 'MOE_structures'
   source=P.parents[1]/'external/abdev-benchmark/data/structures'/folder/row.split/(row.antibody_name+'.pdb')
  output=P/'standardized_v3'/args.predictor/row.split/(row.antibody_name+'.pdb');start=time.time()
  try:
   if not source.exists():raise FileNotFoundError(str(source))
   prepare(source,output,row.VH,row.VL);f=features(output,row.VH,row.VL);rows.append(dict(antibody_name=row.antibody_name,split=row.split,**f));status.append(dict(antibody_name=row.antibody_name,split=row.split,status='success',seconds=time.time()-start))
  except Exception as e:status.append(dict(antibody_name=row.antibody_name,split=row.split,status='failed',error=repr(e),seconds=time.time()-start))
  pd.DataFrame(rows).to_csv(P/f'common_features_{args.predictor}.csv',index=False);pd.DataFrame(status).to_csv(P/f'feature_status_{args.predictor}.csv',index=False)
  print(args.predictor,len(status),'/',len(manifest),status[-1]['status'],round(time.time()-start,2),flush=True)
if __name__=='__main__':main()
