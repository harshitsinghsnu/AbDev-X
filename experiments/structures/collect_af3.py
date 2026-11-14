"""Select AF3's top-ranked sample and preserve exact paired-chain identities."""
from pathlib import Path
import json,time
import pandas as pd
from Bio.PDB import MMCIFParser,PDBIO,PDBParser
from Bio.SeqUtils import seq1
P=Path(__file__).resolve().parent
def collect():
 rows=[]
 for i,row in pd.read_csv(P/'paired_sequences_and_folds.csv').iterrows():
  job=json.loads((P/'af3_inputs'/f'{i:03}.json').read_text())['name'].lower()
  folder=P/'af3_results'/job;src=folder/(job+'_model.cif');out=P/'structures/AlphaFold3'/row['split']/(row.antibody_name+'.pdb')
  if not src.exists():continue
  try:
   structure=MMCIFParser(QUIET=True).get_structure(job,str(src));model=structure[0];found={c.id:''.join(seq1(r.resname) for r in c if r.id[0]==' ') for c in model}
   ids=[]
   for key in ['VH','VL']:
    candidates=[k for k,v in found.items() if v==row[key] and k not in ids]
    if len(candidates)!=1:raise ValueError(str(found))
    ids.append(candidates[0])
   for c in model:c.id='X'+c.id
   for old,new in zip(ids,['H','L']):model['X'+old].id=new
   out.parent.mkdir(parents=True,exist_ok=True);writer=PDBIO();writer.set_structure(structure);writer.save(str(out))
   summary=json.loads((folder/(job+'_summary_confidences.json')).read_text())
   rows.append(dict(predictor='AlphaFold3',split=row['split'],antibody_name=row.antibody_name,status='success',job=job,ranking_score=summary.get('ranking_score'),iptm=summary.get('iptm'),ptm=summary.get('ptm'),selection='official top ranking score among 5 samples; no assay labels used'))
  except Exception as e:rows.append(dict(predictor='AlphaFold3',split=row['split'],antibody_name=row.antibody_name,status='failed',job=job,error=repr(e)))
 pd.DataFrame(rows).to_csv(P/'alphafold3_run.csv',index=False)
 print('AF3 collected',sum(r['status']=='success' for r in rows),'/',326,flush=True)
 return rows
if __name__=='__main__':collect()
