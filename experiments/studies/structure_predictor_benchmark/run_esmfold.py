"""Pinned ESMFold v1, paired-chain encoding matching FAIR's encode_sequence.
VH + 25 glycines + VL, light-chain residue index offset512; linker residues
are excluded from PDB and every structural feature calculation.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import argparse,sys,time,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import EsmForProteinFolding
from transformers.models.esm.openfold_utils import residue_constants
from transformers.models.esm.openfold_utils.protein import Protein,to_pdb
from transformers.models.esm.modeling_esmfold import atom14_to_atom37
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(2)
P=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
data=pd.read_csv(P/'paired_sequences_and_folds.csv');data=data.head(args.limit) if args.limit else data
t=time.time();model=EsmForProteinFolding.from_pretrained(str(P/'weights/ESMFold'),low_cpu_mem_usage=True,device_map={'':'cuda'},torch_dtype=torch.float16)
# Keep the large frozen sequence LM in fp16 and all folding components in fp32.
for name,child in model.named_children():
 if name!='esm':child.float()
for name,param in model.named_parameters():
 if not name.startswith('esm.'):param.data=param.data.float()
model.trunk.set_chunk_size(32);model.eval();print('ESMFold loaded seconds',round(time.time()-t,1),flush=True)
log=P/('esmfold_pilot.csv' if args.limit else 'esmfold_run.csv')
previous=pd.read_csv(log).to_dict('records') if log.exists() else []
prior={(r['split'],r['antibody_name']):r for r in previous}
records=[]
for row in data.itertuples():
 out=P/'structures/ESMFold'/row.split/(row.antibody_name+'.pdb');out.parent.mkdir(parents=True,exist_ok=True);start=time.time();torch.cuda.reset_peak_memory_stats()
 try:
  cached=prior.get((row.split,row.antibody_name))
  if out.exists() and cached and cached['status']=='success':
   records.append(cached);continue
  confidence=np.nan
  if not out.exists():
   h,l=row.VH,row.VL;seq=h+'G'*25+l;ids=torch.tensor([[residue_constants.restype_order_with_x.get(a,20) for a in seq]],device='cuda');positions=torch.arange(len(seq),device='cuda')[None,:];positions[:,len(h)+25:]+=512
   with torch.inference_mode():pred=model(ids,attention_mask=torch.ones_like(ids),position_ids=positions,num_recycles=4)
   coords=atom14_to_atom37(pred.positions[-1],pred)[0].float().cpu().numpy();mask=pred.atom37_atom_exists[0].cpu().numpy();aa=pred.aatype[0].cpu().numpy();plddt=pred.plddt[0].float().cpu().numpy();keep=np.r_[np.arange(len(h)),np.arange(len(h)+25,len(seq))]
   protein=Protein(atom_positions=coords[keep],aatype=aa[keep],atom_mask=mask[keep],residue_index=np.r_[np.arange(1,len(h)+1),np.arange(1,len(l)+1)],b_factors=plddt[keep],chain_index=np.r_[np.zeros(len(h),int),np.ones(len(l),int)])
   pdb=to_pdb(protein);pdb='\n'.join(line[:21]+{'A':'H','B':'L'}.get(line[21],line[21])+line[22:] if line.startswith(('ATOM  ','HETATM','TER   ')) and len(line)>21 else line for line in pdb.splitlines())+'\n';out.write_text(pdb)
   confidence=float(np.sum(plddt[keep]*mask[keep])/mask[keep].sum());del pred,coords,mask,aa,plddt;torch.cuda.empty_cache()
  structure=PDBParser(QUIET=True).get_structure('m',str(out))[0];found={c.id:''.join(seq1(r.resname) for r in c if r.id[0]==' ') for c in structure};assert found.get('H')==row.VH and found.get('L')==row.VL,found
  records.append(dict(predictor='ESMFold',split=row.split,antibody_name=row.antibody_name,status='success',seconds=time.time()-start,peak_gpu_GB=torch.cuda.max_memory_allocated()/2**30,mean_plddt=confidence))
 except Exception as e:
  records.append(dict(predictor='ESMFold',split=row.split,antibody_name=row.antibody_name,status='failed',seconds=time.time()-start,error=repr(e)));torch.cuda.empty_cache()
 pd.DataFrame(records).to_csv(log,index=False);print(len(records),'/',len(data),row.antibody_name,records[-1]['status'],round(time.time()-start,2),records[-1].get('error',''),flush=True)
