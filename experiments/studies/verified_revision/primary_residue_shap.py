"""Permutation SHAP of the actual primary predictor's sequence-descriptor path.

Residues are the coalition players, replaced by alanine when absent. Embeddings,
structure features and metadata stay fixed. This is conditional pathway attribution,
not a proxy model and not full end-to-end sequence/structure attribution.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,pickle,json
import numpy as np,pandas as pd
import verify_primary as V
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
H=Path(__file__).parent;O=H/'evidence';F=H/'figures'
with open(O/'fitted_primary.pkl','rb') as f:Z=pickle.load(f)
g3=Z['g3'];structdir=V.ROOT/'external/abdev-benchmark/data/structures/AntiBodyBuilder3/heldout_test'
name=next(n for n in sorted(g3.antibody_name) if (structdir/(n+'.pdb')).exists());row=g3[g3.antibody_name==name].iloc[0];i=g3.index[g3.antibody_name==name][0]
h=row.vh;l=row.vl;seq=h+l;N=len(seq);base='A'*N;asys=list(V.ASSAYS);W=np.column_stack([Z['models'][a]['weights']['desc'] for a in asys]);rng=np.random.default_rng(42)
def desc(s):return np.array(list(V.build_descriptor_matrix(pd.Series([s[:len(h)]]),pd.Series([s[len(h):]])).iloc[0]),float)
d0=desc(base);d1=desc(seq);assert np.allclose(d1,Z['B3']['desc'][i])
paths=[]
for rep in range(32):
 order=rng.permutation(N)
 for perm in [order,order[::-1]]:
  s=list(base);ds=[d0]
  for j in perm:s[j]=seq[j];ds.append(desc(''.join(s)))
  values=np.array(ds)@W;dif=np.diff(values,axis=0);phi=np.zeros((N,len(asys)));phi[perm]=dif;paths.append(phi)
 if rep%4==0:print('Permutation pairs',rep+1,flush=True)
P=np.stack(paths);phi=P.mean(0);se=P.std(0,ddof=1)/np.sqrt(len(P));expected=(d1-d0)@W
assert np.max(np.abs(phi.sum(0)-expected))<1e-7
struc=PDBParser(QUIET=True).get_structure('m',structdir/(name+'.pdb'));mappings=[]
for chain,chainseq,offset in [('H',h,0),('L',l,len(h))]:
 residues=[r for r in struc[0][chain] if r.id[0]==' '];pdbseq=''.join(seq1(r.resname) for r in residues);assert pdbseq==chainseq,(chain,pdbseq,chainseq)
 for j,r in enumerate(residues):mappings.append((offset+j,chain,r.id[1],r.id[2].strip()))
rows=[]
for ai,a in enumerate(asys):
 for j,chain,pos,ins in mappings:rows.append(dict(assay=a,antibody_name=name,chain=chain,sequence_position=j+1 if chain=='H' else j-len(h)+1,pdb_residue=str(pos)+ins,residue=seq[j],shap=phi[j,ai],mc_standard_error=se[j,ai]))
pd.DataFrame(rows).to_csv(O/'primary_residue_shap.csv',index=False)
meta={'antibody_name':name,'structure':str(structdir/(name+'.pdb')),'selection':'alphabetically first GDPa3 antibody with a predicted structure; selected independently of labels and predictions','reference':'alanine at every VH and VL position','permutations':64,'conditional_inputs':'Only the sequence descriptor block is recomputed, including the original numbering fallback. PLM, bio and subtype inputs held fixed. All four primary expert paths contributing to descriptors are included.','completeness_max_error':float(np.abs(phi.sum(0)-expected).max()),'vh_length':len(h),'vl_length':len(l),'assay_descriptor_changes':dict(zip(asys,expected.tolist()))}
(O/'residue_shap_manifest.json').write_text(json.dumps(meta,indent=2))
import runpy
runpy.run_path(str(H/'render_attribution_surfaces.py'),run_name='__main__')
print('Saved primary descriptor-pathway SHAP and structures.',flush=True)
