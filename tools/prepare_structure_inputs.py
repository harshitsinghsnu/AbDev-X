"""Prepare paired-chain manifests only; never edits manuscript text or results."""
from pathlib import Path
import os,sys,json,pickle
import pandas as pd
R=Path(__file__).resolve().parents[1];P=R/'experiments/studies';O=P/'structures';os.chdir(R)
sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from abdev.data.dataset import load_gdpa1
with open(P/'primary/evidence/fitted_primary.pkl','rb') as f:z=pickle.load(f)
ds=load_gdpa1(str(R/'configs/default.yaml'));rows=[];inputs=O/'af3_inputs';inputs.mkdir(exist_ok=True)
assert ds.df.antibody_name.tolist()==z['train_df'].antibody_name.tolist()
for split,df in [('GDPa1',ds.df),('heldout_test',z['g3'])]:
 for i,row in df.iterrows():
  vh=str(ds.heavy.iloc[i]) if split=='GDPa1' else row.vh;vl=str(ds.light.iloc[i]) if split=='GDPa1' else row.vl
  record=dict(split=split,antibody_name=row.antibody_name,VH=vh,VL=vl,fold=int(z['folds'][i]) if split=='GDPa1' else -1)
  for name,folder in [('ABodyBuilder3','AntiBodyBuilder3'),('MOE','MOE_structures')]:record[name+'_present']=(R/'external/abdev-benchmark/data/structures'/folder/split/(row.antibody_name+'.pdb')).exists()
  index=len(rows);rows.append(record)
  proteins=[dict(protein=dict(id=cid,sequence=seq,unpairedMsa='',pairedMsa='',templates=[])) for cid,seq in [('H',vh),('L',vl)]]
  (inputs/f'{index:03}.json').write_text(json.dumps(dict(name=f'{index:03}_'+row.antibody_name,modelSeeds=[42],sequences=proteins,dialect='alphafold3',version=1)))
pd.DataFrame(rows).to_csv(O/'paired_sequences_and_folds.csv',index=False)
print('Prepared',len(rows),'exact paired chains and original folds')
