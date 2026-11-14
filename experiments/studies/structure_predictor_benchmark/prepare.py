from pathlib import Path
import pandas as pd,pickle,sys,json,shutil,re
O=Path(__file__).resolve().parent;P=O.parent;R=P.parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from abdev.data.dataset import load_gdpa1
with open(P/'verified_revision/evidence/fitted_primary.pkl','rb') as f:z=pickle.load(f)
ds=load_gdpa1(str(R/'configs/default.yaml'));rows=[];jobs=[]
assert z['train_df'].antibody_name.tolist()==ds.df.antibody_name.tolist()
for split,df in [('GDPa1',ds.df),('heldout_test',z['g3'])]:
 for i,row in df.iterrows():
  h=str(ds.heavy.iloc[i]) if split=='GDPa1' else row.vh;l=str(ds.light.iloc[i]) if split=='GDPa1' else row.vl;n=row.antibody_name
  rows.append(dict(split=split,antibody_name=n,VH=h,VL=l,fold=int(z['folds'][i]) if split=='GDPa1' else -1,ABodyBuilder3_present=(R/'external/abdev-benchmark/data/structures/AntiBodyBuilder3'/split/(n+'.pdb')).exists(),MOE_present=(R/'external/abdev-benchmark/data/structures/MOE_structures'/split/(n+'.pdb')).exists()))
  jobs.append({'name':split+'_'+n,'modelSeeds':[42],'sequences':[{'protein':{'id':'H','sequence':h}},{'protein':{'id':'L','sequence':l}}],'dialect':'alphafold3','version':1})
d=pd.DataFrame(rows);d.to_csv(O/'paired_sequences_and_folds.csv',index=False)
# One record per job; this is an input manifest, not a completed inference output.
(O/'alphafold3_jobs.json').write_text(json.dumps(jobs,indent=2))
(O/'igfold_jobs.json').write_text(json.dumps([{'name':r['antibody_name'],'split':r['split'],'sequences':{'H':r['VH'],'L':r['VL']}} for r in rows],indent=2))
protocol={'status':'prepared, not executed','objective':'compare downstream usefulness, not assume one structure predictor is optimal','cohorts':{'GDPa1':246,'GDPa3':80},'candidates':['ABodyBuilder3','IgFold','AlphaFold3','ESMFold'], 'MOE':'supplied additional structural source, not a substitute for the four named predictors','controls':['identical paired VH/VL sequences','same chain mapping and numbering checks','same feature extraction definitions and versions','same refinement policy or explicitly separate raw/refined experiments','same assay-labeled rows and outer cluster-isotype folds','all hyperparameter and predictor selection within training data','GDPa3 used once for frozen comparison; its outcomes have already been inspected in this project, so comparison is retrospective','report prediction/feature failures and common-complete-case sensitivity rather than zero-filling failures','rebuild all structure-dependent primary bio features before claiming a primary AbDev-X predictor comparison'],'endpoints':['per-assay CV Spearman mean and fold SD','paired GDPa3 differences with antibody bootstrap intervals','runtime and peak GPU memory','structural/feature completion rate','descriptor changes and sensitivity','confidence scores reported with their predictor-specific meanings'],'geometry':'assess experimental structure accuracy only for antibodies with appropriate experimental reference structures; do not use inter-predictor agreement as ground truth','ESMFold':'use documented multi-chain input; evaluate VH/VL sequence mapping and relative chain placement; do not infer chains independently and invent interface','prerequisites_missing':['alternative inference packages and weights; AF3 authorized parameters/runtime','a reproducible complete TAP/MOE/A3D/AntiFold feature-generation path matching the 57 cached bio columns'],'interpretation':'existing structural feature-group ablations and numerical feature-noise experiments are not a structure-predictor benchmark'}
(O/'benchmark_protocol.json').write_text(json.dumps(protocol,indent=2))
old=P/'delphi_positioned_revision';shutil.copytree(old/'figures',O/'figures',dirs_exist_ok=True)
s=(old/'AbDevX_DELPHI_comparison_main_SI.tex').read_text(encoding='utf8')
paragraph=r'''Structure-predictor choice remains an untested source of variation. ABodyBuilder3 was used for the available Fv visualization and associated supplied structural analyses; the present experiments do not establish that it maximizes downstream AbDev-X performance. The primary 57-feature bio block merges cached outputs with tool-specific provenance, so its contribution cannot be attributed solely to selecting ABodyBuilder3. A comparative study should regenerate identical feature definitions from paired-chain structures predicted by ABodyBuilder3, IgFold, AlphaFold 3, and ESMFold, then refit the same experts on identical training rows and folds. The ESMFold comparison additionally requires checking paired-chain assembly and the VH--VL interface. Prediction completion, feature sensitivity, compute cost, and paired downstream assay performance would be complementary outcomes. The current structural-feature-group and perturbation studies do not substitute for this experiment.

'''
anchor='Interpretation and transfer impose further requirements.'
assert anchor in s;s=s.replace(anchor,paragraph+anchor,1)
needle='The analysis uses deterministic structures.'
assert needle in s;s=s.replace(needle,'The analysis uses deterministic supplied structures. No matched structure-predictor screen was performed, and no optimal-predictor claim is made.',1)
(O/'AbDevX_structure_scope_main_SI.tex').write_text(s,encoding='utf8')
print(d.groupby('split')[['ABodyBuilder3_present','MOE_present']].sum().to_string());print('Manifest rows',len(d),'Benchmark status: prepared, not executed')
