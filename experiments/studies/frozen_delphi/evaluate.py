"""Frozen-model comparison on identical GDPa3 antibodies; no outcome-based tuning."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
from pathlib import Path
import sys,json,hashlib
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
from scipy.stats import rankdata,norm,spearmanr
import torch
torch.set_num_threads(1)
O=Path(__file__).resolve().parent;P=O.parent;R=P.parents[1];D=R/'baselines/delphi'
sys.path.insert(0,str(D))
from models.transformer_lm import TransformerLMModel
g=pd.read_csv(R/'data/processed/gdpa3_table.csv')
inputs=pd.read_csv(R/'data/processed/gdpa3_delphi_input.csv')
aligned=g.merge(inputs,left_on='antibody_id',right_on='BARCODE',validate='one_to_one')
assert len(aligned)==80 and (aligned.vh==aligned.HSEQ).all() and (aligned.vl==aligned.LSEQ).all()
emb=pd.read_csv(R/'data/processed/gdpa3_delphi_input.csv.ablang.emb.csv').set_index('BARCODE').loc[aligned.BARCODE]
primary=pd.read_csv(P/'primary/evidence/primary_predictions.csv')

def rho_batch(y,p):
    a=rankdata(y,axis=-1);b=rankdata(p,axis=-1)
    a-=a.mean(axis=-1,keepdims=True);b-=b.mean(axis=-1,keepdims=True)
    return np.sum(a*b,axis=-1)/np.sqrt(np.sum(a*a,axis=-1)*np.sum(b*b,axis=-1))

def limits(boot,obs,jack,alpha=.05):
    boot=boot[np.isfinite(boot)];z0=norm.ppf(np.clip(np.mean(boot<obs),1e-6,1-1e-6));v=jack.mean()-jack
    acc=np.sum(v**3)/(6*np.sum(v*v)**1.5+1e-15)
    zz=norm.ppf([alpha/2,1-alpha/2]);q=norm.cdf(z0+(z0+zz)/(1-acc*(z0+zz)))
    return np.quantile(boot,q).tolist()

rows=[];summary=[];checkpoints=[]
for assay,target,ycol,direction in [('PR_CHO','psr_filter','polyreactivity_prscore_cho_avg',-1),('SEC','sec_filter','sec_%monomer_avg',1)]:
    stem='ipi_psr_trainset' if target=='psr_filter' else 'ipi_sec_5000'
    checkpoint=O/'checkpoints'/f'FINAL_{target}_ablang_transformer_lm_{stem}.pt'
    model=TransformerLMModel.load(str(checkpoint),embedding_dim=480,config_path=str(D/'config/transformer_lm.yaml'))
    prob=np.asarray(model.predict_proba(emb),float)
    assert prob.shape==(80,) and np.isfinite(prob).all()
    scores=1-prob if direction==-1 else prob
    ap=primary[primary.assay==assay].set_index('antibody_name').loc[aligned.antibody_name]
    y=aligned[ycol].to_numpy(float);a=ap.predicted.to_numpy(float);d=scores
    assert np.allclose(y,ap.observed,equal_nan=True)
    mask=np.isfinite(y)&np.isfinite(a)&np.isfinite(d);y,a,d=y[mask],a[mask],d[mask];n=len(y)
    observed=np.array([spearmanr(y,a).statistic,spearmanr(y,d).statistic]);delta=observed[0]-observed[1]
    rng=np.random.default_rng(42);bs=[]
    for k in range(40):
        ids=rng.integers(0,n,(500,n));ra=rho_batch(y[ids],a[ids]);rd=rho_batch(y[ids],d[ids]);bs.append(np.column_stack([ra,rd,ra-rd]))
    boot=np.concatenate(bs);jack=[]
    for i in range(n):
        keep=np.arange(n)!=i;ra=spearmanr(y[keep],a[keep]).statistic;rd=spearmanr(y[keep],d[keep]).statistic;jack.append([ra,rd,ra-rd])
    jack=np.array(jack);obs=np.r_[observed,delta]
    for i,name in enumerate(['AbDev-X','DELPHI AbLang2 Transformer','AbDev-X minus DELPHI']):
        lo,hi=limits(boot[:,i],obs[i],jack[:,i]);familylo,familyhi=limits(boot[:,i],obs[i],jack[:,i],.025)
        summary.append(dict(assay=assay,quantity=name,n=n,estimate=obs[i],ci_lo=lo,ci_hi=hi,bonferroni_ci_lo=familylo,bonferroni_ci_hi=familyhi))
    for i,r in aligned.iterrows():rows.append(dict(assay=assay,antibody_id=r.antibody_id,antibody_name=r.antibody_name,VH=r.vh,VL=r.vl,observed=r[ycol],abdevx=ap.loc[r.antibody_name,'predicted'],delphi_pass_probability=prob[i],delphi_oriented=scores[i]))
    checkpoints.append(dict(assay=assay,path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),direction='1-P(PASS)' if direction==-1 else 'P(PASS)'))
    print('RESULT',assay,summary[-3:],flush=True)
pd.DataFrame(rows).to_csv(O/'matched_predictions.csv',index=False);pd.DataFrame(summary).to_csv(O/'paired_comparison.csv',index=False)
(O/'protocol.json').write_text(json.dumps({'comparison':'frozen deployed predictors; different original training sets; not an equal-training algorithm comparison','test':'identical 80 GDPa3 VH/VL pairs; native endpoints PR-CHO and SEC','DELPHI_selection':'AbLang2 Transformer fixed from previously evaluated deployed configuration; corresponding PSR and SEC checkpoints; no test-based model selection','release':'https://zenodo.org/records/21823887','checkpoints':checkpoints,'metric':'signed Spearman correlation, directions defined by PASS labels and assay meaning, not absolute rho','uncertainty':'20000 paired antibody bootstrap resamples, seed42; BCa intervals with leave-one-antibody-out jackknife; same indices for models; positive difference favors AbDev-X','multiplicity':'95% individual BCa and 97.5% Bonferroni intervals for the two endpoint differences','AUC':'not compared to published internal-library AUC; no binary cutoffs fitted on GDPa3','scope':'training data and assays differ between systems; no prospective training-overlap guarantee for proprietary DELPHI data'},indent=2))
