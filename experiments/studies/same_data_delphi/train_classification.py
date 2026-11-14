"""Shared binary task. Native DELPHI classifier and binary-target AbDev-X.
Fixed common cutoffs: PR-CHO PASS <.27; GDPa SEC monomer PASS >=90 percent.
SEC is a common GDPa operational label, not IPI chromatogram-level label recreation.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import sys,json,pickle,random
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(1)
O=Path(__file__).resolve().parent;P=O.parent;R=P.parents[1];D=R/'baselines/delphi';sys.path[:0]=[str(D),str(P/'verified_revision'),str(R/'scripts'),str(R/'src')]
from models.transformer_lm import TransClassifier
import verify_primary as V
with open(P/'verified_revision/evidence/fitted_primary.pkl','rb') as f:z=pickle.load(f)
cfg=yaml.safe_load((D/'config/transformer_lm.yaml').read_text(encoding='utf8'));device='cuda' if torch.cuda.is_available() else 'cpu'
X=z['B1']['ablang2'].astype('float32');XT=z['B3']['ablang2'].astype('float32');df=z['train_df'];g=z['g3'];folds=z['folds'];SEEDS=[42,43,44]
rows=[];metrics=[];choices=[]
def fit(tr,y,epochs,seed,va=None):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 m=TransClassifier(480,hidden_dim=cfg['model']['hidden_dim'],num_layers=cfg['model']['num_layers'],num_heads=cfg['model']['num_heads'],dropout=cfg['model']['dropout'],num_classes=2).to(device)
 opt=torch.optim.AdamW(m.parameters(),lr=float(cfg['training']['lr']),weight_decay=float(cfg['training']['weight_decay']))
 xx=torch.as_tensor(X[tr],device=device);yy=torch.as_tensor(y[tr],dtype=torch.long,device=device);best=float('inf');be=epochs
 for ep in range(1,epochs+1):
  m.train();ids=torch.randperm(len(tr),device=device)
  for st in range(0,len(tr),32):
   ii=ids[st:st+32];opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(xx[ii]),yy[ii]);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if va is not None:
   m.eval()
   with torch.no_grad():vl=float(torch.nn.functional.cross_entropy(m(torch.as_tensor(X[va],device=device)),torch.as_tensor(y[va],dtype=torch.long,device=device)))
   if vl<best:best=vl;be=ep
 return m,be
def predict(m,x):
 m.eval()
 with torch.no_grad():return torch.softmax(m(torch.as_tensor(x,device=device)),1)[:,1].cpu().numpy()
def record(a,model,split,fold,seed,indices,y,p,names):
 ok=np.isfinite(y);yy=y[ok];pp=p[ok]
 metrics.append(dict(assay=a,model=model,split=split,fold=fold,seed=seed,n=int(ok.sum()),n_pass=int(yy.sum()),roc_auc=roc_auc_score(yy,pp) if len(set(yy))==2 else np.nan,fail_average_precision=average_precision_score(1-yy,-pp) if len(set(yy))==2 else np.nan))
 for i,idx in enumerate(indices):rows.append(dict(assay=a,model=model,split=split,fold=fold,seed=seed,antibody_name=names.iloc[idx],observed=y[i],predicted=p[i]))
 pd.DataFrame(rows).to_csv(O/'classification_predictions.csv',index=False);pd.DataFrame(metrics).to_csv(O/'classification_scores.csv',index=False)
cols=z['descriptor_columns']+z['bio_columns']
with threadpool_limits(limits=1):
 for a,col,tc,threshold in [('PR_CHO','PR_CHO','polyreactivity_prscore_cho_avg',.27),('SEC','SEC %Monomer','sec_%monomer_avg',90)]:
  raw=df[col].to_numpy(float);rawtest=g[tc].to_numpy(float);y=np.where(np.isfinite(raw),(raw<threshold if a=='PR_CHO' else raw>=threshold).astype(float),np.nan);yt=np.where(np.isfinite(rawtest),(rawtest<threshold if a=='PR_CHO' else rawtest>=threshold).astype(float),np.nan);selected=[]
  for fold in range(5):
   tr=np.flatnonzero(np.isfinite(y)&(folds!=fold));te=np.flatnonzero(np.isfinite(y)&(folds==fold));inner=(fold+1)%5;itr=tr[folds[tr]!=inner];iva=tr[folds[tr]==inner]
   _,ep=fit(itr,y,55,42,iva);selected.append(ep);choices.append(dict(assay=a,fold=fold,inner_fold=inner,epochs=ep))
   b1={k:v[tr] for k,v in z['B1'].items()};b2={k:v[te] for k,v in z['B1'].items()};model=V.fit_model(b1,b2,y[tr],folds[tr],a,cols)
   record(a,'AbDev-X binary-target','CV',fold,0,te,y[te],model['pred'],df.antibody_name)
   for sd in SEEDS:
    m,_=fit(tr,y,ep,sd);pr=predict(m,X[te]);record(a,'DELPHI native classifier','CV',fold,sd,te,y[te],pr,df.antibody_name)
   print('Classification finished',a,'fold',fold,'epochs',ep,flush=True)
  tr=np.flatnonzero(np.isfinite(y));model=V.fit_model({k:v[tr] for k,v in z['B1'].items()},z['B3'],y[tr],folds[tr],a,cols)
  record(a,'AbDev-X binary-target','GDPa3',-1,0,np.arange(len(g)),yt,model['pred'],g.antibody_name)
  ep=max(1,int(round(np.median(selected))))
  for sd in SEEDS:
   m,_=fit(tr,y,ep,sd);pr=predict(m,XT);record(a,'DELPHI native classifier','GDPa3',-1,sd,np.arange(len(g)),yt,pr,g.antibody_name);torch.save(m.state_dict(),O/f'classifier_{a}_seed{sd}.pt')
  print('Classification external finished',a,flush=True)
pd.DataFrame(choices).to_csv(O/'classification_epoch_selection.csv',index=False)
(O/'classification_protocol.json').write_text(json.dumps({'task':'common GDPa pass/fail; PR-CHO PASS<0.27; SEC monomer PASS>=90 percent','SEC_scope':'does not reproduce IPI trace morphology/retention-time criteria','DELPHI':'native TransClassifier 2 outputs, CE, repository fixed model and optimizer config; fresh initialization; AbLang2 frozen features','AbDev-X':'original four-expert procedure fitted on same 0/1 targets using its ridge heads; retained real scores evaluated by ROC AUC and failure AP','data':'same GDPa1 labeled rows and exact primary outer folds; shared GDPa3 antibodies','selection':'one inner training fold selects epoch by CE, maximum55; full refit median epoch; no test-based choices','seeds':SEEDS,'cutoffs_not_selected_from_test':True,'threshold_selection_at_inference':'none: ROC AUC and AP use continuous predicted scores'},indent=2))
print('CLASSIFICATION COMPLETE',flush=True)
