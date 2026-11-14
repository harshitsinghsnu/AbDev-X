"""DELPHI regression adaptation, using identical GDPa1 rows and primary CV folds.
The public TransClassifier encoder is unchanged; num_classes=1 and MSE replace
the classification head output size and cross-entropy. Not a native DELPHI run.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import sys,json,pickle,random,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf8');torch.set_num_threads(1)
O=Path(__file__).resolve().parent;P=O.parent;R=P.parent;D=R/'baselines/delphi'
sys.path[:0]=[str(D),str(R/'scripts'),str(R/'src')]
from models.transformer_lm import TransClassifier
with open(P/'primary/evidence/fitted_primary.pkl','rb') as f:z=pickle.load(f)
cfg=yaml.safe_load((D/'config/transformer_lm.yaml').read_text(encoding='utf8'))
device='cuda' if torch.cuda.is_available() else 'cpu';SEEDS=[42,43,44];MAXEPOCH=55
X=z['B1']['ablang2'].astype('float32');XT=z['B3']['ablang2'].astype('float32');df=z['train_df'];folds=z['folds'];g=z['g3']
predrows=[];scores=[];selections=[];histories=[]
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def fit(tr,y,epochs,sd,va=None):
 seed(sd);mu=float(np.mean(y[tr]));scale=max(float(np.std(y[tr])),1e-8)
 model=TransClassifier(480,hidden_dim=cfg['model']['hidden_dim'],num_layers=cfg['model']['num_layers'],num_heads=cfg['model']['num_heads'],dropout=cfg['model']['dropout'],num_classes=1).to(device)
 opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['lr']),weight_decay=float(cfg['training']['weight_decay']))
 tx=torch.as_tensor(X[tr],device=device);ty=torch.as_tensor(((y[tr]-mu)/scale).astype('float32'),device=device)
 bestloss=float('inf');bestepoch=epochs
 for ep in range(1,epochs+1):
  model.train();idx=torch.randperm(len(tr),device=device)
  for start in range(0,len(tr),32):
   ids=idx[start:start+32];opt.zero_grad();loss=torch.nn.functional.mse_loss(model(tx[ids]).flatten(),ty[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if va is not None:
   model.eval()
   with torch.no_grad():vp=model(torch.as_tensor(X[va],device=device)).flatten().cpu().numpy()*scale+mu
   vl=float(np.mean((vp-y[va])**2));histories.append(dict(seed=sd,epoch=ep,inner_loss=vl,n_train=len(tr),n_val=len(va)))
   if vl<bestloss:bestloss=vl;bestepoch=ep
 return model,mu,scale,bestepoch
def predict(model,mu,scale,x):
 model.eval()
 with torch.no_grad():return model(torch.as_tensor(x,device=device)).flatten().cpu().numpy()*scale+mu
def save():
 pd.DataFrame(predrows).to_csv(O/'regression_predictions.csv',index=False);pd.DataFrame(scores).to_csv(O/'regression_scores.csv',index=False);pd.DataFrame(selections).to_csv(O/'epoch_selection.csv',index=False)
for assay,col,testcol in [('PR_CHO','PR_CHO','polyreactivity_prscore_cho_avg'),('SEC','SEC %Monomer','sec_%monomer_avg')]:
 y=df[col].to_numpy(float);chosen=[]
 for fold in range(5):
  inner=(fold+1)%5;tr=np.flatnonzero(np.isfinite(y)&(folds!=fold));te=np.flatnonzero(np.isfinite(y)&(folds==fold));itr=tr[folds[tr]!=inner];iva=tr[folds[tr]==inner]
  _,_,_,epoch=fit(itr,y,MAXEPOCH,42,iva);chosen.append(epoch);selections.append(dict(assay=assay,outer_fold=fold,inner_fold=inner,epoch=epoch,outer_test_never_used=True))
  print('Selected epoch',assay,fold,epoch,flush=True)
  for sd in SEEDS:
   model,mu,scale,_=fit(tr,y,epoch,sd);pr=predict(model,mu,scale,X[te]);rho=float(spearmanr(y[te],pr).statistic)
   scores.append(dict(assay=assay,split='CV',fold=fold,seed=sd,rho=rho,n=len(te),epochs=epoch))
   for j,i in enumerate(te):predrows.append(dict(assay=assay,split='CV',fold=fold,seed=sd,antibody_name=df.antibody_name.iloc[i],observed=y[i],predicted=pr[j]))
   print('Finished',assay,'fold',fold,'seed',sd,'rho',round(rho,4),flush=True);save()
 finalepoch=max(1,int(round(np.median(chosen))));tr=np.flatnonzero(np.isfinite(y))
 for sd in SEEDS:
  model,mu,scale,_=fit(tr,y,finalepoch,sd);pr=predict(model,mu,scale,XT);yt=g[testcol].to_numpy(float);ok=np.isfinite(yt)
  scores.append(dict(assay=assay,split='GDPa3',fold=-1,seed=sd,rho=float(spearmanr(yt[ok],pr[ok]).statistic),n=int(ok.sum()),epochs=finalepoch))
  for i,name in enumerate(g.antibody_name):predrows.append(dict(assay=assay,split='GDPa3',fold=-1,seed=sd,antibody_name=name,observed=yt[i],predicted=pr[i]))
  torch.save({'state_dict':model.state_dict(),'target_mean':mu,'target_sd':scale,'epochs':finalepoch,'seed':sd,'adaptation':'DELPHI TransClassifier num_classes=1; MSE'},O/f'regression_{assay}_seed{sd}.pt');save();print('Finished external',assay,sd,flush=True)
(O/'regression_protocol.json').write_text(json.dumps({'model':'DELPHI TransClassifier regression adaptation, not original DELPHI classifier','data':'same labeled GDPa1 antibodies as primary assay model; same frozen paired AbLang2 inputs','outer_split':'exact primary five cluster_isotype fold assignments','epoch_selection':'one deterministic inner cluster-isotype fold within each outer-training set, seed42, minimum raw target MSE over 1..55 epochs; outer evaluation fold excluded; final epoch median of five inner selections','hyperparameters':cfg,'optimizer':'AdamW, fixed LR, norm clipping1; no validation-driven LR scheduling in outer refits','seeds':SEEDS,'external':'full labeled GDPa1 refit; identical GDPa3 paired sequences and measurements; no external tuning','device':device,'source_sha256':hashlib.sha256((D/'models/transformer_lm.py').read_bytes()).hexdigest()},indent=2))
print('REGRESSION COMPLETE',flush=True)
