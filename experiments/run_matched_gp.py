"""New GP benchmarks on primary cached blocks and matched outer folds.
These explicitly specified kernels are not reproductions of the old plotted GPs.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,pickle
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
P=Path(__file__).resolve().parent;R=P.parent;sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import aw_moe as AW
O=P/'gaussian_process/evidence';O.mkdir(parents=True,exist_ok=True)
with open(P/'primary/evidence/fitted_primary.pkl','rb') as f:z=pickle.load(f)
assays=['HIC','Tm2','AC-SINS','PR_CHO','Titer'];folds=z['folds'];B=z['B1'];df=z['train_df'];records=[];predrows=[]

def kernels(dist,ell,kind):
    if kind=='block-additive':
        kk=np.exp(-dist/(2*ell**2));return kk.mean(0),(kk*dist/ell**2).mean(0)
    # First two blocks are PLMs; remaining eligible blocks are physicochemical/metadata.
    p=dist[:2].mean(0);q=dist[2:].mean(0)
    if kind=='stream-additive':
        kp=np.exp(-p/(2*ell**2));kq=np.exp(-q/(2*ell**2));return (kp+kq)/2,(kp*p+kq*q)/(2*ell**2)
    d=p+q;k=np.exp(-d/(2*ell**2));return k,k*d/ell**2

def fit_predict(dt,dv,yy,kind):
    n=len(yy);mu=yy.mean();sd=yy.std();y=(yy-mu)/sd;eye=np.eye(n)
    def objective(t,full=False):
        ell,amp,noise=np.exp(t);k,dk=kernels(dt,ell,kind);K=amp*k+(noise+1e-8)*eye
        cf=cho_factor(K,lower=True,check_finite=False);alpha=cho_solve(cf,y,check_finite=False)
        val=.5*y@alpha+np.log(np.diag(cf[0])).sum()+n*.5*np.log(2*np.pi)
        if full:return alpha
        Q=cho_solve(cf,eye,check_finite=False)-np.outer(alpha,alpha)
        grad=.5*np.array([np.sum(Q*(amp*dk)),np.sum(Q*(amp*k)),noise*np.trace(Q)])
        return val,grad
    fits=[minimize(objective,np.log([ell,1,.2]),jac=True,method='L-BFGS-B',bounds=[(-3,3),(-6,6),(-8,3)],options={'maxiter':200,'ftol':1e-10}) for ell in [.5,2.]]
    opt=min(fits,key=lambda f:f.fun);alpha=objective(opt.x,True);ell,amp,noise=np.exp(opt.x);k,_=kernels(dv,ell,kind)
    return mu+sd*(amp*k@alpha),opt,dict(length_scale=ell,signal_variance=amp,noise_variance=noise)

with threadpool_limits(limits=1):
 for a in assays:
    ycol='AC-SINS_pH7.4' if a=='AC-SINS' else a;y=df[ycol].to_numpy(float)
    keys=[k for k in ['esm2','ablang2','desc','bio','isotype'] if not(k=='isotype' and a in AW.ISO_INELIGIBLE)]
    for fold in sorted(set(folds)):
        tr=np.flatnonzero((folds!=fold)&np.isfinite(y));te=np.flatnonzero((folds==fold)&np.isfinite(y));dt=[];dv=[]
        for k in keys:
            x=B[k];sc=StandardScaler().fit(x[tr]);xt=sc.transform(x[tr]);xv=sc.transform(x[te])
            if x.shape[1]>100:
                pc=PCA(.9,svd_solver='full',random_state=0).fit(xt);xt=pc.transform(xt);xv=pc.transform(xv)
            dt.append(cdist(xt,xt,'sqeuclidean')/xt.shape[1]);dv.append(cdist(xv,xt,'sqeuclidean')/xt.shape[1])
        for kind in ['block-additive','stream-additive','stream-product']:
            pred,opt,h=fit_predict(np.array(dt),np.array(dv),y[tr],kind)
            rho=float(spearmanr(y[te],pred).statistic);records.append(dict(model='GP '+kind,assay=a,fold=int(fold),rho=rho,n_train=len(tr),n_test=len(te),optimizer_success=bool(opt.success),optimizer_message=str(opt.message),negative_log_marginal_likelihood=float(opt.fun),**h))
            for j,i in enumerate(te):predrows.append(dict(model='GP '+kind,assay=a,fold=int(fold),antibody_name=df.antibody_name.iloc[i],observed=y[i],predicted=pred[j]))
        pd.DataFrame(records).to_csv(O/'matched_gp_folds.csv',index=False);pd.DataFrame(predrows).to_csv(O/'matched_gp_predictions.csv',index=False)
        print('GP finished',a,'fold',fold,flush=True)
pd.DataFrame(records).groupby(['model','assay']).rho.agg(['mean','std']).to_csv(O/'matched_gp_summary.csv')
(O/'matched_gp_protocol.json').write_text(json.dumps({'new_experiment':True,'not_reproductions_of_legacy_GP_FA_HA_HP':True,'assays':assays,'outer_split':'same cached primary cluster_isotype folds','fold_ids':list(map(int,sorted(set(folds)))),'preprocessing':'fit StandardScaler per block on labeled outer-training rows; 90% PCA on blocks with >100 input dimensions; no validation rows used','block_distance':'mean squared Euclidean distance per retained transformed dimension','block-additive':'mean_b exp(-D_b/(2 ell^2))','stream-additive':'0.5 exp(-mean(D_ESM,D_AbL)/(2 ell^2)) + 0.5 exp(-mean(D_desc,D_bio,D_iso_if_eligible)/(2 ell^2))','stream-product':'exp(-(mean(D_ESM,D_AbL)+mean(D_desc,D_bio,D_iso_if_eligible))/(2 ell^2))','hyperparameters':'common length scale, signal variance, noise variance learned by outer-training Gaussian log marginal likelihood; two deterministic starts; no held-out tuning','target':'outer-training mean/std normalization','seed':0},indent=2))
print('DONE',flush=True)
