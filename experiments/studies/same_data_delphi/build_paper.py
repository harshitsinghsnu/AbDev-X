from pathlib import Path
import pandas as pd,numpy as np,shutil,re,json
O=Path(__file__).resolve().parent;P=O.parent;F=O/'figures';shutil.copytree(P/'fair_delphi_comparison/figures',F,dirs_exist_ok=True);shutil.copy2(O/'same_data_comparison.pdf',F/'same_data_comparison.pdf')
s=(P/'fair_delphi_comparison/AbDevX_fair_DELPHI_main_SI.tex').read_text(encoding='utf8')
section=r'''\subsection{Training on identical data gives a task-dependent comparison with DELPHI}
To distinguish modeling differences from training-data differences, we retrained the DELPHI AbLang2 Transformer architecture on exactly the GDPa1 antibodies used by AbDev-X: 197 labeled PR-CHO antibodies and 242 labeled SEC antibodies. Both methods used the same five outer cluster--isotype folds, and all final models were refitted on the complete labeled development set before evaluation on the identical 80 GDPa3 antibodies (Figure~\ref{fig:same_data_delphi}). No proprietary IPI training examples or trained DELPHI weights entered these runs.

For continuous targets, we retained the original AbDev-X regressors and adapted DELPHI to a one-output regression head with mean-squared-error loss. AbDev-X had the larger PR-CHO point ($.251$ versus $.133$), whereas the DELPHI regression adaptation had the larger SEC point ($.103$ versus $-.229$). The paired differences were $.119$ (BCa 95\% CI $[-.192,.405]$) for PR-CHO and $-.331$ ($[-.664,.040]$) for SEC. Thus, matched training changes the comparison from the frozen-model result but does not establish superiority on either continuous endpoint.

For classification, both methods learned the same binary targets: PR-CHO PASS below $.27$, and SEC PASS at monomer content at least 90\%. DELPHI used its native two-output classifier; AbDev-X's four-expert procedure was refitted on the same binary labels. External ROC AUC was $.777$ versus $.706$ for PR-CHO and $.441$ versus $.404$ for SEC, favoring AbDev-X in point estimates. The paired AUC differences were $.071$ ($[-.172,.291]$) and $.038$ ($[-.176,.291]$), respectively. Both SEC AUCs are below chance, so their numerical ordering does not establish a useful SEC classifier. All four difference intervals include zero, as do their multiplicity-adjusted intervals. Results from three DELPHI seeds are retained without selecting the best test seed (Supplementary Table~\ref{tab:same_data_seeds}).

This matched-data experiment concerns one DELPHI architecture and fixed configuration. The continuous version is an explicit regression adaptation, and the SEC binary endpoint is a common GDPa monomer-content rule rather than a reconstruction of IPI chromatogram-based labels. It therefore supports a scoped comparison on identical available data, not an unrestricted claim that AbDev-X outperforms the entire DELPHI platform or its published internal-library AUCs.

\figpanel{same_data_comparison.pdf}{\textbf{AbDev-X and DELPHI trained on the same data and evaluated on identical GDPa3 antibodies.} \textbf{A--B}, continuous-target Spearman correlations and paired differences; DELPHI uses its original Transformer encoder with a one-output/MSE regression adaptation. \textbf{C--D}, common binary-target ROC AUC and paired differences; native DELPHI classification and binary-target AbDev-X use identical labels. PR-CHO PASS is $<.27$ and SEC PASS is $\geq90\%$ monomer. The primary displayed DELPHI run uses fixed seed 42; seeds 43/44 and a mean-score ensemble are reported as sensitivity analyses, without best-seed selection. All intervals use 20,000 paired antibody resamples with BCa adjustment. Positive differences favor AbDev-X. Both methods use the same assay-labeled GDPa1 rows, the same outer cluster--isotype fold assignments, and the same 80 GDPa3 paired sequences. Model construction and training-duration selection exclude outer validation folds and external outcomes. These intervals are conditional on fitted models and do not propagate all training uncertainty.}{fig:same_data_delphi}{}
\FloatBarrier
'''
s=s.replace(r'\subsection{The clearest benefits concern',section+r'\subsection{The clearest benefits concern',1)
method=r'''\subsection{DELPHI retraining on matched development data}
The same-data study uses GDPa1 assay availability without extra denoising, balancing, or external labeled data. We reused the frozen 480-dimensional paired AbLang2 representations and exact primary outer fold vector. Native DELPHI's public TransClassifier encoder, input projection, CLS token, and Transformer layers were imported directly. The fixed repository configuration uses hidden dimension 128, three layers, dropout .38, and its constructor's head-divisibility adjustment; AdamW uses learning rate $10^{-5}$ and weight decay $.001$, batch size 32, and gradient clipping at norm 1. The explicit regression adaptation sets one output and MSE loss, standardizing targets using only the relevant training rows. Native classification retains two outputs and cross-entropy. We did not load pretrained task-specific DELPHI weights or fine-tune its PLM.

For each outer fold $f$, fold $(f+1)\bmod5$ from the remaining training folds served as a deterministic inner validation fold. Seed 42 training for 1--55 epochs selected the minimum inner MSE or cross-entropy. Each outer model was reinitialized and fitted on all labeled outer-training antibodies for that selected duration. Thus, no outer test labels chose epochs. The final full-GDPa1 duration was the median of the five inner selections. Seeds 42, 43, and 44 were evaluated in full; seed 42 is the primary run, and no GDPa3 result selected a seed or configuration. Training used fixed learning rates without a validation-driven scheduler in outer refits. Configuration and search budgets are not identical across methods, and the benchmark does not exhaust DELPHI's alternative encoders or classifiers.

For classification, PR-CHO PASS was defined as $<.27$, and SEC PASS as at least 90\% monomer, with the same cutoffs for training and test labels. The SEC definition uses available GDPa monomer measurements and omits unavailable IPI retention-time and peak-morphology criteria. AbDev-X's original four-expert procedure, including training-only source selection and stacking, was fitted anew on these 0/1 targets. Its continuous ridge scores are valid ranking scores for AUC; no test-selected decision threshold was applied. The external panel contains 12 PR-CHO failures and nine SEC failures, limiting precision. We report ROC AUC and failure-class average precision in the saved records.

Paired external intervals use 20,000 shared antibody-index resamples (seed 42) and leave-one-antibody-out BCa adjustment. Bootstrap draws with one class are omitted for AUC. Four primary endpoint/task contrasts additionally use 98.75\% Bonferroni intervals. These are exploratory analyses conditional on fitted models; three seeds assess initialization sensitivity but do not estimate full training-population uncertainty. Identifier/fold manifests, complete per-seed predictions, selected epochs, source hashes, and fitted final models are retained for reproduction.

'''
s=s.replace(r'\subsection{Matched DELPHI evaluation}',method+r'\subsection{Matched DELPHI evaluation}')
# Give the frozen comparison its correct secondary role in Methods.
s=s.replace(r'\subsection{Matched DELPHI evaluation}',r'\subsection{Secondary frozen-checkpoint DELPHI evaluation}')
reg=pd.read_csv(O/'regression_scores.csv');cl=pd.read_csv(O/'classification_scores.csv')
cv=pd.read_csv(P/'verified_revision/evidence/primary_cv_folds.csv');cv=cv[cv.split=='cluster_isotype'];cv2=pd.read_csv(P/'verified_revision/evidence/extension_cv_folds.csv');cv=pd.concat([cv,cv2[cv2.assay=='SEC']])
summary=[]
for task,metric,dd in [('Regression','rho',reg),('Classification','roc_auc',cl)]:
 for a in ['PR_CHO','SEC']:
  for seed in [42,43,44]:
   cc=dd[(dd.assay==a)&(dd.seed==seed)&(dd.split=='CV')];tt=dd[(dd.assay==a)&(dd.seed==seed)&(dd.split=='GDPa3')]
   if task=='Classification':cc=cc[cc.model=='DELPHI native classifier'];tt=tt[tt.model=='DELPHI native classifier']
   summary.append(dict(task=task,assay=a,model='DELPHI',seed=seed,cv_mean=cc[metric].mean(),cv_sd=cc[metric].std(),test=tt[metric].iloc[0]))
  if task=='Regression':c=cv[cv.assay==a].rho;te=.25142766932028077 if a=='PR_CHO' else -.2286504606924677
  else:c=cl[(cl.assay==a)&(cl.model=='AbDev-X binary-target')&(cl.split=='CV')].roc_auc;te=cl[(cl.assay==a)&(cl.model=='AbDev-X binary-target')&(cl.split=='GDPa3')].roc_auc.iloc[0]
  summary.append(dict(task=task,assay=a,model='AbDev-X',seed=0,cv_mean=c.mean(),cv_sd=c.std(),test=te))
q=pd.DataFrame(summary);q.to_csv(O/'CV_and_external_summary.csv',index=False)
table=r'''\subsection{Same-data DELPHI training and seed sensitivity}
\begin{table}[!htbp]\centering\small
\caption{Matched-data cross-validation and external results. CV is the mean $\pm$ sample SD across the same five outer folds. Regression reports Spearman correlation; classification reports ROC AUC for common binary labels. Seeds are shown separately; no test-selected best seed is reported.}\label{tab:same_data_seeds}
\begin{tabular}{llllrr}\toprule Task & Assay & Model & Seed & CV mean $\pm$ SD & GDPa3\\\midrule
'''
for r in q.itertuples():table+=f"{r.task} & {r.assay.replace('_','-')} & {r.model} & {r.seed if r.seed else '--'} & ${r.cv_mean:.3f}\\pm{r.cv_sd:.3f}$ & {r.test:.3f}\\\\\n"
table+=r'''\bottomrule\end{tabular}\end{table}
The primary comparison uses DELPHI seed 42. The saved complete result table also includes a three-seed mean-score ensemble, which is a separate sensitivity analysis. The four paired primary differences and BCa 95\% intervals are: PR-CHO regression $.119$ $[-.192,.405]$; SEC regression $-.331$ $[-.664,.040]$; PR-CHO classification $.071$ $[-.172,.291]$; SEC classification $.038$ $[-.176,.291]$. Every individual and Bonferroni-adjusted difference interval includes zero. This is neither evidence of universal superiority nor an equivalence result.
'''
s=s.replace(r'\end{document}',table+'\n'+r'\end{document}')
(O/'AbDevX_same_data_DELPHI_main_SI.tex').write_text(s,encoding='utf8')
print(q.to_string(index=False))
