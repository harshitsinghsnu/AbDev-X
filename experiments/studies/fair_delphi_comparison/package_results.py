from pathlib import Path
import json,re,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;P=O.parent;OLD=P/'cartoon_revision';F=O/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
d=pd.read_csv(O/'paired_comparison.csv');fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
for k,a in enumerate(['PR_CHO','SEC']):
 for j,(m,c) in enumerate([('AbDev-X','#1A237E'),('DELPHI AbLang2 Transformer','#D55E00')]):
  r=d[(d.assay==a)&(d.quantity==m)].iloc[0];axs[0].errorbar(r.estimate,k+(j-.5)*.16,xerr=[[r.estimate-r.ci_lo],[r.ci_hi-r.estimate]],fmt='o',color=c,capsize=3,label=m if k==0 else None)
 r=d[(d.assay==a)&(d.quantity=='AbDev-X minus DELPHI')].iloc[0];axs[1].errorbar(r.estimate,k,xerr=[[r.estimate-r.ci_lo],[r.ci_hi-r.estimate]],fmt='o',c='#263746',capsize=3)
for ax in axs:ax.axvline(0,color='#aaaaaa',lw=.7);ax.set_yticks([0,1],['PR-CHO','SEC']);ax.set_ylim(1.5,-.5)
axs[0].set(xlabel='Spearman ρ; BCa 95% CI',title='A  Same 80 GDPa3 antibodies');axs[0].legend(loc='lower left',fontsize=6.5)
axs[1].set(xlabel='Δρ (AbDev-X − DELPHI); paired 95% CI',title='B  Paired prediction comparison')
for ext in ['pdf','png','svg']:fig.savefig(O/f'fair_delphi_comparison.{ext}',dpi=350)
plt.close(fig)
# Preserve all existing figure assets and rebuild the benchmark plot with genuine native-task DELPHI intervals.
shutil.copytree(OLD/'figures',F,dirs_exist_ok=True)
source=(P/'finish_benchmark_pymol.py').read_text(encoding='utf8')
part=source[source.index('plt.rcParams.update'):source.index("p=pd.read_csv(P/'verified_revision/evidence/primary_reproduction.csv')")]
E=P/'benchmark_pymol_revision/evidence';R=P.parents[1]/'results'
exec(part,globals())
order=['HIC','Tm2','AC-SINS','PR_CHO','Titer','Tm1','Tonset','SEC'];x=np.arange(8)
pp=pd.read_csv(P/'verified_revision/evidence/primary_reproduction.csv').set_index('assay').loc[order]
b.bar(x-.14,pp.rho,.25,yerr=[pp.rho-pp.ci_lo,pp.ci_hi-pp.rho],capsize=2,color='#1A237E',label='AbDev-X; BCa 95% CI',error_kw={'lw':.7})
j=pd.read_csv(P/'assay_revision/evidence/jiang_verified.csv').set_index('assay')
for i,aa in enumerate(order):
 if aa in j.index:b.scatter(i+.06,j.loc[aa,'best_point'],marker='s',facecolors='none',edgecolors='#657b83',s=20,label='Jiang best published point; CI NR' if i==0 else None)
 if aa in ['PR_CHO','SEC']:
  rr=d[(d.assay==aa)&(d.quantity=='DELPHI AbLang2 Transformer')].iloc[0]
  b.errorbar(i+.26,rr.estimate,yerr=[[rr.estimate-rr.ci_lo],[rr.ci_hi-rr.estimate]],fmt='^',color='#D55E00',capsize=2,ms=4,label='DELPHI native task; BCa 95% CI' if aa=='PR_CHO' else None,lw=.7)
for i,v in enumerate([.708,.392,.356,.337,.310]):b.scatter(i+.40,v,marker='D',facecolors='none',edgecolors='black',s=16,label='Competition champion; CI NR' if i==0 else None)
b.axhline(0,c='#aaaaaa',lw=.5);b.set_xticks(x,[nice(x) for x in order],rotation=25,ha='right');b.set_ylabel('GDPa3 Spearman ρ');b.set_ylim(-.6,1.02);b.set_title('B  External comparison with native-task DELPHI predictions',loc='left',fontweight='bold');b.legend(ncol=2,loc='upper right',fontsize=5.5)
for ext in ['pdf','png','svg']:fig.savefig(F/f'main07_baseline_context.{ext}',dpi=350)
plt.close(fig)
s=(OLD/'AbDevX_PyMOL_cartoon_main_SI.tex').read_text(encoding='utf8')
s=s.replace('stored DELPHI configuration scores','matched frozen DELPHI AbLang2 Transformer predictions on PR-CHO and SEC with BCa 95\\% intervals')
s=s.replace('those results are displayed as open point symbols labeled CI NR','Jiang and competition results are displayed as open point symbols labeled CI NR')
s=s.replace('All bars in B have BCa 95\\% intervals.','All bars in B have BCa 95\\% intervals. DELPHI intervals use 20,000 antibody resamples; existing primary intervals use 2,000.')
s=s.replace('and DELPHI is a software platform whose stored configurations do not supply a matched published benchmark here~\\citep{delphi}.','and DELPHI supplies frozen pretrained predictors whose native PSR and SEC tasks are evaluated on GDPa3 here~\\citep{delphi}.')
text=r'''\subsection{Frozen-model comparison places the PR-CHO and SEC results in context}
We evaluated the deployed AbLang2 Transformer DELPHI configuration and its corresponding SEC checkpoint on the same 80 GDPa3 VH/VL pairs as the primary ensemble. PR-CHO liability uses $1-P(\mathrm{PASS})$, whereas SEC monomer content uses $P(\mathrm{PASS})$; directions were fixed from task semantics. DELPHI correlations were $.354$ on PR-CHO and $.007$ on SEC, compared with $.251$ and $-.229$ for AbDev-X. The paired differences (AbDev-X minus DELPHI) were $-.103$ (BCa 95\% CI $[-.346,.117]$) and $-.235$ ($[-.591,.122]$). Both intervals include zero; simultaneous Bonferroni intervals for the two endpoints do also. AbDev-X therefore does not demonstrate superiority in this comparison, and the data do not statistically resolve a DELPHI advantage either. The systems use different original training sets, so this is a frozen-predictor comparison, not an equal-training comparison of learning algorithms. DELPHI's internal-library classification AUCs are not comparable to these continuous external rank correlations.

'''
s=s.replace(r'\subsection{The clearest benefits concern',text+r'\subsection{The clearest benefits concern')
method=r'''\subsection{Matched DELPHI evaluation}
We fixed the AbLang2 Transformer configuration already used for deployed PSR evaluation before computing the new results and used the corresponding SEC checkpoint. Public IPI-trained checkpoints were obtained from Zenodo release 21823887, with release MD5 checksums and SHA-256 hashes recorded. No fine-tuning, threshold selection, sign optimization, or selection among checkpoints used GDPa3 outcomes. Both heavy and light input sequences matched for all 80 antibodies; prediction records were joined one-to-one by identifiers and verified against the same continuous targets. PSR class 1 denotes PASS, so $1-P(\mathrm{PASS})$ ranks PR-CHO liability; SEC $P(\mathrm{PASS})$ ranks monomer content. Spearman differences were bootstrapped using 20,000 shared antibody-index resamples (seed 42), with BCa adjustment from leave-one-antibody-out jackknife estimates. We report individual 95\% and Bonferroni 97.5\% intervals for the two differences. Original model-training uncertainty is not propagated. Proprietary DELPHI training sequences prevent an independent overlap audit; training data and budgets were not equalized. The analysis evaluates this fixed checkpoint pair and does not claim to cover all DELPHI configurations or reproduce published internal-library AUCs. Predictions, checkpoint hashes, and complete interval tables accompany the comparison.

'''
s=s.replace(r'\subsection{Matched Gaussian-process comparison}',method+r'\subsection{Matched Gaussian-process comparison}')
s=s.replace('Against the verified published Jiang points','Against the verified published Jiang points')
# Add the paired table before the final document end, retaining the existing SI.
table=r'''\subsection{Matched DELPHI prediction comparison}
\begin{table}[!htbp]\centering\small
\caption{Frozen-model comparison on identical GDPa3 antibodies ($n=80$ per endpoint). Intervals use 20,000 paired antibody resamples. Positive differences favor AbDev-X. Training data differ between the systems.}\label{tab:delphi_matched}
\begin{tabular}{llrr}\toprule Assay & Quantity & Estimate & BCa 95\% CI\\\midrule
PR-CHO & AbDev-X & .251 & [.024,.452]\\
 & DELPHI & .354 & [.124,.542]\\
 & Difference & $-.103$ & [$-.346$,.117]\\
SEC & AbDev-X & $-.229$ & [$-.433$,$-.001$]\\
 & DELPHI & .007 & [$-.215$,.232]\\
 & Difference & $-.235$ & [$-.591$,.122]\\\bottomrule
\end{tabular}\end{table}
The Bonferroni 97.5\% difference intervals are $[-.381,.151]$ for PR-CHO and $[-.639,.175]$ for SEC; neither excludes zero. The SEC primary interval is close to zero and differs slightly from the earlier 2,000-resample estimate because this comparison uses 20,000 resamples. A boundary crossing by an individual model interval is not a paired model-difference test.
'''
s=s.replace(r'\end{document}',table+'\n'+r'\end{document}')
s=s.replace('DELPHI points in the main context panel are stored software-configuration outputs, not matched published regression results', 'DELPHI points in the main context panel are matched frozen-checkpoint predictions on PR-CHO and SEC with recomputed intervals; the paired comparison is reported in Table~\\ref{tab:delphi_matched}')
(O/'AbDevX_fair_DELPHI_main_SI.tex').write_text(s,encoding='utf8')
report='Fair frozen-model DELPHI comparison\n\nSame 80 GDPa3 paired sequences and targets; model directions fixed by PASS labels.\n\n'+d.to_string(index=False)+'\n\nConclusion: AbDev-X has lower point estimates on both endpoints. Paired BCa intervals include zero; neither system is statistically established as superior. This is not an equivalence result. Different original training sets; no claim of equal-training algorithm performance. Published internal-library AUCs are not directly comparable.\n\nProtocol and checkpoint hashes: protocol.json. Full antibody predictions: matched_predictions.csv.\n'
(O/'comparison_report.txt').write_text(report,encoding='utf8')
print(report)
