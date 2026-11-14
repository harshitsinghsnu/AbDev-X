from pathlib import Path
import shutil
P=Path(__file__).resolve().parent;O=P.parent/'delphi_positioned_revision';O.mkdir(exist_ok=True)
shutil.copytree(P/'figures',O/'figures',dirs_exist_ok=True)
s=(P/'AbDevX_same_data_DELPHI_main_SI.tex').read_text(encoding='utf8')
needle='Source removal connected the gains to assay-dependent contributions, including structural bio features for HIC and SMAC and antibody embeddings for AC-SINS.'
new=needle+r' With identical GDPa1 training data and folds, AbDev-X also gave higher PR-CHO point estimates than the evaluated DELPHI AbLang2 Transformer: external correlation $.251$ versus $.133$ for a DELPHI regression adaptation, and AUC $.777$ versus $.706$ on common binary labels. The paired difference intervals included zero.'
assert needle in s;s=s.replace(needle,new,1)
s=s.replace(r'\subsection{Training on identical data gives a task-dependent comparison with DELPHI}',r'\subsection{Matched training favors AbDev-X point estimates for PR-CHO ranking and classification}')
needle='No proprietary IPI training examples or trained DELPHI weights entered these runs.'
new=needle+r' This comparison asks whether the assay-aware ensemble can make better use of the same small experimental panel, rather than comparing predictors with different amounts or sources of labeled training data.'
assert needle in s;s=s.replace(needle,new,1)
needle='For continuous targets, we retained the original AbDev-X regressors and adapted DELPHI to a one-output regression head with mean-squared-error loss.'
new=r'''The PR-CHO ordering was consistent between development and external evaluation for the primary seed-42 comparison. Under identical outer folds, mean CV correlation was $.431\pm.184$ for AbDev-X and $.062\pm.086$ for the DELPHI regression adaptation; the corresponding classification AUCs were $.754\pm.183$ and $.602\pm.090$ (mean $\pm$ fold SD). On GDPa3, AbDev-X retained the larger PR-CHO points for both tasks. This provides an observed small-data advantage for the evaluated configuration, with uncertainty assessed using paired external predictions.

'''+needle
assert needle in s;s=s.replace(needle,new,1)
needle='The biological interpretation follows from the contrasts between endpoints.'
new=r'''The matched-data DELPHI experiment adds a complementary view of the polyreactivity result. When both methods learn from the same 197 labeled GDPa1 antibodies and use the same folds, AbDev-X has higher PR-CHO CV and external point estimates for both continuous prediction and common-label classification. The external AUC gain is $.071$, and the correlation gain over the DELPHI regression adaptation is $.119$. These numerical gains suggest that the ensemble can use the available small panel effectively. They remain uncertain under paired resampling and concern one fixed DELPHI architecture; they do not demonstrate superiority over every DELPHI configuration. The frozen proprietary-data-trained DELPHI checkpoint gives a different ordering, emphasizing that training-data regime is part of the comparison. We therefore position the matched PR-CHO results as evidence of a useful observed advantage to validate prospectively.

'''+needle
assert needle in s;s=s.replace(needle,new,1)
needle='The contribution is an assay-aware framework for combining existing molecular information with an explicit evaluation scope.'
new=r'The matched-training comparison additionally identifies higher PR-CHO ranking and classification point estimates than the evaluated DELPHI configuration, with paired uncertainty that motivates larger external validation. '+needle
assert needle in s;s=s.replace(needle,new,1)
(O/'AbDevX_DEL­PHI_comparison_main_SI.tex'.replace('\u00ad','')).write_text(s,encoding='utf8')
print('Updated PR-CHO positioning in abstract, Results, Discussion, and Conclusions.')
