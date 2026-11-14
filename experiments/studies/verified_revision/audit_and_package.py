"""Check scientific provenance and the compiled document; create review artifacts."""
from pathlib import Path
import sys,pickle,re,json,hashlib,shutil
import numpy as np,pandas as pd,fitz
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import verify_primary as V
H=Path(__file__).parent;E=H/'evidence';F=H/'figures';ROOT=H.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with open(E/'fitted_primary.pkl','rb') as f:z=pickle.load(f)
summary=pd.read_csv(ROOT/'results/final_evaluation_summary.csv').set_index('model');rows=[]
with threadpool_limits(1):
 for a in list(V.ASSAYS)[:5]:
  c1,c3=V.ASSAYS[a];y=z['train_df'][c1].to_numpy(float);yt=z['g3'][c3].to_numpy(float);ok=np.isfinite(yt)
  predictions={'A1-E1':z['models'][a]['expert_predictions'][:,0],'A6-MGFS':z['models'][a]['expert_predictions'][:,1],'A7-Dstack':z['models'][a]['expert_predictions'][:,2],'A8-FMoE':z['models'][a]['expert_predictions'][:,3],'AbDev-X':z['models'][a]['pred']}
  for key,block in [('A2-ESM2','esm2'),('A3-AbLang2','ablang2'),('A4-Desc','desc'),('A5-Bio','bio')]:
   pipe=V.AW._head(z['B1'][block]);pipe.fit(z['B1'][block][np.isfinite(y)],y[np.isfinite(y)]);predictions[key]=pipe.predict(z['B3'][block])
  for key,pred in predictions.items():
   rho=spearmanr(yt[ok],pred[ok]).statistic;stored=summary.loc[key,a+'_mean'];rows.append(dict(assay=a,model=key,rho=rho,stored=stored,difference=rho-stored))
components=pd.DataFrame(rows);components.to_csv(E/'primary_component_reproduction.csv',index=False);assert components.difference.abs().max()<1e-8

provenance=[
('architecture_verified','all','primary architecture','scripts/aw_moe.py; scripts/final_evaluation.py','evidence/model_manifest.json'),
('main_results_verified','a-b','primary + matched baseline','verify_primary.py; primary_ablations.py','evidence/primary_cv_folds.csv; evidence/primary_baseline_cv_folds.csv; evidence/primary_reproduction.csv'),
('main_benchmark_context','a-b','primary/default extension + contextual comparators','verify_primary.py; extra_figures.py','evidence/primary_reproduction.csv; evidence/extension_cv_folds.csv; results/benchmark_comparison.csv; results/all7_assays.csv; results/delphi_comparison.csv'),
('main_generalization_verified','a-b,d','primary','primary_ablations.py; verify_primary.py','evidence/primary_cv_folds.csv; evidence/primary_reproduction.csv; evidence/primary_block_IG.csv'),
('main_generalization_verified','c','standalone encoder comparison','scripts/plm_ablation.py','results/plm_ablation.csv'),
('main_encoder_screen','a-f','standalone representation comparisons','scripts/exp_crossab.py; scripts/exp_encoder_screen.py; scripts/exp_struct_uncertainty.py','results/exp_crossab.csv; results/exp_encoder_screen.csv; results/exp_struct_comparison.csv'),
('main_primary_ablations','a-d','primary matched ablations','primary_ablations.py; build_figures.py','evidence/primary_source_ablation.csv; evidence/primary_learning_folds.csv; evidence/primary_paired_significance.csv; evidence/primary_predictions.csv'),
('main_primary_structure_attribution','a-f','primary conditional descriptor pathway','primary_residue_shap.py; render_attribution_surfaces.py','evidence/primary_residue_shap.csv; evidence/residue_shap_manifest.json; evidence/surface_render_manifest.json'),
('main_primary_descriptor_ig','a-e','primary','verify_primary.py','evidence/primary_feature_IG.csv'),
('si_detailed_shap','a-c','physicochemical LightGBM proxies','scripts/run_all.py; src/abdev/explain.py','results/explain/shap_HIC.csv; results/explain/shap_Tm2.csv; results/explain/shap_AC-SINS_pH7.4.csv'),
('si_detailed_shap','d','historical source stack','scripts/publication_figures.py','results/representation_weights.csv'),
('si_detailed_shap','e','expanded physicochemical LightGBM proxy','scripts/explain_validate.py','results/explain/shap_perantibody_HIC.csv; results/explain/shap_perantibody_Tm2.csv; results/explain/shap_perantibody_ACSINS_pH7.4.csv; results/explain/shap_perantibody_PR_CHO.csv; results/explain/shap_perantibody_Titer.csv'),
('si_soft_extensions','a-b','historical source-level soft gate','scripts/soft_moe_gating.py','results/soft_moe/soft_moe_cv.csv; results/soft_moe/gate_weights.csv'),
('si_soft_extensions','c-e','historical constraints/multitask','scripts/physics_constrained.py','results/physics_constrained/physics_hic.csv; results/physics_constrained/physics_rank_reg.csv; results/physics_constrained/multitask_cv.csv'),
('si_soft_extensions','f-h','historical source-stack arm transfer','scripts/eval_gdpa4_bispecifics.py','results/heldout_gdpa4_bispecifics/arm_predictions.csv; data/processed/gdpa4_bispecifics.csv'),
('si_primary_scope','a-b','reproduced primary + default extension','audit_and_package.py; verify_primary.py','evidence/primary_component_reproduction.csv; evidence/primary_reproduction.csv'),
('si_archived_performance','a-c','historical source-stack comparisons','scripts/build_paper_pdf.py; scripts/eval_heldout.py; extra_figures.py','results/benchmark_comparison.csv; results/heldout_configs/gdpa3_esm2_3b_eval.yaml/predictions.csv; results/all_models_splits.csv'),
('si_expert_specialization','a','component selection resampling','scripts/reviewer_response_analyses.py','results/reviewer_response/selection_stability.csv'),
('si_expert_specialization','b-c','primary fitted expert internals','verify_primary.py','evidence/fitted_primary.pkl'),
('si_routing_comparison','a-c','historical alternative architectures/source stack','scripts/abdevx2_ablation.py; scripts/sparse_moe_gdpa3.py; scripts/hybrid_ensemble.py','results/abdevx2_ablation/metrics.csv; results/sparse_moe_gdpa3.csv; results/kronecker_lstm_gdpa3.csv; results/hybrid_ensemble.csv'),
('si_routing_comparison','d','separate chain-aware network','scripts/exp_ig.py; scripts/train_caplm_fast.py','results/exp_ig.csv'),
('si_transfer_ranking','a-b','historical source stack/format reconstruction','scripts/eval_heldout.py; scripts/eval_combined_training.py','results/heldout_gdpa5/spearman_summary.csv; results/combined_training/spearman_summary.csv'),
('si_transfer_ranking','c','primary per-assay candidate ranking','build_figures.py::other_si','evidence/primary_predictions.csv; evidence/primary_precision_at_k.csv'),
('si_robustness','a-d','historical source-stack/structural diagnostics','scripts/exp_batch_correction.py; scripts/exp_struct_uncertainty.py; scripts/exp_calibration_abstention.py','results/exp_batch_correction.csv; results/exp_struct_noise.csv; results/exp_calibration.csv; results/exp_abstention.csv'),
('si_learner_screen','all','standalone regional descriptor models','scripts/harn_experiments.py','results/harn_experiments.csv'),
('si_descriptor_requested','a-d','descriptor-only ridge; zero raw-feature reference; GDPa1','scripts/exp_explain_abdevx.py::compute_descriptor_named_ig','results/explain/abdevx_descriptor_named_ig.csv'),
('si_extension_structure_maps','a-e','unchanged-primary-code default extension; conditional descriptor path','primary_residue_shap.py; render_attribution_surfaces.py','evidence/primary_residue_shap.csv')]
pd.DataFrame(provenance,columns=['figure','panels','model_identity','code','data']).to_csv(E/'figure_provenance.csv',index=False)
hashes={}
for _,_,_,_,data in provenance:
 for item in data.split('; '):
  p=(H/item) if item.startswith('evidence/') else ROOT/item
  assert p.exists(),p
  if p.suffix!='.pkl':hashes[str(p)]=sha(p)
for _,_,_,code,_ in provenance:
 for name in code.split('; '):
  name=name.split('::')[0];p=(ROOT/name) if name.startswith(('scripts/','src/')) else H/name
  assert p.exists(),p
  hashes[str(p)]=sha(p)
(E/'figure_input_hashes.json').write_text(json.dumps(hashes,indent=2))

source_archive=H/'source_archive';source_archive.mkdir(exist_ok=True)
for p in [ROOT/'paper/figures_backup_20261103/panel_performance_comparison.pdf',H.parent/'panel_gen_rep.pdf',H.parent/'panel_baseline_comparison.pdf',H.parent/'panel_descriptor_interpretability.pdf',H.parent/'panel_explainability.pdf',H.parent/'panel_extensions.pdf']:
 shutil.copy2(p,source_archive/p.name)
report='''# Primary AbDev-X architecture and results audit

The five-assay headline result reproduces. Several old figures and the prior ten-assay summary did not use the same primary model. The revised manuscript corrects those identities and uses newly recomputed primary-model results where applicable. Original research scripts, result CSVs, and selected source PDFs are preserved.

## Authoritative primary implementation

- `scripts/aw_moe.py`: `E1Expert` (line 141), `MGFSExpert` (208), `DStackExpert` (231), `FMoEExpert` (274), with their `fit_full`/`predict` implementations.
- `scripts/final_evaluation.py:191`: `pred_awmoe` explicitly averages the four expert predictions. All 45 assay-by-strategy point estimates in its same-five-assay comparison are numerically reproduced (maximum difference < 1e-8).
- The old `AbDevFusion`/source-stack class, chain-aware network, descriptor-only ridge, and LightGBM SHAP proxies are distinct models. Their supplementary figures retain those identities.

## Architecture corrections

| Item | Correct primary behavior |
|---|---|
| Inputs | VH and VL plus separately supplied HC/LC subtype; structural features are cached auxiliary inputs |
| Operational sources | ESM-2 2560; AbLang2 480; descriptors 165; bio 57; subtype 5 |
| Metadata | IgG1/2/4 and kappa/lambda, not germline-gene one-hots |
| E1 | Best eligible block by training-only cluster–isotype CV, then refit |
| A-MGFS | Keyword inclusion/exclusion masks, no greedy feature-selection stage |
| D-Stack | PLM stream = ESM-2 + AbLang2; second stream = descriptors + bio; inner-OOF ridge meta-head, alpha 1 |
| F-MoE | Eligible mechanistic groups; fixed CV-score softmax weights, tau 0.1; groups with fewer than 3 matching features skipped |
| Heads | StandardScaler; PCA 90% for >100 input features; RidgeCV |
| Final fusion | Equal average, 0.25 per expert; not an antibody-dependent learned final gate |
| Model scope | Original primary code defines five assays. Extra five outputs here apply its unchanged fallback rules and are explicitly an extension |

## Primary and ten-assay reproduction

The first five rows match the stored headline scores after rounding. The remaining stored rows do not reproduce under the unchanged expert classes and are replaced with a clearly defined code-default extension. This does not assert that those historical rows were fabricated; their model identity does not match the primary implementation.

| Assay | Old stored rho | Recomputed rho | n | Scope |
|---|---:|---:|---:|---|
'''
points=pd.read_csv(E/'primary_reproduction.csv')
for i,r in points.iterrows():report+=f'| {r.assay} | {r.recorded_rho:.3f} | {r.rho:.6f} | {int(r.n)} | '+('Primary' if i<5 else 'Default extension')+' |\n'
report+=f'\nSame-five primary mean: **{points.rho.iloc[:5].mean():.6f}**. Ten-assay default-extension mean: **{points.rho.mean():.6f}**, replacing the mixed stored mean of {points.recorded_rho.mean():.3f}.\n'
report+='''
## What was recomputed

- Primary GDPa3 predictions for all four experts and their average, plus paired BCa intervals (2000 resamples).
- Same-five nine-strategy component comparison: all 45 point estimates reproduced.
- Primary outer-fold CV under random, cluster and cluster–isotype splits, with fixed equal final fusion. Historical CV used a separate implementation that also fit fold-level fusion weights; its numbers are not assumed identical to a fixed average.
- Refitted primary leave-one-source-out, leave-one-expert-out, 20–100% data-efficiency curves, and matched paired BCa differences versus ESM-2/AbLang2.
- Exact affine input-feature IG, primary D-Stack coefficients and F-MoE weights, and conditional residue permutation SHAP of the descriptor-dependent primary prediction.

The new paired intervals exclude zero for HIC, SMAC and HAC against both PLM baselines. They include zero for AC-SINS and PR-CHO. These exploratory intervals are not multiplicity-adjusted.

## Attribution scope

The full primary predictor is affine in its fitted cached input blocks. Exact IG includes all five blocks and sums to prediction minus the training-mean baseline. The residue maps use the actual ensemble coefficients through every active descriptor path, recomputing descriptors over residue coalitions while keeping PLM, bio and subtype fixed. They are not proxy SHAP or a generic descriptor-color overlay. However, they do not provide full end-to-end sequence-to-structure mutation attribution. The all-alanine reference is artificial; correlations, off-manifold interventions, finite permutation error, and numbering fallback effects limit mechanistic interpretation. Per-residue Monte Carlo standard errors are saved.

The chosen held-out antibody is the alphabetically first entry with a predicted structure, independent of labels/predictions. Both PDB chain sequences are asserted to match the input exactly. Surface meshes are generated directly from the atomic coordinates using scipy/matplotlib; score assignment and rendering parameters are recorded.

## Figure handling and typography

All main and SI figures are rebuilt at **183 mm width**, with **6–7 pt body text**, editable vector lettering and embedded TrueType fonts. Architecture section labels use 8 pt. The tallest panels are 165.1 mm. Scientific molecular-surface renders are raster insets; their surrounding labels remain vector. All tables are in SI. The two requested relocated composites are S3 and S4; detailed SHAP and Soft MoE remain S1 and S2. The requested descriptor-interpretablity panel is re-created from its source CSV as S10, with complete labels and its standalone model identity; primary descriptor IG is in the main paper.

The untraceable tabular-model summary and old composite-ranking summary are also excluded; S9 keeps the traceable regional-descriptor experiment, and S7c uses newly computed primary per-assay precision at K. The archived GP panel has no recoverable raw score table in the inspected results; matching KitAb/Amini numerical results were not verified. They are excluded from primary evidence. The three old scatter labels (.296, -.066, .157) also differ slightly from the available saved ESM-2 3B source-stack prediction rows (.295, -.064, .168); S4 uses the actual available prediction rows and states that provenance.

Nature guidance consulted: https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/ and https://www.nature.com/nature/for-authors/initial-submission . This is a figure-readability revision, not a declaration of journal acceptance or full submission compliance.

`evidence/figure_provenance.csv` maps every figure/panel to model, source code and data. `figure_input_hashes.json` records inputs; `figure_typography_audit.json` and `validation_report.json` record final checks. The original selected PDFs are in `source_archive/`.
'''
(H/'SCIENCE_AUDIT.md').write_text(report,encoding='utf-8')

tex=(H/'AbDevX_verified_main_SI.tex').read_text(encoding='utf-8');aux=(H/'AbDevX_verified_main_SI.aux').read_text(encoding='utf-8');log=(H/'AbDevX_verified_main_SI.log').read_text(encoding='utf-8',errors='replace')
labels=set(re.findall(r'\\newlabel\{([^}]+)\}',aux));refs=set(re.findall(r'\\(?:ref|pageref)\{([^}]+)\}',tex));assert not refs-labels,refs-labels
figs=re.findall(r'\\newlabel\{(fig:[^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux);tables=re.findall(r'\\newlabel\{(tab:[^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)
assert len(figs)==19 and len(tables)==3
assert all(n.startswith('S') for _,n,_ in tables)
assert 'Overfull \\' not in log and 'undefined references' not in log and 'too large' not in log
fontrows=[]
for asset in re.findall(r'\\figpanel\{([^}]+)\}',tex):
 p=F/asset;doc=fitz.open(p);page=doc[0];spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans'] if s['text'].strip()]
 mn=min(s['size'] for s in spans);outside=[s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
 assert mn>=5.99 and abs(page.rect.width*25.4/72-183)<.01 and not outside,(asset,mn,outside)
 fontrows.append(dict(figure=asset,width_mm=183,min_font_pt=mn,height_mm=page.rect.height*25.4/72,outside_page=[]))
(E/'figure_typography_audit.json').write_text(json.dumps(fontrows,indent=2))
pdf=H/'AbDevX_verified_main_SI.pdf';doc=fitz.open(pdf);validation={'pages':len(doc),'figures':[dict(label=x,number=n,page=int(pg)) for x,n,pg in figs],'tables':[dict(label=x,number=n,page=int(pg)) for x,n,pg in tables],'undefined_references':[],'overfull_boxes':0,'all_figure_widths_mm':183,'minimum_figure_font_pt':min(x['min_font_pt'] for x in fontrows),'component_reproduction_max_error':float(components.difference.abs().max()),'primary_mean_rho':float(points.rho.iloc[:5].mean()),'ten_assay_extension_mean_rho':float(points.rho.mean())}
(E/'validation_report.json').write_text(json.dumps(validation,indent=2))
for _,n,pg in figs:
 if n in ['1','7','S1','S2']:doc[int(pg)-1].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(H/f'review_page_{n}.png')
shutil.copy2(pdf,H.parent/pdf.name)
print(json.dumps(validation,indent=2))
