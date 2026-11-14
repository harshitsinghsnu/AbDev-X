# Result-to-code map and reconstruction limits

Figure numbers below refer to `manuscript/main.tex`. Labels are more stable than
numbers if the paper is edited. Primary outputs and historical comparisons are
kept separate throughout.

| Result or panel | Main source code | Numerical evidence and scope |
|---|---|---|
| Four-expert primary and ten endpoints | `scripts/aw_moe.py`, `scripts/final_evaluation.py`, `experiments/studies/verified_revision/verify_primary.py` | Refit from original representations/labels; reproduce all ten assay correlations. Final expert weights are 0.25 each. |
| Primary CV and split comparisons | `scripts/compute_awmoe_cv.py`, `scripts/abdevx_splits.py` | Exact dataset fold assignments; primary and code-default extension identities retained. |
| Nine strategies, PLM baselines, source removal, learning curves | `verified_revision/primary_ablations.py` | Individual predictions, CV rows, and refit results under `verified_revision/evidence/`. |
| Encoder and structural-source-group screen | `scripts/exp_crossab.py`, `scripts/exp_encoder_screen.py`, `scripts/plm_ablation.py`, `assay_revision/compute_uncertainty.py` | Standalone five-competition-assay screens; not a substitution for primary ensemble scores. |
| Baseline/published comparison | `finish_benchmark_pymol.py`, `run_matched_gp.py`, `assay_revision/fetch_comparators.py` | Matched local GP refits are executable; literature values are transcribed published points with protocol scope, not trained here. |
| Exact primary feature contributions | `verified_revision/verify_primary.py` | Affine decomposition of the fitted predictor; training-mean baseline; omitted PLM dimensions remain in the complete model. |
| Primary residue attribution | `verified_revision/primary_residue_shap.py`, `render_chimerax_clean.py` | Conditional sequence-descriptor pathway, 64 permutations; PLM/bio/subtype fixed; no causal hotspot or full molecular recomputation claim. |
| Descriptor-only IG | `scripts/exp_ig.py`, `scripts/exp_explain_abdevx.py` | Separate descriptor/chain-aware studies, not the full primary predictor. |
| S1 proxy SHAP | `assay_revision/compute_uncertainty.py`, `make_revision_figures.py` | LightGBM proxy refits and conditional bootstrap intervals, not direct primary SHAP. |
| S2 source gate / constraints / arm average | `scripts/soft_moe_gating.py`, `scripts/physics_constrained.py`, `scripts/eval_gdpa4_bispecifics.py`, `assay_revision/extend_uncertainty.py` | Historical shuffled-fold/source configurations. Some use globally fitted PCA or reuse OOF predictions; parent dependence limits bispecific inference. |
| S3 headline and complete scope | `verified_revision/verify_primary.py`, `verified_revision/primary_ablations.py` | Same five positive-signal assays across the nine strategies, separate complete ten-assay view. |
| S4 historical baseline/scatter/distributions | supplied `main02_performance.pdf`, original scripts and retained tables | Archival configuration. Its original complete individual-prediction provenance is unavailable; preserved as supplied artwork and labeled accordingly. |
| S5 selection stability / expert refits | `scripts/reviewer_response_analyses.py`, `assay_revision/compute_uncertainty.py`, `revise_requested_bars.py` | Source-selection frequencies are retained summaries; Wilson intervals measure Monte Carlo precision. Primary expert coefficients can be refitted. |
| S6 architecture/routing/pooling | `scripts/abdevx2_ablation.py`, `scripts/sparse_moe_gdpa3.py`, `scripts/hybrid_ensemble.py`, `assay_revision/compute_uncertainty.py` | Fold-level architecture/pooling data retained. Certain routing points lack original replicate/prediction records; their original uncertainty is NR. |
| S7 transfer/ranking | `scripts/eval_heldout.py`, `scripts/eval_combined_training.py`, `scripts/rank_antibodies.py`, `assay_revision/extend_uncertainty.py` | Historical identities retained. Archived composite precision and some combined-training points lack recoverable uncertainty. |
| S8 augmentation/confidence | `scripts/exp_batch_correction.py`, `scripts/exp_calibration_abstention.py`, `scripts/exp_struct_uncertainty.py` | Numerical noise is a sensitivity test, not a conformational ensemble. Some confidence summaries have no retained replicates. |
| S9 learners and regions | `scripts/ablation_tabpfn.py`, `assay_revision/extend_uncertainty.py`, `revise_requested_bars.py` | Stored tabular points have no run-level uncertainty; regional descriptor predictions and BCa intervals are recoverable. Equivalent feature constructions are merged. |
| S10 matched DELPHI | `same_data_delphi/train_regression.py`, `train_classification.py`, `analyze.py` | Same labeled rows/folds/test antibodies. Regression is an explicit one-output adaptation; classification uses agreed common labels. Three fixed seeds retained. |
| S11 structure predictors | `structure_predictor_benchmark/run_igfold.py`, `run_esmfold.py`, `structure_features.py`, `evaluate_common.py` | Four sources, fixed common 20-feature panel, identical complete cases and folds. AF3 is single-sequence without MSA/templates. Conditional paired BCa; not the original bio57 pipeline. |

Unless prefixed with `scripts/`, script names in the table are relative to
`experiments/studies/`. Original figure creation code is retained; the final
requested bar styles are defined in `revise_requested_bars.py`.

## What is not promised

- A summary table or a digitized vector bar cannot reproduce a missing original
  model-training run. We do not manufacture missing error bars or observations.
- The public code alone does not supply gated measurements, original external
  weights, licensed feature software, or cached proprietary-source features.
- A feature-only comparison does not rank structural RMSD without experimental
  reference structures. There is no universal-best-predictor conclusion.
- Confidence intervals condition on fitted models and usually resample
  antibodies, not training datasets or independent antibody families.
- Expected primary metrics are frozen manuscript references. The validation
  scripts expose differences instead of replacing references with a new run.

All copied scientific Python source is syntax-checked. The primary model source
is byte-identical to the audited research checkout. Portability-only path changes
are listed in `reference/source_manifest.json`.
