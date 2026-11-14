# Primary AbDev-X architecture and results audit

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
| HIC | 0.547 | 0.546965 | 79 | Primary |
| AC-SINS | 0.262 | 0.261909 | 80 | Primary |
| PR_CHO | 0.251 | 0.251428 | 80 | Primary |
| SMAC | 0.647 | 0.647238 | 80 | Primary |
| HAC | 0.650 | 0.650132 | 61 | Primary |
| Tm1 | 0.344 | 0.200445 | 80 | Default extension |
| Tm2 | 0.160 | -0.107428 | 74 | Default extension |
| Tonset | 0.211 | 0.132302 | 80 | Default extension |
| SEC | -0.171 | -0.228650 | 80 | Default extension |
| Titer | 0.075 | 0.133380 | 80 | Default extension |

Same-five primary mean: **0.471534**. Ten-assay default-extension mean: **0.248772**, replacing the mixed stored mean of 0.298.

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
