# Running structure-predictor benchmark

The user authorized installing and running all four named structure predictors.
The full cohort is 246 GDPa1 plus 80 GDPa3 paired VH/VL antibodies. The frozen
cluster-isotype fold assignments and exact sequences are in
`paired_sequences_and_folds.csv`.

## Current execution

Run `C:\Users\hs494\ab-dev\.structure-env\Scripts\python.exe status.py` from this
directory for live counts. `execution_status.json` records the persistent queue
PID, stage status, and commands. `queue.log` and individual stage logs retain
execution evidence. A running stage is not a completed scientific result.

- ABodyBuilder3: reuse the supplied GDPa1/held-out structures, with exact-chain
  sequence checks. Two training structures are absent from the supplied cache.
- IgFold 1.0.1 / AntiBERTy 1.0.0: four-model ensemble, CUDA, optional refinement
  and renumbering disabled. All 326 raw predictions completed successfully.
- ESMFold: official v1 weights at Hugging Face revision
  `75a3841ee059df2bf4d56688166c8fb459ddd97a`; Transformers 4.51 implementation;
  four recycles; trunk chunk 32; sequence LM fp16 and folding components fp32.
  VH + 25 glycine linker + VL, light residue index offset 512. Linker atoms are
  excluded. Paired-chain pilot passed; the full run resumes cached predictions.
- AlphaFold3: official v3.0.1 source commit
  `231efc9bb9c13b45cc59e43f7107869084ee9624`, JAX 0.4.34 in isolated WSL Ubuntu
  24.04 environment. Seed 42, default 10 recycles, five diffusion samples,
  official top-ranking-score sample, xla attention, buckets 256/384. Paired chains
  use empty MSAs and no templates. This is explicitly a single-sequence AF3
  condition. The pilot succeeded. Latest official source/JAX 0.10.2 installed
  but failed local GPU allocation; the older official runtime works with explicit
  CUDA library/compiler paths. The downloaded official Google weights are kept
  locally; the installation does not change the primary Windows environment.
- MOE structures: an additional cached source, not a substitute for AF3.

The hidden persistent worker runs AF3 then ESMFold sequentially on the RTX A4000
16GB GPU, extracts common features, then fits the matched downstream experiments.
No main-paper numbers have been replaced by unfinished benchmark results.

## Controlled comparison and scope

`structure_features.py` defines the same 20 open features for every structure
source: SASA, exposed hydrophobic/aromatic/charged area, local surface patches,
radius of gyration, and VH/VL interface geometry. Every chain is matched by exact
sequence. PDBFixer 1.12 / OpenMM 8.6.1 adds missing heavy atoms with seed 42;
hydrogens and terminal OXT atoms are excluded; already complete heavy atoms are
not relaxed. IgFold's raw backbone/CB structures require side-chain completion.
All feature extraction uses the same policy and FreeSASA definitions.

The original 57-feature bio cache combines TAP, MOE, Aggrescan3D, and AntiFold.
Its complete regeneration pipeline is unavailable locally. This experiment is
therefore named **AbDev-X/common-structure20**, with the other four blocks and
the four original expert classes unchanged. It must not be described as an exact
swap of the original primary bio57 pipeline.

`evaluate_common.py` intersects successful structures across all evaluated
predictors, uses identical labeled rows and original outer folds, and repeats
training-only source/head selection inside each outer training partition.
Missing structures are not imputed as zeros. All ten assays are included.
CV mean and fold SD, held-out rho and paired antibody BCa intervals (20,000
replicates), and multiplicity-adjusted interval sensitivity are saved. GDPa3 has
already been inspected in this project: this comparison is retrospective.

Completed analysis files appear under `comparison_<predictor names>/`:
`scores.csv`, `cv_fold_scores.csv`, `heldout_predictions.csv`,
`paired_differences.csv`, `manifest.json`, `RESULTS.md`, and a legible vector PDF
comparison figure. Scientific conclusions should be written only after these
files exist and the stage reports success. Feature-group ablation results are
not evidence that one structure predictor is best.

There are no experimental reference structures matched here, so inter-predictor
agreement is not reported as structural accuracy. Cached baseline inference
runtime is unavailable. Predictor confidence values have different definitions
and must not be ranked against each other.
