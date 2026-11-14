# Inputs and release boundaries

1. GDPa datasets: https://datapoints.ginkgo.bio/dataset-access . The development
   CSV used here is GDPa1 v1.2 (246 antibody rows). The external GDPa3 panel has
   80 antibodies, with assay-specific missing measurements. GDPa2.1, GDPa4,
   and GDPa5 are used only in the stated transfer analyses. Acquire the releases
   used here under their provider's terms; do not replace them with a later
   release selected automatically because it contains more rows.
2. Benchmark features and structures:
   https://github.com/ginkgobioworks/abdev-benchmark . The expected directory is
   `external/abdev-benchmark/`. Exact relative input paths and file hashes are in
   `reference/input_manifest.json`.
3. DELPHI: https://github.com/proteininnovation/delphi . The comparison scripts
   use the frozen MIT-licensed implementation in `baselines/delphi/`, with
   its original model/configuration modules and upstream license. Native
   frozen-checkpoint evaluation additionally requires the original pretrained
   model release and matched embeddings. Matched GDPa1 training does not use
   proprietary IPI training examples or those pretrained checkpoint weights.
4. Cached representations in `data/processed/` include sequence embeddings,
   processed assay tables, and region representations. Regeneration code is in
   `scripts/extract_*.py`, `src/abdev/features/`, and the workflow map. Exact
   model revisions, pooling conventions, row identities, and fold assignments
   must match. They are not interchangeable with arbitrary new embeddings.

This local workspace includes working inputs for verification. Those directories
are excluded by `.gitignore`; preparing this folder does not authorize or cause
public redistribution. Repository users must obtain the data or an authorized
artifact release. Input manifests identify expected files but do not embed
antibody sequences or labels. Use `python tools/check_inputs.py` for file/hash
validation after restoring the required input tree.

`reference/artifact_manifest.json` additionally inventories the retained result
tables, fitted-cache files, and historical plot inputs needed for the cached
artifact reproduction route. Restore an authorized full workspace/artifact tree
with `python tools/restore_inputs.py --from /path/to/authorized/tree`. This copies
only declared paths and validates their checksums. An artifact release must be
published separately when redistribution is allowed; the code-only ZIP does not
contain those Git-ignored artifacts.

The reproducibility source package has no credentials, virtual environments,
large downloaded predictor parameters, or proprietary MOE installation. The
open structural-feature code cannot regenerate licensed MOE measurements or
prove that all columns of the original merged bio57 cache came from one
ABodyBuilder3 workflow. The manuscript preserves that distinction.
