# Validation performed on the assembled folder

Recorded 14 November 2026.

| Check | Result |
|---|---|
| Primary model refitted using this folder's copies of the representations and labels | Completed |
| Ten primary/default-extension external correlations compared with frozen manuscript references | Matched |
| Primary assay BCa confidence limits after the refit | Matched exactly in the recorded environment |
| Matched DELPHI statistics regenerated from retained seed-specific predictions | Completed; seed-42 BCa limits matched exactly |
| Numeric validation from individual predictions, folds and attribution completeness | 172 checks passed |
| Existing CPU descriptor and synthetic pipeline tests | 5 passed |
| Figure regeneration from the assembled folder | Completed, including current bar-style panels and ChimeraX-attribution assembly |
| Main + SI manuscript rebuilt from regenerated figures | Completed |
| Input checksums | 903 declared input files present and matching |
| Python source parsing/integrity | Published Python source files checked, including the baseline implementation |
| Code release audit | No files exceeding the configured size threshold, no common credential patterns, no unexpected raw-input/model-cache files |
| Commit or push | Neither performed |

The audit does not claim that the entire historical research pipeline was
retrained from raw data. In particular, the original encoder caches and cached
bio57 feature source were reused, predictor inference was not repeated, native
pretrained DELPHI checkpoints were not re-downloaded, and original experiments
behind summary-only historical results cannot be reconstructed. Those boundaries
are described in `RESULT_PROVENANCE.md` and `DATA_ACCESS.md`.

Machine-readable reports and logs are under `outputs/validation/` and
`outputs/run_logs/` in the supplied local folder. These are excluded from Git
because they can contain local paths or derived observations. Expected aggregate
metrics, source checksums, and input/artifact inventories are in `reference/`.
