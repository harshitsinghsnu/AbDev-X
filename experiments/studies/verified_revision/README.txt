AbDev-X verified manuscript and figure sources

The combined PDF is AbDevX_verified_main_SI.pdf. The manuscript source uses PDF figures in figures/. All tables are supplementary. SCIENCE_AUDIT.md and evidence/figure_provenance.csv explain model identities and numerical corrections.

To compile only: run pdflatex twice on AbDevX_verified_main_SI.tex from this folder. SVGs retain editable figure lettering. Molecular structures are raster insets rendered from real PDB coordinates, not generated artwork.

To recompute against the original project, place this directory at ab-dev/experiments/studies/verified_revision, run commands with ab-dev as the working directory, and preserve original data, cached embeddings, external feature tables and scripts. Required packages are the project's scientific Python dependencies plus Biopython, matplotlib, scipy, and threadpoolctl.

Order:
1. python experiments/studies/verified_revision/verify_primary.py
2. python experiments/studies/verified_revision/primary_ablations.py
3. python experiments/studies/verified_revision/primary_residue_shap.py
4. python experiments/studies/verified_revision/build_figures.py
5. python experiments/studies/verified_revision/extra_figures.py
6. python experiments/studies/verified_revision/build_manuscript.py
7. Compile twice from this folder.
8. python experiments/studies/verified_revision/audit_and_package.py

The fitted-model pickle is large and intentionally excluded from this bundle. Step 1 recreates it from the original project. Original large embedding caches and benchmark datasets are not duplicated here. The evidence CSVs provide per-antibody predictions, all fold records, direct primary attributions, and source hashes.
