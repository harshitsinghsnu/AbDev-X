# Setup notes

Use Python 3.11 and install the single project requirements file:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py run all
```

This reproduces the supported numerical workflow using the supplied datasets,
cached sequence representations, and structural feature tables. A new GitHub
clone needs those inputs restored first; see `DATA_ACCESS.md`. The same code
runs on CPU, although transformer training benefits from a GPU. Detailed
versions from the working environment are in `environment_record.json`.

## Fresh structure prediction

The numerical comparison can be rerun from its fixed common-feature tables
without reinstalling structure predictors. To predict structures again, install
IgFold, ESMFold, and AF3 in environments supported by their own implementations.
AF3 uses a separate Linux/CUDA/JAX environment and model parameters obtained
under its terms. These applications and large model weights are not installed
by the project requirements file.

The executed settings are recorded in `docs/RESULT_PROVENANCE.md` and the
structure-comparison scripts. Prepare paired inputs using:

```powershell
.\.venv\Scripts\python.exe tools/prepare_structure_inputs.py
```

Then use the predictor scripts in `experiments/studies/structure_predictor_benchmark/`.
The portable AF3 entry point is `tools/run_af3.py`; use it inside the configured
AF3 environment with `--source` and `--weights`. Missing-atom completion and
feature extraction require OpenMM, PDBFixer, and FreeSASA. See the official
installation instructions for those applications before regenerating the
structural features. The cached-feature numerical workflow does not require
these installations.

## Cartoon rendering

Install UCSF ChimeraX separately to recreate the structural attribution images:

```powershell
ChimeraX --script experiments/studies/render_chimerax_clean.py
```

The script uses the existing signed residue attribution scores, without
changing predictions. Numerical attribution calculations are available through
`python reproduce.py run attributions`. Plots go to `outputs/figures/`.
