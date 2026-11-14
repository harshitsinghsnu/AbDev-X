# AbDev-X

Code to reproduce the AbDev-X experiments, numerical results, and plots.
The manuscript is maintained separately and is not included in this repository.

## For users: install and reproduce

Use Python 3.11. In PowerShell:

```powershell
git clone https://github.com/harshitsinghsnu/AbDev-X.git
cd AbDev-X
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Obtain the datasets and required representations using
[Data access](docs/DATA_ACCESS.md). Input data, representation caches, and large
model parameters are not included in the code repository. The numerical
workflow requires these resources; installing the Python packages alone does
not supply them.

## Run

```powershell
# Refit the primary model.
.\.venv\Scripts\python.exe reproduce.py run primary

# Check the saved results against the reference values.
.\.venv\Scripts\python.exe reproduce.py run verify

# Reproduce the complete supported numerical workflow and plots.
.\.venv\Scripts\python.exe reproduce.py run all
```

Use `python reproduce.py list` to see individual experiments. Logs and plots
are written to `outputs/`. Full training and attribution calculations take
longer than checking saved predictions. Fresh structure prediction and cartoon
rendering require their own installed applications; see
[Environment notes](docs/ENVIRONMENTS.md).

## Code layout

| Folder | Purpose |
|---|---|
| `src/abdev/` | Data loading, representations, descriptors, and model utilities |
| `workflows/` | Commands grouped by experiment |
| `scripts/` | Research model and evaluation modules |
| `experiments/studies/` | Experiment implementations and plot functions |
| `baselines/delphi/` | DELPHI comparator implementation and original license |
| `configs/` | Data and primary-model settings |
| `reference/` | Expected metrics and reproducibility manifests |
| `tools/` | Input and result checks |
| `docs/` | Data access and result-to-code map |

The primary model averages four assay-specific experts with equal weights.
The open structural-feature comparison is a separate variant. Results that
retain only historical summary tables are identified in
[Result provenance](docs/RESULT_PROVENANCE.md); missing training records or
error bars are not reconstructed from those summaries.

## For the repository owner: publish the code

Publishing this prepared folder does not require installing the environment
or running any experiments on the publisher's computer.

The remote is already configured as
`https://github.com/harshitsinghsnu/AbDev-X.git`.

```powershell
git add .
git status --short
git commit -m "Add AbDev-X code and reproduction workflows"
git push -u origin main
```

Review the files shown by `git status` before committing. Input data, learned
weights, local environments, generated outputs, and manuscript documents are
excluded by `.gitignore`. No commit or push is performed by the reproduction
commands.

Before announcing complete reproducibility, make the required input resources
accessible to readers, either through their documented original sources or an
authorized data release. A code push alone does not publish the local caches.
Historical results with unavailable original training records remain identified
in the result-to-code map.
