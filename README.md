# AbDev-X

Model, baseline, evaluation, and reproduction code. No manuscript is included.

## Install

Use Python 3.11:

```powershell
git clone https://github.com/harshitsinghsnu/AbDev-X.git
cd AbDev-X
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data

Obtain GDPa datasets from https://datapoints.ginkgo.bio/dataset-access and the
benchmark features from https://github.com/ginkgobioworks/abdev-benchmark.
`data/input_manifest.json` lists the expected input paths and checksums.
Cached result tables needed for historical comparisons are listed in
`data/artifact_manifest.json`. Restore an authorized input/result tree with
`python tools/restore_inputs.py --from PATH`.

Data, model weights, local caches, and generated outputs are excluded from Git.
Some historical comparisons retain summary results rather than complete
training records; their original missing records cannot be regenerated from
the summaries. Fresh structure inference and ChimeraX rendering require those
applications installed separately.

## Reproduce

```powershell
.\.venv\Scripts\python.exe reproduce.py list
.\.venv\Scripts\python.exe reproduce.py run primary
.\.venv\Scripts\python.exe reproduce.py run all
```

`src/` contains reusable model utilities, `scripts/` contains research modules,
`experiments/` contains the experiment implementations, and `runners/`
groups the execution and plotting commands. `baselines/delphi/` retains the
comparator code and its original license. Execution logs and plots are saved under `outputs/`.
