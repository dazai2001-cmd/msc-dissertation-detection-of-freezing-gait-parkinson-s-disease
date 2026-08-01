# Freezing of Gait Dissertation Project

This repository contains the analysis, modelling code, datasets, and dissertation material for detecting freezing-of-gait events from wearable-sensor recordings.

The active modelling workflow uses subject-disjoint train, validation, and test splits. This prevents rows from the same person or recording appearing on both sides of an evaluation split.

## Project structure

```text
Dissertation/
├── notebooks/          Executed EDA and modelling notebooks
├── src/                Reusable leakage-safe data and feature pipeline
├── tests/              Automated checks for splits, features, and sequences
├── scripts/            Notebook generation utilities
├── data/
│   ├── raw/            Local competition datasets (ignored by Git)
│   ├── metadata/       Recording and subject metadata
│   └── submissions/    Submission-format reference data
├── docs/               Dissertation, design, planning, and reference material
├── figures/            Project diagrams and older exported figures
├── models/             Model records and older saved models
└── archive/            Historical project backup
```

## Main notebooks

- `notebooks/defog_dense_subject_generalisation.ipynb` — dense DeFOG baseline evaluated on unseen subjects.
- `notebooks/tdcs_dense_file_generalisation.ipynb` — TDCS baseline using `tdcsfog_metadata.csv` to evaluate unseen subjects; the legacy filename is retained for compatibility.
- `notebooks/defog_lstm_subject_generalisation.ipynb` — chronological DeFOG sequence model evaluated on unseen subjects.
- `notebooks/causal_tcn_subject_generalisation.ipynb` — separate causal TCN ensemble experiment with two-second context and subject-level outer folds.
- `notebooks/defog_model_comparison.ipynb` — fair five-fold MLP/LSTM/TCN comparison on DeFOG.
- `notebooks/tdcsfog_model_comparison.ipynb` — the same fair comparison protocol run separately on TDCS FoG.
- `notebooks/kaggle_fog_competition_submission.ipynb` — self-contained Kaggle GPU workflow that trains separate DeFOG/TDCS CNN–LSTM and TCN ensembles, validates with subject-disjoint mean average precision, and writes the required `submission.csv`.

Each notebook contains EDA, class-balance checks, subject split checks, training history, loss and accuracy plots, final train/validation/test metrics, and confusion matrices.

The TCN notebook defaults to `RUN_PROFILE = "pilot"` (one outer fold and one seed). Change it to `"final"` inside the notebook for the five-fold, three-seed dissertation estimate.

## Reproduce the analysis

From the repository root, create an environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name dissertation-fog --display-name "Dissertation FoG (.venv)"
```

Open a notebook and select **Dissertation FoG (.venv)** as its kernel. Do not use the generic global **Python 3** kernel: that environment may contain NumPy 2 while this TensorFlow and Matplotlib stack requires NumPy 1.26.4.

### GPU comparison environment (WSL2)

The two fair model-comparison notebooks use the separate
**Dissertation FoG GPU (WSL2)** kernel. The Windows CPU environment above remains
available for the historical notebooks.

Open the project through WSL before running a comparison notebook:

```powershell
wsl.exe -d Ubuntu
```

Then, inside Ubuntu:

```bash
cd /mnt/d/Dissertation
code .
```

Select **Dissertation FoG GPU (WSL2)** in the notebook. Verify the GPU from a
WSL terminal, including one synthetic batch through every benchmark model, with:

```bash
$HOME/.venvs/dissertation-gpu/bin/python scripts/verify_gpu.py
```

The reproducible WSL dependency overlay is `requirements-gpu.txt`. TensorFlow
2.17 cannot use CUDA from native Windows, so do not select the Windows `.venv`
for the GPU comparison notebooks.

To recreate the registered environment later:

```bash
$HOME/.local/bin/uv venv --python 3.11 $HOME/.venvs/dissertation-gpu
$HOME/.local/bin/uv pip install \
  --python $HOME/.venvs/dissertation-gpu/bin/python \
  -r /mnt/d/Dissertation/requirements-gpu.txt
$HOME/.venvs/dissertation-gpu/bin/python -m ipykernel install \
  --user \
  --name dissertation-fog-gpu \
  --display-name "Dissertation FoG GPU (WSL2)"
```

Run the automated leakage checks:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Rebuild the notebook source files when the shared template changes:

```powershell
.\.venv\Scripts\python scripts\build_notebooks.py
```

Rebuild only the standalone TCN experiment without touching the other notebooks:

```powershell
.\.venv\Scripts\python scripts\build_tcn_notebook.py
```

Rebuild only the two model-comparison notebooks without touching executed outputs
in the historical notebooks:

```powershell
.\.venv\Scripts\python scripts\build_model_comparison_notebooks.py
```

Rebuild the standalone, self-contained Kaggle competition notebook:

```powershell
.\.venv\Scripts\python scripts\build_competition_submission_notebook.py
```

The comparison notebooks write compact CSV tables to
`results/model_comparison/<dataset>/<profile>/` when you execute their final save cell.

Then execute a notebook from top to bottom in Jupyter. Paths are resolved from the repository root, so the notebooks work from their new `notebooks/` location.

### NumPy compatibility error

If a notebook reports that a module compiled for NumPy 1.x cannot run with NumPy 2.x, the wrong kernel is active. Restart the notebook with **Dissertation FoG (.venv)**. The first setup cell prints the selected Python executable and stops early if an incompatible NumPy major version is detected.

## Data and legacy artifacts

The raw datasets stay in `data/raw/` and are intentionally ignored because they are large. Metadata remains versionable in `data/metadata/`.

Files under `figures/legacy/` and `models/legacy/` predate the current subject-disjoint workflow. Keep them only for historical comparison; use the executed notebooks for the current leakage-safe results.
