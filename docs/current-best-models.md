# Current best FoG artifacts

This index identifies the saved best-result snapshots, supporting code, and the
next untrained experiment. It does not claim a competition win or a new score.

## Measured TDCS results

| Snapshot | Targets | Pooled native-row development mAP | Notebook |
| --- | --- | ---: | --- |
| Retained neural / multiscale / dense-context blend | StartHesitation, Turn, Walking | 0.3876756908850407 | [Dense context](../notebooks/tdcs_dense_context_experiment.ipynb) |
| Target-specific ensemble | StartHesitation, Walking | 0.19629988455476777 | [Two-target ensemble](../notebooks/tdcs_start_hesitation_walking_ensemble.ipynb) |

The records are [tdcs_current_best.json](../results/tdcs_current_best.json) and
[tdcs_two_target_current_best.json](../results/tdcs_two_target_current_best.json).
Both use 815 recordings, 61 subjects, 6,959,762 native rows, and the same five
subject-disjoint folds. AP is pooled over native rows for each target, then
averaged across targets. Three-target and two-target mAP are not directly
comparable. Repeated development-fold selection is exploratory; independent
generalization has not been established.

The two-target ensemble has StartHesitation AP 0.20870336814353063 and Walking AP
0.18389640096600487. Its scores are independent and uncalibrated, not mutually
exclusive probabilities or clinical certainty. Turn-only and no-FOG rows remain
negative examples. Removing the Turn output does not remove those rows.

Small configuration, metric, and independent-validation receipts are committed
under the corresponding fingerprinted results directories. Their original
Windows paths are provenance, not a promise that large assets ship in this repo.

The previously reported DeFOG result of approximately 0.817 is preserved
separately and was not retrained or independently recomputed for this push.
The historical [evidence-weighted Kaggle pipeline](../notebooks/kaggle_fog_evidence_weighted_winning_ensemble.ipynb)
is included as source, not proof of a winning leaderboard submission.

## Prepared next experiment: not a measured best

[tdcs_two_target_neural_training.ipynb](../notebooks/tdcs_two_target_neural_training.ipynb)
trains fresh two-output BiGRU and gated-BiGRU models on fixed subject folds.
It uses 24 fixed epochs per model, EMA weights, short training windows, longer
inference windows, and a predeclared equal neural blend.

The 0.6 two-target mAP requirement remains a target, not an achieved result.
The last recorded launch attempt on 2026-10-07 was rejected by Kaggle for its
weekly GPU quota; see [the launch receipt](../results/tdcs_two_target_neural_launch.json).
This GitHub push does not retry training, submit to the competition, or check
the current Kaggle quota.

The local saved notebook ran only its explicitly labelled synthetic CPU wiring
test, plus a real-data fold-balance audit. It did not train the full TDCS neural
experiment. Synthetic results must not be treated as TDCS performance.

## Run or inspect

The completed TDCS notebooks retain their saved metric tables and plots.
The historical Kaggle pipeline and staged fresh GPU notebook have no saved
full-training outputs. Notebook structure and code syntax were checked for this
publication. The TDCS unit suite passed 42 tests. No expensive experiment was
rerun for the push, and full saved-notebook layout previews have not been checked.

A tested local TDCS package snapshot is in [requirements-tdcs.txt](../requirements-tdcs.txt);
it is separate from the existing TensorFlow environment. From the repository root:

```powershell
python -m pip install -r requirements-tdcs.txt
python -m pytest tests/test_tdcs_candidate_sweep.py tests/test_tdcs_dense_context.py tests/test_tdcs_two_target.py tests/test_tdcs_two_target_neural.py -q
python scripts/tdcs_two_target_neural.py --smoke --output-root tmp/tdcs_neural_smoke
```

The last command is a synthetic wiring check, not a model-quality experiment.
Full training requires a CUDA-enabled PyTorch environment. The notebook fails
explicitly if full training is requested without a GPU.

For a standalone Kaggle run, use the notebook and metadata in
[kaggle-tdcs-two-target-sequence](../kaggle-tdcs-two-target-sequence/).
Attach the Parkinson's Freezing of Gait competition data. The saved metadata
requests the T4 configuration and disables internet. With an authenticated
Kaggle CLI and available GPU quota:

```powershell
python -m kaggle kernels push -p kaggle-tdcs-two-target-sequence -t 25200 --accelerator NvidiaTeslaT4
```

After a real training run completes, download its full fingerprinted output
directory and restore the public training data locally. Independently validate:

```powershell
python scripts/validate_tdcs_two_target_neural.py --run-dir <downloaded-run-directory>
```

Local public data belongs in `data/raw/train/tdcsfog/`; metadata belongs in
`data/metadata/tdcsfog_metadata.csv`. The Kaggle notebook can locate nested
competition input mounts automatically.

Replaying the retained dense/two-target blends additionally requires the
historical neural OOF files under
`kaggle-output/fog-evidence-weighted/competition/neural/tdcsfog/`, and the
multiscale/dense feature caches and trained models. Those assets are not in Git.
The relevant scripts are `tdcs_multiscale_signal_experiment.py`,
`tdcs_candidate_sweep.py`, `tdcs_dense_context_experiment.py`, and
`tdcs_two_target_ensemble.py` under `scripts/`. Existing artifacts must be restored
with valid paths, or regenerated and independently validated before replaying
the saved blend.

## Deliberately excluded

Raw competition data, subject-level prediction dumps, feature caches, model
weights, downloads, API credentials, and temporary files remain local. The
ignore rules protect these outputs; only selected compact aggregate reports
and receipts are explicitly versioned. Unrelated pre-FOG work and existing
local README/requirements edits are not part of this publication.
