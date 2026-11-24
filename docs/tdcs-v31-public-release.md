# TDCS experiments through v31

[Metadata suite (M00–M23)](../notebooks/tdcs_metadata_comprehensive_v26_public.ipynb) ·
[Sensor suite (S00–S12)](../notebooks/tdcs_sensor_comprehensive_v26_public.ipynb) ·
[Historical method scores](tdcs-v31/metrics.csv)

The strongest exploratory TDCS blend before the paired-crop comparison had
**0.450329 mAP**: StartHesitation **0.595277 AP**, Walking **0.305380 AP**.
This is pooled native-row out-of-fold average precision over 6,959,762 rows,
815 recordings, and 61 subjects on the same five frozen subject folds. All
negative recordings are included. The two-target mAP gives both targets equal
weight; Turn is not scored here. This is a development result selected on
reused folds, not a leaderboard or independent-test result. The 0.6 target
has not been reached and the retained models and previous DeFOG result are
unchanged.

The exploratory blend uses 75% of each retained head and 25% of the corresponding
ungated candidate: M21 for StartHesitation and M22 for Walking. M22's Walking
blend beats the previous exploratory Walking blend in four of five folds and
the retained Walking head in all five. M22 uses typed DeFOG labels plus weak
binary Event labels in warmup; weak positives are never given invented
subtypes. Both S10 (typed transfer) and M22 pass saved-model CPU replay.
[Aggregate results and replay limits](tdcs-v31/summary.json).

## Current matched comparison

M23 and S12 both start from the M22 warmup recipe. They use two targeted
fine-tuning crops per batch (Walking, then alternating Turn and
StartHesitation) and keep six natural crops. S12 additionally uses the S11
Walking ranking loss; M23 is its cross-entropy control. The [predeclared
comparison](tdcs-v31/plan.json) fixes the folds, training budget, and blend
weights. The [launch record](tdcs-v31/launch.json) is a point-in-time receipt,
not a live status or score.

## Reproduction

These public notebook copies preserve the pinned training code and labelled
experiment cells. **Run All defaults to synthetic inputs**, including on
Kaggle. Synthetic metrics and plots check wiring only. Use Python 3.11 with
[`requirements-tdcs.txt`](../requirements-tdcs.txt) plus XGBoost 3.2.0,
LightGBM 4.6.0, and PyArrow 20.0.0.

Full runs are opt-in (`TDCS_RUN_FULL=1`) and require a private Kaggle CUDA
session, the authorized competition input, the pinned feature-cache parent,
and the historical result artifacts named by the suite. Set
`TDCS_FROZEN_GATE_FILE` to the authorized private frozen-gate JSON before
running; its canonical JSON SHA-256 must match the notebook's pin. Full mode
loads prior test results and fits only the new experiment in each suite.
No dataset files, per-record gate scores, model checkpoints, or out-of-fold
predictions are included in this release.
