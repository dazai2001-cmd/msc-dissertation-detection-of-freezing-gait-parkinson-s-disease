"""Independently validate two-target native-row OOF scores and confidence losses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_tdcs_experiment import independent_ap

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('StartHesitation', 'Walking')
EXPECTED_FOLD_POSITIVES = {0: [239789, 143269], 1: [14606, 31959], 2: [19549, 15904],
                           3: [23918, 4729], 4: [6928, 11654]}


def validate(run_dir: Path) -> dict:
    decision = json.loads((run_dir / 'decision.json').read_text())
    configuration = json.loads((run_dir / 'configuration.json').read_text())
    selected = Path(decision['selected_candidate_dir'])
    manifest = pd.read_csv(selected / 'frozen_tdcs_manifest.csv', dtype={'recording_id': str, 'subject': str})
    metadata = pd.read_csv(ROOT / 'data/metadata/tdcsfog_metadata.csv', dtype={'Id': str, 'Subject': str})
    joined = manifest.merge(metadata[['Id', 'Subject']], left_on='recording_id', right_on='Id', validate='one_to_one')
    assert len(joined) == len(manifest) == 815 and manifest.subject.nunique() == 61
    assert (joined.subject == joined.Subject).all()
    assert joined.groupby('subject').fold.nunique().max() == 1
    assert '4dc2f8' not in set(joined.subject) and set(manifest.fold) == set(range(5))
    assert {p.stem for p in (selected / 'oof').glob('*.npz')} == set(manifest.recording_id)
    assert configuration['targets'] == list(TARGETS)
    assert configuration['turn_specialist_loaded'] is False
    assert configuration['turn_only_rows_dropped'] is False and configuration['calibrator_fit'] is False
    assert configuration['source_files']['three_class_registry'] == hashlib.sha256((ROOT / 'results/tdcs_current_best.json').read_bytes()).hexdigest()
    parts_y, parts_p, folds = [], [], []
    positives = {fold: np.zeros(2, np.int64) for fold in range(5)}
    for row in manifest.itertuples(index=False):
        # Load Turn only for this diagnostic: ensure Turn-only rows were retained,
        # not used as features or excluded from fitting/scoring the two targets.
        frame = pd.read_csv(row.path, usecols=['StartHesitation', 'Turn', 'Walking'])
        y = frame.loc[:, list(TARGETS)].to_numpy()
        with np.load(selected / 'oof' / f'{row.recording_id}.npz', allow_pickle=False) as payload:
            p = payload['prediction']
            assert payload['targets'].tolist() == list(TARGETS)
            assert payload['confidence_status'].item() == 'uncalibrated'
        assert y.shape == p.shape and p.shape[1] == 2
        assert np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        positives[row.fold] += y.sum(axis=0, dtype=np.int64)
        parts_y.append(y.astype(np.uint8))
        parts_p.append(p)
        folds.append(np.full(len(y), row.fold, np.int8))
    y, p = np.concatenate(parts_y), np.concatenate(parts_p)
    assert len(y) == 6959762
    assert {fold: v.tolist() for fold, v in positives.items()} == EXPECTED_FOLD_POSITIVES
    independent = {target: independent_ap(y[:, i], p[:, i]) for i, target in enumerate(TARGETS)}
    independent['two_target_mAP'] = float(np.mean(list(independent.values())))
    reported = pd.read_csv(selected / 'native_oof_metrics.csv').iloc[0]
    for key, value in independent.items():
        assert abs(value - reported[key]) < 1e-10, (key, value, reported[key])
    confidence = pd.read_csv(run_dir / 'confidence_metrics.csv')
    confidence = confidence.loc[confidence.method.eq(decision['selected_method'])].set_index('target')
    bins = pd.read_csv(run_dir / 'reliability_bins.csv')
    verified_losses = {}
    for i, target in enumerate(TARGETS):
        target_y, target_p = y[:, i].astype(np.float64), p[:, i].astype(np.float64)
        clipped = np.clip(target_p, 1e-7, 1 - 1e-7)
        brier = float(np.mean((target_p - target_y) ** 2))
        log_loss = float(-np.mean(target_y * np.log(clipped) + (1 - target_y) * np.log1p(-clipped)))
        assert abs(brier - confidence.loc[target, 'brier_loss']) < 1e-12
        assert abs(log_loss - confidence.loc[target, 'log_loss']) < 1e-12
        selected_bins = bins.loc[(bins.method == decision['selected_method']) & (bins.target == target)]
        index = np.minimum(np.floor(target_p * 10).astype(np.int64), 9)
        counts = np.bincount(index, minlength=10)
        positive_counts = np.bincount(index, weights=target_y, minlength=10)
        assert np.array_equal(counts, selected_bins.native_rows.to_numpy())
        assert np.array_equal(positive_counts, selected_bins.positive_rows.to_numpy())
        assert counts.sum() == len(y)
        verified_losses[target] = {'brier_loss': brier, 'log_loss': log_loss,
                                   'native_positive_rows': int(target_y.sum())}
    receipt = {
        'status': 'passed', 'native_rows': len(y), 'recordings': len(manifest), 'subjects': int(manifest.subject.nunique()),
        'targets': TARGETS, 'subject_disjoint': True, 'prediction_ids_exact_match': True,
        'independent_rank_based_AP': independent, 'confidence_losses': verified_losses,
        'fold_positive_rows': {fold: dict(zip(TARGETS, values.tolist())) for fold, values in positives.items()},
        'all_native_rows_retained': True, 'turn_output_removed': True, 'old_three_class_registry_unchanged': True,
        'calibration_bin_counts_verified': True, 'confidence_is_calibrated': False,
        'independent_generalization_established': False,
    }
    (selected / 'validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or Path(json.loads((ROOT / 'results/tdcs_two_target/latest_run.json').read_text())['run_dir'])
    validate(run_dir)
