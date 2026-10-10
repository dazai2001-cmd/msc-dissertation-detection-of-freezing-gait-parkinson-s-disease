"""Validate downloaded real neural OOF, resolving cloud paths to local raw data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tdcs_two_target_neural import TARGETS, build_manifest, locate_data
from validate_tdcs_experiment import independent_ap


def validate(run_dir: Path):
    decision = json.loads((run_dir / 'decision.json').read_text())
    if decision['smoke_only'] or decision['status'] != 'complete':
        raise ValueError('Only completed REAL TDCS experiments can be validated or promoted')
    configuration = json.loads((run_dir / 'configuration.json').read_text())['configuration']
    assert configuration['targets'] == list(TARGETS) and configuration['folds'] == list(range(5))
    selected = Path(run_dir / 'selected_candidate')
    saved = pd.read_csv(selected / 'frozen_tdcs_manifest.csv', dtype={'recording_id': str, 'subject': str})
    raw_root, metadata = locate_data()
    local = build_manifest(raw_root, metadata)
    columns = ['recording_id', 'subject', 'fold', 'source_bytes']
    pd.testing.assert_frame_equal(saved[columns].reset_index(drop=True), local[columns].reset_index(drop=True), check_dtype=False)
    assert {p.stem for p in (selected / 'oof').glob('*.npz')} == set(local.recording_id)
    for variant in configuration['variants']:
        for fold in range(5):
            receipt = json.loads((run_dir / variant / f'fold_{fold}' / 'training_receipt.json').read_text())
            train_subjects = set(local.loc[local.fold.ne(fold), 'subject'])
            test_subjects = set(local.loc[local.fold.eq(fold), 'subject'])
            assert set(receipt['training_subjects']) == train_subjects
            assert set(receipt['held_out_subjects']) == test_subjects
            assert train_subjects.isdisjoint(test_subjects)
            assert receipt['targets'] == list(TARGETS)
            assert receipt['outer_labels_used_for_checkpoint_selection'] is False
            assert receipt['fixed_epochs'] == configuration['epochs']
    label_parts, score_parts = [], []
    counts = {fold: np.zeros(2, np.int64) for fold in range(5)}
    for row in local.itertuples(index=False):
        labels = pd.read_csv(row.path, usecols=list(TARGETS)).loc[:, list(TARGETS)].to_numpy()
        with np.load(selected / 'oof' / f'{row.recording_id}.npz', allow_pickle=False) as payload:
            scores = payload['prediction']
            assert payload['targets'].tolist() == list(TARGETS)
        assert labels.shape == scores.shape and labels.shape[1] == 2
        assert np.isin(labels, [0, 1]).all()
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        counts[row.fold] += labels.sum(axis=0, dtype=np.int64)
        label_parts.append(labels.astype(np.uint8))
        score_parts.append(scores)
    y, p = np.concatenate(label_parts), np.concatenate(score_parts)
    assert len(y) == 6959762
    independent = {target: independent_ap(y[:, i], p[:, i]) for i, target in enumerate(TARGETS)}
    independent['two_target_mAP'] = float(np.mean(list(independent.values())))
    reported = pd.read_csv(selected / 'native_oof_metrics.csv').iloc[0]
    for metric, value in independent.items():
        assert abs(value - reported[metric]) < 1e-10, (metric, value, reported[metric])
    validation = {
        'status': 'passed', 'native_rows': len(y), 'recordings': len(local), 'subjects': int(local.subject.nunique()),
        'targets': TARGETS, 'subject_disjoint': True, 'prediction_ids_exact_match': True,
        'all_negative_native_rows_retained': True, 'outer_labels_not_used_for_checkpoints': True,
        'independent_rank_based_AP': independent,
        'fold_positive_rows': {fold: dict(zip(TARGETS, values.tolist())) for fold, values in counts.items()},
        'goal_two_target_mAP': 0.6, 'goal_reached': independent['two_target_mAP'] >= 0.6,
        'confidence_calibrated': False, 'independent_generalization_established': False,
        'competition_submission_created': False,
    }
    (selected / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps(validation, indent=2), flush=True)
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Downloaded fingerprint directory containing decision.json')
    validate(parser.parse_args().run_dir)
