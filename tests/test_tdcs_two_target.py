"""Two-target semantics, independent scores, and confidence-bin boundaries."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import tdcs_two_target_ensemble as experiment


def members(values):
    return {name: np.asarray(p, np.float32) for name, p in zip(experiment.MEMBERS, values)}


def test_exactly_two_targets_and_no_turn_specialist():
    assert experiment.TARGETS == ('StartHesitation', 'Walking')
    assert experiment.SOURCE_INDICES == (0, 2)
    assert 'tdcs_turn_specialist' not in experiment.MEMBERS
    for weights in experiment.FIXED_CANDIDATES.values():
        assert set(weights) == set(experiment.TARGETS)
        for target in experiment.TARGETS:
            assert sum(weights[target]) == pytest.approx(1)


def test_weights_are_target_specific_not_softmax():
    p = members([[[0.1, 0.3]], [[0.2, 0.4]], [[0.9, 0.8]]])
    result = experiment.combine_members(p, experiment.FIXED_CANDIDATES['hesitation_dense_90_walking_retained'])
    np.testing.assert_allclose(result, [[0.825, 0.575]], atol=1e-7)
    assert result.sum() > 1  # Independent scores are deliberately not normalized together.


def test_both_scores_can_be_low_without_a_turn_output():
    p = members([[[0.01, 0.02]]] * 3)
    result = experiment.combine_members(p, experiment.FIXED_CANDIDATES['current_model_two_target_reference'])
    np.testing.assert_allclose(result, [[0.01, 0.02]])


@pytest.mark.parametrize('scores', [np.zeros((3, 3)), [[np.nan, 0.2]], [[-0.1, 0.2]], [[0.2, 1.1]]])
def test_invalid_confidences_are_rejected(scores):
    with pytest.raises(ValueError):
        experiment.check_scores(scores)


def test_native_alignment_required():
    p = members([np.zeros((3, 2)), np.zeros((2, 2)), np.zeros((3, 2))])
    with pytest.raises(ValueError, match='row counts'):
        experiment.combine_members(p, experiment.FIXED_CANDIDATES['current_model_two_target_reference'])


def test_bad_weights_are_rejected():
    p = members([np.zeros((2, 2))] * 3)
    with pytest.raises(ValueError, match='summing to one'):
        experiment.combine_members(p, {'StartHesitation': [1, 1, 1], 'Walking': [0, 0, 1]})


def test_zero_and_one_scores_are_included_in_bins():
    bins = experiment.reliability_bins(np.array([0, 1, 0, 1]), np.array([0.0, 0.1, 0.9, 1.0]))
    assert bins.native_rows.sum() == 4 and bins.positive_rows.sum() == 2
    assert bins.iloc[0].native_rows == 1 and bins.iloc[-1].native_rows == 2
    assert bins.iloc[-1].observed_fraction == 0.5
    assert np.isnan(bins.iloc[2].observed_fraction)  # Empty bins are unknown, not zero.


def test_two_target_map_does_not_count_turn_or_remove_negative_rows():
    y = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], np.uint8)
    p = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.8, 0.8]])
    scores = experiment.score_targets(y, p)
    assert set(scores) == {'StartHesitation', 'Walking', 'two_target_mAP'}
    assert scores['two_target_mAP'] == 1


def test_confidence_metrics_and_volume_reconcile():
    y = np.array([[0, 0], [1, 1]], np.uint8)
    p = np.array([[0.2, 0.4], [0.8, 0.6]])
    rows, bins = experiment.confidence_metrics(y, p, 'test')
    assert rows[0]['brier_loss'] == pytest.approx(0.04)
    assert rows[1]['brier_loss'] == pytest.approx(0.16)
    assert rows[0]['ece_10_uniform_bins'] == pytest.approx(0.2)
    assert all(table.native_rows.sum() == 2 for table in bins)
    assert all(row['confidence_status'].startswith('uncalibrated') for row in rows)
