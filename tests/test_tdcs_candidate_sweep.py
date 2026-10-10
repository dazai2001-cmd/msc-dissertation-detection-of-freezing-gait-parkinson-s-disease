"""Fast safety tests for fixed TDCS candidate blending and held-out fitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tdcs_candidate_sweep as sweep
from validate_tdcs_experiment import independent_ap


def test_fixed_blends_preserve_target_order_and_normalization():
    predictions = {
        "previous_tdcs_blend": np.full((5, 3), 0.2, np.float32),
        "multiscale_sensor_hgb": np.full((5, 3), 0.6, np.float32),
        "multiscale_catboost": np.full((5, 3), 0.8, np.float32),
    }
    np.testing.assert_allclose(sweep.blended_prediction("neural_hgb_equal", predictions), 0.4)
    np.testing.assert_allclose(sweep.blended_prediction("hgb_catboost_equal", predictions), 0.7)
    np.testing.assert_allclose(sweep.blended_prediction("neural_sensor_three_way", predictions), 0.45)
    assert sweep.TARGETS == ("StartHesitation", "Turn", "Walking")
    assert all(abs(sum(weights.values()) - 1) < 1e-12 for weights in sweep.FIXED_BLENDS.values())
    assert np.allclose(np.sum(list(sweep.NEURAL_WEIGHTS.values()), axis=0), 1)


@pytest.mark.parametrize("probability", [np.nan, np.inf, -0.01, 1.01])
def test_prediction_checks_reject_invalid_probabilities(probability):
    with pytest.raises(ValueError):
        sweep.validate_prediction(np.zeros((2, 3), np.uint8), np.full((2, 3), probability))


def test_independent_ap_groups_exact_score_ties():
    labels = np.array([0, 1, 0, 1], np.uint8)
    probabilities = np.array([0.2, 0.9, 0.9, 0.1], np.float32)
    # Two thresholds add positives: precision 1/2 and 2/4, with recall gain 1/2 each.
    assert independent_ap(labels, probabilities) == pytest.approx(0.5)


def test_prediction_checks_reject_row_misalignment():
    with pytest.raises(ValueError):
        sweep.validate_prediction(np.zeros((2, 3), np.uint8), np.full((3, 3), 0.5))


def test_catboost_never_receives_outer_validation_labels(monkeypatch, tmp_path):
    arrays = {
        "x": np.arange(20).reshape(10, 2).astype(np.float32),
        "y": np.tile(np.array([[0, 0, 0], [1, 1, 1]], np.uint8), (5, 1)),
        "fold": np.repeat(np.arange(5, dtype=np.int8), 2),
        "feature_names": np.array(["a", "b"]),
    }
    fitting_calls, predicting_calls = [], []

    class FakeCatBoost:
        feature_importances_ = np.array([60.0, 40.0])

        def __init__(self, **kwargs):
            self.fold = kwargs["random_seed"] - 42
            assert kwargs["use_best_model"] is False

        def fit(self, x, y, **kwargs):
            assert set(kwargs) == {"sample_weight"}  # No eval_set or early_stopping.
            np.testing.assert_array_equal(x, arrays["x"][arrays["fold"] != self.fold])
            assert len(y) == 8
            fitting_calls.append(self.fold)

        def predict_proba(self, x):
            np.testing.assert_array_equal(x, arrays["x"][arrays["fold"] == self.fold])
            predicting_calls.append(self.fold)
            return np.tile([0.5, 0.5], (len(x), 1))

        def save_model(self, path):
            Path(path).touch()

    monkeypatch.setattr(sweep, "CatBoostClassifier", FakeCatBoost)
    prediction = sweep.fit_catboost(arrays, tmp_path)
    np.testing.assert_allclose(prediction, 0.5)
    assert len(fitting_calls) == len(predicting_calls) == 15
    assert np.bincount(fitting_calls).tolist() == [3, 3, 3, 3, 3]


def test_native_interpolation_stays_inside_each_recording(tmp_path):
    manifest = pd.DataFrame({"recording_id": ["first", "second"]})
    arrays = {"record": np.array([0, 0, 1, 1]), "grid": np.array([0, 4, 0, 2]), "row_counts": np.array([5, 3])}
    sampled = np.array([[0, 0, 0], [1, 1, 1], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9]], np.float32)
    (tmp_path / "catboost").mkdir()
    p = sweep.native_catboost(manifest, arrays, sampled, tmp_path)
    np.testing.assert_allclose(p[:5, 0], [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(p[5:, 0], [0.7, 0.8, 0.9])
    assert p.shape == (8, 3)
