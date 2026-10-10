"""Small deterministic checks for spectral features and held-out TDCS fitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tdcs_dense_context_experiment as experiment


def test_spectral_features_recover_known_frequency():
    time = np.arange(2048) / 128
    raw = np.column_stack([np.sin(2 * np.pi * 4 * time), np.sin(2 * np.pi * 2 * time), np.sin(2 * np.pi * 6 * time)])
    features, names = experiment.spectral_features(raw, np.array([512, 1024, 1536]))
    for window in experiment.FFT_WINDOWS:
        np.testing.assert_allclose(features[:, names.index(f"fft_peak_hz_AccV_w{window}")], 4)
        np.testing.assert_allclose(features[:, names.index(f"fft_peak_hz_AccML_w{window}")], 2)
        np.testing.assert_allclose(features[:, names.index(f"fft_peak_hz_AccAP_w{window}")], 6)
        fractions = [features[:, names.index(f"fft_{band}_fraction_AccV_w{window}")] for band in ["low", "mid", "high"]]
        np.testing.assert_allclose(np.sum(fractions, axis=0), 1, atol=1e-6)
        assert np.all(features[:, names.index(f"fft_mid_fraction_AccV_w{window}")] > 0.99)


def test_constant_short_recordings_remain_finite():
    grid, features, names = experiment.expanded_features(np.ones((8, 3)))
    assert len(grid) == 2 and grid[0] == 0 and grid[-1] == 7
    assert features.shape == (2, 213) and len(set(names)) == 213
    assert np.isfinite(features).all()


def test_context_clamps_to_recording_boundaries():
    names = [f"std_{axis}_w256" for axis in [*experiment.base.AXES, "dynamic_magnitude"]]
    names += [f"mean_{axis}_w256" for axis in experiment.base.AXES]
    names += [f"log_mid_low_energy_ratio_{axis}_w256" for axis in experiment.base.AXES]
    original = np.tile(np.array([2, 4, 8], np.float32)[:, None], (1, 10))
    context, context_names = experiment.temporal_context(original, names, np.array([0, 16, 32]))
    np.testing.assert_allclose(context[:, :20], 2)
    np.testing.assert_allclose(context[:, 20:], 8)
    assert len(context_names) == 40


def test_finer_grid_includes_last_native_row():
    raw = np.random.default_rng(7).normal(size=(100, 3))
    grid, features, names = experiment.expanded_features(raw)
    np.testing.assert_array_equal(grid, [0, 16, 32, 48, 64, 80, 96, 99])
    assert len(features) == len(grid)
    assert np.isfinite(features).all()
    assert len(names) == features.shape[1] == 213


def test_models_do_not_train_on_held_out_rows(monkeypatch, tmp_path):
    arrays = {"x": np.arange(20).reshape(10, 2).astype(np.float32),
              "y": np.tile(np.array([[0, 0, 0], [1, 1, 1]], np.uint8), (5, 1)),
              "fold": np.repeat(np.arange(5, dtype=np.int8), 2)}
    calls = []

    class FakeHGB:
        def __init__(self, **kwargs):
            self.fold = kwargs["random_state"] - 42
            assert kwargs["early_stopping"] is False

        def fit(self, x, y, **kwargs):
            np.testing.assert_array_equal(x, arrays["x"][arrays["fold"] != self.fold])
            assert len(y) == 8 and set(kwargs) == {"sample_weight"}
            calls.append(self.fold)

        def predict_proba(self, x):
            np.testing.assert_array_equal(x, arrays["x"][arrays["fold"] == self.fold])
            return np.tile([0.6, 0.4], (len(x), 1))

    monkeypatch.setattr(experiment, "HistGradientBoostingClassifier", FakeHGB)
    monkeypatch.setattr(experiment.joblib, "dump", lambda model, path: Path(path).touch())
    prediction = experiment.train_models(arrays, tmp_path)
    np.testing.assert_allclose(prediction, 0.4)
    assert np.bincount(calls).tolist() == [3, 3, 3, 3, 3]


def test_new_blend_weights_are_fixed_and_normalized():
    assert all(abs(sum(weights.values()) - 1) < 1e-12 for weights in experiment.FIXED_BLENDS.values())
    predictions = {"retained_neural_hgb": np.full((2, 3), 0.2, np.float32),
                   "previous_tdcs_blend": np.full((2, 3), 0.4, np.float32),
                   "dense_context_hgb": np.full((2, 3), 0.8, np.float32)}
    np.testing.assert_allclose(experiment.get_prediction("retained_dense_quarter", predictions), 0.35)
    np.testing.assert_allclose(experiment.get_prediction("retained_dense_equal", predictions), 0.5)
    np.testing.assert_allclose(experiment.get_prediction("neural_dense_equal", predictions), 0.6)
