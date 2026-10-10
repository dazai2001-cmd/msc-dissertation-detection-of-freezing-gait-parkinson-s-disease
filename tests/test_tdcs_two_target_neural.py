"""Sequence shapes, target-only fitting, loss masking, and native-row coverage."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import tdcs_two_target_neural as experiment


def tiny_records():
    records = []
    for fold in range(5):
        y = np.zeros((64, 2), np.float32)
        y[10:20, 0] = 1
        y[40:50, 1] = 1
        records.append({'recording_id': str(fold), 'subject': f'subject_{fold}', 'fold': fold,
                        'x': np.zeros((64, 15), np.float32), 'y': y,
                        'negative': np.flatnonzero(y.sum(axis=1) == 0),
                        'centers': [experiment.event_centers(y[:, i]) for i in range(2)]})
    return records


@pytest.mark.parametrize('variant', experiment.CFG.variants)
def test_network_has_only_two_native_row_outputs(variant):
    cfg = replace(experiment.CFG, hidden=8, width=16, gru_layers=1)
    network = experiment.TwoTargetGRU(cfg, variant)
    x = torch.randn(2, 33, 15)
    logits = network(x)
    assert logits.shape == (2, 33, 2) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert experiment.TARGETS == ('StartHesitation', 'Walking')


def test_sensor_features_use_no_labels_and_handle_constant_record():
    x = experiment.sensor_features(np.ones((19, 3)))
    assert x.shape == (19, 15) and np.isfinite(x).all()
    np.testing.assert_allclose(x[:, :12], 0)
    assert x[0, 12] == 0 and x[-1, 12] == 1


def test_sampler_excludes_all_outer_subject_records():
    records = tiny_records()
    cfg = replace(experiment.CFG, batch_size=3, train_length=32)
    sampler = experiment.EventSampler(records, 2, cfg)
    assert 'subject_2' not in sampler.subjects
    rng = np.random.default_rng(7)
    for _ in range(100):
        record, start = sampler.plan(rng)
        assert record['fold'] != 2 and record['subject'] != 'subject_2'
    x, y, mask = sampler.batch(rng)
    assert x.shape == (3, 32, 15) and y.shape == mask.shape == (3, 32, 2)


def test_outer_labels_cannot_change_sampling():
    original = tiny_records()
    changed = tiny_records()
    changed[0]['y'][:] = 1
    changed[0]['centers'] = [np.array([32]), np.array([32])]
    cfg = replace(experiment.CFG, batch_size=2, train_length=32)
    a, b = experiment.EventSampler(original, 0, cfg), experiment.EventSampler(changed, 0, cfg)
    rng_a, rng_b = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(30):
        ra, sa = a.plan(rng_a)
        rb, sb = b.plan(rng_b)
        assert ra['recording_id'] == rb['recording_id'] and sa == sb


def test_padding_is_excluded_from_loss():
    x = np.ones((10, 15), np.float32)
    y = np.ones((10, 2), np.float32)
    window, labels, mask = experiment.padded_window(x, y, -3, 16)
    assert window.shape == (16, 15) and mask.sum() == 20
    logits = torch.zeros((1, 16, 2), requires_grad=True)
    loss = experiment.sequence_loss(logits, torch.from_numpy(labels[None]), torch.from_numpy(mask[None]))
    assert float(loss.detach()) == pytest.approx(np.log(2))
    loss.backward()
    assert torch.all(logits.grad[0, :3] == 0) and torch.all(logits.grad[0, 13:] == 0)


def test_ranking_loss_is_finite_with_absent_positive_class():
    logits = torch.randn(2, 20, 2, requires_grad=True)
    y = torch.zeros_like(logits)
    y[:, :4, 0] = 1
    loss = experiment.sequence_loss(logits, y, torch.ones_like(y), rank_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('length', [1, 63, 64, 65, 129, 197])
def test_center_crop_inference_covers_every_native_row(length):
    denominator = np.zeros(length)
    for start in experiment.inference_starts(length, 64):
        count = min(64, length - start)
        denominator[start:start + count] += experiment.inference_weight(64, start == 0, start + 64 >= length)[:count]
    assert np.all(denominator > 0)


def test_streamed_inference_preserves_native_alignment():
    cfg = replace(experiment.CFG, inference_length=64, inference_batch=2)

    class ConstantModel(torch.nn.Module):
        def forward(self, x):
            return torch.full((*x.shape[:2], 2), 0.0)

    p = experiment.predict_record(ConstantModel(), np.zeros((197, 15), np.float32), cfg, torch.device('cpu'))
    assert p.shape == (197, 2)
    np.testing.assert_allclose(p, 0.5)


def test_training_schedule_and_metric_target_are_fixed():
    assert experiment.CFG.epochs == 24 and experiment.CFG.goal_two_target_map == 0.6
    assert experiment.learning_rate_multiplier(0, 100) > 0
    assert experiment.learning_rate_multiplier(99, 100) == 0.02
    y = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    p = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]], np.float32)
    assert experiment.average_precision(y, p)['two_target_mAP'] == 1
