"""Independent CNN / patch-attention TDCS experiments; never overwrite retained heads.

Dense three-target adaptations inspired by InceptionTime and patch-based attention,
not reproductions of either paper. Two-target and three-target mAP stay separate.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

import tdcs_two_target_neural as common

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('StartHesitation', 'Turn', 'Walking')
ARCHITECTURES = ('multiscale_cnn', 'patch_transformer')


@dataclass(frozen=True)
class Config:
    name: str = 'tdcs_alternatives_v2'
    targets: tuple = TARGETS
    folds: tuple = (0, 1, 2, 3, 4)
    epochs: int = 24
    steps_per_epoch: int = 96
    batch_size: int = 16
    window: int = 2048
    inference_batch: int = 4
    width: int = 64
    dilations: tuple = (1, 2, 4, 8, 16, 32)
    patch: int = 32
    attention_width: int = 96
    attention_layers: int = 3
    dropout: float = 0.25
    weight_decay: float = 0.01
    learning_rate: float = 0.0007
    event_fraction: float = 0.5
    negative_fraction: float = 0.25
    ema_decay: float = 0.995
    seed: int = 1207


CFG = Config()


def prepare_records(manifest):
    records = []
    for index, row in enumerate(manifest.itertuples(index=False)):
        frame = pd.read_csv(row.path, usecols=[*common.AXES, *TARGETS])
        y = frame.loc[:, list(TARGETS)].to_numpy(np.float32)
        assert np.isin(y, [0, 1]).all()
        # Sensor-only: remove the three absolute recording-progress features.
        x = common.sensor_features(frame.loc[:, list(common.AXES)].to_numpy(np.float32))[:, :12].copy()
        negatives = np.flatnonzero(y.sum(axis=1) == 0)
        negatives = negatives[::max(1, len(negatives) // 2000)]
        records.append({'recording_id': row.recording_id, 'subject': row.subject, 'fold': row.fold,
            'x': x, 'y': y, 'negative': negatives, 'centers': [common.event_centers(y[:, i]) for i in range(3)]})
        if (index + 1) % 100 == 0:
            print(f'Prepared {index+1}/{len(manifest)} recordings', flush=True)
    assert sum(len(record['y']) for record in records) == 6959762
    return records


def window(record, start, length):
    indices = np.arange(start, start + length)
    valid = (indices >= 0) & (indices < len(record['x']))
    x = record['x'][np.clip(indices, 0, len(record['x']) - 1)].copy()
    y = np.zeros((length, 3), np.float32)
    y[valid] = record['y'][indices[valid]]
    return x, y, np.repeat(valid[:, None], 3, axis=1).astype(np.float32)


class Sampler:
    def __init__(self, records, fold, cfg):
        self.cfg = cfg
        self.records = [record for record in records if record['fold'] != fold]
        self.subjects, self.negative = {}, {}
        self.positive = [{}, {}, {}]
        for record in self.records:
            subject = record['subject']
            self.subjects.setdefault(subject, []).append(record)
            if len(record['negative']):
                self.negative.setdefault(subject, []).append(record)
            for target in range(3):
                if len(record['centers'][target]):
                    self.positive[target].setdefault(subject, []).append(record)
        if not self.subjects or any(not pool for pool in self.positive):
            raise ValueError('All targets require training events; no held-out records may be sampled')

    def plan(self, rng):
        draw = rng.random()
        if draw < self.cfg.event_fraction:
            target = int(rng.integers(3))
            record = common.EventSampler.choose_record(self.positive[target], rng)
            center = int(rng.choice(record['centers'][target]))
            center += int(rng.integers(-self.cfg.window // 4, self.cfg.window // 4 + 1))
        elif draw < self.cfg.event_fraction + self.cfg.negative_fraction and self.negative:
            record = common.EventSampler.choose_record(self.negative, rng)
            center = int(rng.choice(record['negative']))
        else:
            record = common.EventSampler.choose_record(self.subjects, rng)
            center = int(rng.integers(len(record['x'])))
        center = int(np.clip(center, 0, len(record['x']) - 1))
        return record, center - self.cfg.window // 2

    def batch(self, rng):
        xs, ys, masks = [], [], []
        for _ in range(self.cfg.batch_size):
            record, start = self.plan(rng)
            x, y, mask = window(record, start, self.cfg.window)
            gains = np.tile(rng.uniform(.9, 1.1, 3).astype(np.float32), 4)
            x *= gains
            angle = rng.uniform(-.15, .15)
            rotation = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                                    [0, np.sin(angle), np.cos(angle)]], np.float32)
            for begin in range(0, 12, 3):
                x[:, begin:begin+3] = x[:, begin:begin+3] @ rotation.T
            if rng.random() < .25:
                begin = int(rng.integers(len(x)))
                x[begin:begin + min(64, len(x) // 8)] = 0
            xs.append(x)
            ys.append(y)
            masks.append(mask)
        return tuple(torch.from_numpy(np.stack(parts)) for parts in [xs, ys, masks])


class MultiscaleBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(width, width, kernel, groups=width,
            dilation=dilation, padding=dilation * (kernel - 1) // 2) for kernel in (3, 7, 15)])
        self.mix = nn.Sequential(nn.Conv1d(width * 3, width, 1), nn.GroupNorm(8, width),
                                 nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.mix(torch.cat([branch(x) for branch in self.branches], dim=1))


class AlternativeNetwork(nn.Module):
    def __init__(self, architecture, cfg=CFG):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError('Unknown architecture')
        self.architecture = architecture
        self.stem = nn.Sequential(nn.Conv1d(12, cfg.width, 7, padding=3), nn.GroupNorm(8, cfg.width), nn.GELU())
        if architecture == 'multiscale_cnn':
            self.temporal = nn.Sequential(*[MultiscaleBlock(cfg.width, dilation, cfg.dropout) for dilation in cfg.dilations])
            self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Conv1d(cfg.width, 3, 1))
        else:
            self.patch = nn.Conv1d(cfg.width, cfg.attention_width, cfg.patch,
                                    stride=max(1, cfg.patch // 2), padding=cfg.patch // 2)
            layer = nn.TransformerEncoderLayer(cfg.attention_width, 4, cfg.attention_width * 3,
                dropout=cfg.dropout, activation='gelu', batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, cfg.attention_layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Conv1d(cfg.width + cfg.attention_width, cfg.width, 3, padding=1),
                                      nn.GELU(), nn.Dropout(cfg.dropout), nn.Conv1d(cfg.width, 3, 1))

    def forward(self, x):
        h = self.stem(x.transpose(1, 2))
        if self.architecture == 'multiscale_cnn':
            return self.head(self.temporal(h)).transpose(1, 2)
        patches = self.patch(h).transpose(1, 2)
        position = torch.arange(patches.shape[1], device=x.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, patches.shape[2], 2, device=x.device) * (-math.log(10000) / patches.shape[2]))
        encoding = torch.zeros(patches.shape[1:], device=x.device, dtype=torch.float32)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        patches = self.temporal(patches + encoding.to(patches.dtype))
        context = F.interpolate(patches.transpose(1, 2), size=h.shape[2], mode='linear', align_corners=False)
        return self.head(torch.cat([h, context], dim=1)).transpose(1, 2)


def masked_loss(logits, y, mask):
    loss = F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction='none')
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def train_model(records, fold, architecture, cfg, directory, device, fingerprint):
    directory.mkdir(parents=True, exist_ok=True)
    seed = cfg.seed + 1009 * fold
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = AlternativeNetwork(architecture, cfg).to(device)
    final = directory / 'final_ema.pt'
    if final.exists():
        state = torch.load(final, map_location=device, weights_only=False)
        assert state['fingerprint'] == fingerprint and tuple(state['targets']) == TARGETS
        model.load_state_dict(state['model'])
        return model
    sampler = Sampler(records, fold, cfg)
    training = sorted(sampler.subjects)
    held_out = sorted({record['subject'] for record in records if record['fold'] == fold})
    assert set(training).isdisjoint(held_out)
    common.write_json(directory / 'training_receipt.json', {'targets': TARGETS, 'training_subjects': training,
        'held_out_subjects': held_out, 'fixed_epochs': cfg.epochs, 'outer_labels_used_for_checkpoint_selection': False,
        'architecture': architecture, 'sensor_features': 12, 'training_window': cfg.window,
        'inference_window': cfg.window, 'previous_models_overwritten': False, 'seed': seed})
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: common.learning_rate_multiplier(step, cfg.epochs * cfg.steps_per_epoch))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    rng = np.random.default_rng(seed)
    resume, history, first = directory / 'resume.pt', [], 0
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        assert state['fingerprint'] == fingerprint
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
        first, history = state['next_epoch'], state['history']
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    for epoch in range(first, cfg.epochs):
        started, losses = time.monotonic(), []
        model.train()
        print(f'{architecture} fold {fold} epoch {epoch+1}/{cfg.epochs}', flush=True)
        for _ in range(cfg.steps_per_epoch):
            x, y, mask = [value.to(device) for value in sampler.batch(rng)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                loss = masked_loss(model(x), y, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            with torch.no_grad():
                for destination, source in zip(ema.parameters(), common.core_model(model).parameters()):
                    destination.lerp_(source.detach(), 1-cfg.ema_decay)
            losses.append(float(loss.detach().cpu()))
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'seconds': time.monotonic()-started,
                        'learning_rate': schedule.get_last_lr()[0]})
        pd.DataFrame(history).to_csv(directory / 'training_log.csv', index=False)
        state = {'fingerprint': fingerprint, 'model': common.core_model(model).state_dict(), 'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': schedule.state_dict(), 'scaler': scaler.state_dict(),
            'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'next_epoch': epoch+1, 'history': history}
        temporary = directory / 'resume.tmp.pt'
        torch.save(state, temporary)
        os.replace(temporary, resume)
        common.write_json(directory.parent / 'progress.json', {'stage': 'training', 'architecture': architecture,
            'fold': fold, 'epoch': epoch+1, 'fixed_epochs': cfg.epochs, 'training_loss': history[-1]['loss']})
    torch.save({'fingerprint': fingerprint, 'model': ema.state_dict(), 'targets': TARGETS}, final)
    return ema


def predict_record(model, features, cfg, device):
    length, n = cfg.window, len(features)
    starts = common.inference_starts(n, length)
    total, counts = np.zeros((n, 3), np.float32), np.zeros(n, np.float32)
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(starts), cfg.inference_batch):
            batch = starts[begin:begin+cfg.inference_batch]
            windows = [features[np.clip(np.arange(start, start+length), 0, n-1)] for start in batch]
            x = torch.from_numpy(np.stack(windows)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                p = torch.sigmoid(model(x).float()).cpu().numpy()
            for i, start in enumerate(batch):
                count = min(length, n-start)
                weight = common.inference_weight(length, start == 0, start+length >= n)[:count]
                total[start:start+count] += p[i, :count] * weight[:, None]
                counts[start:start+count] += weight
    if (counts <= 0).any():
        raise ValueError('Uncovered native rows')
    result = np.clip(total / counts[:, None], 1e-6, 1-1e-6)
    assert np.isfinite(result).all()
    return result


def score(y, p):
    assert y.shape == p.shape and y.shape[1] == 3
    values = {target: float(average_precision_score(y[:, i], p[:, i])) for i, target in enumerate(TARGETS)}
    return {**values, 'three_target_mAP': float(np.mean(list(values.values()))),
            'two_target_mAP': float((values['StartHesitation'] + values['Walking']) / 2)}


def run_experiment(architecture, cfg=CFG, output_root=None, synthetic_records=None):
    torch.set_num_threads(4)
    if architecture not in ARCHITECTURES or tuple(cfg.targets) != TARGETS:
        raise ValueError('Unexpected architecture or target order')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if synthetic_records is None and device.type != 'cuda':
        raise RuntimeError('Full training requires a CUDA GPU; no automatic smoke fallback')
    manifest = None
    if synthetic_records is None:
        raw, metadata = common.locate_data()
        manifest = common.build_manifest(raw, metadata)
        population = manifest[['recording_id', 'subject', 'fold', 'source_bytes']].to_csv(index=False, lineterminator='\n')
    else:
        population = 'synthetic wiring only; not TDCS performance'
    configuration = {'configuration': asdict(cfg), 'architecture': architecture, 'targets': TARGETS,
        'source_sha256': globals().get('SOURCE_SHA256') or hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'common_source_sha256': globals().get('COMMON_SOURCE_SHA256') or hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest(),
        'population_sha256': hashlib.sha256(population.encode()).hexdigest(), 'smoke_only': synthetic_records is not None,
        'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__, 'gpu_count': torch.cuda.device_count()},
        'walking_policy': 'candidate only; retained Walking is not changed by training',
        'checkpoint_selection': 'fixed final EMA; no outer-label epoch selection',
        'inference_mode': 'offline, whole-record robust sensor scaling', 'confidence': 'uncalibrated sigmoid scores',
        'references': ['https://arxiv.org/abs/1909.04939', 'https://arxiv.org/abs/2211.14730'],
        'competition_submission_created': False}
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:20]
    output_root = Path(output_root or ROOT / 'results/tdcs_alternative_architectures')
    run = output_root / fingerprint
    run.mkdir(parents=True, exist_ok=True)
    common.write_json(run / 'configuration.json', configuration)
    if manifest is not None:
        manifest.to_csv(run / 'frozen_tdcs_manifest.csv', index=False)
    records = prepare_records(manifest) if synthetic_records is None else synthetic_records
    rows, labels, predictions = [], [], []
    (run / 'oof').mkdir(exist_ok=True)
    for fold in cfg.folds:
        model = train_model(records, fold, architecture, cfg, run / f'fold_{fold}', device, fingerprint)
        ys, ps = [], []
        for record in records:
            if record['fold'] != fold:
                continue
            pred = predict_record(model, record['x'], cfg, device)
            assert pred.shape == record['y'].shape
            np.savez_compressed(run / 'oof' / f'{record["recording_id"]}.npz', prediction=pred, targets=np.asarray(TARGETS))
            ys.append(record['y'])
            ps.append(pred)
        rows.append({'fold': fold, **score(np.concatenate(ys), np.concatenate(ps))})
        pd.DataFrame(rows).to_csv(run / 'fold_metrics.csv', index=False)
        labels.extend(ys)
        predictions.extend(ps)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    metrics = score(np.concatenate(labels), np.concatenate(predictions))
    pd.DataFrame([metrics]).to_csv(run / 'native_oof_metrics.csv', index=False)
    decision = {'run_dir': str(run), 'status': 'complete', 'architecture': architecture, **metrics,
        'smoke_only': synthetic_records is not None, 'native_rows': sum(len(y) for y in labels),
        'goal_two_target_mAP': .6, 'goal_reached': metrics['two_target_mAP'] >= .6 and synthetic_records is None,
        'retained_heads_unchanged': True, 'independent_generalization_established': False,
        'competition_submission_created': False}
    common.write_json(run / 'decision.json', decision)
    common.write_json(run / 'progress.json', decision)
    common.write_json(output_root / 'latest_run.json', decision)
    print(json.dumps(decision, indent=2), flush=True)
    return decision


def smoke_experiment(architecture, output_root=None):
    cfg = replace(CFG, name='synthetic_wiring_only', epochs=1, steps_per_epoch=1, batch_size=2,
        window=64, width=16, dilations=(1, 2), patch=8, attention_width=32, attention_layers=1)
    rng, records = np.random.default_rng(1), []
    for fold in range(5):
        x = common.sensor_features(rng.normal(size=(96+fold, 3)).astype(np.float32))[:, :12].copy()
        y = np.zeros((len(x), 3), np.float32)
        for target, start in enumerate((12, 35, 62)):
            y[start:start+12, target] = 1
        records.append({'recording_id': f'smoke_{fold}', 'subject': f'synthetic_{fold}', 'fold': fold,
            'x': x, 'y': y, 'negative': np.flatnonzero(y.sum(axis=1) == 0),
            'centers': [common.event_centers(y[:, i]) for i in range(3)]})
    return run_experiment(architecture, cfg, output_root or ROOT / 'tmp' / f'tdcs_{architecture}_smoke', records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=ARCHITECTURES, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    smoke_experiment(args.architecture, args.output_root) if args.smoke else run_experiment(args.architecture, output_root=args.output_root)
