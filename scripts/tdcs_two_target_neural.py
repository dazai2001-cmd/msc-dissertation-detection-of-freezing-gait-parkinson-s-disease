"""Fresh two-output TDCS sequence models, with frozen subject-held-out evaluation.

No Turn head, no prior neural checkpoint, no leaderboard/submission call. Models
use event/subject-balanced training and longer context only at inference. Fixed
epochs and EMA checkpoints avoid selecting epochs on outer evaluation labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from tdcs_multiscale_signal_experiment import AXES, FROZEN_SUBJECT_FOLDS

TARGETS = ('StartHesitation', 'Walking')
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    name: str = 'two_target_sequence_v1'
    targets: tuple = TARGETS
    variants: tuple = ('bigru_bce', 'gated_bigru_rank')
    folds: tuple = (0, 1, 2, 3, 4)
    epochs: int = 24
    steps_per_epoch: int = 144
    batch_size: int = 32
    train_length: int = 1000
    inference_length: int = 5000
    inference_batch: int = 4
    hidden: int = 128
    gru_layers: int = 3
    width: int = 64
    dropout: float = 0.15
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    event_fraction: float = 0.5
    negative_fraction: float = 0.25
    rank_weight: float = 0.1
    ema_decay: float = 0.995
    seed: int = 731
    goal_two_target_map: float = 0.6


CFG = Config()


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(temporary, path)


def locate_data():
    candidates = [ROOT / 'data/metadata', Path.cwd() / 'data/metadata']
    mount = Path('/kaggle/input')
    if mount.exists():
        candidates = [p.parent for p in sorted(mount.rglob('tdcsfog_metadata.csv'))] + candidates
    for metadata_root in candidates:
        metadata_path = metadata_root / 'tdcsfog_metadata.csv'
        for raw in [metadata_root / 'train/tdcsfog', metadata_root.parent / 'raw/train/tdcsfog']:
            if metadata_path.exists() and raw.is_dir():
                return raw, metadata_path
    raise FileNotFoundError('Attach the competition data: need tdcsfog_metadata.csv and train/tdcsfog')


def build_manifest(raw_root: Path, metadata_path: Path):
    metadata = pd.read_csv(metadata_path, dtype={'Id': str, 'Subject': str})
    if metadata.Id.duplicated().any():
        raise ValueError('Duplicate recording metadata')
    subject_by_id = metadata.set_index('Id').Subject.to_dict()
    rows = []
    for path in sorted(raw_root.glob('*.csv')):
        subject = subject_by_id.get(path.stem)
        if subject == '4dc2f8':
            continue
        if subject not in FROZEN_SUBJECT_FOLDS:
            raise ValueError(f'Unexpected/missing subject for recording {path.stem}')
        rows.append({'recording_id': path.stem, 'subject': subject,
                     'fold': FROZEN_SUBJECT_FOLDS[subject], 'path': str(path), 'source_bytes': path.stat().st_size})
    manifest = pd.DataFrame(rows)
    if len(manifest) != 815 or manifest.subject.nunique() != 61:
        raise ValueError('Frozen development population must be 815 recordings / 61 subjects')
    if manifest.groupby('subject').fold.nunique().max() != 1:
        raise ValueError('Subject leakage between folds')
    return manifest


def sensor_features(raw: np.ndarray):
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError('Need a finite recording with at least two rows and three accelerometer axes')
    lag = np.vstack([np.zeros((1, 3), np.float32), np.diff(raw, axis=0)])
    lead = np.vstack([np.diff(raw, axis=0), np.zeros((1, 3), np.float32)])
    integrated = np.cumsum(raw - np.median(raw, axis=0, keepdims=True), axis=0, dtype=np.float64).astype(np.float32)
    values = np.concatenate([raw, lag, lead, integrated], axis=1)
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    values = (values - median) / np.maximum(q75 - q25, 1e-4)
    position = np.arange(len(raw), dtype=np.float64)
    seconds = position / 128.0
    timing = np.column_stack([position / max(len(raw) - 1, 1), np.log1p(seconds) / 8,
                              np.log1p((len(raw) - 1 - position) / 128.0) / 8])
    features = np.column_stack([np.clip(values, -20, 20), timing]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Invalid robust features')
    return features


def event_centers(labels: np.ndarray):
    labels = np.asarray(labels, bool)
    starts = np.flatnonzero(labels & ~np.r_[False, labels[:-1]])
    stops = np.flatnonzero(labels & ~np.r_[labels[1:], False]) + 1
    return ((starts + stops - 1) // 2).astype(np.int64)


def prepare_records(manifest: pd.DataFrame):
    records = []
    for index, row in enumerate(manifest.itertuples(index=False)):
        frame = pd.read_csv(row.path, usecols=[*AXES, *TARGETS])
        y = frame.loc[:, list(TARGETS)].to_numpy()
        if not np.isin(y, [0, 1]).all():
            raise ValueError('Targets must be binary')
        x = sensor_features(frame.loc[:, list(AXES)].to_numpy(np.float32))
        negative = np.flatnonzero(y.sum(axis=1) == 0)
        negative = negative[::max(1, len(negative) // 2000)]
        records.append({'recording_id': row.recording_id, 'subject': row.subject, 'fold': row.fold,
                        'x': x, 'y': y.astype(np.float32), 'negative': negative,
                        'centers': [event_centers(y[:, i]) for i in range(2)]})
        if (index + 1) % 100 == 0:
            print(f'Prepared sensor records: {index + 1}/{len(manifest)}', flush=True)
    if sum(len(r['y']) for r in records) != 6959762:
        raise ValueError('Native population changed')
    return records


def padded_window(x, y, start, length):
    indices = np.arange(start, start + length)
    valid = (indices >= 0) & (indices < len(x))
    window = x[np.clip(indices, 0, len(x) - 1)].copy()
    labels = np.zeros((length, 2), np.float32)
    labels[valid] = y[indices[valid]]
    return window, labels, np.repeat(valid[:, None], 2, axis=1).astype(np.float32)


class EventSampler:
    def __init__(self, records, held_out_fold, cfg):
        self.cfg = cfg
        self.records = [r for r in records if r['fold'] != held_out_fold]
        self.held_out_fold = held_out_fold
        if not self.records or any(r['fold'] == held_out_fold for r in self.records):
            raise ValueError('Training records cross the held-out fold')
        self.subjects = {}
        self.positive = [{}, {}]
        self.negative = {}
        for record in self.records:
            subject = record['subject']
            self.subjects.setdefault(subject, []).append(record)
            if len(record['negative']):
                self.negative.setdefault(subject, []).append(record)
            for target in range(2):
                if len(record['centers'][target]):
                    self.positive[target].setdefault(subject, []).append(record)
        self.available_targets = [t for t in range(2) if self.positive[t]]
        if len(self.available_targets) != 2:
            raise ValueError('Both targets require training events')

    @staticmethod
    def choose_record(pool, rng):
        subjects = sorted(pool)
        subject = subjects[int(rng.integers(len(subjects)))]
        choices = pool[subject]
        return choices[int(rng.integers(len(choices)))]

    def plan(self, rng):
        draw = rng.random()
        if draw < self.cfg.event_fraction:
            target = int(rng.choice(self.available_targets))
            record = self.choose_record(self.positive[target], rng)
            center = int(rng.choice(record['centers'][target]))
            center += int(rng.integers(-self.cfg.train_length // 5, self.cfg.train_length // 5 + 1))
        elif draw < self.cfg.event_fraction + self.cfg.negative_fraction and self.negative:
            record = self.choose_record(self.negative, rng)
            center = int(rng.choice(record['negative']))
        else:
            record = self.choose_record(self.subjects, rng)
            center = int(rng.integers(len(record['x'])))
        center = int(np.clip(center, 0, len(record['x']) - 1))
        return record, center - self.cfg.train_length // 2

    def batch(self, rng):
        x_parts, y_parts, masks = [], [], []
        for _ in range(self.cfg.batch_size):
            record, start = self.plan(rng)
            x, y, mask = padded_window(record['x'], record['y'], start, self.cfg.train_length)
            # Mild amplitude/axis perturbation, applied coherently to all four 3-axis blocks.
            gains = rng.uniform(0.95, 1.05, 3).astype(np.float32)
            x[:, :12] *= np.tile(gains, 4)
            theta = float(rng.uniform(-0.10, 0.10))
            rotation = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                                 [0, np.sin(theta), np.cos(theta)]], np.float32)
            for begin in range(0, 12, 3):
                x[:, begin:begin + 3] = x[:, begin:begin + 3] @ rotation.T
            x_parts.append(x)
            y_parts.append(y)
            masks.append(mask)
        return tuple(torch.from_numpy(np.stack(parts)) for parts in [x_parts, y_parts, masks])


class GatedBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.filter = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.gate = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.output = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        h = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        return x + self.output(self.norm(h))


class TwoTargetGRU(nn.Module):
    def __init__(self, cfg=CFG, variant='bigru_bce'):
        super().__init__()
        if variant not in ('bigru_bce', 'gated_bigru_rank'):
            raise ValueError('Unknown neural variant')
        self.input = nn.Sequential(nn.Linear(15, cfg.width), nn.LayerNorm(cfg.width), nn.GELU())
        self.gated = nn.Sequential(*[GatedBlock(cfg.width, d) for d in (1, 2, 4, 8)]) if variant.startswith('gated') else None
        self.gru = nn.GRU(cfg.width, cfg.hidden, num_layers=cfg.gru_layers,
                          dropout=cfg.dropout if cfg.gru_layers > 1 else 0,
                          batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * cfg.hidden, 128), nn.LayerNorm(128), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(128, 2))

    def forward(self, x):
        h = self.input(x)
        if self.gated is not None:
            h = self.gated(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h.contiguous())
        return self.head(h)


def sequence_loss(logits, labels, mask, rank_weight=0.0):
    # Sampling already increases event frequency: do NOT also apply inverse-frequency positive weights.
    bce = (F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction='none') * mask).sum()
    loss = bce / mask.sum().clamp_min(1)
    rank_terms = []
    if rank_weight:
        for target in range(2):
            valid = mask[..., target] > 0
            pos = logits[..., target][valid & (labels[..., target] > 0.5)].float()
            neg = logits[..., target][valid & (labels[..., target] < 0.5)].float()
            if pos.numel() and neg.numel():
                pos = pos[torch.randperm(len(pos), device=pos.device)[:192]]
                # Hard negatives are selected from this TRAINING batch, never from outer OOF labels.
                neg = torch.topk(neg, min(192, len(neg))).values
                rank_terms.append(F.softplus(neg[None, :] - pos[:, None]).mean())
    if rank_terms:
        loss = loss + rank_weight * torch.stack(rank_terms).mean()
    return loss


def core_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def learning_rate_multiplier(step, total_steps):
    warmup = max(1, int(total_steps * 0.05))
    if step < warmup:
        return max(0.001, (step + 1) / warmup)
    return max(0.02, (total_steps - step) / max(total_steps - warmup, 1))


def train_model(records, fold, variant, cfg, run_dir, device, fingerprint):
    fold_dir = run_dir / variant / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    final_path = fold_dir / 'final_ema.pt'
    seed = cfg.seed + 1009 * fold + (1 if variant.startswith('gated') else 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    network = TwoTargetGRU(cfg, variant).to(device)
    if final_path.exists():
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        if checkpoint['fingerprint'] != fingerprint:
            raise ValueError('Cannot reuse a checkpoint from another configuration')
        network.load_state_dict(checkpoint['model'])
        print(f'Reusing completed {variant} fold {fold}', flush=True)
        return network
    sampler = EventSampler(records, fold, cfg)
    training_subjects = sorted(sampler.subjects)
    held_out_subjects = sorted({r['subject'] for r in records if r['fold'] == fold})
    if set(training_subjects) & set(held_out_subjects):
        raise ValueError('Outer subject leakage')
    write_json(fold_dir / 'training_receipt.json', {'targets': TARGETS, 'training_subjects': training_subjects,
        'held_out_subjects': held_out_subjects, 'outer_labels_used_for_checkpoint_selection': False,
        'event_fraction': cfg.event_fraction, 'negative_fraction': cfg.negative_fraction,
        'positive_class_weights': [1, 1], 'fixed_epochs': cfg.epochs, 'seed': seed})
    ema = copy.deepcopy(network).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_multiplier(step, total_steps))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    start_epoch = 0
    rng = np.random.default_rng(seed)
    resume_path = fold_dir / 'resume.pt'
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Stale training resume state')
        network.load_state_dict(saved['model'])
        ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if use_amp and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        start_epoch = saved['next_epoch']
    if use_amp and torch.cuda.device_count() > 1:
        network = nn.DataParallel(network)
    history = []
    history_path = fold_dir / 'training_log.csv'
    if history_path.exists():
        history = pd.read_csv(history_path).to_dict('records')
    for epoch in range(start_epoch, cfg.epochs):
        network.train()
        started = time.monotonic()
        losses, event_rows, valid_rows = [], np.zeros(2, np.float64), np.zeros(2, np.float64)
        print(f'{variant} fold {fold} epoch {epoch + 1}/{cfg.epochs}: training', flush=True)
        for step in range(cfg.steps_per_epoch):
            x, y, mask = sampler.batch(rng)
            event_rows += (y.numpy() * mask.numpy()).sum(axis=(0, 1))
            valid_rows += mask.numpy().sum(axis=(0, 1))
            if use_amp:
                x, y, mask = [v.pin_memory().to(device, non_blocking=True) for v in (x, y, mask)]
            else:
                x, y, mask = [v.to(device) for v in (x, y, mask)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = network(x)
                loss = sequence_loss(logits, y, mask, cfg.rank_weight if variant.startswith('gated') else 0)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(network.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            with torch.no_grad():
                for destination, source in zip(ema.parameters(), core_model(network).parameters()):
                    destination.lerp_(source.detach(), 1 - cfg.ema_decay)
            losses.append(float(loss.detach().cpu()))
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)),
                        'StartHesitation_sampled_fraction': float(event_rows[0] / valid_rows[0]),
                        'Walking_sampled_fraction': float(event_rows[1] / valid_rows[1]),
                        'learning_rate': scheduler.get_last_lr()[0], 'seconds': time.monotonic() - started})
        pd.DataFrame(history).to_csv(history_path, index=False)
        state = {'fingerprint': fingerprint, 'model': core_model(network).state_dict(), 'ema': ema.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                 'next_epoch': epoch + 1, 'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if use_amp else None}
        temporary = fold_dir / 'resume.tmp.pt'
        torch.save(state, temporary)
        os.replace(temporary, resume_path)
        write_json(run_dir / 'progress.json', {'stage': 'training', 'variant': variant, 'fold': fold,
                   'epoch_completed': epoch + 1, 'epochs': cfg.epochs, 'latest_training_loss': history[-1]['loss']})
        print(f'{variant} fold {fold}: epoch {epoch + 1}, loss {history[-1]["loss"]:.5f}, {history[-1]["seconds"]:.1f}s', flush=True)
    torch.save({'fingerprint': fingerprint, 'model': ema.state_dict(), 'targets': TARGETS}, final_path)
    return ema


def inference_starts(n, length):
    if n <= length:
        return [0]
    starts = list(range(0, n - length + 1, max(1, length // 2)))
    if starts[-1] != n - length:
        starts.append(n - length)
    return starts


def inference_weight(length, first, last):
    weight = np.zeros(length, np.float32)
    quarter = length // 4
    weight[quarter:length - quarter] = 1
    if first:
        weight[:quarter] = 1
    if last:
        weight[length - quarter:] = 1
    return weight


def predict_record(model, features, cfg, device):
    n, length = len(features), cfg.inference_length
    starts = inference_starts(n, length)
    total = np.zeros((n, 2), np.float32)
    denominator = np.zeros(n, np.float32)
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(starts), cfg.inference_batch):
            selected = starts[begin:begin + cfg.inference_batch]
            windows = [features[np.clip(np.arange(start, start + length), 0, n - 1)] for start in selected]
            x = torch.from_numpy(np.stack(windows)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                p = torch.sigmoid(model(x).float()).cpu().numpy()
            for i, start in enumerate(selected):
                count = min(length, n - start)
                weights = inference_weight(length, start == 0, start + length >= n)[:count]
                total[start:start + count] += p[i, :count] * weights[:, None]
                denominator[start:start + count] += weights
    if (denominator <= 0).any():
        raise ValueError('Inference windows left native rows uncovered')
    prediction = total / denominator[:, None]
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite prediction')
    return np.clip(prediction, 1e-6, 1 - 1e-6)


def average_precision(y, p):
    result = {target: float(average_precision_score(y[:, i], p[:, i])) for i, target in enumerate(TARGETS)}
    result['two_target_mAP'] = float(np.mean(list(result.values())))
    return result


def run_experiment(cfg=CFG, output_root=None, synthetic_records=None):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if synthetic_records is None and device.type != 'cuda':
        raise RuntimeError('Full sequence training requires a GPU; run the smoke mode locally instead')
    if output_root is None:
        output_root = Path('/kaggle/working/tdcs_two_target_neural') if Path('/kaggle/working').exists() else ROOT / 'results/tdcs_two_target_neural'
    output_root = Path(output_root)
    if synthetic_records is None:
        raw_root, metadata_path = locate_data()
        manifest = build_manifest(raw_root, metadata_path)
        source_population = manifest[['recording_id', 'subject', 'fold', 'source_bytes']].to_csv(index=False)
        metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    else:
        manifest = pd.DataFrame([{key: record[key] for key in ['recording_id', 'subject', 'fold']} for record in synthetic_records])
        source_population = manifest.to_csv(index=False)
        metadata_sha = 'synthetic smoke only'
    source_sha = globals().get('SOURCE_SHA256') or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = {'configuration': asdict(cfg), 'source_sha256': source_sha,
              'runtime_versions': {'torch': str(torch.__version__), 'numpy': np.__version__, 'pandas': pd.__version__},
              'frozen_fold_sha256': hashlib.sha256(json.dumps(FROZEN_SUBJECT_FOLDS, sort_keys=True).encode()).hexdigest(),
              'metadata_sha256': metadata_sha, 'population_sha256': hashlib.sha256(source_population.encode()).hexdigest(),
              'validation': 'frozen outer subject folds; all native rows; no outer-label checkpoint selection',
              'confidence': 'independent sigmoid scores; event oversampling means not calibrated probability',
              'leaderboard_scoring_used': False, 'competition_submission_created': False,
              'previous_two_target_reference': 0.19629988455476777, 'goal_is_measured_not_guaranteed': 0.6,
              'smoke_only': synthetic_records is not None}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    run_dir = output_root / fingerprint
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'configuration.json', config)
    manifest.to_csv(run_dir / 'frozen_tdcs_manifest.csv', index=False)
    write_json(output_root / 'latest_run.json', {'run_dir': str(run_dir), 'status': 'running'})
    print(f'Fresh TDCS neural run: {run_dir}; torch {torch.__version__}; {device}; GPUs {torch.cuda.device_count()}', flush=True)
    records = prepare_records(manifest) if synthetic_records is None else synthetic_records
    model_metrics = []
    for variant in cfg.variants:
        for fold in cfg.folds:
            model = train_model(records, fold, variant, cfg, run_dir, device, fingerprint)
            oof_dir = run_dir / variant / 'oof'
            oof_dir.mkdir(parents=True, exist_ok=True)
            ys, ps = [], []
            for record in records:
                if record['fold'] != fold:
                    continue
                path = oof_dir / f'{record["recording_id"]}.npz'
                if path.exists():
                    with np.load(path, allow_pickle=False) as payload:
                        p = payload['prediction']
                else:
                    p = predict_record(model, record['x'], cfg, device)
                    np.savez_compressed(path, prediction=p, targets=np.asarray(TARGETS))
                if p.shape != record['y'].shape or not np.isfinite(p).all():
                    raise ValueError('Native-row OOF alignment failure')
                ys.append(record['y'])
                ps.append(p)
            score = {'variant': variant, 'fold': fold, **average_precision(np.concatenate(ys), np.concatenate(ps))}
            model_metrics.append(score)
            pd.DataFrame(model_metrics).to_csv(run_dir / 'fold_metrics.csv', index=False)
            write_json(run_dir / 'progress.json', {'stage': 'fold_scored', **score})
            print(json.dumps(score), flush=True)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    y = np.concatenate([record['y'] for record in records if record['fold'] in cfg.folds])
    predictions = {}
    for variant in cfg.variants:
        parts = []
        for record in records:
            if record['fold'] in cfg.folds:
                with np.load(run_dir / variant / 'oof' / f'{record["recording_id"]}.npz', allow_pickle=False) as payload:
                    parts.append(payload['prediction'])
        predictions[variant] = np.concatenate(parts)
    predictions['neural_equal_ensemble'] = sum(predictions.values()) / len(predictions)
    metrics = pd.DataFrame([{'method': method, **average_precision(y, p)} for method, p in predictions.items()])
    metrics.to_csv(run_dir / 'candidate_metrics.csv', index=False)
    selected = metrics.sort_values('two_target_mAP', ascending=False).iloc[0]
    selected_dir = run_dir / 'selected_candidate'
    (selected_dir / 'oof').mkdir(parents=True, exist_ok=True)
    manifest.to_csv(selected_dir / 'frozen_tdcs_manifest.csv', index=False)
    pd.DataFrame([selected]).to_csv(selected_dir / 'native_oof_metrics.csv', index=False)
    offset = 0
    for record in records:
        if record['fold'] not in cfg.folds:
            continue
        n = len(record['y'])
        np.savez_compressed(selected_dir / 'oof' / f'{record["recording_id"]}.npz',
                            prediction=predictions[str(selected.method)][offset:offset + n], targets=np.asarray(TARGETS))
        offset += n
    decision = {'run_dir': str(run_dir), 'selected_candidate_dir': str(selected_dir), 'status': 'complete',
                'selected_method': str(selected.method), 'two_target_mAP': float(selected.two_target_mAP),
                'class_average_precision': {target: float(selected[target]) for target in TARGETS},
                'reference_two_target_map': 0.19629988455476777,
                'target_two_target_map': 0.6, 'target_reached': bool(selected.two_target_mAP >= 0.6) and synthetic_records is None,
                'smoke_only': synthetic_records is not None, 'native_rows_scored': len(y),
                'no_outer_labels_in_training': True, 'competition_submission_created': False,
                'leaderboard_scoring_used': False, 'independent_generalization_established': False,
                'retained_defog_reported_oof_map': 0.817}
    write_json(run_dir / 'decision.json', decision)
    write_json(output_root / 'latest_run.json', decision)
    write_json(run_dir / 'progress.json', {'stage': 'complete', **decision})
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    return decision


def smoke_experiment(output_root=None):
    cfg = replace(CFG, name='synthetic_smoke_only', epochs=1, steps_per_epoch=2, batch_size=2,
                  train_length=32, inference_length=64, inference_batch=2, width=16, hidden=8, gru_layers=1)
    rng = np.random.default_rng(1)
    records = []
    for fold in range(5):
        raw = rng.normal(size=(96 + fold, 3)).astype(np.float32)
        y = np.zeros((len(raw), 2), np.float32)
        y[16:28, 0] = 1
        y[55:67, 1] = 1
        records.append({'recording_id': f'smoke_{fold}', 'subject': f'smoke_subject_{fold}', 'fold': fold,
                        'x': sensor_features(raw), 'y': y, 'negative': np.flatnonzero(y.sum(axis=1) == 0),
                        'centers': [event_centers(y[:, target]) for target in range(2)]})
    return run_experiment(cfg, output_root or ROOT / 'tmp/tdcs_neural_smoke', records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Run synthetic CPU wiring checks, not a performance experiment')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.smoke:
        smoke_experiment(args.output_root)
    else:
        run_experiment(output_root=args.output_root)
