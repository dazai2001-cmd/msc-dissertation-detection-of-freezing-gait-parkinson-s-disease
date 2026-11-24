"""Lower-memory three-target TDCS candidates; immutable v2 helpers are reused.

These are new pooled-context candidates, not a claim that the v2 kernel-death
cause is known. Retained heads and DeFOG are never read for training or replaced.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import tdcs_alternative_architectures as legacy

common = legacy.common
ROOT = Path(__file__).resolve().parents[1]
TARGETS = legacy.TARGETS
ARCHITECTURES = ('pooled_cnn', 'gated_gru32')


@dataclass(frozen=True)
class Config:
    name: str = 'tdcs_resource_safe_v3'
    targets: tuple = TARGETS
    folds: tuple = (0, 1, 2, 3, 4)
    epochs: int = 24
    steps_per_epoch: int = 96
    batch_size: int = 16
    micro_batch: int = 2
    window: int = 2048
    inference_batch: int = 1
    width: int = 48
    pool_stride: int = 4
    dilations: tuple = (1, 2, 4, 8)
    hidden: int = 64
    gru_layers: int = 2
    dropout: float = .25
    weight_decay: float = .01
    learning_rate: float = .0007
    event_fraction: float = .5
    negative_fraction: float = .25
    ema_decay: float = .995
    gate_aux_weight: float = .2
    rank_weight: float = .03
    checkpoint_steps: int = 12
    max_seconds: int = 21600
    seed: int = 1701


CFG = Config()


def optional_text(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return None


def check_config(cfg):
    if tuple(cfg.targets) != TARGETS or tuple(cfg.folds) != (0, 1, 2, 3, 4):
        raise ValueError('All three targets and all five frozen folds are required')
    if cfg.micro_batch < 1 or cfg.batch_size % cfg.micro_batch:
        raise ValueError('Effective batch must be divisible by the microbatch')
    if cfg.width % 8 or min(cfg.window, cfg.pool_stride, cfg.epochs, cfg.steps_per_epoch,
                            cfg.checkpoint_steps, cfg.max_seconds) < 1:
        raise ValueError('Invalid width, context or training budget')
    if min(cfg.event_fraction, cfg.negative_fraction) < 0 or cfg.event_fraction + cfg.negative_fraction > 1:
        raise ValueError('Invalid sampling fractions')


class ResourceLog:
    """Best-effort host/GPU allocator observations plus a native-stack watchdog."""
    def __init__(self, directory, device, interval=30):
        self.directory, self.device, self.interval = Path(directory), device, interval
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stage, self.details = 'initializing', {}
        self.lock, self.stop = threading.Lock(), threading.Event()

    def emit(self, stage=None, **details):
        with self.lock:
            if stage is not None:
                self.stage, self.details = stage, details
            row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'stage': self.stage, **self.details}
            status = optional_text('/proc/self/status')
            if status is not None:
                for line in status.splitlines():
                    if line.startswith(('VmRSS:', 'VmHWM:', 'VmSwap:')):
                        key, value = line.split(':', 1)
                        row[key + '_KiB'] = int(value.split()[0])
            for name in ('memory.current', 'memory.max', 'memory.events'):
                value = optional_text(Path('/sys/fs/cgroup') / name)
                if value is not None:
                    row['cgroup_' + name] = value.strip()
            if self.device.type == 'cuda':
                row.update(gpu_allocated_bytes=torch.cuda.memory_allocated(self.device),
                           gpu_reserved_bytes=torch.cuda.memory_reserved(self.device),
                           gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device))
            with (self.directory / 'resources.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
                stream.flush()

    def _watch(self):
        while not self.stop.wait(self.interval):
            try:
                self.emit()
            except Exception:
                # A missing optional telemetry source must not stop training.
                pass

    def __enter__(self):
        self.trace = (self.directory / 'native_stacks.log').open('a', encoding='utf-8')
        self.watchdog_enabled = False
        try:
            faulthandler.enable(file=self.trace, all_threads=True)
            faulthandler.dump_traceback_later(180, repeat=True, file=self.trace)
            self.watchdog_enabled = True
        except (RuntimeError, OSError):
            pass
        self.emit('starting')
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.watchdog_enabled:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()
        self.trace.close()


class Network(nn.Module):
    def __init__(self, architecture, cfg=CFG):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError('Unknown candidate')
        self.architecture = architecture
        self.stem = nn.Sequential(nn.Conv1d(12, cfg.width, 7, padding=3),
                                  nn.GroupNorm(8, cfg.width), nn.GELU())
        self.pool = nn.AvgPool1d(cfg.pool_stride, cfg.pool_stride, ceil_mode=True,
                                 count_include_pad=False)
        if architecture == 'pooled_cnn':
            self.temporal = nn.Sequential(*[legacy.MultiscaleBlock(cfg.width, d, cfg.dropout)
                                             for d in cfg.dilations])
            context_width = cfg.width
        else:
            self.temporal = nn.GRU(cfg.width, cfg.hidden, cfg.gru_layers, batch_first=True,
                bidirectional=True, dropout=cfg.dropout if cfg.gru_layers > 1 else 0)
            context_width = cfg.hidden * 2
        self.fuse = nn.Sequential(nn.Conv1d(cfg.width + context_width, cfg.width, 1),
                                  nn.GroupNorm(8, cfg.width), nn.GELU(), nn.Dropout(cfg.dropout))
        self.types = nn.Conv1d(cfg.width, 3, 1)
        self.gate = nn.Conv1d(cfg.width, 1, 1) if architecture == 'gated_gru32' else None

    def parts(self, x):
        local = self.stem(x.transpose(1, 2))
        pooled = self.pool(local)
        if self.architecture == 'pooled_cnn':
            context = self.temporal(pooled)
        else:
            context, _ = self.temporal(pooled.transpose(1, 2))
            context = context.transpose(1, 2)
        context = F.interpolate(context, size=local.shape[2], mode='linear', align_corners=False)
        fused = self.fuse(torch.cat([local, context], dim=1))
        logits = self.types(fused).transpose(1, 2).float()
        gate = self.gate(fused).transpose(1, 2).float() if self.gate is not None else None
        if gate is not None:
            probability = (torch.sigmoid(logits) * torch.sigmoid(gate)).clamp(1e-6, 1-1e-6)
            logits = torch.logit(probability)
        return logits, gate

    def forward(self, x):
        return self.parts(x)[0]


def ranking_loss(logits, y, mask):
    terms = []
    for target in (0, 2):
        valid = mask[..., target].bool()
        positives = logits[..., target][valid & (y[..., target] > .5)]
        negatives = logits[..., target][valid & (y[..., target] < .5)]
        if len(positives) and len(negatives):
            pi = torch.linspace(0, len(positives)-1, min(256, len(positives)), device=logits.device).long()
            ni = torch.linspace(0, len(negatives)-1, min(256, len(negatives)), device=logits.device).long()
            count = min(len(pi), len(ni))
            terms.append(F.softplus(negatives[ni[:count]] - positives[pi[:count]]).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def loss_for_parts(logits, gate, y, mask, cfg):
    value = legacy.masked_loss(logits, y, mask)
    if gate is not None:
        gate_mask, gate_y = mask.amax(-1, keepdim=True), y.amax(-1, keepdim=True)
        value = value + cfg.gate_aux_weight * legacy.masked_loss(gate, gate_y, gate_mask)
        if cfg.rank_weight:
            value = value + cfg.rank_weight * ranking_loss(logits, y, mask)
    return value


def backward_microbatches(model, batch, cfg, device, scaler):
    """One optimizer batch, weighted by valid rows even when padding differs."""
    total_valid = batch[2].sum().clamp_min(1)
    value = 0.
    for start in range(0, len(batch[0]), cfg.micro_batch):
        x, y, mask = [part[start:start+cfg.micro_batch].to(device) for part in batch]
        weight = float(mask.sum().detach().cpu() / total_valid)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits, gate = model.parts(x)
            loss = loss_for_parts(logits, gate, y, mask, cfg) * weight
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite microbatch loss')
        scaler.scale(loss).backward()
        value += float(loss.detach().cpu())
    return value


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(item) for item in value)
    return value


def atomic_checkpoint(state, path):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(cpu_state(state), temporary)
    os.replace(temporary, path)


def train_model(records, fold, architecture, cfg, directory, device, fingerprint, health, deadline):
    directory.mkdir(parents=True, exist_ok=True)
    seed = cfg.seed + 1009 * fold
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    model = Network(architecture, cfg).to(device)
    final = directory / 'final_ema.pt'
    if final.is_file():
        state = torch.load(final, map_location='cpu', weights_only=False)
        assert state['fingerprint'] == fingerprint and tuple(state['targets']) == TARGETS
        model.load_state_dict(state['model'])
        return model
    sampler = legacy.Sampler(records, fold, cfg)
    training = sorted(sampler.subjects)
    heldout = sorted({record['subject'] for record in records if record['fold'] == fold})
    assert set(training).isdisjoint(heldout)
    common.write_json(directory / 'training_receipt.json', {
        'targets': TARGETS, 'architecture': architecture, 'training_subjects': training,
        'held_out_subjects': heldout, 'fixed_epochs': cfg.epochs,
        'outer_labels_used_for_checkpoint_selection': False,
        'training_window': cfg.window, 'inference_window': cfg.window,
        'effective_batch': cfg.batch_size, 'micro_batch': cfg.micro_batch,
        'gradient_accumulation': cfg.batch_size // cfg.micro_batch,
        'data_parallel': False, 'device': str(device), 'seed': seed,
        'previous_models_overwritten': False})
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: common.learning_rate_multiplier(step, cfg.epochs * cfg.steps_per_epoch))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    rng = np.random.default_rng(seed)
    resume, history, first_epoch, first_step, partial_sum = directory/'resume.pt', [], 0, 0, 0.
    if resume.is_file():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        assert state['fingerprint'] == fingerprint and tuple(state['targets']) == TARGETS
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], device)
        first_epoch, first_step = state['next_epoch'], state['next_step']
        history, partial_sum = state['history'], state['partial_loss_sum']
    for epoch in range(first_epoch, cfg.epochs):
        started, loss_sum = time.monotonic(), partial_sum if epoch == first_epoch else 0.
        begin = first_step if epoch == first_epoch else 0
        model.train()
        print(f'{architecture} fold {fold} epoch {epoch+1}/{cfg.epochs}', flush=True)
        for step in range(begin, cfg.steps_per_epoch):
            if time.monotonic() >= deadline:
                raise TimeoutError('Six-hour safety budget reached; latest checkpoint is retained')
            health.emit('training', fold=fold, epoch=epoch+1, step=step+1)
            optimizer.zero_grad(set_to_none=True)
            loss_sum += backward_microbatches(model, sampler.batch(rng), cfg, device, scaler)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            with torch.no_grad():
                for destination, source in zip(ema.parameters(), model.parameters()):
                    destination.lerp_(source.detach(), 1-cfg.ema_decay)
            last = step+1 == cfg.steps_per_epoch
            if last:
                history.append({'epoch': epoch+1, 'loss': loss_sum/cfg.steps_per_epoch,
                    'seconds': time.monotonic()-started, 'learning_rate': schedule.get_last_lr()[0]})
                pd.DataFrame(history).to_csv(directory/'training_log.csv', index=False)
            if last or (step+1) % cfg.checkpoint_steps == 0:
                health.emit('checkpoint', fold=fold, epoch=epoch+1, step=step+1)
                atomic_checkpoint({'fingerprint': fingerprint, 'targets': TARGETS,
                    'model': model.state_dict(), 'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': schedule.state_dict(),
                    'scaler': scaler.state_dict(), 'numpy_rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                    'next_epoch': epoch+1 if last else epoch, 'next_step': 0 if last else step+1,
                    'partial_loss_sum': 0. if last else loss_sum, 'history': history}, resume)
                common.write_json(directory.parent/'progress.json', {'stage': 'training',
                    'architecture': architecture, 'fold': fold, 'epoch': epoch+1,
                    'step': step+1, 'fixed_epochs': cfg.epochs, 'checkpoint': str(resume)})
    atomic_checkpoint({'fingerprint': fingerprint, 'targets': TARGETS, 'model': ema.state_dict()}, final)
    return ema


def source_hash(module, override):
    return globals().get(override) or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def run_experiment(architecture, cfg=CFG, output_root=None, synthetic_records=None):
    check_config(cfg)
    if architecture not in ARCHITECTURES:
        raise ValueError('Unknown architecture')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if synthetic_records is None and device.type != 'cuda':
        raise RuntimeError('Full training requires CUDA; no silent synthetic fallback')
    manifest = None
    if synthetic_records is None:
        raw, metadata = common.locate_data()
        manifest = common.build_manifest(raw, metadata)
        population = manifest[['recording_id','subject','fold','source_bytes']].to_csv(index=False, lineterminator='\n')
    else:
        population = 'synthetic wiring only; not TDCS performance'
    configuration = {'configuration': asdict(cfg), 'architecture': architecture, 'targets': TARGETS,
        'source_sha256': globals().get('SOURCE_SHA256') or hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'legacy_source_sha256': source_hash(legacy, 'LEGACY_SOURCE_SHA256'),
        'common_source_sha256': source_hash(common, 'COMMON_SOURCE_SHA256'),
        'population_sha256': hashlib.sha256(population.encode()).hexdigest(),
        'smoke_only': synthetic_records is not None,
        'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__,
                    'visible_gpu_count': torch.cuda.device_count(), 'used_gpu_indices': [0] if device.type=='cuda' else []},
        'checkpoint_selection': 'fixed final EMA; no outer-label epoch selection',
        'inference_mode': 'offline; native-row output with pooled sensor context and local skip',
        'confidence': 'uncalibrated independent sigmoid scores',
        'failure_cause_of_v2_confirmed': False, 'retained_heads_overwritten': False,
        'competition_submission_created': False}
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:20]
    output_root = Path(output_root or ROOT/'results/tdcs_resource_safe')
    run = output_root/fingerprint
    run.mkdir(parents=True, exist_ok=True)
    common.write_json(run/'configuration.json', configuration)
    if manifest is not None:
        manifest.to_csv(run/'frozen_tdcs_manifest.csv', index=False)
    (run/'oof').mkdir(exist_ok=True)
    deadline = time.monotonic() + cfg.max_seconds
    with ResourceLog(run, device) as health:
        health.emit('preparing_records')
        records = legacy.prepare_records(manifest) if synthetic_records is None else synthetic_records
        rows, labels, predictions = [], [], []
        for fold in cfg.folds:
            model = train_model(records, fold, architecture, cfg, run/f'fold_{fold}', device, fingerprint, health, deadline)
            ys, ps = [], []
            for record in records:
                if record['fold'] != fold:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError('Six-hour safety budget reached during inference')
                health.emit('inference', fold=fold, recording_id=record['recording_id'])
                pred = legacy.predict_record(model, record['x'], cfg, device)
                assert pred.shape == record['y'].shape
                np.savez_compressed(run/'oof'/f'{record["recording_id"]}.npz', prediction=pred, targets=np.asarray(TARGETS))
                ys.append(record['y'])
                ps.append(pred)
            rows.append({'fold': fold, **legacy.score(np.concatenate(ys), np.concatenate(ps))})
            pd.DataFrame(rows).to_csv(run/'fold_metrics.csv', index=False)
            labels.extend(ys)
            predictions.extend(ps)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        metrics = legacy.score(np.concatenate(labels), np.concatenate(predictions))
        pd.DataFrame([metrics]).to_csv(run/'native_oof_metrics.csv', index=False)
        result = {'run_dir': str(run), 'status': 'complete', 'architecture': architecture, **metrics,
            'smoke_only': synthetic_records is not None, 'native_rows': sum(len(y) for y in labels),
            'goal_two_target_mAP': .6,
            'goal_reached': metrics['two_target_mAP'] >= .6 and synthetic_records is None,
            'retained_heads_unchanged': True, 'independent_generalization_established': False,
            'competition_submission_created': False}
        common.write_json(run/'decision.json', result)
        common.write_json(run/'progress.json', result)
        common.write_json(output_root/'latest_run.json', result)
        health.emit('complete', native_rows=result['native_rows'])
    print(json.dumps(result, indent=2), flush=True)
    return result


def smoke_experiment(architecture, output_root):
    cfg = replace(CFG, name='synthetic_resource_safe_wiring', epochs=1, steps_per_epoch=2,
        batch_size=4, micro_batch=2, window=64, width=16, dilations=(1,2),
        hidden=8, gru_layers=1, checkpoint_steps=1)
    rng, records = np.random.default_rng(1), []
    for fold in range(5):
        x = common.sensor_features(rng.normal(size=(96+fold,3)).astype(np.float32))[:,:12].copy()
        y = np.zeros((len(x),3), np.float32)
        for target, begin in enumerate((12,35,62)):
            y[begin:begin+12,target] = 1
        records.append({'recording_id':f'smoke_{fold}', 'subject':f'synthetic_{fold}',
            'fold':fold, 'x':x, 'y':y, 'negative':np.flatnonzero(y.sum(1)==0),
            'centers':[common.event_centers(y[:,i]) for i in range(3)]})
    return run_experiment(architecture, cfg, output_root, records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=ARCHITECTURES, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke_experiment(args.architecture, args.output_root)
    else:
        run_experiment(args.architecture, output_root=args.output_root)
