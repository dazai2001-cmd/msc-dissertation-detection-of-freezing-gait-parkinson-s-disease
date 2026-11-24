"""S11: S10's raw transfer recipe plus a Walking-vs-other-FOG ranking term.

Same typed input, warmup, architecture, optimizer, crop plan, seeds, fixed
budget and inference as S10. Only TDCS fine-tune loss changes. No held-out
labels, predictions, or thresholds participate in fitting. Preserve S10 and
the frozen v28 reader/model source by pinning this variant separately.
"""
from contextlib import contextmanager
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import tdcs_raw_transfer_v28_recovery as typed

raw = typed.raw
ORIGINAL_TRAIN = raw.train_phase
METHOD = 'S11'
TERM_WEIGHT, MARGIN, HARD_NEGATIVES = .2, .2, 16
SPEC = {'suite':'sensor', 'transfer':True, 'matched_control':'S10', 'rank_term_weight':TERM_WEIGHT,
    'rank_margin_log_odds':MARGIN, 'hard_negatives_per_microbatch':HARD_NEGATIVES,
    'ranking_positives':'all valid Walking rows in the microbatch',
    'ranking_negatives':'at most 16 highest Walking log-odds among valid SH/Turn rows',
    'normalization':'mean over positive-negative pairs; average over all microbatches, including zero terms',
    'training_change':'add rank term to TDCS fine-tune cross entropy only'}


def source_hash():
    return globals().get('SOURCE_SHA256') or raw.data.digest(__file__)


def effective_source_hash():
    sources = {'typed_control_source':typed.effective_source_hash(), 'Walking_rank_source':source_hash()}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def rank_term(logits, labels, valid):
    """Return mean softplus ranking loss and its exact pair count."""
    if logits.shape != (*labels.shape, 4) or labels.shape != valid.shape:
        raise ValueError('Aligned four-state logits, labels and masks are required')
    walking = valid & labels.eq(3)
    other_fog = valid & (labels.eq(1) | labels.eq(2))
    score = logits[..., 3] - torch.logsumexp(logits[..., :3], dim=-1)
    positives = score[walking]
    negatives = score[other_fog]
    if not positives.numel() or not negatives.numel():
        return score.sum() * 0., 0
    hard = torch.topk(negatives, min(negatives.numel(), HARD_NEGATIVES), sorted=False).values
    return F.softplus(hard[None, :] - positives[:, None] + MARGIN).mean(), positives.numel() * hard.numel()


def train_fine(model, records, cfg, steps, lr, seed, deadline, ema):
    """Keep S10's sample stream, weight mass, optimizer and EMA arithmetic."""
    if ema is None or any(r['domain'] != 'tdcs' for r in records):
        raise ValueError('Ranking is allowed only in the TDCS fine-tune phase')
    torch.manual_seed(seed)
    if next(model.parameters()).is_cuda: torch.cuda.manual_seed_all(seed)
    device = next(model.parameters()).device; rng = np.random.default_rng(seed)
    records, p = raw.sampling_plan(records)
    weights, counts = raw.class_weights(records); weight = torch.from_numpy(weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    trace = []; eligible_micros = 0; pairs = 0
    micro_count = cfg.batch_size // cfg.micro_batch
    model.train(); started = time.monotonic()
    for step in range(steps):
        if time.monotonic() >= deadline: raise TimeoutError('Raw sequence runtime budget reached')
        bx, by, mask, crops = raw.batch(records, p, rng, cfg)
        mass = float((weights[by] * mask).sum())
        if mass <= 0: raise ValueError('Sampled batch contains no annotated positions')
        trace.extend(crops); optimizer.zero_grad(set_to_none=True)
        for start in range(0, cfg.batch_size, cfg.micro_batch):
            stop = start + cfg.micro_batch; x = torch.from_numpy(bx[start:stop]).to(device)
            y = torch.from_numpy(by[start:stop]).to(device); valid = torch.from_numpy(mask[start:stop]).to(device)
            logits = model(x)
            base_loss = F.cross_entropy(logits.transpose(1,2), y, weight=weight, reduction='none')
            rank, n_pairs = rank_term(logits, y, valid)
            pairs += n_pairs; eligible_micros += int(n_pairs > 0)
            loss = (base_loss * valid.float()).sum()/mass + (TERM_WEIGHT / micro_count) * rank
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite Walking ranking loss')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        with torch.no_grad():
            for dst, src in zip(ema.parameters(), model.parameters()): dst.lerp_(src, 1-cfg.ema_decay)
        if (step+1)%128 == 0:
            print(f'Walking rank phase: {step+1}/{steps} updates; {eligible_micros} eligible microbatches', flush=True)
    if not cfg.smoke and not eligible_micros: raise ValueError('No valid Walking/other-FOG ranking pairs in the fit')
    return {'updates':steps,'seed':seed,'class_counts':counts.tolist(),'class_weights':weights.tolist(),
        'subjects':sorted({r['subject'] for r in records}),'elapsed_seconds':time.monotonic()-started,'trace':trace,
        'rank_usage':{'eligible_microbatches':eligible_micros, 'total_microbatches':steps*micro_count, 'pairs':pairs}}


@contextmanager
def applied(usage=None):
    assert raw.source_hash() == typed.FROZEN_RAW_SHA256 and METHOD not in raw.SPECS
    prior_prepare, prior_train = raw.prepare, raw.train_phase
    had_override = 'SOURCE_SHA256' in raw.__dict__; override = raw.__dict__.get('SOURCE_SHA256')

    def phase(model, records, cfg, steps, lr, seed, deadline, ema=None):
        if ema is None: return ORIGINAL_TRAIN(model, records, cfg, steps, lr, seed, deadline)
        result = train_fine(model, records, cfg, steps, lr, seed, deadline, ema)
        if usage is not None: usage.append({'seed':seed, **result['rank_usage']})
        return result

    raw.prepare = typed.prepare; raw.train_phase = phase
    raw.SOURCE_SHA256 = effective_source_hash(); raw.SPECS[METHOD] = SPEC
    try: yield
    finally:
        raw.prepare = prior_prepare; raw.train_phase = prior_train
        if had_override: raw.SOURCE_SHA256 = override
        else: raw.__dict__.pop('SOURCE_SHA256', None)
        raw.SPECS.pop(METHOD)


def run(session):
    usage = []
    with applied(usage): result = raw.run(session,METHOD)
    if result['status'] == 'complete':
        cfg = raw.smoke_config() if session['cfg'].smoke else raw.Config()
        # A rerun can reuse existing folds; record exact usage only on the first full fit.
        destination = Path(result['run_directory'])/'walking_rank_usage.json'
        if len(usage) == 5:
            for fold, info in enumerate(usage):
                info['fold'] = fold
                assert info['seed'] == cfg.seed+1009*fold+200000
            raw.base.write(destination, {'source_sha256':effective_source_hash(), 'folds':usage,
                'config':SPEC, 'warmup_received_ranking_term':False, 'smoke_only':cfg.smoke})
        elif not destination.is_file():
            raise ValueError('Rank usage receipt missing for reused folds; rerun in a fresh root')
    return result


def finish(session):
    with applied(): return raw.finish(session)
