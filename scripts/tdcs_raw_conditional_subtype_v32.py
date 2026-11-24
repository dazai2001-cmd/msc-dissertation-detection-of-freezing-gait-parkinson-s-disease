"""S13: a Walking-vs-other-FOG term on the unchanged M22 natural stream."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import tdcs_raw_weak_transfer_v29 as weak


raw = weak.raw
METHOD = "S13"
COEFFICIENT = 0.5
SPEC = {
    **weak.SPEC,
    "suite": "sensor",
    "matched_control": "M22",
    "fine_tune_changed": True,
    "conditional_target": "Walking vs StartHesitation or Turn given an annotated FOG row",
    "conditional_logit": "Walking logit - logsumexp(StartHesitation logit, Turn logit)",
    "conditional_coefficient": COEFFICIENT,
    "conditional_normalization": "same class weights and full valid batch mass as base cross entropy",
    "conditional_unknown_or_background_rows": "excluded from conditional term",
    "natural_crop_stream_changed": False,
    "pair_or_rank_loss": False,
}


def source_hash():
    return globals().get("SOURCE_SHA256") or raw.data.digest(__file__)


def effective_source_hash():
    pins = {"weak_M22_source": weak.effective_source_hash(), "conditional_source": source_hash()}
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def conditional_losses(logits, labels, valid):
    """Per-row binary loss over known FOG subtypes; ignore invalid/background rows."""
    if logits.shape != (*labels.shape, 4) or labels.shape != valid.shape:
        raise ValueError("Aligned four-state logits, labels and masks required")
    log_odds = logits[..., 3] - torch.logsumexp(logits[..., 1:3], dim=-1)
    values = F.binary_cross_entropy_with_logits(log_odds, labels.eq(3).to(log_odds.dtype), reduction="none")
    return values.masked_fill(~(valid & labels.ge(1) & labels.le(3)), 0.)


def train_fine(model, records, cfg, steps, lr, seed, deadline, ema):
    if ema is None or not records or any(r["domain"] != "tdcs" for r in records):
        raise ValueError("Conditional subtype loss is restricted to outer fitting TDCS")
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    records, probability = raw.sampling_plan(records)
    weights, counts = raw.class_weights(records)
    weight = torch.from_numpy(weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    trace, seen = [], np.zeros(4, np.int64)
    micros_with_fog = 0
    model.train()
    started = time.monotonic()
    for step in range(steps):
        if time.monotonic() >= deadline:
            raise TimeoutError("Conditional subtype runtime budget reached")
        bx, by, mask, crops = raw.batch(records, probability, rng, cfg)
        mass = float((weights[by] * mask).sum())
        if mass <= 0:
            raise ValueError("No valid fitting rows in the sampled batch")
        trace.extend(crops)
        seen += np.bincount(by[mask], minlength=4)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, cfg.batch_size, cfg.micro_batch):
            stop = start + cfg.micro_batch
            x = torch.from_numpy(bx[start:stop]).to(device)
            y = torch.from_numpy(by[start:stop]).to(device)
            valid = torch.from_numpy(mask[start:stop]).to(device)
            micros_with_fog += int(np.any(mask[start:stop] & (by[start:stop] > 0)))
            logits = model(x)
            base = F.cross_entropy(logits.transpose(1, 2), y, weight=weight, reduction="none")
            cond = conditional_losses(logits, y, valid)
            loss = (base * valid).sum() / mass + COEFFICIENT * (cond * weight[y]).sum() / mass
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite conditional subtype loss")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            for dst, src in zip(ema.parameters(), model.parameters()):
                dst.lerp_(src, 1-cfg.ema_decay)
        if (step + 1) % 128 == 0:
            print(f"Conditional subtype phase: {step+1}/{steps} updates; {micros_with_fog} FOG microbatches", flush=True)
    return {
        "updates": steps,
        "seed": seed,
        "class_counts": counts.tolist(),
        "class_weights": weights.tolist(),
        "subjects": sorted({r["subject"] for r in records}),
        "elapsed_seconds": time.monotonic()-started,
        "trace": trace,
        "conditional_usage": {
            "valid_sampled_rows_by_class": seen.tolist(),
            "micros_with_any_FOG": micros_with_fog,
            "total_microbatches": steps * (cfg.batch_size // cfg.micro_batch),
        },
    }


@contextmanager
def applied(usage=None):
    with weak.applied():
        prior_source, prior_train = raw.SOURCE_SHA256, raw.train_phase

        def phase(model, records, cfg, steps, lr, seed, deadline, ema=None):
            if ema is None:
                return prior_train(model, records, cfg, steps, lr, seed, deadline)
            out = train_fine(model, records, cfg, steps, lr, seed, deadline, ema)
            if usage is not None:
                usage.append({"seed": seed, **out["conditional_usage"]})
            return out

        assert METHOD not in raw.SPECS
        raw.train_phase = phase
        raw.SOURCE_SHA256 = effective_source_hash()
        raw.SPECS[METHOD] = SPEC
        try:
            yield
        finally:
            raw.train_phase = prior_train
            raw.SOURCE_SHA256 = prior_source
            raw.SPECS.pop(METHOD)


def run(session):
    usage = []
    with applied(usage):
        result = raw.run(session, METHOD)
    if result["status"] == "complete":
        destination = Path(result["run_directory"]) / "conditional_usage.json"
        if len(usage) == 5:
            cfg = raw.smoke_config() if session["cfg"].smoke else raw.Config()
            for fold, row in enumerate(usage):
                row["fold"] = fold
                assert row["seed"] == cfg.seed + 1009*fold + 200000
            raw.base.write(destination, {
                "id": METHOD, "specification": SPEC, "source_sha256": effective_source_hash(),
                "smoke_only": cfg.smoke, "folds": usage, "outer_labels_used_for_training": False,
            })
        elif not destination.is_file():
            raise ValueError("Incomplete conditional loss usage receipt")
    return result


def finish(session):
    import tdcs_raw_paired_crops_v31 as paired

    with applied():
        prior = {k: raw.SPECS[k] for k in ("S11", "S12") if k in raw.SPECS}
        assert not prior
        raw.SPECS.update(S11=paired.rank.SPEC, S12=paired.SPECS["S12"])
        try:
            return raw.finish(session)
        finally:
            raw.SPECS.pop("S11")
            raw.SPECS.pop("S12")
