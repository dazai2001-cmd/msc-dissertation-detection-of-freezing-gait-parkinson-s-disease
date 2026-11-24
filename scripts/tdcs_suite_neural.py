"""Metadata-aware sequence models and fold-safe masked reconstruction pretraining."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import tdcs_suite_data as data


class Network(nn.Module):
    def __init__(self, features, architecture, width=64, auxiliary=False):
        super().__init__()
        self.features, self.architecture, self.width, self.auxiliary = features, architecture, width, auxiliary
        self.input = nn.Sequential(nn.Linear(features, width), nn.LayerNorm(width), nn.GELU())
        if architecture == "tcn":
            self.context = nn.ModuleList([nn.Conv1d(width, width, 5, padding=2*d, dilation=d) for d in (1, 2, 4, 8)])
            size = width
        elif architecture in ("bigru", "bilstm"):
            cls = nn.GRU if architecture == "bigru" else nn.LSTM
            self.context = cls(width, width, 2, batch_first=True, bidirectional=True, dropout=.15)
            size = width * 2
        elif architecture == "attention":
            block = nn.TransformerEncoderLayer(width, 4, width * 2, .15, batch_first=True)
            self.context = nn.TransformerEncoder(block, 2)
            size = width
        else:
            raise ValueError("Unknown sequence architecture")
        self.output = nn.Linear(size, 3 if auxiliary else 2)
        self.decoder = nn.Linear(size, features)

    def encode(self, x):
        h = self.input(x)
        if self.architecture == "tcn":
            for layer in self.context:
                h = (h + F.dropout(F.gelu(layer(h.transpose(1, 2)).transpose(1, 2)), .15, self.training)) / math.sqrt(2)
        elif self.architecture in ("bigru", "bilstm"):
            h = self.context(h)[0]
        else:
            positions = torch.arange(h.shape[1], device=h.device, dtype=h.dtype)[:, None]
            frequencies = torch.exp(torch.arange(0, self.width, 2, device=h.device, dtype=h.dtype) * (-math.log(10000.) / self.width))
            position = torch.zeros(h.shape[1], self.width, device=h.device, dtype=h.dtype)
            position[:, 0::2], position[:, 1::2] = torch.sin(positions * frequencies), torch.cos(positions * frequencies)
            h = self.context(h + position)
        return h

    def forward(self, x):
        return self.output(self.encode(x))


def preprocessing(x, y, fitting):
    count, mean, m2 = 0, np.zeros(x.shape[1], np.float64), np.zeros(x.shape[1], np.float64)
    rows = np.flatnonzero(fitting)
    for start in range(0, len(rows), 4096):
        block = x[rows[start:start+4096]].astype(np.float64)
        n = len(block); delta = block.mean(0) - mean; total = count + n
        m2 += ((block - block.mean(0)) ** 2).sum(0) + delta ** 2 * count * n / total
        mean += delta * n / total; count = total
    scale = np.sqrt(np.maximum(m2 / max(count, 1), 0))
    scale[scale < 1e-5] = 1
    positive = y[rows].sum(0)
    pw = np.clip(np.sqrt((len(rows) - positive) / np.maximum(positive, 1)), 1, 8)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32), "positive_weight": pw.astype(np.float32)}


def transform(x, prep):
    return np.clip((x - prep["mean"]) / prep["scale"], -8, 8).astype(np.float32)


def sampling_groups(arrays, fitting):
    records = []
    for record in np.unique(arrays["record"][fitting]):
        rows = np.flatnonzero((arrays["record"] == record) & fitting)
        records.append(rows)
    subjects = np.asarray([str(arrays["subject"][r[0]]) for r in records])
    totals = {s: sum(len(r) for r, subject in zip(records, subjects) if subject == s) for s in set(subjects)}
    probability = np.asarray([len(r) / math.sqrt(totals[s]) for r, s in zip(records, subjects)], float)
    probability /= probability.sum()
    return records, probability


def batch(records, probability, rng, x, labels, cfg):
    values, ys, masks, trace = [], [], [], []
    for _ in range(cfg.batch_size):
        i = int(rng.choice(len(records), p=probability))
        rows = records[i]
        start = int(rng.integers(len(rows))) - cfg.window // 2
        positions = start + np.arange(cfg.window)
        valid = (positions >= 0) & (positions < len(rows))
        chosen = rows[np.clip(positions, 0, len(rows) - 1)]
        values.append(x[chosen]); ys.append(labels[chosen]); masks.append(valid)
        trace.append((int(rows[0]), start))
    return np.stack(values), np.stack(ys), np.stack(masks), trace


def train(context, x, fold, spec, cfg, directory, device, fingerprint, external=None, deadline=float("inf")):
    a = context["arrays"]
    fitting = a["fold"] != fold
    train_subjects, heldout_subjects = set(a["subject"][fitting]), set(a["subject"][~fitting])
    assert train_subjects.isdisjoint(heldout_subjects)
    seed = cfg.seed + 1009 * fold
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    prep = preprocessing(x, a["y"], fitting)
    np.savez_compressed(directory / "preprocessing.npz", **prep)
    xx = transform(x, prep)
    model = Network(x.shape[1], spec["architecture"], cfg.width, spec.get("kinetic", False)).to(device)
    labels = np.column_stack([a["y"], context["kinetic"]]).astype(np.float32)
    records, probability = sampling_groups(a, fitting)
    rng = np.random.default_rng(seed)
    pretrain_indices = np.empty(0, np.int64)
    if spec.get("pretrain"):
        if external is None:
            raise ValueError("Self-supervised method requires mapped external unlabeled records")
        pretrain_indices = data.external_training_indices(external, heldout_subjects)
        ux = transform(external["x"], prep)
        urecords = [np.flatnonzero((external["record"] == i) & np.isin(np.arange(len(ux)), pretrain_indices))
            for i in np.unique(external["record"][pretrain_indices])]
        urecords = [r for r in urecords if len(r)]
        up = np.asarray([len(r) for r in urecords], float); up /= up.sum()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.01)
        ulabels = np.zeros((len(ux), 4), np.float32)
        for _ in range(cfg.pretrain_steps):
            if time.monotonic() >= deadline:
                raise TimeoutError("Suite runtime budget reached during pretraining")
            bx, _, mask, _ = batch(urecords, up, rng, ux, ulabels, cfg)
            values = torch.from_numpy(bx).to(device)
            selected = torch.rand_like(values) < .2
            reconstruction = model.decoder(model.encode(values.masked_fill(selected, 0.)))
            valid = selected & torch.from_numpy(mask).to(device)[..., None]
            loss = ((reconstruction - values) ** 2)[valid].mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        torch.save({"model": model.state_dict(), "pretraining_steps": cfg.pretrain_steps,
            "fitting_external_indices": pretrain_indices, "fingerprint": fingerprint}, directory / "pretrained.pt")
        del ux
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.01)
    positive_weight = torch.from_numpy(prep["positive_weight"]).to(device)
    traces = []
    model.train()
    for epoch in range(cfg.neural_epochs):
        for _ in range(cfg.steps_per_epoch):
            if time.monotonic() >= deadline:
                raise TimeoutError("Suite runtime budget reached during supervised training")
            bx, by, mask, trace = batch(records, probability, rng, xx, labels, cfg)
            traces.extend(trace)
            inputs = torch.from_numpy(bx).to(device)
            truth = torch.from_numpy(by).to(device)
            valid = torch.from_numpy(mask).to(device).float()
            logits = model(inputs)
            losses = F.binary_cross_entropy_with_logits(logits[..., :2], truth[..., :2], pos_weight=positive_weight, reduction="none")
            loss = (losses * valid[..., None]).sum() / (valid.sum().clamp_min(1) * 2)
            if spec.get("kinetic"):
                kmask = valid * truth[..., 3]
                auxiliary = F.binary_cross_entropy_with_logits(logits[..., 2], truth[..., 2], reduction="none")
                loss = loss + .2 * (auxiliary * kmask).sum() / kmask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite neural loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            with torch.no_grad():
                for dst, src in zip(ema.parameters(), model.parameters()):
                    dst.lerp_(src, .005)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "completed_epochs": epoch + 1, "fingerprint": fingerprint}, directory / "epoch_checkpoint.pt")
        print(f"{spec['id']} fold {fold}: epoch {epoch+1}/{cfg.neural_epochs}", flush=True)
    np.savez_compressed(directory / "training_trace.npz", crops=np.asarray(traces, np.int64), external_indices=pretrain_indices)
    torch.save({"model": ema.state_dict(), "features": x.shape[1], "architecture": spec["architecture"],
        "width": cfg.width, "auxiliary": spec.get("kinetic", False), "fingerprint": fingerprint,
        "fold": fold}, directory / "model.pt")
    receipt = {"backend": str(device), "optimizer_updates": cfg.neural_epochs * cfg.steps_per_epoch,
        "fixed_epochs": cfg.neural_epochs, "final_weights": "EMA_0.995", "seed": seed,
        "fitting_subjects": sorted(map(str, train_subjects)), "heldout_subjects": sorted(map(str, heldout_subjects)),
        "external_pretraining_rows": len(pretrain_indices),
        "kinetic_supervised_fitting_grid_rows": int(context["kinetic"][fitting, 1].sum()) if spec.get("kinetic") else 0,
        "outer_labels_used_for_training_or_stopping": False}
    return ema.eval(), prep, receipt


def predict(model, x, prep, cfg, device):
    model.eval()
    xx = transform(x, prep)
    starts = list(range(-cfg.window // 2, len(x), cfg.window // 2))
    result, counts = np.zeros((len(x), 2), np.float64), np.zeros(len(x), np.float64)
    blend = np.maximum(np.hanning(cfg.window), .1)
    with torch.no_grad():
        for offset in range(0, len(starts), cfg.batch_size):
            group = starts[offset:offset + cfg.batch_size]
            windows = np.stack([xx[np.clip(s + np.arange(cfg.window), 0, len(xx)-1)] for s in group])
            values = torch.sigmoid(model(torch.from_numpy(windows).to(device))[..., :2]).cpu().numpy()
            for start, values in zip(group, values):
                left, right = max(0, start), min(len(x), start + cfg.window)
                if right <= left:
                    continue
                local = left - start; weights = blend[local:local+right-left]
                result[left:right] += values[local:local+right-left] * weights[:, None]
                counts[left:right] += weights
    assert (counts > 0).all()
    return (result / counts[:, None]).astype(np.float32)


def load(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = Network(payload["features"], payload["architecture"], payload["width"], payload["auxiliary"]).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()
