"""Reuse the frozen 213 local + 112 recording sensor-feature recipes on Kaggle."""
from __future__ import annotations
import ast
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import tdcs_multiscale_signal_experiment as base
import tdcs_native_context_experiment as native

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCES = {
    "tdcs_dense_context_experiment.py": "7877338e575f9cc824a5ab9227f7ec1ce6ce4d7b1f240e71bfba3a07bff96ba4",
    "tdcs_recording_context.py": "7501dfdd906f90ded1c691c9f712ad721f10f25b77d5efda0f436bb6f1db4c3d"}
FEATURE_CFG = replace(base.CFG, feature_version="dense_temporal_spectral_v2", training_stride=16)
DENSE = ROOT / "results/tdcs_dense_context/e1844437c340c0b75d39/dense_context_features.npz"
GLOBAL = ROOT / "results/tdcs_recording_context/a92c6da6e016faa1cc85/recording_features.npz"
CACHE_HASHES = {
    "dense": "19935bb84e9456616de70e2174a7839ac80fc8aa0e285bae3e1db4e2e0cd6f50",
    "global": "1e79e1b74b9c2b3f07a6222f4406c5240e463efb73ac3e3d588a5adea084b23d"}


def recipe_functions():
    """Compile only unchanged feature functions, not unrelated model pipelines."""
    namespace = {"np": np, "base": base, "native": native, "common": native.common,
        "sliding_window_view": sliding_window_view, "FEATURE_CFG": FEATURE_CFG,
        "FFT_WINDOWS": (256, 1024), "CONTEXT_SHIFTS": (-256, -64, 64, 256)}
    selections = {
        "tdcs_dense_context_experiment.py": ("spectral_features", "temporal_context", "expanded_features"),
        "tdcs_recording_context.py": ("recording_features",)}
    embedded = globals().get("FROZEN_FEATURE_SOURCES", {})
    for filename, names in selections.items():
        source = embedded.get(filename)
        if source is None:
            source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        if hashlib.sha256(source.encode()).hexdigest() != FROZEN_SOURCES[filename]:
            raise ValueError("Frozen feature source changed: " + filename)
        functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        tree = ast.Module(body=[functions[name] for name in names], type_ignores=[])
        exec(compile(tree, filename, "exec"), namespace)
    return namespace["expanded_features"], namespace["recording_features"]


def recording_features(raw):
    dense_function, global_function = recipe_functions()
    grid, local, names = dense_function(raw)
    whole, whole_names = global_function(raw)
    x = np.column_stack([local, np.broadcast_to(whole, (len(grid), 112))]).astype(np.float32)
    names = [*names, *["file_" + name for name in whole_names]]
    assert x.shape == (len(grid), 325) and len(set(names)) == 325 and np.isfinite(x).all()
    return grid, x, names


def prepare_arrays(manifest):
    local_function, global_function = recipe_functions()
    parts = {key: [] for key in ("x", "y", "fold", "record", "grid")}
    lengths, names = [], None
    for i, row in enumerate(manifest.itertuples(index=False)):
        frame = pd.read_csv(row.path, usecols=[*base.AXES, *base.TARGETS])
        raw = frame.loc[:, list(base.AXES)].to_numpy()
        labels = frame.loc[:, list(base.TARGETS)].to_numpy(np.uint8)
        assert np.isin(labels, [0, 1]).all() and (labels.sum(1) <= 1).all()
        grid, local, local_names = local_function(raw)
        whole, whole_names = global_function(raw)
        current_names = [*local_names, *["file_" + name for name in whole_names]]
        assert names is None or names == current_names
        names = current_names
        x = np.column_stack([local, np.broadcast_to(whole, (len(grid), 112))]).astype(np.float32)
        assert x.shape[1] == 325 and np.isfinite(x).all()
        parts["x"].append(x); parts["y"].append(labels[grid][:, [0, 2]])
        parts["fold"].append(np.full(len(grid), row.fold, np.int8))
        parts["record"].append(np.full(len(grid), i, np.int16)); parts["grid"].append(grid)
        lengths.append(len(frame))
        if (i + 1) % 100 == 0:
            print(f"Frozen engineered sensor features: {i + 1}/815 recordings", flush=True)
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    arrays.update(lengths=np.asarray(lengths, np.int32), feature_names=np.asarray(names),
        recording_ids=manifest.recording_id.to_numpy(str),
        subject=manifest.subject.to_numpy(str)[arrays["record"]])
    assert arrays["x"].shape == (436128, 325) and int(arrays["lengths"].sum()) == 6959762
    return arrays


def load_local_arrays(manifest):
    """Read exact, already-validated local caches; this does not fit a model."""
    for path, name in ((DENSE, "dense"), (GLOBAL, "global")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == CACHE_HASHES[name]
    with np.load(DENSE, allow_pickle=False) as payload:
        local = {key: payload[key].copy() for key in payload.files}
    with np.load(GLOBAL, allow_pickle=False) as payload:
        whole = {key: payload[key].copy() for key in payload.files}
    ids = manifest.recording_id.to_numpy(str)
    assert np.array_equal(local["recording_ids"], ids) and np.array_equal(whole["recording_ids"], ids)
    rows = local["record"]
    result = {"x": np.column_stack([local["x"], whole["x"][rows]]).astype(np.float32),
        "y": local["y"][:, [0, 2]].copy(), "fold": local["fold"], "record": rows,
        "grid": local["grid"], "lengths": local["row_counts"], "recording_ids": ids,
        "feature_names": np.r_[local["feature_names"], np.char.add("file_", whole["feature_names"])],
        "subject": manifest.subject.to_numpy(str)[rows]}
    assert result["x"].shape == (436128, 325) and np.isfinite(result["x"]).all()
    return result


def feature_digest(arrays):
    value = hashlib.sha256()
    for key in ("x", "y", "fold", "record", "grid", "lengths"):
        a = np.ascontiguousarray(arrays[key])
        value.update(key.encode()); value.update(a.dtype.str.encode())
        value.update(str(a.shape).encode()); value.update(a.tobytes())
    value.update("\n".join(arrays["feature_names"].tolist()).encode())
    value.update("\n".join(arrays["recording_ids"].tolist()).encode())
    return value.hexdigest()
