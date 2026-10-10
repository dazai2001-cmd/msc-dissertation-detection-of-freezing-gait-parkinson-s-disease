"""TDCS-only dense temporal/spectral HGB experiment; preserve previous winners.

One new feature/model configuration and four fixed candidate comparisons against
the retained 0.3800 blend. Settings are written before evaluation. No leaderboard
calls, learned OOF stacker, outer-label early stopping, or DeFOG mutations.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import tdcs_multiscale_signal_experiment as base
import tdcs_candidate_sweep as previous


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "tdcs_dense_context"
TARGETS = base.TARGETS
FEATURE_CFG = replace(base.CFG, feature_version="dense_temporal_spectral_v2", training_stride=16)
MODEL_PARAMS = {
    "learning_rate": 0.07, "max_iter": 260, "max_leaf_nodes": 31,
    "min_samples_leaf": 200, "l2_regularization": 20.0,
    "max_bins": 128, "early_stopping": False,
}
FFT_WINDOWS = (256, 1024)
CONTEXT_SHIFTS = (-256, -64, 64, 256)
FIXED_BLENDS = {
    "neural_dense_equal": {"previous_tdcs_blend": 0.5, "dense_context_hgb": 0.5},
    "retained_dense_quarter": {"retained_neural_hgb": 0.75, "dense_context_hgb": 0.25},
    "retained_dense_equal": {"retained_neural_hgb": 0.5, "dense_context_hgb": 0.5},
}
PLOT_LABELS = {
    "retained_neural_hgb": "Retained neural + HGB",
    "dense_context_hgb": "Dense spectral/context HGB",
    "neural_dense_equal": "Neural + dense HGB (equal)",
    "retained_dense_quarter": "Retained + dense HGB (25%)",
    "retained_dense_equal": "Retained + dense HGB (50%)",
}


def spectral_features(raw: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Centered Hann-window spectra; real accelerometer/timing inputs only."""
    raw = np.asarray(raw, np.float64)
    center = raw.mean(axis=0)
    z = (raw - center) / np.maximum(raw.std(axis=0), 0.05)
    dynamic = np.linalg.norm(raw - center, axis=1)
    dynamic /= max(float(np.sqrt(np.mean(dynamic ** 2))), 0.05)
    channels = np.column_stack([z, dynamic])
    channel_names = [*base.AXES, "orientation_invariant_dynamic_magnitude"]
    statistics = ["log_power", "entropy", "centroid_hz", "spread_hz", "peak_hz", "peak_fraction",
                  "low_fraction", "mid_fraction", "high_fraction"]
    names, parts = [], []
    for window in FFT_WINDOWS:
        left = window // 2
        padded = np.pad(channels, ((left, window - 1 - left), (0, 0)), mode="edge")
        views = sliding_window_view(padded, window_shape=window, axis=0)
        frequency = np.fft.rfftfreq(window, d=1 / FEATURE_CFG.sample_hz)
        selected = (frequency >= 0.5) & (frequency <= 15.0)
        frequency = frequency[selected]
        band_masks = [(frequency < 3), ((frequency >= 3) & (frequency < 8)), (frequency >= 8)]
        taper = np.hanning(window)
        result = np.empty((len(grid), len(statistics) * 4), np.float32)
        for start in range(0, len(grid), 128):
            rows = grid[start:start + 128]
            segments = views[rows].copy()
            segments -= segments.mean(axis=-1, keepdims=True)
            spectrum = np.fft.rfft(segments * taper, axis=-1)
            power = np.abs(spectrum[..., selected]) ** 2 / np.sum(taper ** 2)
            total = power.sum(axis=-1)
            normalized = power / np.maximum(total[..., None], 1e-12)
            entropy = -(normalized * np.log(np.maximum(normalized, 1e-12))).sum(axis=-1) / np.log(len(frequency))
            centroid = (normalized * frequency).sum(axis=-1)
            spread = np.sqrt(np.maximum((normalized * (frequency - centroid[..., None]) ** 2).sum(axis=-1), 0))
            peak = frequency[np.argmax(power, axis=-1)]
            peak = np.where(total > 1e-12, peak, 0)
            values = [np.log1p(total), entropy, centroid, spread, peak, normalized.max(axis=-1)]
            values.extend(normalized[..., mask].sum(axis=-1) for mask in band_masks)
            result[start:start + len(rows)] = np.stack(values, axis=1).reshape(len(rows), -1)
        parts.append(result)
        names.extend(f"fft_{statistic}_{channel}_w{window}" for statistic in statistics for channel in channel_names)
    return np.concatenate(parts, axis=1), names


def temporal_context(features: np.ndarray, names: list[str], grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Look backward/forward inside one recording, never across its boundary."""
    selected_names = [f"std_{axis}_w256" for axis in [*base.AXES, "dynamic_magnitude"]]
    selected_names += [f"mean_{axis}_w256" for axis in base.AXES]
    selected_names += [f"log_mid_low_energy_ratio_{axis}_w256" for axis in base.AXES]
    indices = [names.index(name) for name in selected_names]
    shifted, shifted_names = [], []
    for shift in CONTEXT_SHIFTS:
        shifted.append(np.column_stack([np.interp(grid + shift, grid, features[:, index]) for index in indices]))
        shifted_names.extend(f"context_shift_{shift}_{name}" for name in selected_names)
    return np.concatenate(shifted, axis=1).astype(np.float32), shifted_names


def expanded_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grid, original, names = base.sensor_features(raw, cfg=FEATURE_CFG)
    spectrum, spectrum_names = spectral_features(raw, grid)
    context, context_names = temporal_context(original, names, grid)
    x = np.concatenate([original, spectrum, context], axis=1)
    names = [*names, *spectrum_names, *context_names]
    if x.shape[1] != len(names) or len(set(names)) != len(names) or not np.isfinite(x).all():
        raise ValueError("Invalid expanded sensor features")
    return grid, x, names


def prepare_arrays(manifest: pd.DataFrame, run_dir: Path) -> dict:
    cache = run_dir / "dense_context_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        if not np.array_equal(arrays["recording_ids"], manifest.recording_id.to_numpy(str)):
            raise ValueError("Cached feature recording order differs")
        return arrays
    parts = {name: [] for name in ["x", "y", "fold", "record", "grid"]}
    row_counts, feature_names = [], None
    for record_index, row in enumerate(manifest.itertuples(index=False)):
        frame = pd.read_csv(row.path, usecols=[*base.AXES, *TARGETS])
        labels = frame.loc[:, list(TARGETS)].to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Invalid labels: {row.recording_id}")
        grid, x, names = expanded_features(frame.loc[:, list(base.AXES)].to_numpy())
        if feature_names is not None and feature_names != names:
            raise ValueError("Feature names changed")
        feature_names = names
        parts["x"].append(x)
        parts["y"].append(labels[grid].astype(np.uint8))
        parts["fold"].append(np.full(len(grid), row.fold, np.int8))
        parts["record"].append(np.full(len(grid), record_index, np.int16))
        parts["grid"].append(grid)
        row_counts.append(len(frame))
        if (record_index + 1) % 100 == 0:
            print(f"Dense spectral/context features: {record_index + 1}/{len(manifest)}", flush=True)
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    arrays.update(row_counts=np.asarray(row_counts, np.int32),
                  feature_names=np.asarray(feature_names, dtype=str), recording_ids=manifest.recording_id.to_numpy(str))
    print(f"Saving cache: {len(arrays['x']):,} rows x {arrays['x'].shape[1]} features", flush=True)
    np.savez_compressed(cache, **arrays)
    return arrays


def train_models(arrays: dict, run_dir: Path) -> np.ndarray:
    x, y, folds = arrays["x"], arrays["y"], arrays["fold"]
    predictions = np.full(y.shape, np.nan, np.float32)
    training_log = []
    with threadpool_limits(limits=4):
        for fold in range(5):
            fitting, validation = folds != fold, folds == fold
            fold_dir = run_dir / "dense_hgb" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            for target_index, target in enumerate(TARGETS):
                start = time.monotonic()
                model_path = fold_dir / f"{target}.joblib"
                prediction_path = fold_dir / f"{target}_sampled_prediction.npy"
                labels = y[fitting, target_index]
                positives = int(labels.sum())
                if positives == 0 or positives == len(labels):
                    raise ValueError("Both target classes required in training")
                weight = float(min(8.0, np.sqrt((len(labels) - positives) / positives)))
                reused = model_path.exists() and prediction_path.exists()
                if reused:
                    p = np.load(prediction_path, allow_pickle=False)
                else:
                    print(f"Training dense HGB fold {fold}, {target} ...", flush=True)
                    model = HistGradientBoostingClassifier(**MODEL_PARAMS, random_state=42 + fold)
                    model.fit(x[fitting], labels, sample_weight=np.where(labels == 1, weight, 1.0))
                    p = model.predict_proba(x[validation])[:, 1].astype(np.float32)
                    joblib.dump(model, model_path)
                    np.save(prediction_path, p, allow_pickle=False)
                    del model
                    gc.collect()
                if len(p) != int(validation.sum()) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
                    raise ValueError("Invalid sampled OOF predictions")
                predictions[validation, target_index] = p
                training_log.append({"fold": fold, "target": target, "positive_weight": weight,
                                     "training_rows": int(fitting.sum()), "validation_rows": int(validation.sum()),
                                     "elapsed_seconds": time.monotonic() - start, "reused": reused})
                pd.DataFrame(training_log).to_csv(run_dir / "training_log.csv", index=False)
                print(f"Dense HGB fold {fold} / {target}: complete ({training_log[-1]['elapsed_seconds']:.1f}s)", flush=True)
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete sampled OOF predictions")
    np.save(run_dir / "dense_hgb_sampled_oof.npy", predictions, allow_pickle=False)
    return predictions


def expand_native(manifest: pd.DataFrame, arrays: dict, sampled: np.ndarray, run_dir: Path) -> np.ndarray:
    output = run_dir / "dense_hgb" / "oof"
    output.mkdir(exist_ok=True)
    parts = []
    for index, row in enumerate(manifest.itertuples(index=False)):
        mask = arrays["record"] == index
        grid, sparse = arrays["grid"][mask], sampled[mask]
        n_rows = int(arrays["row_counts"][index])
        p = np.column_stack([np.interp(np.arange(n_rows), grid, sparse[:, target]) for target in range(3)]).astype(np.float32)
        np.savez_compressed(output / f"{row.recording_id}.npz", prediction=p)
        parts.append(p)
    return np.concatenate(parts)


def get_prediction(method: str, predictions: dict) -> np.ndarray:
    if method in predictions:
        return predictions[method]
    result = np.zeros_like(predictions["retained_neural_hgb"])
    weights = FIXED_BLENDS[method]
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("Blend weights must sum to one")
    for name, weight in weights.items():
        result += np.float32(weight) * predictions[name]
    return result


def evaluate_candidates(y: np.ndarray, folds: np.ndarray, predictions: dict, run_dir: Path) -> pd.DataFrame:
    rows, fold_rows = [], []
    for method in ["retained_neural_hgb", "dense_context_hgb", *FIXED_BLENDS]:
        p = get_prediction(method, predictions)
        previous.validate_prediction(y, p)
        rows.append({"method": method, **base.score_targets(y, p)})
        for fold in range(5):
            mask = folds == fold
            fold_rows.append({"method": method, "fold": fold, **base.score_targets(y[mask], p[mask])})
        print(f"{method}: TDCS native-row mAP {rows[-1]['mAP']:.6f}", flush=True)
    metrics = pd.DataFrame(rows)
    if abs(float(metrics.iloc[0].mAP) - 0.3800059184564884) > 1e-8:
        raise ValueError("The retained 0.3800 reference did not reproduce")
    metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "candidate_fold_metrics.csv", index=False)
    return metrics


def save_selected(manifest: pd.DataFrame, arrays: dict, p: np.ndarray, metric: pd.Series, run_dir: Path) -> Path:
    selected_dir = run_dir / "selected_candidate"
    prediction_dir = selected_dir / "oof"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(selected_dir / "frozen_tdcs_manifest.csv", index=False)
    pd.DataFrame([{"domain": "tdcsfog", **metric.to_dict()}]).to_csv(selected_dir / "native_oof_metrics.csv", index=False)
    offset = 0
    for index, row in enumerate(manifest.itertuples(index=False)):
        n_rows = int(arrays["row_counts"][index])
        np.savez_compressed(prediction_dir / f"{row.recording_id}.npz", prediction=p[offset:offset + n_rows])
        offset += n_rows
    if offset != len(p):
        raise ValueError("Selected prediction row alignment changed")
    return selected_dir


def make_figures(metrics: pd.DataFrame, run_dir: Path) -> list[Path]:
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    ordered = metrics.sort_values("mAP")
    fig, ax = plt.subplots(figsize=(10.5, 5.2), layout="constrained")
    colors = ["#8b929c" if method == "retained_neural_hgb" else "#2463a0" for method in ordered.method]
    bars = ax.barh([PLOT_LABELS[name] for name in ordered.method], ordered.mAP, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pooled native-row TDCS OOF mAP")
    ax.set_title("Dense temporal/spectral TDCS candidates: 61 subjects, 5 folds")
    ax.grid(axis="x", alpha=0.2)
    first = run_dir / "tdcs_dense_candidate_map.png"
    fig.savefig(first, dpi=150)
    plt.close(fig)
    reference = metrics.loc[metrics.method.eq("retained_neural_hgb")].iloc[0]
    selected = metrics.sort_values("mAP", ascending=False).iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    positions = np.arange(3)
    for i, (row, label, color) in enumerate([
        (reference, "Retained neural + HGB", "#8b929c"),
        (selected, f"Selected: {PLOT_LABELS[selected.method]}", "#2463a0"),
    ]):
        bars = ax.bar(positions + (i - 0.5) * 0.34, row.loc[list(TARGETS)].to_numpy(float), 0.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set_xticks(positions, ["Start hesitation", "Turn", "Walking"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pooled native-row average precision")
    ax.set_title("TDCS per-class tradeoffs against the retained 0.3800 blend")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    second = run_dir / "tdcs_dense_candidate_classes.png"
    fig.savefig(second, dpi=150)
    plt.close(fig)
    return [first, second]


def run_experiment() -> dict:
    sensor_run, manifest, old_arrays, old_audit = previous.load_inputs()
    del old_arrays
    gc.collect()
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                     [Path(__file__), ROOT / "scripts" / "tdcs_multiscale_signal_experiment.py",
                      ROOT / "scripts" / "tdcs_candidate_sweep.py"]}
    configuration = {
        "feature_configuration": asdict(FEATURE_CFG), "model_parameters": MODEL_PARAMS,
        "fft_windows": FFT_WINDOWS, "fft_range_hz": [0.5, 15], "context_shifts_native_rows": CONTEXT_SHIFTS,
        "fixed_blends": FIXED_BLENDS, "positive_weight": "sqrt(negative/positive), cap 8, training labels only",
        "sensor_input_fingerprint": old_audit["fingerprint"], "source_hashes": source_hashes,
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "reference_mAP": 0.3800059184564884,
        "selection_caveat": "development folds reused across experiments; exploratory selection",
    }
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:20]
    run_dir = OUTPUT_ROOT / fingerprint
    run_dir.mkdir(parents=True, exist_ok=True)
    previous.write_json(run_dir / "predeclared_configuration.json", configuration)
    manifest.to_csv(run_dir / "frozen_tdcs_manifest.csv", index=False)
    print(f"Dense TDCS artifacts: {run_dir}", flush=True)
    arrays = prepare_arrays(manifest, run_dir)
    if int(arrays["row_counts"].sum()) != 6959762:
        raise ValueError("Native population changed")
    audit = {
        "domain": "tdcsfog", "recordings": len(manifest), "subjects": int(manifest.subject.nunique()),
        "native_rows": int(arrays["row_counts"].sum()), "sampled_training_rows": len(arrays["x"]),
        "feature_count": int(arrays["x"].shape[1]), "training_stride": FEATURE_CFG.training_stride,
        "validation": "same five frozen subject-disjoint folds; outer labels used for scoring only",
        "feature_inputs": "accelerometer and recording timing only",
        "offline_assumption": "centered spectra, past/future context and whole-record normalization",
        "retained_defog_reported_oof_map": 0.817, "competition_submission_created": False,
        "model_selection_caveat": configuration["selection_caveat"],
    }
    previous.write_json(run_dir / "audit.json", audit)
    sampled = train_models(arrays, run_dir)
    del arrays["x"]
    gc.collect()
    dense = expand_native(manifest, arrays, sampled, run_dir)
    y, folds, original = previous.load_native_inputs(sensor_run, manifest)
    predictions = {"previous_tdcs_blend": original["previous_tdcs_blend"],
                   "retained_neural_hgb": previous.blended_prediction("neural_hgb_equal", original),
                   "dense_context_hgb": dense}
    del original
    gc.collect()
    metrics = evaluate_candidates(y, folds, predictions, run_dir)
    best = metrics.sort_values("mAP", ascending=False).iloc[0]
    reference = float(metrics.iloc[0].mAP)
    selected_dir = save_selected(manifest, arrays, get_prediction(str(best.method), predictions), best, run_dir)
    figures = make_figures(metrics, run_dir)
    decision = {"run_dir": str(run_dir), "selected_candidate_dir": str(selected_dir),
                "selected_method": str(best.method), "candidate_tdcs_oof_map": float(best.mAP),
                "retained_reference_tdcs_oof_map": reference, "delta": float(best.mAP - reference),
                "decision": "retain new TDCS candidate" if best.mAP > reference else "retain previous 0.3800 blend",
                "retained_defog_reported_oof_map": 0.817, "competition_submission_created": False,
                "selection_scope": "one new model and three fixed blends on reused development folds"}
    previous.write_json(run_dir / "decision.json", decision)
    previous.write_json(OUTPUT_ROOT / "latest_run.json", decision)
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    return {"run_dir": run_dir, "metrics": metrics, "decision": decision, "figures": figures}


if __name__ == "__main__":
    run_experiment()
