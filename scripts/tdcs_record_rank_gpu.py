"""Fixed per-record rank feature trials; fit only on Kaggle CUDA.

Compare replacing local features by their within-record mid-percentiles with
appending those percentiles. A frozen cross-fitted Walking file gate is applied
exactly once after native interpolation. No gate score is a fitting feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

import tdcs_subject_weight_gpu as parent

shared = parent.shared
gate_helper = parent.parent.parent_loader
ROOT = Path(__file__).resolve().parents[1]
TARGETS = parent.TARGETS
VARIANTS = ("rank_replace", "rank_append")
FEATURE_COUNTS = {"rank_replace": 325, "rank_append": 538}
PARAMETERS = parent.PARAMETERS
LOCAL_COLUMNS = 213
GATE_FLOOR = .05


@dataclass(frozen=True)
class Config:
    name: str = "tdcs_record_rank_gpu_v16"
    targets: tuple = TARGETS
    folds: tuple = (0, 1, 2, 3, 4)
    negative_divisor: int = 4
    subject_weight_exponent: float = .5
    seed: int = 5107
    smoke_rounds: int = 4
    max_seconds: int = 21600


CFG = Config()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(value.dtype.str.encode())
    result.update(str(value.shape).encode())
    result.update(memoryview(value).cast("B"))
    return result.hexdigest()


def source_hashes():
    return {"source_sha256": globals().get("SOURCE_SHA256") or digest(__file__),
        "parent_sources": parent.source_hashes()}


def check_config(cfg):
    if tuple(cfg.targets) != TARGETS or tuple(cfg.folds) != tuple(range(5)):
        raise ValueError("Both targets and all five frozen subject folds required")
    if cfg.subject_weight_exponent != .5:
        raise ValueError("Fixed square-root training-subject weights required")
    if min(cfg.negative_divisor, cfg.smoke_rounds, cfg.max_seconds) < 1:
        raise ValueError("Positive sampling and training budget required")


def record_percentiles(x, records):
    """Label-free (average rank - .5)/n, independently within each recording."""
    x, records = np.asarray(x), np.asarray(records)
    if x.ndim != 2 or x.shape[1] != 325 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Nonempty finite 325-channel feature matrix required")
    if records.shape != (len(x),) or not np.issubdtype(records.dtype, np.integer):
        raise ValueError("One integer recording index per feature row required")
    names, counts = np.unique(records, return_counts=True)
    if not np.array_equal(names, np.arange(len(names))) or np.any(np.diff(records) < 0):
        raise ValueError("Contiguous ordered recording blocks numbered from zero required")
    result = np.empty((len(x), LOCAL_COLUMNS), np.float32)
    begin = 0
    for count in counts:
        end = begin + int(count)
        ranks = rankdata(x[begin:end, :LOCAL_COLUMNS], method="average", axis=0, nan_policy="raise")
        result[begin:end] = ((ranks - .5) / count).astype(np.float32)
        begin = end
    assert begin == len(x) and np.isfinite(result).all()
    assert ((result > 0) & (result < 1)).all()
    return result


def transform_features(arrays, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown fixed rank-feature trial")
    names = np.asarray(arrays["feature_names"], str)
    if names.shape != (325,) or len(set(names)) != 325:
        raise ValueError("325 unique frozen feature names required")
    ranks = record_percentiles(arrays["x"], arrays["record"])
    rank_names = np.char.add("record_midpercentile_", names[:LOCAL_COLUMNS])
    if variant == "rank_replace":
        x = np.column_stack([ranks, arrays["x"][:, LOCAL_COLUMNS:]]).astype(np.float32)
        output_names = np.r_[rank_names, names[LOCAL_COLUMNS:]]
    else:
        x = np.column_stack([arrays["x"], ranks]).astype(np.float32)
        output_names = np.r_[names, rank_names]
    assert x.shape == (len(ranks), FEATURE_COUNTS[variant]) and np.isfinite(x).all()
    assert len(set(output_names)) == x.shape[1]
    return x, output_names, array_digest(ranks)


def gate_scores(payload, manifest):
    if payload["recording_ids"] != manifest.recording_id.tolist() or payload["folds"] != manifest.fold.tolist():
        raise ValueError("Frozen gate IDs and folds must match the full evaluation cohort")
    q = np.asarray(payload["scores"], np.float32)
    if q.shape != (len(manifest),) or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("One finite frozen file-gate score in [0,1] required per recording")
    return q


def combine_native(grid, raw_scores, length, file_score):
    if not np.isfinite(file_score) or not 0 <= file_score <= 1:
        raise ValueError("Finite frozen file-gate score required")
    raw_scores = np.asarray(raw_scores)
    if raw_scores.shape != (len(grid), 2) or not np.isfinite(raw_scores).all() or ((raw_scores < 0) | (raw_scores > 1)).any():
        raise ValueError("Finite raw SH/Walking grid scores in [0,1] required")
    p = shared.expand_native(grid, raw_scores, length)
    factor = (GATE_FLOOR + (1 - GATE_FLOOR) * np.asarray([file_score], np.float32)).astype(np.float32)[0]
    p[:, 1] = np.asarray(p[:, 1] * factor, np.float32)
    return p


def synthetic_inputs():
    arrays, manifest, labels = parent.synthetic_inputs()
    arrays["feature_names"] = np.asarray([f"local_{i}" for i in range(213)] + [f"file_{i}" for i in range(112)])
    q = np.asarray([.2, .4, .6, .8, .1], np.float32)
    gates = {"recording_ids": manifest.recording_id.tolist(), "folds": manifest.fold.tolist(), "scores": q.tolist()}
    return arrays, manifest, labels, gates


def run_experiment(variant, cfg=CFG, output_root=None, synthetic=None):
    check_config(cfg)
    if variant not in VARIANTS:
        raise ValueError("Unknown rank-feature variant")
    smoke = synthetic is not None
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("Real-data rank-feature fitting requires Kaggle CUDA; no CPU fallback")
    output_root = Path(output_root or ROOT / "results/tdcs_record_rank_gpu" / variant)
    output_root.mkdir(parents=True, exist_ok=True)
    if smoke:
        arrays, manifest, native_labels, gates = synthetic
        parent_run, parent_config = Path("synthetic"), {}
    else:
        parent_run, parent_config, arrays, manifest = gate_helper.load_parent()
        native_labels, gates = None, gate_helper.gate_payload()
    q = gate_scores(gates, manifest)
    started = time.monotonic()
    print(f"{variant}: calculating per-record label-free percentiles", flush=True)
    x, names, rank_hash = transform_features(arrays, variant)
    print(f"{variant}: {x.shape[0]:,} rows x {x.shape[1]} features; {x.nbytes / 2**20:.1f} MiB", flush=True)
    population = ("synthetic wiring only; not TDCS performance" if smoke else
        manifest[["recording_id", "subject", "fold", "source_bytes"]].to_csv(index=False, lineterminator="\n"))
    config = {"configuration": asdict(cfg), "variant": variant, "targets": TARGETS,
        "learner": "catboost", "parameters": PARAMETERS, "sources": source_hashes(), "smoke_only": smoke,
        "feature_count": len(names), "feature_names": names.tolist(), "feature_array_sha256": array_digest(x),
        "rank_feature_sha256": rank_hash, "local_columns_ranked": LOCAL_COLUMNS,
        "rank_formula": "(within-record average rank - 0.5) / number of grid rows in that recording; ties averaged",
        "rank_scope": "Only the current recording's sensor-derived features, including its full offline time extent; no labels, other records or subjects",
        "parent_fingerprint": parent_run.name,
        "parent_feature_array_sha256": parent_config.get("feature_array_sha256"),
        "parent_cache_sha256": parent_config.get("feature_cache_sha256"),
        "population_sha256": hashlib.sha256(population.encode()).hexdigest(),
        "gate_payload_sha256": gate_helper.payload_digest(gates), "walking_gate_floor": GATE_FLOOR,
        "Walking_raw_scores_gated_exactly_once_after_native_interpolation": True,
        "StartHesitation_gate_applied": False, "gate_used_as_fitting_feature": False,
        "runtime": {name: version(name) for name in ("catboost", "numpy", "scipy", "torch")},
        "used_gpu_indices": [] if smoke else [0], "subject_weight_exponent": cfg.subject_weight_exponent,
        "training_scope": "All training recordings including negative files; no conditional file exclusion",
        "negative_sampling": "All positive grid rows and a seeded quarter of negatives, inverse sampling correction",
        "prediction_backend": "CPU scoring of saved GPU models; native in-record interpolation, then frozen Walking-only gate",
        "early_stopping": False, "outer_labels_used_for_fitting_or_stopping": False,
        "base_OOF_predictions_used_as_fitting_features": False, "extra_class_weights": False,
        "IDs_or_subjects_used_as_features": False, "metadata_used_as_features": False,
        "event_boundaries_used_as_features": False, "task_intervals_used_as_features": False,
        "retained_heads_overwritten": False, "competition_submission_created": False,
        "confidence_calibrated": False, "independent_generalization_established": False}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    run = output_root / fingerprint
    run.mkdir(parents=True, exist_ok=True)
    shared.common.write_json(run / "configuration.json", config)
    shared.common.write_json(run / "frozen_gate_payload.json", gates)
    manifest.to_csv(run / "frozen_tdcs_manifest.csv", index=False)
    (run / "oof").mkdir(exist_ok=True)
    scores = np.full(arrays["y"].shape, np.nan, np.float32)
    deadline = started + cfg.max_seconds
    for fold in cfg.folds:
        heldout = arrays["fold"] == fold
        training_subjects = sorted(set(arrays["subject"][~heldout]))
        heldout_subjects = sorted(set(arrays["subject"][heldout]))
        assert training_subjects and heldout_subjects and set(training_subjects).isdisjoint(heldout_subjects)
        directory = run / f"fold_{fold}"
        directory.mkdir(exist_ok=True)
        for index, target in enumerate(TARGETS):
            if time.monotonic() >= deadline:
                raise TimeoutError("Rank-feature experiment safety budget reached")
            seed = cfg.seed + 1009 * fold + index
            chosen, weights, probability = parent.training_plan(arrays["y"][:, index], arrays["fold"],
                arrays["subject"], fold, seed, cfg.negative_divisor, cfg.subject_weight_exponent)
            sample_path, model_path = directory / (target + "_training_sample.npz"), directory / (target + ".cbm")
            receipt_path = directory / (target + "_training_receipt.json")
            existing = [p.is_file() for p in (sample_path, model_path, receipt_path)]
            if any(existing):
                if not all(existing):
                    raise ValueError("Incomplete model/sample/receipt; preserve artifacts and choose a fresh output root")
                receipt = json.loads(receipt_path.read_text())
                assert receipt["fingerprint"] == fingerprint and receipt["variant"] == variant and receipt["target"] == target
                assert receipt["seed"] == seed and receipt["parameters"] == PARAMETERS
                assert receipt["smoke_only"] == smoke and receipt["sample_sha256"] == digest(sample_path)
                assert receipt["feature_count"] == len(names)
                assert receipt["trees"] == (cfg.smoke_rounds if smoke else PARAMETERS["iterations"])
                assert receipt["actual_training_backend"] == ("CPU" if smoke else "GPU")
                with np.load(sample_path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["indices"], chosen)
                    np.testing.assert_array_equal(saved["weights"], weights)
                model = shared.load_model("catboost", model_path)
            else:
                np.savez_compressed(sample_path, indices=chosen, weights=weights)
                print(f"{variant} fold {fold} {target}: {len(chosen):,} sampled rows", flush=True)
                fit_started = time.monotonic()
                model, backend = parent.parent.fit_model("regularized_depth6", x[chosen], arrays["y"][chosen, index],
                    weights, seed, gpu=not smoke, rounds=cfg.smoke_rounds if smoke else None)
                shared.save_model(model, "catboost", model_path)
                shared.common.write_json(receipt_path, {"fingerprint": fingerprint, "variant": variant,
                    "target": target, "seed": seed, "parameters": PARAMETERS, "feature_count": len(names),
                    "feature_array_sha256": config["feature_array_sha256"], "subject_weight_exponent": .5,
                    "training_subjects": training_subjects, "held_out_subjects": heldout_subjects,
                    "training_grid_rows_before_sampling": int((~heldout).sum()), "sampled_grid_rows": len(chosen),
                    "negative_sampling_probability": probability, "all_positive_training_rows_kept": True,
                    "all_training_recordings_eligible": True, "inverse_sampling_correction": True,
                    "subject_counts_from_full_training_rows_only": True,
                    "outer_labels_used_for_fitting_or_stopping": False, "gate_used_as_fitting_feature": False,
                    "early_stopping": False, "extra_class_weights": False,
                    "sample_sha256": digest(sample_path), "smoke_only": smoke,
                    "seconds": time.monotonic() - fit_started, **backend})
            scores[heldout, index] = shared.predict_model(model, "catboost", x[heldout])
            del model
    assert np.isfinite(scores).all()
    np.savez_compressed(run / "grid_scores.npz", prediction=scores, targets=np.asarray(TARGETS))
    ys, ps, raw_ps = [], [], []
    for i, row in enumerate(manifest.itertuples(index=False)):
        selected = arrays["record"] == i
        raw = shared.expand_native(arrays["grid"][selected], scores[selected], int(arrays["lengths"][i]))
        p = combine_native(arrays["grid"][selected], scores[selected], int(arrays["lengths"][i]), q[i])
        y = (native_labels[i] if smoke else
            pd.read_csv(row.path, usecols=list(TARGETS)).loc[:, list(TARGETS)].to_numpy(np.uint8))
        assert p.shape == y.shape and np.isin(y, [0, 1]).all()
        np.savez_compressed(run / "oof" / (row.recording_id + ".npz"), prediction=p, targets=np.asarray(TARGETS))
        ys.append(y)
        ps.append(p)
        raw_ps.append(raw)
    y, p, raw = np.concatenate(ys), np.concatenate(ps), np.concatenate(raw_ps)
    values = shared.common.average_precision(y, p)
    raw_values = shared.common.average_precision(y, raw)
    pd.DataFrame([values]).to_csv(run / "native_oof_metrics.csv", index=False)
    pd.DataFrame([raw_values]).to_csv(run / "native_raw_oof_metrics.csv", index=False)
    result = {"status": "complete", "run_dir": str(run), "variant": variant, "targets": TARGETS,
        **values, "raw_ungated_scores_diagnostic_only": raw_values, "native_rows": len(y), "smoke_only": smoke,
        "retained_heads_unchanged": True, "confidence_calibrated": False,
        "independent_generalization_established": False, "competition_submission_created": False}
    shared.common.write_json(run / "decision.json", result)
    shared.common.write_json(output_root / "latest_run.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def smoke_experiment(variant, output_root):
    return run_experiment(variant, replace(CFG, smoke_rounds=4), output_root, synthetic_inputs())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    (smoke_experiment if args.smoke else run_experiment)(args.variant, output_root=args.output_root)
