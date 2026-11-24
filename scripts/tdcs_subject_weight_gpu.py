"""Matched GPU trials of weaker training-subject balancing on TDCS.

The only difference from the v10 regularized recipe is the subject-count
exponent: 0 (row-uniform) or 0.5 (square-root balancing), rather than 1.
Sampling, seeds, model parameters, features and evaluation population stay fixed.
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

import tdcs_gpu_regularized_context as parent

shared = parent.shared
ROOT = Path(__file__).resolve().parents[1]
TARGETS = parent.TARGETS
EXPONENTS = {"row_uniform": 0., "subject_sqrt": .5}
VARIANTS = tuple(EXPONENTS)
PARAMETERS = parent.PARAMETERS["regularized_depth6"]


@dataclass(frozen=True)
class Config:
    name: str = "tdcs_subject_weight_gpu_v12"
    targets: tuple = TARGETS
    folds: tuple = (0, 1, 2, 3, 4)
    negative_divisor: int = 4
    seed: int = 5107
    smoke_rounds: int = 4
    max_seconds: int = 21600


CFG = Config()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {"source_sha256": globals().get("SOURCE_SHA256") or digest(__file__),
        "parent_sources": parent.source_hashes()}


def check_config(cfg):
    if tuple(cfg.targets) != TARGETS or tuple(cfg.folds) != tuple(range(5)):
        raise ValueError("Both difficult targets and all five frozen folds required")
    if min(cfg.negative_divisor, cfg.smoke_rounds, cfg.max_seconds) < 1:
        raise ValueError("Positive sampling and budget required")


def training_plan(y, folds, subjects, fold, seed, divisor, exponent):
    if exponent not in (0., .5, 1.):
        raise ValueError("Only fixed exponents 0, 0.5 and control 1 are supported")
    chosen, _, probability = shared.training_plan(y, folds, subjects, fold, seed, divisor)
    y, folds, subjects = np.asarray(y), np.asarray(folds), np.asarray(subjects, str)
    names, counts = np.unique(subjects[folds != fold], return_counts=True)
    lookup = dict(zip(names, counts))
    weights = np.asarray([float(lookup[s]) ** -exponent for s in subjects[chosen]], np.float64)
    weights[y[chosen] == 0] /= probability
    weights /= weights.mean()
    assert np.isfinite(weights).all() and (weights > 0).all()
    return chosen, weights, probability


def synthetic_inputs():
    rng = np.random.default_rng(71)
    lengths = np.asarray([32, 48, 64, 80, 96], np.int32)
    record = np.repeat(np.arange(5), lengths).astype(np.int16)
    grid = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])
    y = np.column_stack([grid % 13 == 0, grid % 11 == 1]).astype(np.uint8)
    y[record == 4] = 0
    manifest = pd.DataFrame({"recording_id": [f"smoke_{i}" for i in range(5)],
        "subject": [f"subject_{i}" for i in range(5)], "fold": range(5)})
    arrays = {"x": rng.normal(size=(len(record), 325)).astype(np.float32),
        "y": y, "fold": record.astype(np.int8), "record": record, "grid": grid,
        "subject": manifest.subject.to_numpy(str)[record], "lengths": lengths,
        "recording_ids": manifest.recording_id.to_numpy(str)}
    return arrays, manifest, [y[record == i].copy() for i in range(5)]


def run_experiment(variant, cfg=CFG, output_root=None, synthetic=None):
    check_config(cfg)
    if variant not in VARIANTS:
        raise ValueError("Unknown weighting trial")
    smoke = synthetic is not None
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("Real-data weighting fitting requires Kaggle CUDA; no CPU fallback")
    output_root = Path(output_root or ROOT / "results/tdcs_subject_weight_gpu" / variant)
    output_root.mkdir(parents=True, exist_ok=True)
    if smoke:
        arrays, manifest, native_labels = synthetic
        parent_run, parent_config = Path("synthetic"), {}
    else:
        parent_run, parent_config, arrays, manifest = parent.parent_loader.load_parent()
        native_labels = None
    population = ("synthetic wiring only; not TDCS performance" if smoke else
        manifest[["recording_id", "subject", "fold", "source_bytes"]].to_csv(index=False, lineterminator="\n"))
    exponent = EXPONENTS[variant]
    config = {"configuration": asdict(cfg), "variant": variant, "targets": TARGETS,
        "learner": "catboost", "parameters": PARAMETERS, "sources": source_hashes(),
        "smoke_only": smoke, "feature_count": 325, "subject_weight_exponent": exponent,
        "weight_formula": "training_subject_full_grid_row_count ** (-exponent), times inverse negative-sampling probability, normalized to mean 1",
        "parent_fingerprint": parent_run.name,
        "parent_feature_array_sha256": parent_config.get("feature_array_sha256"),
        "parent_cache_sha256": parent_config.get("feature_cache_sha256"),
        "population_sha256": hashlib.sha256(population.encode()).hexdigest(),
        "runtime": {name: version(name) for name in ("catboost", "numpy", "torch")},
        "used_gpu_indices": [] if smoke else [0],
        "prediction_backend": "CPU saved-model scoring; in-record interpolation to native rows",
        "training_scope": "all training recordings, including negative files",
        "negative_sampling": "all positives plus seeded quarter negatives with inverse correction",
        "conditional_training": False, "file_gate_applied": False, "early_stopping": False,
        "extra_positive_class_weights": False, "base_OOF_predictions_used_as_features": False,
        "event_boundaries_used_as_features": False, "task_intervals_used_as_features": False,
        "IDs_or_subjects_used_as_features": False, "metadata_used_as_features": False,
        "outer_labels_used_for_fitting_or_stopping": False, "retained_heads_overwritten": False,
        "competition_submission_created": False, "confidence_calibrated": False,
        "independent_generalization_established": False}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    run = output_root / fingerprint
    run.mkdir(parents=True, exist_ok=True)
    shared.common.write_json(run / "configuration.json", config)
    manifest.to_csv(run / "frozen_tdcs_manifest.csv", index=False)
    (run / "oof").mkdir(exist_ok=True)
    scores = np.full(arrays["y"].shape, np.nan, np.float32)
    deadline = time.monotonic() + cfg.max_seconds
    for fold in cfg.folds:
        heldout = arrays["fold"] == fold
        training_subjects = sorted(set(arrays["subject"][~heldout]))
        heldout_subjects = sorted(set(arrays["subject"][heldout]))
        assert training_subjects and heldout_subjects and set(training_subjects).isdisjoint(heldout_subjects)
        directory = run / f"fold_{fold}"
        directory.mkdir(exist_ok=True)
        for index, target in enumerate(TARGETS):
            if time.monotonic() >= deadline:
                raise TimeoutError("Subject-weight experiment safety budget reached")
            seed = cfg.seed + 1009 * fold + index
            chosen, weights, probability = training_plan(arrays["y"][:, index], arrays["fold"],
                arrays["subject"], fold, seed, cfg.negative_divisor, exponent)
            sample_path = directory / (target + "_training_sample.npz")
            model_path = directory / (target + ".cbm")
            receipt_path = directory / (target + "_training_receipt.json")
            existing = [p.is_file() for p in (sample_path, model_path, receipt_path)]
            if any(existing):
                if not all(existing):
                    raise ValueError("Incomplete model/sample/receipt; preserve artifacts and choose a fresh root")
                receipt = json.loads(receipt_path.read_text())
                assert receipt["fingerprint"] == fingerprint and receipt["target"] == target
                assert receipt["variant"] == variant and receipt["seed"] == seed
                assert receipt["subject_weight_exponent"] == exponent and receipt["parameters"] == PARAMETERS
                assert receipt["smoke_only"] == smoke and receipt["sample_sha256"] == digest(sample_path)
                assert receipt["trees"] == (cfg.smoke_rounds if smoke else PARAMETERS["iterations"])
                assert receipt["actual_training_backend"] == ("CPU" if smoke else "GPU")
                with np.load(sample_path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["indices"], chosen)
                    np.testing.assert_array_equal(saved["weights"], weights)
                model = shared.load_model("catboost", model_path)
            else:
                np.savez_compressed(sample_path, indices=chosen, weights=weights)
                print(f"{variant} fold {fold} {target}: {len(chosen):,} rows; exponent {exponent}", flush=True)
                started = time.monotonic()
                model, backend = parent.fit_model("regularized_depth6", arrays["x"][chosen],
                    arrays["y"][chosen, index], weights, seed, gpu=not smoke,
                    rounds=cfg.smoke_rounds if smoke else None)
                shared.save_model(model, "catboost", model_path)
                shared.common.write_json(receipt_path, {"fingerprint": fingerprint, "variant": variant,
                    "target": target, "seed": seed, "parameters": PARAMETERS,
                    "feature_count": 325, "subject_weight_exponent": exponent,
                    "training_subjects": training_subjects, "held_out_subjects": heldout_subjects,
                    "training_grid_rows_before_sampling": int((~heldout).sum()),
                    "sampled_grid_rows": len(chosen), "negative_sampling_probability": probability,
                    "all_positive_training_rows_kept": True, "inverse_sampling_correction": True,
                    "subject_counts_from_full_training_rows_only": True,
                    "outer_labels_used_for_fitting_or_stopping": False,
                    "base_OOF_predictions_used_as_features": False, "early_stopping": False,
                    "file_gate_applied": False, "extra_positive_class_weights": False,
                    "sample_sha256": digest(sample_path), "smoke_only": smoke,
                    "seconds": time.monotonic() - started, **backend})
            scores[heldout, index] = shared.predict_model(model, "catboost", arrays["x"][heldout])
            del model
    assert np.isfinite(scores).all()
    np.savez_compressed(run / "grid_scores.npz", prediction=scores, targets=np.asarray(TARGETS))
    ys, ps = [], []
    for i, row in enumerate(manifest.itertuples(index=False)):
        selected = arrays["record"] == i
        p = shared.expand_native(arrays["grid"][selected], scores[selected], int(arrays["lengths"][i]))
        y = (native_labels[i] if smoke else
            pd.read_csv(row.path, usecols=list(TARGETS)).loc[:, list(TARGETS)].to_numpy(np.uint8))
        assert p.shape == y.shape and np.isin(y, [0, 1]).all()
        np.savez_compressed(run / "oof" / (row.recording_id + ".npz"), prediction=p, targets=np.asarray(TARGETS))
        ys.append(y); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    values = shared.common.average_precision(y, p)
    pd.DataFrame([values]).to_csv(run / "native_oof_metrics.csv", index=False)
    result = {"status": "complete", "run_dir": str(run), "variant": variant,
        "targets": TARGETS, **values, "native_rows": len(y), "smoke_only": smoke,
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
