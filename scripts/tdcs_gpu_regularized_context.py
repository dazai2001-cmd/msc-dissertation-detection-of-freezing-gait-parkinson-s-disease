"""Two fixed lower-capacity GPU CatBoost trials on the full TDCS training cohort.

The v8 feature cache and helpers are immutable. This module does not use the
conditional v9 training population or its file gate; only its parent-cache loader.
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

import tdcs_walking_conditional_gpu as parent_loader

shared = parent_loader.shared
ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("StartHesitation", "Walking")
VARIANTS = ("shallow_depth4", "regularized_depth6")
PARAMETERS = {
    variant: {**shared.PARAMETERS["catboost"], "depth": depth, "l2_leaf_reg": 100.}
    for variant, depth in zip(VARIANTS, (4, 6))
}


@dataclass(frozen=True)
class Config:
    name: str = "tdcs_gpu_regularized_context_v10"
    targets: tuple = TARGETS
    folds: tuple = (0, 1, 2, 3, 4)
    negative_divisor: int = 4
    seed: int = 5107
    smoke_rounds: int = 4
    max_seconds: int = 21600


CFG = Config()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_config(cfg):
    if tuple(cfg.targets) != TARGETS or tuple(cfg.folds) != tuple(range(5)):
        raise ValueError("Both difficult targets and all five frozen subject folds required")
    if min(cfg.negative_divisor, cfg.smoke_rounds, cfg.max_seconds) < 1:
        raise ValueError("Positive sampling divisor and training budget required")


def source_hashes():
    return {"source_sha256": globals().get("SOURCE_SHA256") or digest(__file__),
        "parent_loader_sha256": getattr(parent_loader, "SOURCE_SHA256", None) or digest(parent_loader.__file__),
        "shared_sources": shared.source_hashes()}


def fit_model(variant, x, y, weights, seed, gpu=True, rounds=None):
    if variant not in VARIANTS:
        raise ValueError("Unknown fixed regularization trial")
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("Real-data regularized fitting requires Kaggle CUDA; no CPU fallback")
    from catboost import CatBoostClassifier
    params = {**PARAMETERS[variant], "task_type": "GPU" if gpu else "CPU", "random_seed": seed}
    if gpu:
        params.update(devices="0", gpu_ram_part=.6)
    if rounds is not None:
        params["iterations"] = rounds
    model = CatBoostClassifier(**params)
    model.fit(x, y, sample_weight=weights, verbose=200 if rounds is None else False)
    actual = model.get_all_params()["task_type"]
    if gpu and actual != "GPU":
        raise RuntimeError("CatBoost did not use the requested GPU backend")
    assert model.tree_count_ == params["iterations"]
    return model, {"actual_training_backend": actual, "trees": model.tree_count_}


def run_experiment(variant, cfg=CFG, output_root=None, synthetic=None):
    check_config(cfg)
    if variant not in VARIANTS:
        raise ValueError("Unknown fixed regularization trial")
    smoke = synthetic is not None
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("Real-data regularized fitting requires Kaggle CUDA; no CPU fallback")
    output_root = Path(output_root or ROOT / "results/tdcs_gpu_regularized_context" / variant)
    output_root.mkdir(parents=True, exist_ok=True)
    if smoke:
        arrays, manifest, native_labels = synthetic
        parent, parent_config = Path("synthetic"), {}
    else:
        parent, parent_config, arrays, manifest = parent_loader.load_parent()
        native_labels = None
    population = ("synthetic wiring only; not TDCS performance" if smoke else
        manifest[["recording_id", "subject", "fold", "source_bytes"]].to_csv(index=False, lineterminator="\n"))
    config = {"configuration": asdict(cfg), "variant": variant, "targets": TARGETS,
        "learner": "catboost", "feature_count": 325, "parameters": PARAMETERS[variant],
        "sources": source_hashes(), "smoke_only": smoke,
        "population_sha256": hashlib.sha256(population.encode()).hexdigest(),
        "parent_fingerprint": parent.name,
        "parent_feature_array_sha256": parent_config.get("feature_array_sha256"),
        "parent_cache_sha256": parent_config.get("feature_cache_sha256"),
        "runtime": {name: version(name) for name in ("catboost", "numpy", "torch")},
        "training_scope": "all training recordings; positive and negative files eligible",
        "negative_sampling": "all positives plus seeded one-quarter negatives with inverse correction",
        "subject_weights": "inverse full-fitting-row counts from training subjects only; normalized",
        "conditional_training": False, "file_gate_applied": False, "early_stopping": False,
        "outer_labels_used_for_fitting_or_stopping": False, "base_OOF_predictions_used_as_features": False,
        "extra_positive_class_weights": False, "used_gpu_indices": [] if smoke else [0],
        "prediction_backend": "CPU scoring of saved GPU models; in-record interpolation to native rows",
        "reused_development_folds": True, "confidence_calibrated": False,
        "retained_heads_overwritten": False, "competition_submission_created": False}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    run = output_root / fingerprint; run.mkdir(parents=True, exist_ok=True)
    shared.common.write_json(run / "configuration.json", config)
    manifest.to_csv(run / "frozen_tdcs_manifest.csv", index=False)
    (run / "oof").mkdir(exist_ok=True)
    grid_scores = np.full(arrays["y"].shape, np.nan, np.float32)
    deadline = time.monotonic() + cfg.max_seconds
    for fold in cfg.folds:
        heldout = arrays["fold"] == fold
        training_subjects = sorted(set(arrays["subject"][~heldout]))
        heldout_subjects = sorted(set(arrays["subject"][heldout]))
        assert training_subjects and heldout_subjects and set(training_subjects).isdisjoint(heldout_subjects)
        directory = run / f"fold_{fold}"; directory.mkdir(exist_ok=True)
        for index, target in enumerate(TARGETS):
            if time.monotonic() >= deadline:
                raise TimeoutError("Regularized CatBoost safety budget reached")
            seed = cfg.seed + 1009 * fold + index
            chosen, weights, probability = shared.training_plan(arrays["y"][:, index], arrays["fold"],
                arrays["subject"], fold, seed, cfg.negative_divisor)
            sample_path = directory / (target + "_training_sample.npz")
            model_path = directory / (target + ".cbm")
            receipt_path = directory / (target + "_training_receipt.json")
            existing = (model_path.is_file(), receipt_path.is_file(), sample_path.is_file())
            if any(existing):
                # Never rewrite sampling provenance underneath an existing model.
                if not all(existing):
                    raise ValueError("Incomplete model/sample/receipt; preserve artifacts and use a fresh output root")
                receipt = json.loads(receipt_path.read_text())
                assert receipt["fingerprint"] == fingerprint
                assert receipt["target"] == target and receipt["variant"] == variant
                assert receipt["seed"] == seed and receipt["parameters"] == PARAMETERS[variant]
                assert receipt["trees"] == (cfg.smoke_rounds if smoke else PARAMETERS[variant]["iterations"])
                assert receipt["actual_training_backend"] == ("CPU" if smoke else "GPU")
                assert receipt["smoke_only"] == smoke
                assert receipt["training_sample_sha256"] == digest(sample_path)
                with np.load(sample_path, allow_pickle=False) as saved_sample:
                    np.testing.assert_array_equal(saved_sample["indices"], chosen)
                    np.testing.assert_array_equal(saved_sample["weights"], weights)
                model = shared.load_model("catboost", model_path)
            else:
                np.savez_compressed(sample_path, indices=chosen, weights=weights)
                print(f"{variant} fold {fold} {target}: {len(chosen):,} fitting grid rows", flush=True)
                started = time.monotonic()
                model, backend = fit_model(variant, arrays["x"][chosen], arrays["y"][chosen, index],
                    weights, seed, gpu=not smoke, rounds=cfg.smoke_rounds if smoke else None)
                shared.save_model(model, "catboost", model_path)
                shared.common.write_json(receipt_path, {"fingerprint": fingerprint, "target": target,
                    "variant": variant, "parameters": PARAMETERS[variant], "seed": seed,
                    "training_subjects": training_subjects, "held_out_subjects": heldout_subjects,
                    "training_grid_rows_before_sampling": int((~heldout).sum()),
                    "sampled_grid_rows": len(chosen), "all_positive_training_rows_kept": True,
                    "negative_sampling_probability": probability, "inverse_negative_sampling_correction": True,
                    "subject_weights_training_only": True, "conditional_training": False,
                    "outer_labels_used_for_fitting_or_stopping": False, "early_stopping": False,
                    "base_OOF_predictions_used_as_features": False, "file_gate_applied": False,
                    "extra_positive_class_weights": False, "training_sample_file": sample_path.name,
                    "training_sample_sha256": digest(sample_path), "smoke_only": smoke,
                    "seconds": time.monotonic() - started, **backend})
            grid_scores[heldout, index] = shared.predict_model(model, "catboost", arrays["x"][heldout])
            del model
    assert np.isfinite(grid_scores).all()
    np.savez_compressed(run / "grid_scores.npz", prediction=grid_scores, targets=np.asarray(TARGETS))
    ys, ps = [], []
    for i, row in enumerate(manifest.itertuples(index=False)):
        selected = arrays["record"] == i
        p = shared.expand_native(arrays["grid"][selected], grid_scores[selected], int(arrays["lengths"][i]))
        y = (native_labels[i] if smoke else pd.read_csv(row.path, usecols=list(TARGETS)).loc[:, list(TARGETS)].to_numpy(np.uint8))
        assert p.shape == y.shape and np.isin(y, [0, 1]).all()
        np.savez_compressed(run / "oof" / (row.recording_id + ".npz"), prediction=p, targets=np.asarray(TARGETS))
        ys.append(y); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    scores = shared.common.average_precision(y, p)
    pd.DataFrame([scores]).to_csv(run / "native_oof_metrics.csv", index=False)
    result = {"status": "complete", "run_dir": str(run), "variant": variant, "targets": TARGETS,
        **scores, "native_rows": len(y), "smoke_only": smoke, "retained_heads_unchanged": True,
        "independent_generalization_established": False, "competition_submission_created": False}
    shared.common.write_json(run / "decision.json", result)
    shared.common.write_json(output_root / "latest_run.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def smoke_experiment(variant, output_root):
    rng = np.random.default_rng(1)
    folds = np.repeat(np.arange(5), 64).astype(np.int8); record = folds.astype(np.int16)
    y = np.zeros((320, 2), np.uint8)
    y[:, 0] = (np.arange(320) % 13 == 0).astype(np.uint8)
    y[:, 1] = (np.arange(320) % 11 == 1).astype(np.uint8)
    y[folds == 4] = 0
    manifest = pd.DataFrame({"recording_id": [f"smoke_{i}" for i in range(5)],
        "subject": [f"subject_{i}" for i in range(5)], "fold": range(5)})
    arrays = {"x": rng.normal(size=(320, 325)).astype(np.float32), "y": y, "fold": folds,
        "subject": manifest.subject.to_numpy(str)[record], "record": record,
        "grid": np.tile(np.arange(64, dtype=np.int32), 5), "lengths": np.full(5, 64),
        "recording_ids": manifest.recording_id.to_numpy(str)}
    native = [y[folds == i].copy() for i in range(5)]
    return run_experiment(variant, cfg=replace(CFG, smoke_rounds=4), output_root=output_root,
        synthetic=(arrays, manifest, native))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    (smoke_experiment if args.smoke else run_experiment)(args.variant, output_root=args.output_root)
