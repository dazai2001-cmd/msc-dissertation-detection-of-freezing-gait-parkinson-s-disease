"""Fixed-budget GPU boosted trees on the frozen TDCS sensor/context features."""
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

import tdcs_gpu_boosted_features as features

common = features.native.common
ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("StartHesitation", "Walking")
LEARNERS = ("catboost", "xgboost")
PARAMETERS = {
    "catboost": {"iterations": 1000, "depth": 6, "learning_rate": .04,
        "l2_leaf_reg": 20., "loss_function": "Logloss", "bootstrap_type": "Bernoulli",
        "subsample": .85, "border_count": 128, "boosting_type": "Plain",
        "random_strength": 1., "allow_writing_files": False, "thread_count": 2},
    "xgboost": {"num_boost_round": 900, "max_depth": 4, "learning_rate": .035,
        "min_child_weight": 30., "reg_lambda": 30., "subsample": .85,
        "colsample_bytree": .85, "max_bin": 128, "objective": "binary:logistic",
        "eval_metric": "logloss", "tree_method": "hist", "nthread": 2}}


@dataclass(frozen=True)
class Config:
    name: str = "tdcs_gpu_boosted_context_v8"
    targets: tuple = TARGETS
    folds: tuple = (0, 1, 2, 3, 4)
    negative_divisor: int = 4
    seed: int = 3107
    smoke_rounds: int = 12
    max_seconds: int = 21600


CFG = Config()


def check_config(cfg):
    if tuple(cfg.targets) != TARGETS or tuple(cfg.folds) != tuple(range(5)):
        raise ValueError("Two difficult targets and all five frozen subject folds required")
    if min(cfg.negative_divisor, cfg.smoke_rounds, cfg.max_seconds) < 1:
        raise ValueError("Invalid sampling or budget")


def source_hashes():
    modules = {"source": None, "feature_helper": features, "multiscale": features.base,
        "native": features.native, "common": common, "legacy": features.native.legacy,
        "recovery": features.native.safe}
    result = {}
    for name, module in modules.items():
        if module is None:
            digest = globals().get("SOURCE_SHA256") or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        else:
            digest = getattr(module, "SOURCE_SHA256", None) or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        result[name + "_sha256"] = digest
    result["frozen_feature_recipes"] = features.FROZEN_SOURCES
    return result


def training_plan(y, folds, subjects, fold, seed, divisor=4):
    y, folds, subjects = np.asarray(y), np.asarray(folds), np.asarray(subjects, str)
    if y.ndim != 1 or len(y) != len(folds) or len(y) != len(subjects) or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned binary labels and split metadata required")
    fitting = np.flatnonzero(folds != fold)
    if not len(fitting) or divisor < 1:
        raise ValueError("Nonempty fitting split and positive negative-sampling divisor required")
    positives, negatives = fitting[y[fitting] == 1], fitting[y[fitting] == 0]
    rng = np.random.default_rng(seed)
    count = max(1, int(np.ceil(len(negatives) / divisor))) if len(negatives) else 0
    sampled = rng.choice(negatives, size=count, replace=False) if count else np.empty(0, np.int64)
    chosen = np.r_[positives, sampled]; rng.shuffle(chosen)
    _, inverse, counts = np.unique(subjects[fitting], return_inverse=True, return_counts=True)
    weights = np.zeros(len(y), np.float64)
    weights[fitting] = 1. / counts[inverse]
    probability = count / len(negatives) if len(negatives) else 1.
    weights[sampled] /= probability
    selected_weights = weights[chosen]; selected_weights /= selected_weights.mean()
    assert not (folds[chosen] == fold).any() and set(positives) <= set(chosen)
    return chosen, selected_weights, probability


def fit_model(learner, x, y, weights, seed, gpu=True, rounds=None):
    if learner not in LEARNERS:
        raise ValueError("Unknown learner")
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU fitting requires CUDA; no CPU fallback")
    params = dict(PARAMETERS[learner])
    if learner == "catboost":
        from catboost import CatBoostClassifier
        params.update(task_type="GPU" if gpu else "CPU", random_seed=seed)
        if gpu:
            params.update(devices="0", gpu_ram_part=.6)
        if rounds is not None:
            params["iterations"] = rounds
        model = CatBoostClassifier(**params)
        model.fit(x, y, sample_weight=weights, verbose=200 if rounds is None else False)
        actual = model.get_all_params()["task_type"]
        if gpu and actual != "GPU":
            raise RuntimeError("CatBoost did not use its GPU backend")
        assert model.tree_count_ == params["iterations"]
        backend = {"actual_training_backend": actual, "trees": model.tree_count_}
    else:
        import xgboost as xgb
        params.update(device="cuda:0" if gpu else "cpu", seed=seed)
        count = rounds if rounds is not None else params.pop("num_boost_round")
        params.pop("num_boost_round", None)
        data = xgb.QuantileDMatrix(x, label=y, weight=weights, max_bin=params["max_bin"], nthread=2)
        model = xgb.train(params, data, num_boost_round=count, verbose_eval=False)
        actual = json.loads(model.save_config())["learner"]["generic_param"]["device"]
        if gpu and not actual.startswith("cuda"):
            raise RuntimeError("XGBoost silently fell back from CUDA")
        assert model.num_boosted_rounds() == count
        backend = {"actual_training_backend": actual, "trees": model.num_boosted_rounds()}
    return model, backend


def predict_model(model, learner, x):
    """CPU scoring of GPU-trained models is explicit and portable."""
    if learner == "catboost":
        p = model.predict_proba(x, task_type="CPU", thread_count=2)[:, 1]
    else:
        import xgboost as xgb
        model.set_param({"device": "cpu", "nthread": 2})
        p = model.predict(xgb.DMatrix(x, nthread=2))
    p = np.asarray(p, np.float32)
    assert p.shape == (len(x),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def save_model(model, learner, path):
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    model.save_model(str(temporary))
    temporary.replace(path)


def load_model(learner, path):
    if learner == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
    else:
        from xgboost import Booster
        model = Booster(params={"nthread": 2, "device": "cpu"})
    if learner == "catboost":
        model.load_model(str(path))
    else:
        # Native buffer loading also avoids Windows filesystem-canonicalization
        # failures in older XGBoost builds; the model bytes are unchanged.
        model.load_model(bytearray(path.read_bytes()))
    return model


def expand_native(grid, probability, length):
    if length < 1 or not len(grid) or grid[0] != 0 or grid[-1] >= length or np.any(np.diff(grid) <= 0):
        raise ValueError("Ordered in-record grid beginning at zero required")
    if probability.shape != (len(grid), 2):
        raise ValueError("Two sampled output columns required")
    return np.column_stack([np.interp(np.arange(length), grid, probability[:, i]) for i in range(2)]).astype(np.float32)


def run_experiment(learner, cfg=CFG, output_root=None, synthetic=None):
    check_config(cfg)
    if learner not in LEARNERS:
        raise ValueError("Unknown learner")
    smoke = synthetic is not None
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("Real-data boosted-tree training requires Kaggle CUDA; no CPU fallback")
    output_root = Path(output_root or ROOT / "results/tdcs_gpu_boosted_context" / learner)
    output_root.mkdir(parents=True, exist_ok=True)
    raw, metadata = common.locate_data() if not smoke else (None, None)
    manifest = common.build_manifest(raw, metadata) if not smoke else synthetic[1]
    population = (manifest[["recording_id", "subject", "fold", "source_bytes"]].to_csv(index=False, lineterminator="\n")
                  if not smoke else "synthetic wiring only; not TDCS performance")
    settings = {"configuration": asdict(cfg), "learner": learner, "targets": TARGETS,
        "parameters": PARAMETERS[learner], "sources": source_hashes(),
        "population_sha256": hashlib.sha256(population.encode()).hexdigest(), "smoke_only": smoke,
        "negative_sampling": "all positives + seeded one-quarter negatives with inverse sampling correction",
        "subject_weights": "inverse full-fitting-row count per training subject, then normalized",
        "positive_class_weights": [1., 1.], "early_stopping": False,
        "outer_labels_used_for_fitting_or_stopping": False, "base_OOF_predictions_used_as_features": False,
        "file_gate_applied": False, "retained_heads_overwritten": False, "competition_submission_created": False}
    common.write_json(output_root / "predeclared_settings.json", settings)
    if not smoke:
        # Fail before expensive feature preparation if this installed library
        # cannot really train on CUDA. This tiny random-data check is not an AP trial.
        rng = np.random.default_rng(9)
        probe, backend = fit_model(learner, rng.normal(size=(128, 16)).astype(np.float32),
            np.tile([0, 1], 64), np.ones(128), 9, gpu=True, rounds=2)
        print("Verified GPU backend:", backend, flush=True); del probe
        arrays = features.prepare_arrays(manifest)
    else:
        arrays = synthetic[0]
    cache = output_root / "engineered_features.npz"
    np.savez_compressed(cache, **arrays)
    config = {**settings, "feature_count": arrays["x"].shape[1],
        "feature_array_sha256": features.feature_digest(arrays), "feature_cache_file": cache.name,
        "feature_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "runtime": {name: version(name) for name in ("numpy", "pandas", "torch", learner)},
        "visible_gpu_count": torch.cuda.device_count(), "used_gpu_indices": [0] if not smoke else [],
        "prediction_backend": "CPU scoring of final GPU-trained model; all native rows by in-record interpolation",
        "reused_development_folds": True, "confidence_calibrated": False}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    run = output_root / fingerprint; run.mkdir(parents=True, exist_ok=True)
    common.write_json(run / "configuration.json", config)
    manifest.to_csv(run / "frozen_tdcs_manifest.csv", index=False)
    (run / "oof").mkdir(exist_ok=True)
    grid_scores = np.full(arrays["y"].shape, np.nan, np.float32)
    deadline = time.monotonic() + cfg.max_seconds
    for fold in cfg.folds:
        heldout = arrays["fold"] == fold
        training_subjects = sorted(set(arrays["subject"][~heldout]))
        heldout_subjects = sorted(set(arrays["subject"][heldout]))
        assert set(training_subjects).isdisjoint(heldout_subjects)
        directory = run / f"fold_{fold}"; directory.mkdir(exist_ok=True)
        for index, target in enumerate(TARGETS):
            if time.monotonic() >= deadline:
                raise TimeoutError("Boosted-tree safety budget reached")
            seed = cfg.seed + 1009 * fold + index
            chosen, weights, probability = training_plan(arrays["y"][:, index], arrays["fold"],
                arrays["subject"], fold, seed, cfg.negative_divisor)
            sample_path = directory / (target + "_training_sample.npz")
            np.savez_compressed(sample_path, indices=chosen, weights=weights)
            path = directory / (target + (".cbm" if learner == "catboost" else ".ubj"))
            receipt_path = directory / (target + "_training_receipt.json")
            if path.is_file() and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text()); assert receipt["fingerprint"] == fingerprint
                model = load_model(learner, path)
            else:
                print(f"{learner} fold {fold} {target}: {len(chosen):,} fitting grid rows", flush=True)
                started = time.monotonic()
                model, actual = fit_model(learner, arrays["x"][chosen], arrays["y"][chosen, index],
                    weights, seed, gpu=not smoke, rounds=cfg.smoke_rounds if smoke else None)
                save_model(model, learner, path)
                common.write_json(receipt_path, {"fingerprint": fingerprint, "target": target, "learner": learner,
                    "training_subjects": training_subjects, "held_out_subjects": heldout_subjects,
                    "training_grid_rows_before_sampling": int((~heldout).sum()), "sampled_grid_rows": len(chosen),
                    "all_positive_training_rows_kept": True, "negative_sampling_probability": probability,
                    "inverse_negative_sampling_correction": True, "subject_weights_training_only": True,
                    "outer_labels_used_for_fitting_or_stopping": False, "early_stopping": False,
                    "base_OOF_predictions_used_as_features": False, "extra_positive_class_weight": False,
                    "parameters": PARAMETERS[learner], "seed": seed, "seconds": time.monotonic() - started,
                    "training_sample_file": sample_path.name,
                    "training_sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                    "smoke_only": smoke, **actual})
            grid_scores[heldout, index] = predict_model(model, learner, arrays["x"][heldout])
            del model
    assert np.isfinite(grid_scores).all()
    np.savez_compressed(run / "grid_scores.npz", prediction=grid_scores, targets=np.asarray(TARGETS))
    ys, ps = [], []
    for i, row in enumerate(manifest.itertuples(index=False)):
        selected = arrays["record"] == i
        p = expand_native(arrays["grid"][selected], grid_scores[selected], int(arrays["lengths"][i]))
        if smoke:
            y = synthetic[2][i]
        else:
            frame = pd.read_csv(row.path, usecols=list(TARGETS))
            y = frame.loc[:, list(TARGETS)].to_numpy(np.uint8)
        assert p.shape == y.shape and np.isin(y, [0, 1]).all()
        np.savez_compressed(run / "oof" / (row.recording_id + ".npz"), prediction=p, targets=np.asarray(TARGETS))
        ys.append(y); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    scores = common.average_precision(y, p)
    pd.DataFrame([scores]).to_csv(run / "native_oof_metrics.csv", index=False)
    decision = {"status": "complete", "run_dir": str(run), "learner": learner, **scores,
        "native_rows": len(y), "smoke_only": smoke, "targets": TARGETS,
        "retained_heads_unchanged": True, "goal_two_target_mAP": .6,
        "goal_reached": not smoke and scores["two_target_mAP"] >= .6,
        "independent_generalization_established": False, "competition_submission_created": False}
    common.write_json(run / "decision.json", decision)
    common.write_json(output_root / "latest_run.json", decision)
    print(json.dumps(decision, indent=2), flush=True)
    return decision


def smoke_experiment(learner, output_root):
    rng, xs, labels, folds, records, grids, native_labels = np.random.default_rng(1), [], [], [], [], [], []
    for fold in range(5):
        x = rng.normal(size=(48, 12)).astype(np.float32)
        y = np.column_stack([np.arange(48) % 7 == 0, np.arange(48) % 11 == 0]).astype(np.uint8)
        xs.append(x); labels.append(y); folds.extend([fold] * 48); records.extend([fold] * 48)
        grids.extend(range(48)); native_labels.append(y)
    manifest = pd.DataFrame({"recording_id": [f"smoke_{i}" for i in range(5)],
        "subject": [f"subject_{i}" for i in range(5)], "fold": range(5)})
    arrays = {"x": np.concatenate(xs), "y": np.concatenate(labels), "fold": np.asarray(folds, np.int8),
        "record": np.asarray(records, np.int16), "grid": np.asarray(grids, np.int32),
        "lengths": np.full(5, 48, np.int32), "recording_ids": manifest.recording_id.to_numpy(str),
        "subject": manifest.subject.to_numpy(str)[records],
        "feature_names": np.asarray([f"feature_{i}" for i in range(12)])}
    return run_experiment(learner, replace(CFG, smoke_rounds=4), output_root, (arrays, manifest, native_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner", choices=LEARNERS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke_experiment(args.learner, args.output_root)
    else:
        run_experiment(args.learner, output_root=args.output_root)
