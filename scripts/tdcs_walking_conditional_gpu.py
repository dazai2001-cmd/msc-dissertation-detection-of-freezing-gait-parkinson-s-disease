"""Walking-only conditional CatBoost trials, with a frozen cross-fitted file gate."""
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
from sklearn.metrics import average_precision_score

import tdcs_gpu_boosted_context as shared

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Walking",)
VARIANTS = ("local_213", "context_325")
PARENT_FINGERPRINT = "7a2aa76886a59c9dc66c"
PARENT_LOCAL = ROOT / "kaggle-output/tdcs-gpu-catboost-context-v8-version-1/tdcs_gpu_catboost_v8" / PARENT_FINGERPRINT
GATE_FILE = ROOT / "results/tdcs_walking_conditional_gpu_gate_payload.json"
FLOOR = .05


@dataclass(frozen=True)
class Config:
    name: str = "tdcs_walking_conditional_gpu_v9"
    targets: tuple = TARGETS
    folds: tuple = (0, 1, 2, 3, 4)
    negative_divisor: int = 4
    seed: int = 4107
    smoke_rounds: int = 4
    max_seconds: int = 21600


CFG = Config()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_config(cfg):
    if tuple(cfg.targets) != TARGETS or tuple(cfg.folds) != tuple(range(5)):
        raise ValueError("Walking only and all five frozen folds required")
    if min(cfg.negative_divisor, cfg.smoke_rounds, cfg.max_seconds) < 1:
        raise ValueError("Invalid sampling or training budget")


def source_hashes():
    return {"source_sha256": globals().get("SOURCE_SHA256") or digest(__file__),
            "shared_sources": shared.source_hashes()}


def gate_payload():
    payload = globals().get("FROZEN_GATE_PAYLOAD")
    return payload if payload is not None else json.loads(GATE_FILE.read_text())


def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_parent():
    if Path("/kaggle/input").exists():
        matches = []
        for path in Path("/kaggle/input").rglob("configuration.json"):
            config = json.loads(path.read_text())
            if path.parent.name == PARENT_FINGERPRINT and config.get("learner") == "catboost":
                matches.append(path.parent)
        if len(matches) != 1:
            raise ValueError("Attach exactly the completed GPU CatBoost v8 notebook output")
        parent = matches[0]
    else:
        parent = PARENT_LOCAL
    config = json.loads((parent / "configuration.json").read_text())
    assert not config["smoke_only"] and config["sources"] == shared.source_hashes()
    cache = parent.parent / config["feature_cache_file"]
    assert digest(cache) == config["feature_cache_sha256"]
    with np.load(cache, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    assert shared.features.feature_digest(arrays) == config["feature_array_sha256"]
    raw, metadata = shared.common.locate_data()
    manifest = shared.common.build_manifest(raw, metadata)
    assert arrays["recording_ids"].tolist() == manifest.recording_id.tolist()
    assert arrays["x"].shape == (436128, 325)
    return parent, config, arrays, manifest


def native_presence(arrays, manifest):
    presence, positives, rows = [], 0, 0
    for i, row in enumerate(manifest.itertuples(index=False)):
        y = pd.read_csv(row.path, usecols=["Walking"]).Walking.to_numpy(np.uint8)
        assert np.isin(y, [0, 1]).all() and len(y) == arrays["lengths"][i]
        selected = arrays["record"] == i
        np.testing.assert_array_equal(y[arrays["grid"][selected]], arrays["y"][selected, 1])
        presence.append(int(y.any())); positives += int(y.sum()); rows += len(y)
    assert rows == 6959762 and positives == 207515 and sum(presence) == 66
    return np.asarray(presence, np.uint8)


def conditional_plan(y, folds, subjects, presence, fold, seed, divisor=4):
    y, folds, subjects, presence = map(np.asarray, (y, folds, subjects, presence))
    if y.ndim != 1 or any(len(a) != len(y) for a in (folds, subjects, presence)):
        raise ValueError("Aligned grid labels, folds, subjects and file presence required")
    if not np.isin(y, [0, 1]).all() or not np.isin(presence, [0, 1]).all() or divisor < 1:
        raise ValueError("Binary labels and positive sampling divisor required")
    training = folds != fold
    if ((y[training] == 1) & (presence[training] == 0)).any():
        raise ValueError("Positive training rows in a file marked absent")
    eligible = np.flatnonzero(training & (presence == 1))
    if not len(eligible):
        raise ValueError("No Walking-positive training files")
    positive, negative = eligible[y[eligible] == 1], eligible[y[eligible] == 0]
    count = max(1, int(np.ceil(len(negative) / divisor))) if len(negative) else 0
    rng = np.random.default_rng(seed)
    sampled = rng.choice(negative, count, replace=False) if count else np.empty(0, np.int64)
    chosen = np.r_[positive, sampled]; rng.shuffle(chosen)
    _, inverse, counts = np.unique(subjects[eligible], return_inverse=True, return_counts=True)
    weights = np.zeros(len(y), np.float64); weights[eligible] = 1. / counts[inverse]
    probability = count / len(negative) if len(negative) else 1.
    weights[sampled] /= probability
    values = weights[chosen]; values /= values.mean()
    assert not (folds[chosen] == fold).any()
    return chosen, values, probability, eligible


def combine_native(grid, raw_probability, length, file_score):
    if not 0 <= file_score <= 1 or not np.isfinite(file_score):
        raise ValueError("Finite cross-fitted file score required")
    if raw_probability.shape != (len(grid),):
        raise ValueError("One raw Walking score per grid row required")
    if grid[0] != 0 or grid[-1] >= length or np.any(np.diff(grid) <= 0):
        raise ValueError("Ordered in-record grid required")
    if not np.isfinite(raw_probability).all() or ((raw_probability < 0) | (raw_probability > 1)).any():
        raise ValueError("Invalid local score")
    raw = np.interp(np.arange(length), grid, raw_probability).astype(np.float32)
    # A one-dimensional float32 gate matches the original gate implementation
    # and avoids NumPy 1.x/2.x differences in zero-dimensional scalar promotion.
    factor = (FLOOR + (1 - FLOOR) * np.asarray([file_score], np.float32)).astype(np.float32)[0]
    return (raw * factor).astype(np.float32)[:, None]


def run_experiment(variant, cfg=CFG, output_root=None, synthetic=None):
    check_config(cfg)
    if variant not in VARIANTS:
        raise ValueError("Unknown feature comparison")
    smoke = synthetic is not None
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("Real-data Walking fitting requires Kaggle CUDA; no CPU fallback")
    columns = 213 if variant == "local_213" else 325
    output_root = Path(output_root or ROOT / "results/tdcs_walking_conditional_gpu" / variant)
    output_root.mkdir(parents=True, exist_ok=True)
    if not smoke:
        parent, parent_config, arrays, manifest = load_parent()
        presence = native_presence(arrays, manifest)
        gates = gate_payload()
        assert gates["recording_ids"] == manifest.recording_id.tolist()
        assert gates["folds"] == manifest.fold.tolist()
        q = np.asarray(gates["scores"], np.float32)
    else:
        arrays, manifest, presence, q, _ = synthetic
        parent, parent_config = Path("synthetic"), {}
        gates = {"recording_ids": manifest.recording_id.tolist(), "folds": manifest.fold.tolist(), "scores": q.tolist()}
    assert q.shape == (len(manifest),) and np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
    population = ("synthetic wiring only; not TDCS performance" if smoke else
        manifest[["recording_id", "subject", "fold", "source_bytes"]].to_csv(index=False, lineterminator="\n"))
    config = {"configuration": asdict(cfg), "targets": TARGETS, "variant": variant,
        "feature_count": columns, "sources": source_hashes(), "smoke_only": smoke,
        "population_sha256": hashlib.sha256(population.encode()).hexdigest(),
        "parent_fingerprint": parent.name, "parent_feature_array_sha256": parent_config.get("feature_array_sha256"),
        "parent_cache_sha256": parent_config.get("feature_cache_sha256"),
        "gate_payload_sha256": payload_digest(gates), "gate_floor": FLOOR,
        "runtime": {name: version(name) for name in ("catboost", "numpy", "torch")},
        "used_gpu_indices": [] if smoke else [0],
        "parameters": shared.PARAMETERS["catboost"], "training_scope": "Walking-positive training files only",
        "gate_used_as_fitting_feature": False, "base_OOF_predictions_used_as_features": False,
        "outer_labels_used_for_fitting_or_stopping": False, "early_stopping": False,
        "raw_local_scores_gated_exactly_once": True, "retained_heads_overwritten": False,
        "competition_submission_created": False}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    run = output_root / fingerprint; run.mkdir(parents=True, exist_ok=True)
    shared.common.write_json(run / "configuration.json", config)
    shared.common.write_json(run / "frozen_gate_payload.json", gates)
    manifest.to_csv(run / "frozen_tdcs_manifest.csv", index=False)
    (run / "oof").mkdir(exist_ok=True)
    raw_scores = np.full(len(arrays["y"]), np.nan, np.float32)
    deadline = time.monotonic() + cfg.max_seconds
    for fold in cfg.folds:
        selected, weights, probability, eligible = conditional_plan(arrays["y"][:, 1],
            arrays["fold"], arrays["subject"], presence[arrays["record"]], fold, cfg.seed + 1009 * fold,
            cfg.negative_divisor)
        heldout = arrays["fold"] == fold
        training_subjects = set(arrays["subject"][eligible]); test_subjects = set(arrays["subject"][heldout])
        assert training_subjects.isdisjoint(test_subjects)
        if time.monotonic() >= deadline:
            raise TimeoutError("Walking safety budget reached")
        directory = run / f"fold_{fold}"; directory.mkdir(exist_ok=True)
        model_path = directory / "Walking.cbm"; receipt_path = directory / "Walking_training_receipt.json"
        np.savez_compressed(directory / "Walking_training_sample.npz", indices=selected, weights=weights)
        if model_path.is_file() and receipt_path.is_file():
            assert json.loads(receipt_path.read_text())["fingerprint"] == fingerprint
            model = shared.load_model("catboost", model_path)
        else:
            print(f"{variant} fold {fold}: {len(selected):,} rows from {len(training_subjects)} Walking-positive training subjects", flush=True)
            model, backend = shared.fit_model("catboost", arrays["x"][selected, :columns],
                arrays["y"][selected, 1], weights, cfg.seed + 1009 * fold, gpu=not smoke,
                rounds=cfg.smoke_rounds if smoke else None)
            shared.save_model(model, "catboost", model_path)
            shared.common.write_json(receipt_path, {"fingerprint": fingerprint, "target": "Walking",
                "training_subjects": sorted(training_subjects), "held_out_subjects": sorted(test_subjects),
                "eligible_training_grid_rows": len(eligible), "sampled_grid_rows": len(selected),
                "training_recording_ids": arrays["recording_ids"][np.unique(arrays["record"][eligible])].tolist(),
                "all_positive_eligible_rows_kept": True, "negative_sampling_probability": probability,
                "inverse_sampling_correction": True, "subject_weights_from_eligible_training_rows_only": True,
                "outer_labels_used_for_fitting_or_stopping": False, "early_stopping": False,
                "gate_used_as_fitting_feature": False, "base_OOF_predictions_used_as_features": False,
                "feature_count": columns, "seed": cfg.seed + 1009 * fold, "smoke_only": smoke,
                "sample_sha256": digest(directory / "Walking_training_sample.npz"), **backend})
        raw_scores[heldout] = shared.predict_model(model, "catboost", arrays["x"][heldout, :columns])
        del model
    assert np.isfinite(raw_scores).all()
    np.savez_compressed(run / "grid_scores.npz", prediction=raw_scores[:, None], targets=np.asarray(TARGETS))
    ys, ps = [], []
    for i, row in enumerate(manifest.itertuples(index=False)):
        selected = arrays["record"] == i
        p = combine_native(arrays["grid"][selected], raw_scores[selected], int(arrays["lengths"][i]), q[i])
        y = (synthetic[4][i] if smoke else
             pd.read_csv(row.path, usecols=["Walking"]).loc[:, ["Walking"]].to_numpy(np.uint8))
        assert p.shape == y.shape
        np.savez_compressed(run / "oof" / (row.recording_id + ".npz"), prediction=p, targets=np.asarray(TARGETS))
        ys.append(y); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    ap = float(average_precision_score(y[:, 0], p[:, 0]))
    pd.DataFrame([{"Walking": ap}]).to_csv(run / "native_oof_metrics.csv", index=False)
    result = {"status": "complete", "run_dir": str(run), "variant": variant, "Walking": ap,
        "native_rows": len(y), "targets": TARGETS, "smoke_only": smoke,
        "no_two_or_three_target_mAP_claim": True, "retained_heads_unchanged": True,
        "independent_generalization_established": False, "competition_submission_created": False}
    shared.common.write_json(run / "decision.json", result)
    shared.common.write_json(output_root / "latest_run.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def smoke_experiment(variant, output_root):
    rng = np.random.default_rng(1); folds = np.repeat(np.arange(5), 64).astype(np.int8)
    records = folds.astype(np.int16); y = np.zeros((320, 2), np.uint8)
    native = []
    for fold in range(5):
        labels = ((np.arange(64) % 11 == 0) if fold != 4 else np.zeros(64, bool)).astype(np.uint8)
        y[fold * 64:(fold + 1) * 64, 1] = labels; native.append(labels[:, None])
    manifest = pd.DataFrame({"recording_id": [f"smoke_{i}" for i in range(5)],
        "subject": [f"subject_{i}" for i in range(5)], "fold": range(5)})
    arrays = {"x": rng.normal(size=(320, 325)).astype(np.float32), "y": y, "fold": folds,
        "subject": manifest.subject.to_numpy(str)[records], "record": records,
        "grid": np.tile(np.arange(64, dtype=np.int32), 5), "lengths": np.full(5, 64),
        "recording_ids": manifest.recording_id.to_numpy(str)}
    return run_experiment(variant, cfg=replace(CFG, smoke_rounds=4), output_root=output_root,
        synthetic=(arrays, manifest, np.asarray([1, 1, 1, 1, 0], np.uint8), np.full(5, .5, np.float32), native))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    (smoke_experiment if args.smoke else run_experiment)(args.variant, output_root=args.output_root)
