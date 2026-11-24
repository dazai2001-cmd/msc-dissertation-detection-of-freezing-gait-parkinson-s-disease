"""Two labelled suites in two Kaggle runs; never updates best-model registries."""
from __future__ import annotations

import gc
import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

import tdcs_suite_data as data
import tdcs_suite_models as trees
import tdcs_suite_neural as neural

ROOT = Path(__file__).resolve().parents[1]
TARGETS = data.TARGETS


def method(id, title, suite, kind="catboost", encoding="record_metadata", **options):
    return {"id": id, "title": title, "suite": suite, "kind": kind, "encoding": encoding, **options}


CATALOG = [
    method("M00", "Control: all three recording metadata fields", "metadata"),
    method("M01", "Within-record ranks plus recording metadata", "metadata", encoding="rank_metadata"),
    method("M02", "Metadata-aware temporal CNN", "metadata", "neural", architecture="tcn"),
    method("M03", "Metadata-aware bidirectional GRU", "metadata", "neural", architecture="bigru"),
    method("M04", "Metadata-aware bidirectional LSTM", "metadata", "neural", architecture="bilstm"),
    method("M05", "Metadata-aware attention encoder", "metadata", "neural", architecture="attention"),
    method("M06", "XGBoost plus recording metadata", "metadata", "xgboost"),
    method("M07", "Depthwise CatBoost plus metadata", "metadata", policy="Depthwise"),
    method("M08", "Lossguide CatBoost plus metadata", "metadata", policy="Lossguide"),
    method("M09", "213 local sensor columns plus metadata", "metadata", encoding="local_metadata"),
    method("M10", "213 local rank columns plus metadata", "metadata", encoding="local_rank_metadata"),
    method("M11", "Recording and clinical metadata plus sensors", "metadata", encoding="clinical_metadata"),
    method("M12", "Metadata-only baseline: no sensor predictors or sensor gate", "metadata", encoding="metadata_only", gate=False),
    method("M13", "Field ablation: Visit only plus sensors", "metadata", encoding="visit_only"),
    method("M14", "Field ablation: Test only plus sensors", "metadata", encoding="test_only"),
    method("M15", "Field ablation: Medication only plus sensors", "metadata", encoding="medication_only"),
    method("M16", "Medication-specific experts with a global fallback", "metadata", experts="medication"),
    method("M17", "Test-specific experts with a global fallback", "metadata", experts="test"),
    method("M18", "Auxiliary Kinetic training label with metadata-aware CNN", "metadata", "neural", architecture="tcn", kinetic=True),
    method("S00", "Sensor-only CatBoost control", "sensor", encoding="sensor"),
    method("S01", "LightGBM: explicit Kaggle CPU training", "sensor", "lightgbm", encoding="sensor"),
    method("S02", "Supervised sensor CNN control for pretraining", "sensor", "neural", "sensor", architecture="tcn"),
    method("S03", "Masked reconstruction pretraining on mapped unlabeled recordings", "sensor", "neural", "sensor", architecture="tcn", pretrain=True),
    method("S04", "Teacher-student pseudo-label training", "sensor", "pseudo", "sensor"),
    method("S05", "Sensor-only binary XGBoost control for ranking", "sensor", "xgboost", "sensor"),
    method("S06", "Per-record MAP-focused tree ranking", "sensor", "rank_map", "sensor"),
]


@dataclass(frozen=True)
class Config:
    seed: int = 27103
    tree_rounds: int = 1000
    neural_epochs: int = 12
    steps_per_epoch: int = 64
    pretrain_steps: int = 512
    batch_size: int = 16
    window: int = 256
    width: int = 64
    max_seconds: int = 21600
    folds: tuple = (0, 1, 2, 3, 4)
    smoke: bool = False


def smoke_config():
    return Config(tree_rounds=4, neural_epochs=1, steps_per_epoch=1, pretrain_steps=1,
        batch_size=4, window=32, width=16, max_seconds=600, smoke=True)


def source_hashes():
    modules = {"runner": __file__, "data": data.__file__, "trees": trees.__file__, "neural": neural.__file__}
    result = {}
    import sys
    for name, path in modules.items():
        module = sys.modules[__name__] if name == "runner" else {"data": data, "trees": trees, "neural": neural}[name]
        result[name] = getattr(module, "SOURCE_SHA256", None) or data.digest(path)
    result["frozen_rank_helpers"] = data.frozen.source_hashes()
    return result


def write(path, value):
    data.frozen.shared.common.write_json(Path(path), value)


def initialize(suite, output, cfg=None):
    if suite not in ("metadata", "sensor"):
        raise ValueError("Unknown suite")
    cfg = cfg or Config()
    if cfg.folds != (0, 1, 2, 3, 4) or min(cfg.tree_rounds, cfg.neural_epochs, cfg.steps_per_epoch) < 1:
        raise ValueError("All five folds and positive fixed training budgets are required")
    if not cfg.smoke and (not Path("/kaggle/working").exists() or not torch.cuda.is_available()):
        raise RuntimeError("Real-data fitting is authorized only on Kaggle CUDA; local execution must be synthetic")
    torch.set_num_threads(2)
    context = data.synthetic_context() if cfg.smoke else data.load_context()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    session = {"suite": suite, "output": output, "cfg": cfg, "context": context,
        "deadline": time.monotonic() + cfg.max_seconds, "results": []}
    versions = {}
    for name in ("numpy", "torch", "catboost", "xgboost", "lightgbm"):
        try:
            versions[name] = version(name)
        except Exception:
            versions[name] = None
    session["runtime"] = versions
    write(output / "suite_manifest.json", {"suite": suite, "methods": [s for s in CATALOG if s["suite"] == suite],
        "configuration": asdict(cfg), "sources": source_hashes(), "runtime": versions,
        "clinical_join_audit": context["clinical_audit"], "input_sources": context["source_inputs"],
        "all_negative_recordings_retained": True, "outer_stopping": False,
        "independent_generalization_established": False, "retained_heads_changed": False,
        "independent_validation_required_before_promotion": True, "competition_submission_created": False})
    return session


def condition_values(context, field):
    m = context["record_metadata"]
    if field == "medication":
        return np.where(m[:, 6] == 1, 0, np.where(m[:, 7] == 1, 1, 2))
    return np.where(m[:, 2] == 1, 0, np.where(m[:, 3] == 1, 1, np.where(m[:, 4] == 1, 2, 3)))


def fit_tree_fold(session, spec, x, fold, directory, fingerprint, external):
    c, cfg = session["context"], session["cfg"]
    a = c["arrays"]
    heldout = a["fold"] == fold
    fitting_subjects = sorted(set(a["subject"][~heldout]))
    heldout_subjects = sorted(set(a["subject"][heldout]))
    assert set(fitting_subjects).isdisjoint(heldout_subjects)
    predictions, models = np.empty((int(heldout.sum()), 2), np.float32), []
    for target in range(2):
        seed = cfg.seed + 1009 * fold + target
        chosen, weights, probability = data.frozen.parent.training_plan(a["y"][:, target], a["fold"],
            a["subject"], fold, seed, 4, .5)
        kind = "catboost" if spec["kind"] == "pseudo" else spec["kind"]
        groups = None
        if kind == "rank_map":
            chosen, groups, weights = trees.ranking_plan(chosen, weights, a["record"])
        np.savez_compressed(directory / (TARGETS[target] + "_sample.npz"), indices=chosen, weights=weights,
            groups=np.asarray([] if groups is None else groups))
        model, backend = trees.fit_tree(kind, x[chosen], a["y"][chosen, target], weights, seed,
            cfg, not cfg.smoke, spec.get("policy", "SymmetricTree"), groups)
        info = {"target": TARGETS[target], "kind": kind, "training_sample_probability": probability,
            "sample_sha256": data.digest(directory / (TARGETS[target] + "_sample.npz")), "backend": backend,
            "pseudo_rows_added": 0, "experts": [], "expert_routing": []}
        if spec["kind"] == "pseudo":
            teacher_path = trees.save_tree(model, kind, directory / (TARGETS[target] + "_teacher"))
            info["teacher_file"], info["teacher_sha256"] = teacher_path.name, data.digest(teacher_path)
            eligible = data.external_training_indices(external, heldout_subjects)
            probabilities = trees.predict_tree(model, kind, external["x"][eligible])
            selected = np.flatnonzero((probabilities >= .90) | (probabilities <= .01))
            if len(selected) > 40000:
                selected = np.random.default_rng(seed).choice(selected, 40000, replace=False)
            indices = eligible[selected]
            pseudo = (probabilities[selected] >= .90).astype(np.uint8)
            np.savez_compressed(directory / (TARGETS[target] + "_pseudo.npz"), indices=indices, labels=pseudo,
                teacher_probabilities=probabilities[selected], weights=np.full(len(indices), .1))
            info["pseudo_rows_added"] = len(indices)
            if len(indices):
                combined_weights = np.r_[weights, np.full(len(indices), .1)]
                combined_weights /= combined_weights.mean()
                model, backend = trees.fit_tree(kind, np.vstack([x[chosen], external["x"][indices]]),
                    np.r_[a["y"][chosen, target], pseudo], combined_weights, seed, cfg, not cfg.smoke)
                info["student_backend"] = backend
        path = trees.save_tree(model, kind, directory / TARGETS[target])
        info["model_file"], info["model_sha256"] = path.name, data.digest(path)
        model = trees.load_tree(kind, path)
        predictions[:, target] = trees.predict_tree(model, kind, x[heldout])
        if spec.get("experts"):
            conditions = condition_values(c, spec["experts"])[a["record"]]
            for value in np.unique(conditions):
                mask = conditions[chosen] == value
                labels = a["y"][chosen[mask], target]
                routing = {"condition": int(value), "training_rows": int(mask.sum()),
                    "heldout_rows": int((conditions[heldout] == value).sum())}
                # Fixed small-group fallback; no held-out labels decide routing.
                if mask.sum() < 128 or labels.sum() < 5 or (1 - labels).sum() < 5:
                    info['expert_routing'].append({**routing, 'route': 'global_small_fitting_group'})
                    continue
                expert, actual = trees.fit_tree(kind, x[chosen[mask]], labels, weights[mask], seed,
                    cfg, not cfg.smoke, spec.get("policy", "SymmetricTree"))
                expert_path = trees.save_tree(expert, kind, directory / (TARGETS[target] + f"_expert_{value}"))
                selected = conditions[heldout] == value
                if selected.any():
                    predictions[selected, target] = trees.predict_tree(expert, kind, x[heldout][selected])
                info['expert_routing'].append({**routing, 'route': 'fitting_group_expert'})
                info["experts"].append({"condition": int(value), "model_file": expert_path.name,
                    "model_sha256": data.digest(expert_path), "training_rows": int(mask.sum()), "backend": actual})
        models.append(info)
    return predictions, {"fingerprint": fingerprint, "fold": fold, "models": models,
        "fitting_subjects": list(map(str, fitting_subjects)), "heldout_subjects": list(map(str, heldout_subjects)),
        "outer_labels_used_for_training_or_stopping": False,
        "ranking_weights_are_per_query_not_per_row": spec["kind"] == "rank_map"}


def evaluate(context, scores, apply_gate=True):
    a = context["arrays"]
    native = np.empty_like(context["native_y"], dtype=np.float32)
    native_folds = np.empty(len(native), np.int8)
    for i, row in enumerate(context["manifest"].itertuples(index=False)):
        selected = a["record"] == i
        left, right = context["native_offsets"][i:i+2]
        for target in range(2):
            native[left:right, target] = np.interp(np.arange(right-left), a["grid"][selected], scores[selected, target]).astype(np.float32)
        if apply_gate:
            factor = (np.asarray([context["gates"][i]], np.float32) * .95 + .05)[0]
            native[left:right, 1] *= factor
        native_folds[left:right] = row.fold
    assert np.isfinite(native).all() and ((native >= 0) & (native <= 1)).all()
    values = {target: float(average_precision_score(context["native_y"][:, j], native[:, j]))
        for j, target in enumerate(TARGETS)}
    folds = []
    for fold in range(5):
        selected = native_folds == fold
        row = {"fold": fold}
        for j, target in enumerate(TARGETS):
            row[target] = (float(average_precision_score(context["native_y"][selected, j], native[selected, j]))
                if context["native_y"][selected, j].any() else None)
        folds.append(row)
    values["two_target_mAP"] = float(np.mean(list(values.values())))
    return values, folds, data.array_digest(native)


def run_method(session, method_id):
    spec = next(s for s in CATALOG if s["id"] == method_id and s["suite"] == session["suite"])
    c, cfg, output = session["context"], session["cfg"], session["output"]
    status = {"id": method_id, "title": spec["title"], "suite": spec["suite"], "status": "pending"}
    try:
        if time.monotonic() >= session["deadline"]:
            raise TimeoutError("Suite runtime budget reached before this test")
        x = data.input_features(c, spec["encoding"])
        a = c["arrays"]
        external = data.external_pool(c, output) if spec.get("pretrain") or spec["kind"] == "pseudo" else None
        if spec.get("kinetic") and not c["kinetic"][:, 1].any():
            raise ValueError("No mapped Kinetic auxiliary labels are available")
        config = {"method": spec, "configuration": asdict(cfg), "sources": source_hashes(),
            "runtime": session["runtime"],
            "input_feature_count": x.shape[1], "input_feature_sha256": data.array_digest(x),
            "population_recordings": c["manifest"].recording_id.tolist(),
            "population_subjects": c["manifest"].subject.tolist(), "population_folds": c["manifest"].fold.tolist(),
            "source_inputs": c["source_inputs"], "gate_applied": spec.get("gate", True),
            "external_pool_sha256": ({key: data.array_digest(value) for key, value in external.items()}
                if external is not None else None),
            "smoke_only": cfg.smoke, "outer_stopping": False, "automatic_promotion": False}
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
        run = output / "methods" / method_id / fingerprint
        run.mkdir(parents=True, exist_ok=True)
        write(run / "configuration.json", config)
        scores = np.full((len(x), 2), np.nan, np.float32)
        for fold in cfg.folds:
            if time.monotonic() >= session["deadline"]:
                raise TimeoutError("Suite runtime budget reached between folds")
            directory = run / f"fold_{fold}"
            directory.mkdir(exist_ok=True)
            saved, receipt_path = directory / "heldout_grid.npz", directory / "training_receipt.json"
            heldout = a["fold"] == fold
            if saved.is_file() and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["fingerprint"] == fingerprint and receipt["fold"] == fold
                assert receipt["grid_sha256"] == data.digest(saved)
                for filename, expected in receipt["artifact_sha256"].items():
                    assert data.digest(directory / filename) == expected
                with np.load(saved, allow_pickle=False) as payload:
                    np.testing.assert_array_equal(payload["indices"], np.flatnonzero(heldout))
                    scores[heldout] = payload["prediction"]
                print(f"{method_id} fold {fold}: reused verified complete checkpoint", flush=True)
                continue
            if any(directory.iterdir()):
                raise ValueError("Incomplete fold artifacts: preserve them and use a fresh output root; no silent overwrite")
            if spec["kind"] == "neural":
                device = torch.device("cpu" if cfg.smoke else "cuda:0")
                model, prep, receipt = neural.train(c, x, fold, spec, cfg, directory, device, fingerprint,
                    external=external, deadline=session["deadline"])
                parts = np.empty((int(heldout.sum()), 2), np.float32)
                location = np.full(len(x), -1, np.int64); location[heldout] = np.arange(heldout.sum())
                for record in np.unique(a["record"][heldout]):
                    rows = np.flatnonzero((a["record"] == record) & heldout)
                    parts[location[rows]] = neural.predict(model, x[rows], prep, cfg, device)
                receipt.update(fingerprint=fingerprint, fold=fold)
                del model
            else:
                parts, receipt = fit_tree_fold(session, spec, x, fold, directory, fingerprint, external)
            assert parts.shape == (int(heldout.sum()), 2) and np.isfinite(parts).all()
            scores[heldout] = parts
            np.savez_compressed(saved, indices=np.flatnonzero(heldout), prediction=parts)
            receipt["grid_sha256"] = data.digest(saved)
            receipt["artifact_sha256"] = {p.name: data.digest(p) for p in directory.iterdir()
                if p.is_file() and p.name not in ("training_receipt.json", "heldout_grid.npz")}
            write(receipt_path, receipt)
            print(f"{method_id} fold {fold}: training and held-out prediction complete", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        np.savez_compressed(run / "grid_scores.npz", prediction=scores, targets=np.asarray(TARGETS))
        values, folds, native_hash = evaluate(c, scores, spec.get("gate", True))
        status.update(status="complete", **values, run_directory=str(run), native_prediction_sha256=native_hash,
            independent_validation_pending=True, smoke_only=cfg.smoke, actual_grid_rows=len(scores))
        write(run / "decision.json", {**status, "folds": folds, "all_negative_rows_retained": True,
            "retained_heads_changed": False, "confidence_calibrated": False,
            "independent_generalization_established": False, "competition_submission_created": False})
        del x
    except Exception as error:
        status.update(status="not_run_budget" if isinstance(error, TimeoutError) else "failed",
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(f"{method_id}: {status['status']}: {error}", flush=True)
    session["results"].append(status)
    write(output / (method_id + "_status.json"), status)
    pd.DataFrame([{key: value for key, value in row.items() if key != "traceback"} for row in session["results"]]).to_csv(output / "suite_summary.csv", index=False)
    return {key: value for key, value in status.items() if key != "traceback"}


def finish(session):
    expected = {s["id"] for s in CATALOG if s["suite"] == session["suite"]}
    actual = {s["id"] for s in session["results"]}
    complete = [s["id"] for s in session["results"] if s["status"] == "complete"]
    result = {"status": "all_methods_completed_validation_pending" if len(complete) == len(expected) else "partial_coverage",
        "expected_tests": sorted(expected), "attempted_tests": sorted(actual), "completed_tests": complete,
        "missing_tests": sorted(expected - actual), "smoke_only": session["cfg"].smoke,
        "all_complete_results_require_independent_replay_before_promotion": True,
        "retained_heads_changed": False, "competition_submission_created": False,
        "confidence_calibrated": False, "independent_generalization_established": False}
    write(session["output"] / "suite_completion.json", result)
    return result
