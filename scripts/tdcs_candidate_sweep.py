"""Bounded TDCS-only candidates on the existing frozen subject folds.

No leaderboard calls, inference submission, learned stacking, or DeFOG writes.
Configurations and blend weights are saved before any new candidate is scored.
Repeated use of these folds makes model selection exploratory, not an independent
estimate of the final selected model's generalization.
"""

# %% Setup and fixed candidate definitions
from __future__ import annotations

import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import catboost
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import tdcs_multiscale_signal_experiment as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "tdcs_candidate_sweep"
NEURAL_ROOT = ROOT / "kaggle-output" / "fog-evidence-weighted" / "competition" / "neural" / "tdcsfog"
TARGETS = base.TARGETS
MODEL_PARAMS = {
    "iterations": 400, "learning_rate": 0.05, "depth": 6,
    "l2_leaf_reg": 10.0, "loss_function": "Logloss",
    "border_count": 128, "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.8, "random_strength": 1.0,
    "rsm": 0.85, "thread_count": 4, "task_type": "CPU",
    "use_best_model": False, "allow_writing_files": False,
    "verbose": False,
}
# The neural blend exactly matches the old predeclared TDCS blend.
NEURAL_WEIGHTS = {
    "patch_bilstm": [0.5, 0.375, 0.5],
    "engineered_bigru": [0.5, 0.375, 0.5],
    "tdcs_turn_specialist": [0.0, 0.25, 0.0],
}
# Three deliberately bounded, fixed blends; no weight search on OOF labels.
FIXED_BLENDS = {
    "neural_hgb_equal": {"previous_tdcs_blend": 0.5, "multiscale_sensor_hgb": 0.5},
    "hgb_catboost_equal": {"multiscale_sensor_hgb": 0.5, "multiscale_catboost": 0.5},
    "neural_sensor_three_way": {
        "previous_tdcs_blend": 0.5, "multiscale_sensor_hgb": 0.25, "multiscale_catboost": 0.25,
    },
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


# %% Verify cached features and native-row prediction coverage
def load_inputs() -> tuple[Path, pd.DataFrame, dict, dict]:
    latest = json.loads((ROOT / "results" / "tdcs_multiscale_signal" / "latest_run.json").read_text())
    sensor_run = Path(latest["run_dir"])
    audit = json.loads((sensor_run / "audit.json").read_text())
    validation = json.loads((sensor_run / "validation.json").read_text())
    if validation["status"] != "passed" or audit["configuration"] != json.loads(json.dumps(base.asdict(base.CFG))):
        raise ValueError("The feature cache must come from the independently verified sensor model")
    manifest = pd.read_csv(sensor_run / "frozen_tdcs_manifest.csv", dtype={"recording_id": str, "subject": str})
    if len(manifest) != 815 or manifest.subject.nunique() != 61:
        raise ValueError("TDCS validation population changed")
    if manifest.groupby("subject").fold.nunique().max() != 1:
        raise ValueError("Subjects cross folds")
    for row in manifest.itertuples(index=False):
        if base.FROZEN_SUBJECT_FOLDS[row.subject] != row.fold:
            raise ValueError("Frozen subject assignment changed")
    with np.load(sensor_run / "sampled_features.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    if len(arrays["x"]) != len(arrays["y"]) or arrays["x"].shape[1] != 101:
        raise ValueError("Feature cache shape changed")
    if int(arrays["row_counts"].sum()) != 6959762:
        raise ValueError("Native row count changed")
    for index, row in enumerate(manifest.itertuples(index=False)):
        mask = arrays["record"] == index
        if not mask.any() or not np.all(arrays["fold"][mask] == row.fold):
            raise ValueError("Sampled features have wrong folds")
    return sensor_run, manifest, arrays, audit


def load_native_inputs(sensor_run: Path, manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    neural_indexes = {}
    expected = set(manifest.recording_id)
    for branch in NEURAL_WEIGHTS:
        paths = list((NEURAL_ROOT / branch).glob("fold_*/seed_42/oof/*.npz"))
        index = {path.stem: path for path in paths}
        if len(paths) != len(index) or set(index) != expected:
            raise ValueError(f"Incomplete or duplicate OOF recordings in {branch}")
        neural_indexes[branch] = index
    if {p.stem for p in (sensor_run / "oof").glob("*.npz")} != expected:
        raise ValueError("Sensor OOF coverage differs")
    labels, folds, neural, sensor = [], [], [], []
    for row in manifest.itertuples(index=False):
        y = pd.read_csv(row.path, usecols=list(TARGETS)).loc[:, list(TARGETS)].to_numpy(np.uint8)
        blend = np.zeros(y.shape, np.float32)
        for branch, weights in NEURAL_WEIGHTS.items():
            path = neural_indexes[branch][row.recording_id]
            if f"fold_{row.fold}" not in path.parts:
                raise ValueError("Neural OOF prediction is from the wrong fold")
            with np.load(path, allow_pickle=False) as payload:
                p = payload["prediction"]
            validate_prediction(y, p)
            blend += p * np.asarray(weights, np.float32)
        with np.load(sensor_run / "oof" / f"{row.recording_id}.npz", allow_pickle=False) as payload:
            hgb = payload["prediction"].copy()
        validate_prediction(y, hgb)
        labels.append(y)
        neural.append(blend)
        sensor.append(hgb)
        folds.append(np.full(len(y), row.fold, np.int8))
    y = np.concatenate(labels)
    if len(y) != 6959762:
        raise ValueError("Native label coverage changed")
    return y, np.concatenate(folds), {
        "previous_tdcs_blend": np.concatenate(neural),
        "multiscale_sensor_hgb": np.concatenate(sensor),
    }


def validate_prediction(y: np.ndarray, p: np.ndarray) -> None:
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Label/prediction shape or labels invalid")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid probabilities")


# %% Train an independent CatBoost model for each target and held-out fold
def fit_catboost(arrays: dict, run_dir: Path) -> np.ndarray:
    x, y, folds = arrays["x"], arrays["y"], arrays["fold"]
    prediction = np.full(y.shape, np.nan, np.float32)
    rows = []
    for fold in range(5):
        fitting, validation = folds != fold, folds == fold
        fold_dir = run_dir / "catboost" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for target_index, target in enumerate(TARGETS):
            start = time.monotonic()
            labels = y[fitting, target_index]
            positives = int(labels.sum())
            if positives == 0 or positives == len(labels):
                raise ValueError("Training fold lacks both target classes")
            weight = float(min(8.0, np.sqrt((len(labels) - positives) / positives)))
            model_path = fold_dir / f"{target}.cbm"
            prediction_path = fold_dir / f"{target}_sampled_prediction.npy"
            reused = model_path.exists() and prediction_path.exists()
            if reused:
                p = np.load(prediction_path, allow_pickle=False)
            else:
                print(f"Training CatBoost fold {fold}, {target} ...", flush=True)
                model = CatBoostClassifier(**MODEL_PARAMS, random_seed=42 + fold)
                # No eval_set or early stopping: outer labels cannot select trees.
                model.fit(x[fitting], labels, sample_weight=np.where(labels == 1, weight, 1.0))
                p = model.predict_proba(x[validation])[:, 1].astype(np.float32)
                model.save_model(str(model_path))
                np.save(prediction_path, p, allow_pickle=False)
                pd.DataFrame({"feature": arrays["feature_names"],
                              "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(
                    fold_dir / f"{target}_feature_importance.csv", index=False)
                del model
                gc.collect()
            if len(p) != int(validation.sum()) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
                raise ValueError("Invalid CatBoost sampled predictions")
            prediction[validation, target_index] = p
            rows.append({"fold": fold, "target": target, "positive_weight": weight,
                         "training_rows": int(fitting.sum()), "validation_rows": int(validation.sum()),
                         "elapsed_seconds": time.monotonic() - start, "reused": reused})
            pd.DataFrame(rows).to_csv(run_dir / "catboost_training_log.csv", index=False)
            print(f"CatBoost fold {fold} / {target}: complete ({rows[-1]['elapsed_seconds']:.1f}s)", flush=True)
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete CatBoost OOF")
    np.save(run_dir / "catboost_sampled_oof.npy", prediction, allow_pickle=False)
    return prediction


def native_catboost(manifest: pd.DataFrame, arrays: dict, sampled: np.ndarray, run_dir: Path) -> np.ndarray:
    oof_dir = run_dir / "catboost" / "oof"
    oof_dir.mkdir(exist_ok=True)
    parts = []
    for record_index, row in enumerate(manifest.itertuples(index=False)):
        mask = arrays["record"] == record_index
        grid, sparse = arrays["grid"][mask], sampled[mask]
        n_rows = int(arrays["row_counts"][record_index])
        p = np.column_stack([np.interp(np.arange(n_rows), grid, sparse[:, i]) for i in range(3)]).astype(np.float32)
        np.savez_compressed(oof_dir / f"{row.recording_id}.npz", prediction=p)
        parts.append(p)
    return np.concatenate(parts)


# %% Fixed blending, exact native-row mAP, and experiment decision
def blended_prediction(name: str, predictions: dict) -> np.ndarray:
    if name in predictions:
        return predictions[name]
    weights = FIXED_BLENDS[name]
    if abs(sum(weights.values()) - 1) > 1e-12:
        raise ValueError("Blend weights do not sum to one")
    result = np.zeros_like(next(iter(predictions.values())))
    for method, weight in weights.items():
        result += predictions[method] * np.float32(weight)
    return result


def evaluate_methods(y: np.ndarray, folds: np.ndarray, predictions: dict, names: list[str], run_dir: Path) -> pd.DataFrame:
    metric_rows, fold_rows = [], []
    for name in names:
        p = blended_prediction(name, predictions)
        validate_prediction(y, p)
        metric = {"method": name, **base.score_targets(y, p)}
        metric_rows.append(metric)
        for fold in range(5):
            mask = folds == fold
            fold_rows.append({"method": name, "fold": fold, **base.score_targets(y[mask], p[mask])})
        print(f"{name}: native-row TDCS mAP {metric['mAP']:.6f}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "candidate_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "candidate_fold_metrics.csv", index=False)
    return metrics


def save_selected(manifest: pd.DataFrame, arrays: dict, predictions: dict, selected: str, run_dir: Path) -> Path:
    selected_dir = run_dir / "selected_candidate"
    prediction_dir = selected_dir / "oof"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(selected_dir / "frozen_tdcs_manifest.csv", index=False)
    p = blended_prediction(selected, predictions)
    offset = 0
    for index, row in enumerate(manifest.itertuples(index=False)):
        n_rows = int(arrays["row_counts"][index])
        np.savez_compressed(prediction_dir / f"{row.recording_id}.npz", prediction=p[offset:offset + n_rows])
        offset += n_rows
    if offset != len(p):
        raise ValueError("Selected candidate row alignment failed")
    return selected_dir


def make_figures(metrics: pd.DataFrame, run_dir: Path) -> list[Path]:
    labels = {
        "previous_tdcs_blend": "Previous neural blend",
        "multiscale_sensor_hgb": "Sensor HGB",
        "multiscale_catboost": "Sensor CatBoost",
        "neural_hgb_equal": "Neural + HGB (equal)",
        "hgb_catboost_equal": "HGB + CatBoost (equal)",
        "neural_sensor_three_way": "Neural + HGB + CatBoost",
    }
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    ordered = metrics.sort_values("mAP")
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    colors = ["#8b929c" if name == "previous_tdcs_blend" else "#2463a0" for name in ordered.method]
    bars = ax.barh([labels[name] for name in ordered.method], ordered.mAP, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pooled native-row TDCS OOF mAP")
    ax.set_title("Fixed TDCS candidates: 61 held-out subjects, 5 folds")
    ax.grid(axis="x", alpha=0.2)
    path = run_dir / "tdcs_candidate_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    selected = metrics.sort_values("mAP", ascending=False).iloc[0]
    previous = metrics.loc[metrics.method.eq("previous_tdcs_blend")].iloc[0]
    sensor = metrics.loc[metrics.method.eq("multiscale_sensor_hgb")].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    positions = np.arange(3)
    for i, (row, label, color) in enumerate([
        (previous, "Previous neural blend", "#8b929c"),
        (sensor, "Sensor HGB", "#a26916"),
        (selected, f"Selected: {labels[selected.method]}", "#2463a0"),
    ]):
        bars = ax.bar(positions + (i - 1) * 0.25, row.loc[list(TARGETS)].to_numpy(float), 0.25, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_xticks(positions, ["Start hesitation", "Turn", "Walking"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pooled native-row average precision")
    ax.set_title("TDCS class performance: previous and selected candidate")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    second = run_dir / "tdcs_candidate_classes.png"
    fig.savefig(second, dpi=150)
    plt.close(fig)
    return [path, second]


def run_sweep() -> dict:
    sensor_run, manifest, arrays, sensor_audit = load_inputs()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = {"catboost_parameters": MODEL_PARAMS, "positive_weight": "sqrt(negative/positive), cap 8; training labels only",
              "random_seeds": {fold: 42 + fold for fold in range(5)}, "fixed_blends": FIXED_BLENDS,
              "neural_weights": NEURAL_WEIGHTS, "sensor_cache_fingerprint": sensor_audit["fingerprint"],
              "source_sha256": source_hash, "catboost_version": catboost.__version__, "python": platform.python_version()}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    run_dir = OUTPUT_ROOT / fingerprint
    run_dir.mkdir(parents=True, exist_ok=True)
    # This record exists before reading native OOF labels or fitting the candidate.
    write_json(run_dir / "predeclared_configuration.json", config)
    manifest.to_csv(run_dir / "frozen_tdcs_manifest.csv", index=False)
    write_json(run_dir / "audit.json", {
        "domain": "tdcsfog", "recordings": len(manifest), "subjects": int(manifest.subject.nunique()),
        "native_rows": int(arrays["row_counts"].sum()), "feature_count": int(arrays["x"].shape[1]),
        "sampled_training_rows": len(arrays["x"]), "validation": "same five frozen subject-disjoint folds",
        "feature_inputs": "accelerometer and recording timing only; no identity or clinical inputs",
        "evaluation_grain": "every native row, pooled AP per target, arithmetic mean of 3 targets",
        "model_selection_caveat": "same development folds reused across experiments; selection is exploratory",
        "retained_defog_reported_oof_map": 0.817, "competition_submission_created": False,
        "catboost_docs": "https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier",
    })
    print(f"TDCS candidate artifacts: {run_dir}", flush=True)
    y, folds, predictions = load_native_inputs(sensor_run, manifest)
    preliminary = evaluate_methods(y, folds, predictions,
                                   [*predictions, "neural_hgb_equal"], run_dir)
    if abs(float(preliminary.iloc[0].mAP) - 0.35118769305798225) > 1e-8:
        raise ValueError("Neural baseline did not reproduce")
    if abs(float(preliminary.iloc[1].mAP) - 0.36421111222258024) > 1e-8:
        raise ValueError("Sensor baseline did not reproduce")
    sampled = fit_catboost(arrays, run_dir)
    predictions["multiscale_catboost"] = native_catboost(manifest, arrays, sampled, run_dir)
    metrics = evaluate_methods(y, folds, predictions, [*predictions, *FIXED_BLENDS], run_dir)
    best = metrics.sort_values("mAP", ascending=False).iloc[0]
    selected_dir = save_selected(manifest, arrays, predictions, str(best.method), run_dir)
    pd.DataFrame([{"domain": "tdcsfog", **best.to_dict()}]).to_csv(selected_dir / "native_oof_metrics.csv", index=False)
    figures = make_figures(metrics, run_dir)
    decision = {
        "run_dir": str(run_dir), "selected_candidate_dir": str(selected_dir),
        "selected_method": str(best.method), "candidate_tdcs_oof_map": float(best.mAP),
        "previous_neural_tdcs_oof_map": float(preliminary.iloc[0].mAP),
        "previous_sensor_tdcs_oof_map": float(preliminary.iloc[1].mAP),
        "delta_vs_neural": float(best.mAP - preliminary.iloc[0].mAP),
        "delta_vs_sensor": float(best.mAP - preliminary.iloc[1].mAP),
        "selection_scope": "best of six fixed candidates on reused development OOF; fresh validation needed",
        "retained_defog_reported_oof_map": 0.817, "competition_submission_created": False,
    }
    write_json(run_dir / "decision.json", decision)
    write_json(OUTPUT_ROOT / "latest_run.json", decision)
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    return {"run_dir": run_dir, "metrics": metrics, "decision": decision, "figures": figures}


if __name__ == "__main__":
    run_sweep()
