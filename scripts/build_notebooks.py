"""Regenerate the three canonical leakage-safe modelling notebooks."""

from pathlib import Path
from textwrap import dedent
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(text: str):
    source = dedent(text).strip()
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(f"markdown:{source}".encode("utf-8")).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(text: str):
    source = dedent(text).strip()
    return {
        "cell_type": "code",
        "id": hashlib.sha1(f"code:{source}".encode("utf-8")).hexdigest()[:12],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def validate_notebook(notebook: dict) -> None:
    if notebook.get("nbformat") != 4:
        raise ValueError("Notebook must use nbformat 4")
    if not isinstance(notebook.get("cells"), list) or not notebook["cells"]:
        raise ValueError("Notebook must contain cells")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") not in {"markdown", "code"}:
            raise ValueError(f"Unsupported cell type at index {index}")
        if not isinstance(cell.get("source"), list):
            raise ValueError(f"Cell {index} source must be a list of strings")
        if cell["cell_type"] == "code" and "outputs" not in cell:
            raise ValueError(f"Code cell {index} must contain outputs")


def save_notebook(
    name: str,
    cells: list,
    *,
    kernel_name: str = "dissertation-fog",
    kernel_display_name: str = "Dissertation FoG (.venv)",
) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": kernel_display_name,
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    validate_notebook(notebook)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    with (NOTEBOOK_DIR / name).open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=1)
        handle.write("\n")


COMMON_IMPORTS = """
from pathlib import Path
import sys
import gc
import importlib
import numpy as np

print(f"Python executable: {sys.executable}")
print(f"NumPy version: {np.__version__}")
if int(np.__version__.split(".")[0]) >= 2:
    raise RuntimeError(
        "This notebook requires NumPy 1.26.x. Select the "
        "'Dissertation FoG (.venv)' kernel and restart the notebook."
    )

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
from IPython.display import display

get_ipython().run_line_magic("matplotlib", "inline")

PROJECT_ROOT = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "src" / "fog_pipeline.py").exists()
)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

importlib.invalidate_caches()
import fog_pipeline as _fog_pipeline
_fog_pipeline = importlib.reload(_fog_pipeline)

from fog_pipeline import (
    ANY_FOG_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    align_binary_predictions,
    assert_subject_disjoint,
    evaluate_episode_predictions,
    load_recordings,
    prepare_tabular_splits,
    select_episode_threshold,
    split_by_subject,
    split_summary,
)

RANDOM_STATE = 42
WINDOW_SIZE = 5
EDA_SAMPLE_SIZE = 200_000
EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
BATCH_SIZE = 4096
RUN_TRAINING = True
EPISODE_IOU_THRESHOLD = 0.25
# No temporal smoothing is imposed without validation evidence.
MERGE_GAP_SECONDS = 0.0
MIN_PREDICTED_EPISODE_SECONDS = 0.0
# Set a small integer for a quick smoke test, or None for the full dataset.
MAX_RECORDINGS = None
# Bounded smoke tests may not contain every rare label; full runs must.
REQUIRE_ALL_TARGETS = MAX_RECORDINGS is None

sns.set_theme(style="whitegrid", context="notebook")
"""


EDA_OVERVIEW = """
overview = pd.Series(
    {
        "rows": len(raw_data),
        "recordings": raw_data["RecordingId"].nunique(),
        "subjects_or_file_groups": raw_data["Subject"].nunique(),
        "time_min": raw_data["Time"].min(),
        "time_max": raw_data["Time"].max(),
    },
    name="dataset_overview",
)
display(overview.to_frame())

missing_summary = raw_data[
    [
        "Time",
        *FEATURE_COLUMNS[:3],
        ANY_FOG_COLUMN,
        *TARGET_COLUMNS,
        "RecordingId",
        "Subject",
    ]
].isna().sum().rename("missing_values")
display(missing_summary.to_frame())

numeric_summary = raw_data[
    ["Time", *FEATURE_COLUMNS[:3], ANY_FOG_COLUMN, *TARGET_COLUMNS]
].describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]).T
display(numeric_summary)

recording_lengths = raw_data.groupby("RecordingId", observed=True).size()
display(
    recording_lengths.describe().rename("rows_per_recording").to_frame()
)

event_columns = [ANY_FOG_COLUMN, *TARGET_COLUMNS]
event_summary = pd.DataFrame(
    {
        "positive_rows": raw_data.loc[:, event_columns].sum(),
        "prevalence_percent": raw_data.loc[:, event_columns].mean() * 100,
    }
)
display(event_summary)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
event_summary["positive_rows"].plot.bar(ax=axes[0], color="#4c78a8")
axes[0].set_yscale("log")
axes[0].set_title("Positive FoG-labelled rows (log scale)")
axes[0].set_ylabel("Rows")
event_summary["prevalence_percent"].plot.bar(ax=axes[1], color="#f58518")
axes[1].set_title("Any-FoG and event-type prevalence")
axes[1].set_ylabel("Percent of rows")
for axis in axes:
    axis.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
"""


EDA_TIME_SERIES = """
recording_event_totals = raw_data.groupby(
    "RecordingId", observed=True
)[list(TARGET_COLUMNS)].sum()
sample_recording_id = recording_event_totals.sum(axis=1).idxmax()
sample_recording = raw_data.loc[
    raw_data["RecordingId"] == sample_recording_id
].sort_values("Time", kind="stable")

event_positions = np.flatnonzero(
    sample_recording.loc[:, TARGET_COLUMNS].any(axis=1).to_numpy()
)
centre = int(event_positions[0]) if len(event_positions) else len(sample_recording) // 2
window_start = max(0, centre - 750)
window_end = min(len(sample_recording), centre + 1250)
plot_window = sample_recording.iloc[window_start:window_end]

sensor_event_pairs = [
    ("AccV", "StartHesitation", "#e45756"),
    ("AccML", "Turn", "#72b7b2"),
    ("AccAP", "Walking", "#54a24b"),
]
fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
for axis, (sensor, target, colour) in zip(axes, sensor_event_pairs):
    sensor_values = plot_window[sensor].copy()
    positive_time_steps = plot_window["Time"].diff().dropna()
    positive_time_steps = positive_time_steps.loc[positive_time_steps > 0]
    typical_step = float(positive_time_steps.median()) if len(positive_time_steps) else 1.0
    sensor_values.loc[plot_window["Time"].diff() > 5 * typical_step] = np.nan
    axis.plot(plot_window["Time"], sensor_values, linewidth=0.9, color="#4c78a8")
    axis.fill_between(
        plot_window["Time"],
        0,
        1,
        where=plot_window[ANY_FOG_COLUMN].astype(bool),
        transform=axis.get_xaxis_transform(),
        step="post",
        color="#e45756",
        alpha=0.12,
        label="Any FoG",
    )
    events = plot_window.loc[plot_window[target] == 1]
    axis.scatter(events["Time"], events[sensor], s=18, color=colour, label=target, zorder=3)
    axis.set_ylabel(sensor)
    axis.legend(loc="upper right")
axes[-1].set_xlabel("Recording time index")
fig.suptitle(f"Representative event window: recording {sample_recording_id}")
plt.tight_layout()
plt.show()
"""


EDA_RELATIONSHIPS = """
eda_sample = raw_data.sample(
    n=min(EDA_SAMPLE_SIZE, len(raw_data)),
    random_state=RANDOM_STATE,
)
correlation_columns = [*FEATURE_COLUMNS[:3], ANY_FOG_COLUMN, *TARGET_COLUMNS]
correlation_matrix = eda_sample.loc[:, correlation_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    square=True,
)
plt.title(f"Sensor/event correlations (sample n={len(eda_sample):,})")
plt.tight_layout()
plt.show()

sensor_means = []
for target in (ANY_FOG_COLUMN, *TARGET_COLUMNS):
    grouped_means = eda_sample.groupby(target, observed=True)[list(FEATURE_COLUMNS[:3])].mean()
    grouped_means["target"] = target
    grouped_means["state"] = grouped_means.index.astype(int)
    sensor_means.append(grouped_means.reset_index(drop=True))
sensor_means = pd.concat(sensor_means, ignore_index=True)
display(sensor_means.set_index(["target", "state"]))

cooccurrence = raw_data.loc[:, TARGET_COLUMNS].value_counts().rename("rows").reset_index()
display(cooccurrence.head(10))
"""


SPLIT_DIAGNOSTICS = """
split_table = split_summary(subject_splits)
display(split_table)

rate_targets = [ANY_FOG_COLUMN, *TARGET_COLUMNS]
rate_columns = [f"{target}_rate" for target in rate_targets]
rate_plot = split_table.loc[:, rate_columns].rename(
    columns={f"{target}_rate": target for target in rate_targets}
) * 100
rate_plot.plot.bar(figsize=(11, 5))
plt.title("FoG prevalence by subject-disjoint split")
plt.ylabel("Percent of rows")
plt.xlabel("Split")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""


DENSE_MODEL = """
model = None
history = None

if RUN_TRAINING:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    fog_train = prepared.y_any_fog_train
    fog_validation = prepared.y_any_fog_validation
    fog_negatives = float((1.0 - fog_train).sum())
    fog_positives = float(fog_train.sum())
    fog_positive_weight = float(
        np.clip(fog_negatives / max(fog_positives, 1.0), 1.0, 200.0)
    )
    type_negatives = (1.0 - prepared.y_train).sum(axis=0)
    type_positives = prepared.y_train.sum(axis=0)
    type_positive_weights = np.clip(
        type_negatives / np.maximum(type_positives, 1.0), 1.0, 200.0
    )
    print(f"Any-FoG positive-class loss weight: {fog_positive_weight:.2f}")
    print(
        "Auxiliary event-type weights:",
        dict(zip(TARGET_COLUMNS, type_positive_weights.round(2))),
    )

    def make_weighted_binary_crossentropy(positive_weights):
        weights = tf.constant(positive_weights, dtype=tf.float32)

        def weighted_binary_crossentropy(y_true, y_pred):
            y_pred = tf.clip_by_value(
                y_pred,
                tf.keras.backend.epsilon(),
                1.0 - tf.keras.backend.epsilon(),
            )
            loss = -(
                weights * y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
            )
            return tf.reduce_mean(loss)

        return weighted_binary_crossentropy

    inputs = tf.keras.layers.Input(shape=(len(FEATURE_COLUMNS),), name="sensor_features")
    shared = tf.keras.layers.Dense(128, activation="relu")(inputs)
    shared = tf.keras.layers.Dense(128, activation="relu")(shared)
    shared = tf.keras.layers.Dense(64, activation="relu")(shared)
    fog_output = tf.keras.layers.Dense(1, activation="sigmoid", name="fog")(shared)
    event_type_output = tf.keras.layers.Dense(
        len(TARGET_COLUMNS), activation="sigmoid", name="event_type"
    )(shared)
    model = tf.keras.Model(
        inputs=inputs,
        outputs={"fog": fog_output, "event_type": event_type_output},
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "fog": make_weighted_binary_crossentropy(fog_positive_weight),
            "event_type": make_weighted_binary_crossentropy(type_positive_weights),
        },
        loss_weights={"fog": 1.0, "event_type": 0.2},
        metrics={
            "fog": [
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            ],
            "event_type": [
                tf.keras.metrics.AUC(
                    curve="PR",
                    multi_label=True,
                    num_labels=len(TARGET_COLUMNS),
                    name="pr_auc",
                )
            ],
        },
    )
    history = model.fit(
        prepared.X_train,
        {"fog": fog_train, "event_type": prepared.y_train},
        validation_data=(
            prepared.X_validation,
            {"fog": fog_validation, "event_type": prepared.y_validation},
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_fog_pr_auc",
                mode="max",
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
else:
    print("Training is disabled. Set RUN_TRAINING=True after reviewing the split summary.")
"""


LEARNING_CURVES = """
if history is not None:
    history_frame = pd.DataFrame(history.history)
    display(history_frame)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].plot(history_frame["fog_loss"], label="train")
    axes[0].plot(history_frame["val_fog_loss"], label="validation")
    axes[0].set_title("Primary any-FoG weighted loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_frame["fog_pr_auc"], label="train")
    axes[1].plot(history_frame["val_fog_pr_auc"], label="validation")
    axes[1].set_title("Any-FoG precision-recall AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    axes[2].plot(history_frame["fog_binary_accuracy"], label="train")
    axes[2].plot(
        history_frame["val_fog_binary_accuracy"], label="validation"
    )
    axes[2].set_title("Any-FoG binary accuracy (secondary)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    plt.tight_layout()
    plt.show()
else:
    print("Learning curves are unavailable because training is disabled.")
"""


FINAL_EVALUATION = """
if model is not None:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
    )

    def unpack_model_outputs(outputs):
        if isinstance(outputs, dict):
            return outputs["fog"], outputs["event_type"]
        output_by_name = dict(zip(model.output_names, outputs))
        return output_by_name["fog"], output_by_name["event_type"]

    validation_outputs = model.predict(
        prepared.X_validation,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    validation_fog_probabilities, validation_type_probabilities = unpack_model_outputs(
        validation_outputs
    )
    decision_threshold, threshold_search = select_episode_threshold(
        prepared.alignment_splits.validation,
        prepared.y_any_fog_validation,
        validation_fog_probabilities,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        thresholds=np.linspace(0.05, 0.95, 19),
        minimum_iou=EPISODE_IOU_THRESHOLD,
        merge_gap_seconds=MERGE_GAP_SECONDS,
        minimum_predicted_duration_seconds=MIN_PREDICTED_EPISODE_SECONDS,
    )
    print(f"Validation-selected any-FoG threshold: {decision_threshold:.2f}")
    display(threshold_search)

    plt.figure(figsize=(10, 5))
    for metric, colour in (
        ("episode_precision", "#4c78a8"),
        ("episode_recall", "#f2a900"),
        ("episode_f1", "#e45756"),
    ):
        plt.plot(
            threshold_search["threshold"],
            threshold_search[metric],
            label=metric.replace("episode_", "").title(),
            color=colour,
        )
    plt.axvline(decision_threshold, color="#333333", linestyle="--", label="Selected")
    plt.title(f"Validation episode metrics at IoU ≥ {EPISODE_IOU_THRESHOLD:.2f}")
    plt.xlabel("Any-FoG probability threshold")
    plt.ylabel("Episode metric")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Freeze the model and validation-selected threshold before examining test.
    split_evaluations = []
    split_probabilities = {}
    for split_name, X_values, y_fog_values, y_type_values in (
        ("train", prepared.X_train, prepared.y_any_fog_train, prepared.y_train),
        (
            "validation",
            prepared.X_validation,
            prepared.y_any_fog_validation,
            prepared.y_validation,
        ),
        ("test", prepared.X_test, prepared.y_any_fog_test, prepared.y_test),
    ):
        evaluation = model.evaluate(
            X_values,
            {"fog": y_fog_values, "event_type": y_type_values},
            batch_size=BATCH_SIZE,
            return_dict=True,
            verbose=0,
        )
        outputs = model.predict(X_values, batch_size=BATCH_SIZE, verbose=0)
        fog_probabilities, type_probabilities = unpack_model_outputs(outputs)
        split_probabilities[split_name] = (fog_probabilities, type_probabilities)
        fog_true = y_fog_values.reshape(-1)
        fog_score = fog_probabilities.reshape(-1)
        fog_predicted = (fog_score >= decision_threshold).astype(np.int8)
        split_evaluations.append(
            {
                "split": split_name,
                "fog_loss": evaluation["fog_loss"],
                "any_fog_prevalence": fog_true.mean(),
                "average_precision": average_precision_score(fog_true, fog_score),
                "balanced_accuracy": balanced_accuracy_score(
                    fog_true, fog_predicted
                ),
                "precision": precision_score(
                    fog_true, fog_predicted, zero_division=0
                ),
                "recall": recall_score(fog_true, fog_predicted, zero_division=0),
                "f1": f1_score(fog_true, fog_predicted, zero_division=0),
            }
        )

    split_evaluation_frame = pd.DataFrame(split_evaluations).set_index("split")
    display(split_evaluation_frame)
    split_colours = ["#4c78a8", "#f2a900", "#e45756"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    split_evaluation_frame["fog_loss"].plot.bar(ax=axes[0], color=split_colours)
    axes[0].set_title("Frozen any-FoG weighted loss")
    axes[0].set_ylabel("Loss")
    axes[0].tick_params(axis="x", rotation=0)
    for axis, metric, title in (
        (axes[1], "average_precision", "Any-FoG average precision"),
        (axes[2], "balanced_accuracy", "Balanced accuracy (secondary)"),
        (axes[3], "f1", "Timestep FoG F1"),
    ):
        split_evaluation_frame[metric].plot.bar(ax=axis, color=split_colours)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()

    test_fog_probabilities, test_type_probabilities = split_probabilities["test"]
    test_fog_true = prepared.y_any_fog_test.reshape(-1)
    test_fog_score = test_fog_probabilities.reshape(-1)
    test_fog_predictions = (test_fog_score >= decision_threshold).astype(np.int8)
    point_metrics = pd.Series(
        {
            "threshold_selected_on_validation": decision_threshold,
            "positive_prevalence": test_fog_true.mean(),
            "average_precision": average_precision_score(
                test_fog_true, test_fog_score
            ),
            "balanced_accuracy": balanced_accuracy_score(
                test_fog_true, test_fog_predictions
            ),
            "precision": precision_score(
                test_fog_true, test_fog_predictions, zero_division=0
            ),
            "recall": recall_score(
                test_fog_true, test_fog_predictions, zero_division=0
            ),
            "f1": f1_score(test_fog_true, test_fog_predictions, zero_division=0),
        },
        name="held_out_test",
    )
    display(point_metrics.to_frame())

    episode_evaluation = evaluate_episode_predictions(
        prepared.alignment_splits.test,
        prepared.y_any_fog_test,
        test_fog_probabilities,
        threshold=decision_threshold,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        minimum_iou=EPISODE_IOU_THRESHOLD,
        merge_gap_seconds=MERGE_GAP_SECONDS,
        minimum_predicted_duration_seconds=MIN_PREDICTED_EPISODE_SECONDS,
    )
    episode_metrics = pd.Series(
        {
            "true_episodes": episode_evaluation.true_count,
            "predicted_episodes": episode_evaluation.predicted_count,
            "matched_episodes": episode_evaluation.true_positive_count,
            "false_positive_episodes": episode_evaluation.false_positive_count,
            "missed_episodes": episode_evaluation.false_negative_count,
            "episode_precision": episode_evaluation.precision,
            "episode_recall": episode_evaluation.recall,
            "episode_f1": episode_evaluation.f1,
            "false_alarms_per_minute": episode_evaluation.false_alarms_per_minute,
            "mean_onset_delay_seconds": episode_evaluation.mean_onset_delay_seconds,
            "median_onset_delay_seconds": episode_evaluation.median_onset_delay_seconds,
            "mean_absolute_onset_error_seconds": episode_evaluation.mean_absolute_onset_error_seconds,
            "mean_matched_duration_iou": episode_evaluation.mean_duration_iou,
            "evaluated_minutes": episode_evaluation.evaluation_minutes,
        },
        name="held_out_test",
    )
    display(episode_metrics.to_frame())

    episode_bar_values = episode_metrics.loc[
        [
            "episode_precision",
            "episode_recall",
            "episode_f1",
            "mean_matched_duration_iou",
        ]
    ]
    ax = episode_bar_values.plot.bar(
        figsize=(10, 5),
        color=["#4c78a8", "#f2a900", "#e45756", "#72b7b2"],
    )
    ax.set_title(f"Held-out episode detection at IoU ≥ {EPISODE_IOU_THRESHOLD:.2f}")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.show()

    matrix = confusion_matrix(
        test_fog_true,
        test_fog_predictions,
        labels=[0, 1],
    )
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Held-out timestep confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0.5, 1.5], ["No FoG", "FoG"])
    plt.yticks([0.5, 1.5], ["No FoG", "FoG"], rotation=0)
    plt.tight_layout()
    plt.show()

    # Event type remains an auxiliary description, not the primary objective.
    type_thresholds = []
    type_metric_rows = []
    for target_index, target in enumerate(TARGET_COLUMNS):
        validation_precision, validation_recall, candidate_thresholds = precision_recall_curve(
            prepared.y_validation[:, target_index],
            validation_type_probabilities[:, target_index],
        )
        validation_f1 = (
            2
            * validation_precision[:-1]
            * validation_recall[:-1]
            / np.maximum(
                validation_precision[:-1] + validation_recall[:-1], 1e-12
            )
        )
        type_threshold = float(
            candidate_thresholds[int(np.nanargmax(validation_f1))]
        )
        type_thresholds.append(type_threshold)
        y_true = prepared.y_test[:, target_index]
        y_score = test_type_probabilities[:, target_index]
        y_pred = (y_score >= type_threshold).astype(np.int8)
        type_metric_rows.append(
            {
                "target": target,
                "threshold": type_threshold,
                "positive_prevalence": y_true.mean(),
                "average_precision": average_precision_score(y_true, y_score),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    display(pd.DataFrame(type_metric_rows).set_index("target"))

    # Fixed-rule example: first held-out recording containing an annotated FoG row.
    aligned_test = align_binary_predictions(
        prepared.alignment_splits.test,
        prepared.y_any_fog_test,
        test_fog_probabilities,
        threshold=decision_threshold,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    recording_totals = aligned_test.groupby(
        "RecordingId", observed=True
    )["AnyFoGTrue"].sum()
    positive_recordings = sorted(
        recording_totals.loc[recording_totals > 0].index,
        key=str,
    )
    if positive_recordings:
        example_recording_id = positive_recordings[0]
        example = aligned_test.loc[
            aligned_test["RecordingId"] == example_recording_id
        ].sort_values("TimeSeconds", kind="stable")
        first_positive_time = float(
            example.loc[example["AnyFoGTrue"] == 1, "TimeSeconds"].iloc[0]
        )
        example_window = example.loc[
            example["TimeSeconds"].between(
                max(0.0, first_positive_time - 5.0),
                first_positive_time + 10.0,
            )
        ]

        fig, axis = plt.subplots(figsize=(14, 5))
        axis.plot(
            example_window["TimeSeconds"],
            example_window["AnyFoGScore"],
            color="#4c78a8",
            label="Predicted P(FoG now)",
        )
        axis.step(
            example_window["TimeSeconds"],
            example_window["AnyFoGTrue"],
            where="post",
            color="#e45756",
            alpha=0.85,
            label="Annotated FoG",
        )
        axis.axhline(
            decision_threshold,
            color="#333333",
            linestyle="--",
            label="Validation threshold",
        )
        axis.set_title(
            f"Held-out FoG timing example: recording {example_recording_id}"
        )
        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Probability / label")
        axis.set_ylim(-0.05, 1.05)
        axis.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
    else:
        print("No positive held-out recording was available for a timing example.")
else:
    print("No final test evaluation was run because training is disabled.")
"""


def build_base_notebook() -> None:
    cells = [
        markdown(
            """
            # Subject-Generalising DeFOG Episode Detector

            ## Goal

            Predict whether freezing of gait is occurring at each timestamp for
            subjects that never appear in training. Event type is retained as a
            low-weight auxiliary output, while episode timing is the primary result.

            ### Key assumptions

            - `defog_metadata.csv` is the authoritative recording-to-subject mapping.
            - The intended deployment is inference on a previously unseen subject.
            - Ground-truth event labels are unavailable at prediction time, so no
              target or target-derived rolling count is used as a feature.
            """
        ),
        markdown("## Setup"),
        code(COMMON_IMPORTS),
        markdown("## Data — load identities before concatenation"),
        code(
            """
            raw_data = load_recordings(
                recordings_dir=PROJECT_ROOT / "data" / "raw" / "train" / "defog",
                metadata_path=PROJECT_ROOT / "data" / "metadata" / "defog_metadata.csv",
                limit_recordings=MAX_RECORDINGS,
                filter_valid_task=True,
            )
            SAMPLING_RATE_HZ = 100.0
            raw_data.head()
            """
        ),
        markdown("### Dataset overview and class balance"),
        code(EDA_OVERVIEW),
        markdown("### Representative sensor and event window"),
        code(EDA_TIME_SERIES),
        markdown("### Correlations, event-conditioned means, and label combinations"),
        code(EDA_RELATIONSHIPS),
        markdown("## Methods — subject-disjoint train, validation, and test splits"),
        code(
            """

            subject_splits = split_by_subject(
                raw_data,
                test_size=0.20,
                validation_size=0.20,
                random_state=RANDOM_STATE,
                require_all_targets=REQUIRE_ALL_TARGETS,
            )
            print("Selected split seed:", subject_splits.random_state)
            split_summary(subject_splits)
            """
        ),
        markdown("## Checks — prove subjects and labels cannot leak"),
        code(
            """
            assert_subject_disjoint(subject_splits)
            assert set(TARGET_COLUMNS).isdisjoint(FEATURE_COLUMNS)
            assert ANY_FOG_COLUMN not in FEATURE_COLUMNS

            print("Subjects are disjoint across train, validation, and test.")
            print("Model features:", list(FEATURE_COLUMNS))
            print("Primary target:", ANY_FOG_COLUMN)
            print("Auxiliary event types:", list(TARGET_COLUMNS))
            """
        ),
        code(SPLIT_DIAGNOSTICS),
        markdown("## Features — fit imputation and scaling on training subjects only"),
        code(
            """
            del raw_data
            gc.collect()
            prepared = prepare_tabular_splits(
                subject_splits,
                window_size=WINDOW_SIZE,
            )
            del subject_splits
            gc.collect()

            pd.DataFrame(
                {
                    "split": ["train", "validation", "test"],
                    "samples": [
                        len(prepared.X_train),
                        len(prepared.X_validation),
                        len(prepared.X_test),
                    ],
                    "features": [
                        prepared.X_train.shape[1],
                        prepared.X_validation.shape[1],
                        prepared.X_test.shape[1],
                    ],
                }
            )
            """
        ),
        markdown("## Model — use validation subjects for training decisions"),
        code(DENSE_MODEL),
        markdown("### Learning curves"),
        code(LEARNING_CURVES),
        markdown("## Final test — detect complete FoG episodes once"),
        code(FINAL_EVALUATION),
        markdown(
            """
            ## Takeaways

            - Rows from a subject can occur in only one split.
            - Rolling features reset at recording boundaries.
            - Inputs contain accelerometer-derived values only.
            - The imputer and scaler learn from training subjects only.
            - The test subjects are not used for early stopping or model selection.
            - The validation split selects the FoG probability threshold; test
              episodes are then scored by one-to-one temporal overlap.
            """
        ),
    ]
    save_notebook("defog_dense_subject_generalisation.ipynb", cells)


def build_tdcs_notebook() -> None:
    cells = [
        markdown(
            """
            # Subject-Generalising TDCS FoG Episode Detector

            ## Goal

            Predict whether freezing of gait is occurring at each TDCS timestamp
            and reconstruct complete episodes on genuinely unseen subjects.

            ### Identity mapping

            `tdcsfog_metadata.csv` is the authoritative mapping from each recording
            ID to its subject. All recordings belonging to one person stay in the
            same train, validation, or test split.
            """
        ),
        markdown("## Setup"),
        code(COMMON_IMPORTS),
        markdown("## Data — map every recording to its subject"),
        code(
            """
            raw_data = load_recordings(
                recordings_dir=PROJECT_ROOT / "data" / "raw" / "train" / "tdcsfog",
                metadata_path=PROJECT_ROOT / "data" / "metadata" / "tdcsfog_metadata.csv",
                limit_recordings=MAX_RECORDINGS,
                filter_valid_task=False,
            )
            SAMPLING_RATE_HZ = 128.0
            raw_data.head()
            """
        ),
        markdown("### Dataset overview and class balance"),
        code(EDA_OVERVIEW),
        markdown("### Representative sensor and event window"),
        code(EDA_TIME_SERIES),
        markdown("### Correlations, event-conditioned means, and label combinations"),
        code(EDA_RELATIONSHIPS),
        markdown("## Methods — subject-disjoint train, validation, and test splits"),
        code(
            """

            subject_splits = split_by_subject(
                raw_data,
                test_size=0.20,
                validation_size=0.20,
                random_state=RANDOM_STATE,
                require_all_targets=REQUIRE_ALL_TARGETS,
            )
            print("Selected split seed:", subject_splits.random_state)
            split_summary(subject_splits)
            """
        ),
        markdown("## Checks — prove subjects, recordings, and labels cannot leak"),
        code(
            """
            assert_subject_disjoint(subject_splits)
            assert set(TARGET_COLUMNS).isdisjoint(FEATURE_COLUMNS)
            assert ANY_FOG_COLUMN not in FEATURE_COLUMNS
            print("Real TDCS subjects are disjoint across all splits.")
            print("Model features:", list(FEATURE_COLUMNS))
            """
        ),
        code(SPLIT_DIAGNOSTICS),
        markdown("## Features — fit imputation and scaling on training subjects only"),
        code(
            """
            del raw_data
            gc.collect()
            prepared = prepare_tabular_splits(
                subject_splits,
                window_size=WINDOW_SIZE,
            )
            del subject_splits
            gc.collect()

            pd.DataFrame(
                {
                    "split": ["train", "validation", "test"],
                    "samples": [
                        len(prepared.X_train),
                        len(prepared.X_validation),
                        len(prepared.X_test),
                    ],
                }
            )
            """
        ),
        markdown("## Model — train one general model"),
        code(DENSE_MODEL),
        markdown("### Learning curves"),
        code(LEARNING_CURVES),
        markdown("## Final test — detect episodes on unseen subjects"),
        code(FINAL_EVALUATION),
        markdown(
            """
            ## Takeaways

            This evaluates generalisation to people absent from model development.
            Episode F1, false alarms, onset timing, and duration overlap are primary;
            event type remains auxiliary.
            """
        ),
    ]
    save_notebook("tdcs_dense_file_generalisation.ipynb", cells)


def build_lstm_notebook() -> None:
    cells = [
        markdown(
            """
            # Subject-Generalising DeFOG LSTM Episode Detector

            ## Goal

            Predict whether FoG is occurring at the final timestamp of each genuine
            chronological window. Subject splitting occurs first, windows cannot
            cross recordings or missing-time gaps, and validation/test inference
            uses stride one for precise episode timing.
            """
        ),
        markdown("## Setup"),
        code(
            """
            from pathlib import Path
            import sys
            import gc
            import importlib
            import numpy as np

            print(f"Python executable: {sys.executable}")
            print(f"NumPy version: {np.__version__}")
            if int(np.__version__.split(".")[0]) >= 2:
                raise RuntimeError(
                    "This notebook requires NumPy 1.26.x. Select the "
                    "'Dissertation FoG (.venv)' kernel and restart the notebook."
                )

            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from IPython import get_ipython
            from IPython.display import display

            get_ipython().run_line_magic("matplotlib", "inline")

            PROJECT_ROOT = next(
                candidate
                for candidate in (Path.cwd(), *Path.cwd().parents)
                if (candidate / "src" / "fog_pipeline.py").exists()
            )
            SRC_DIR = PROJECT_ROOT / "src"
            if str(SRC_DIR) not in sys.path:
                sys.path.insert(0, str(SRC_DIR))

            importlib.invalidate_caches()
            import fog_pipeline as _fog_pipeline
            _fog_pipeline = importlib.reload(_fog_pipeline)

            from fog_pipeline import (
                ANY_FOG_COLUMN,
                FEATURE_COLUMNS,
                TARGET_COLUMNS,
                align_binary_predictions,
                assert_subject_disjoint,
                evaluate_episode_predictions,
                load_recordings,
                prepare_sequence_splits,
                select_episode_threshold,
                split_by_subject,
                split_summary,
            )

            RANDOM_STATE = 42
            TIMESTEPS = 5
            STRIDE = 5
            EVALUATION_STRIDE = 1
            FEATURE_WINDOW_SIZE = 5
            EDA_SAMPLE_SIZE = 200_000
            EPOCHS = 40
            EARLY_STOPPING_PATIENCE = 8
            BATCH_SIZE = 4096
            RUN_TRAINING = True
            SAMPLING_RATE_HZ = 100.0
            EPISODE_IOU_THRESHOLD = 0.25
            MERGE_GAP_SECONDS = 0.0
            MIN_PREDICTED_EPISODE_SECONDS = 0.0
            MAX_RECORDINGS = None
            REQUIRE_ALL_TARGETS = MAX_RECORDINGS is None

            sns.set_theme(style="whitegrid", context="notebook")
            """
        ),
        markdown("## Data — split complete subjects before making windows"),
        code(
            """
            raw_data = load_recordings(
                recordings_dir=PROJECT_ROOT / "data" / "raw" / "train" / "defog",
                metadata_path=PROJECT_ROOT / "data" / "metadata" / "defog_metadata.csv",
                limit_recordings=MAX_RECORDINGS,
                filter_valid_task=True,
            )
            SAMPLING_RATE_HZ = 100.0
            raw_data.head()
            """
        ),
        markdown("### Dataset overview and class balance"),
        code(EDA_OVERVIEW),
        markdown("### Representative sensor and event window"),
        code(EDA_TIME_SERIES),
        markdown("## Methods — subject-disjoint sequence splits"),
        code(
            """
            subject_splits = split_by_subject(
                raw_data,
                test_size=0.20,
                validation_size=0.20,
                random_state=RANDOM_STATE,
                require_all_targets=REQUIRE_ALL_TARGETS,
            )
            assert_subject_disjoint(subject_splits)
            assert set(TARGET_COLUMNS).isdisjoint(FEATURE_COLUMNS)
            assert ANY_FOG_COLUMN not in FEATURE_COLUMNS
            print("Selected split seed:", subject_splits.random_state)
            split_summary(subject_splits)
            """
        ),
        code(SPLIT_DIAGNOSTICS),
        markdown("## Sequences — preserve time and recording boundaries"),
        code(
            """
            del raw_data
            gc.collect()
            prepared = prepare_sequence_splits(
                subject_splits,
                timesteps=TIMESTEPS,
                stride=STRIDE,
                evaluation_stride=EVALUATION_STRIDE,
                feature_window_size=FEATURE_WINDOW_SIZE,
            )
            del subject_splits
            gc.collect()

            sequence_summary = pd.DataFrame(
                {
                    "split": ["train", "validation", "test"],
                    "sequences": [
                        len(prepared.X_train),
                        len(prepared.X_validation),
                        len(prepared.X_test),
                    ],
                    "timesteps": TIMESTEPS,
                    "prediction_stride": [
                        STRIDE,
                        EVALUATION_STRIDE,
                        EVALUATION_STRIDE,
                    ],
                    "features": len(FEATURE_COLUMNS),
                }
            )
            display(sequence_summary)

            sequence_event_rates = pd.DataFrame(
                {
                    "train": np.concatenate(
                        [prepared.y_any_fog_train, prepared.y_train], axis=1
                    ).mean(axis=0),
                    "validation": np.concatenate(
                        [prepared.y_any_fog_validation, prepared.y_validation], axis=1
                    ).mean(axis=0),
                    "test": np.concatenate(
                        [prepared.y_any_fog_test, prepared.y_test], axis=1
                    ).mean(axis=0),
                },
                index=[ANY_FOG_COLUMN, *TARGET_COLUMNS],
            ) * 100
            display(sequence_event_rates.rename_axis("target"))
            sequence_event_rates.T.plot.bar(figsize=(11, 5))
            plt.title("Sequence-label prevalence by split")
            plt.ylabel("Percent of sequences")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown("## Model — tune only with validation subjects"),
        code(
            """
            model = None
            history = None

            if RUN_TRAINING:
                import tensorflow as tf

                tf.keras.utils.set_random_seed(RANDOM_STATE)
                fog_train = prepared.y_any_fog_train
                fog_validation = prepared.y_any_fog_validation
                fog_negatives = float((1.0 - fog_train).sum())
                fog_positives = float(fog_train.sum())
                fog_positive_weight = float(
                    np.clip(
                        fog_negatives / max(fog_positives, 1.0), 1.0, 200.0
                    )
                )
                type_negatives = (1.0 - prepared.y_train).sum(axis=0)
                type_positives = prepared.y_train.sum(axis=0)
                type_positive_weights = np.clip(
                    type_negatives / np.maximum(type_positives, 1.0),
                    1.0,
                    200.0,
                )
                print(
                    f"Any-FoG positive-class loss weight: {fog_positive_weight:.2f}"
                )
                print(
                    "Auxiliary event-type weights:",
                    dict(zip(TARGET_COLUMNS, type_positive_weights.round(2))),
                )

                def make_weighted_binary_crossentropy(positive_weights):
                    weights = tf.constant(positive_weights, dtype=tf.float32)

                    def weighted_binary_crossentropy(y_true, y_pred):
                        y_pred = tf.clip_by_value(
                            y_pred,
                            tf.keras.backend.epsilon(),
                            1.0 - tf.keras.backend.epsilon(),
                        )
                        loss = -(
                            weights * y_true * tf.math.log(y_pred)
                            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
                        )
                        return tf.reduce_mean(loss)

                    return weighted_binary_crossentropy

                inputs = tf.keras.layers.Input(
                    shape=(TIMESTEPS, len(FEATURE_COLUMNS)),
                    name="sensor_sequence",
                )
                shared = tf.keras.layers.LSTM(64, activation="tanh")(inputs)
                shared = tf.keras.layers.Dense(64, activation="relu")(shared)
                fog_output = tf.keras.layers.Dense(
                    1, activation="sigmoid", name="fog"
                )(shared)
                event_type_output = tf.keras.layers.Dense(
                    len(TARGET_COLUMNS),
                    activation="sigmoid",
                    name="event_type",
                )(shared)
                model = tf.keras.Model(
                    inputs=inputs,
                    outputs={"fog": fog_output, "event_type": event_type_output},
                )
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss={
                        "fog": make_weighted_binary_crossentropy(
                            fog_positive_weight
                        ),
                        "event_type": make_weighted_binary_crossentropy(
                            type_positive_weights
                        ),
                    },
                    loss_weights={"fog": 1.0, "event_type": 0.2},
                    metrics={
                        "fog": [
                            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                            tf.keras.metrics.Precision(name="precision"),
                            tf.keras.metrics.Recall(name="recall"),
                            tf.keras.metrics.BinaryAccuracy(
                                name="binary_accuracy"
                            ),
                        ],
                        "event_type": [
                            tf.keras.metrics.AUC(
                                curve="PR",
                                multi_label=True,
                                num_labels=len(TARGET_COLUMNS),
                                name="pr_auc",
                            )
                        ],
                    },
                )
                history = model.fit(
                    prepared.X_train,
                    {"fog": fog_train, "event_type": prepared.y_train},
                    validation_data=(
                        prepared.X_validation,
                        {
                            "fog": fog_validation,
                            "event_type": prepared.y_validation,
                        },
                    ),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_fog_pr_auc",
                            mode="max",
                            min_delta=1e-4,
                            patience=EARLY_STOPPING_PATIENCE,
                            restore_best_weights=True,
                        )
                    ],
                    verbose=2,
                )
            else:
                print("Training is disabled. Set RUN_TRAINING=True when ready.")
            """
        ),
        markdown("### Learning curves"),
        code(LEARNING_CURVES),
        markdown("## Final test — evaluate once"),
        code(FINAL_EVALUATION),
        markdown(
            """
            ## Takeaways

            The model now measures performance on unseen subjects, and no LSTM
            window can mix recordings, cross unobserved gaps, or contain shuffled
            timesteps. Its primary output is P(FoG now) at each window endpoint.
            """
        ),
    ]
    save_notebook("defog_lstm_subject_generalisation.ipynb", cells)


if __name__ == "__main__":
    build_base_notebook()
    build_tdcs_notebook()
    build_lstm_notebook()
    print(f"Rebuilt leakage-safe notebooks in {NOTEBOOK_DIR}.")
