"""Fair, subject-disjoint MLP/LSTM/TCN comparison utilities.

This module deliberately fixes the data, folds, timestamps, preprocessing,
training budget, decoder, and metrics while changing only the model family.
DeFOG and TDCS use the same causal 25 Hz representation and the same two-second
endpoint windows, but they are evaluated as separate experiments.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from fog_pipeline import (
    ANY_FOG_COLUMN,
    SEGMENT_COLUMN,
    SENSOR_COLUMNS,
    TARGET_COLUMNS,
    evaluate_episode_predictions,
)


ARCHITECTURES = ("MLP", "LSTM", "TCN")
MODEL_FEATURES = (*SENSOR_COLUMNS, "AccMagnitude")
PRIMARY_OPERATING_POINT = "macro_f1"
ALARM_BUDGET_OPERATING_POINT = "alarm_budget"


@dataclass(frozen=True)
class BenchmarkSettings:
    """Every setting shared by the three compared architectures."""

    native_sampling_rate_hz: float
    benchmark_sampling_rate_hz: float = 25.0
    folds_to_run: tuple[int, ...] = (0, 1, 2, 3, 4)
    ensemble_seeds: tuple[int, ...] = (42,)
    n_outer_folds: int = 5
    random_state: int = 42
    window_seconds: float = 2.0
    training_stride_bins: int = 5
    early_stop_stride_bins: int = 1
    batch_size: int = 256
    prediction_batch_size: int = 512
    epochs: int = 40
    early_stopping_patience: int = 8
    learning_rate: float = 1e-3
    subject_balanced_training: bool = True
    auxiliary_type_loss_weight: float = 0.2
    class_weight_power: float = 0.5
    max_class_weight: float = 20.0
    mlp_hidden_units: tuple[int, int] = (80, 32)
    lstm_units: int = 64
    shared_dense_units: int = 32
    tcn_filters: int = 28
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    model_dropout: float = 0.15
    decoder_on_thresholds: tuple[float, ...] = (
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.99,
    )
    decoder_hysteresis_gap: float = 0.10
    decoder_on_confirm_seconds: float = 0.05
    decoder_off_confirm_seconds: float = 0.10
    episode_iou_threshold: float = 0.25
    false_alarm_budget_per_minute: float = 1.0
    max_pr_curve_points: int = 2_000

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.benchmark_sampling_rate_hz))

    @property
    def decoder_on_confirm_samples(self) -> int:
        return max(
            1,
            int(
                np.ceil(
                    self.decoder_on_confirm_seconds
                    * self.benchmark_sampling_rate_hz
                )
            ),
        )

    @property
    def decoder_off_confirm_samples(self) -> int:
        return max(
            1,
            int(
                np.ceil(
                    self.decoder_off_confirm_seconds
                    * self.benchmark_sampling_rate_hz
                )
            ),
        )

    def validate(self) -> None:
        if self.native_sampling_rate_hz <= 0:
            raise ValueError("native_sampling_rate_hz must be positive")
        if self.benchmark_sampling_rate_hz <= 0:
            raise ValueError("benchmark_sampling_rate_hz must be positive")
        if self.benchmark_sampling_rate_hz > self.native_sampling_rate_hz:
            raise ValueError("The benchmark must not upsample a recording")
        if self.window_samples < 2:
            raise ValueError("window_seconds is too short")
        if self.n_outer_folds < 3:
            raise ValueError("At least three outer folds are required")
        if not self.folds_to_run:
            raise ValueError("At least one outer fold must be requested")
        invalid_folds = sorted(
            set(self.folds_to_run).difference(range(self.n_outer_folds))
        )
        if invalid_folds:
            raise ValueError(f"Invalid outer folds: {invalid_folds}")
        if not self.ensemble_seeds:
            raise ValueError("At least one common model seed is required")
        if self.training_stride_bins < 1 or self.early_stop_stride_bins < 1:
            raise ValueError("Window strides must be positive")
        if self.batch_size < 1 or self.prediction_batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        if self.epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("Epoch and patience settings must be positive")
        if self.false_alarm_budget_per_minute < 0:
            raise ValueError("The false-alarm budget must be non-negative")


@dataclass(frozen=True)
class BenchmarkResults:
    """Compact artifacts; full timestamp probabilities are not retained."""

    histories: pd.DataFrame
    fold_results: pd.DataFrame
    overall_results: pd.DataFrame
    decoder_tables: pd.DataFrame
    subject_results: pd.DataFrame
    confusion_counts: pd.DataFrame
    pr_curves: pd.DataFrame
    model_parameters: pd.DataFrame
    fold_partitions: pd.DataFrame
    matches: pd.DataFrame


def validate_benchmark_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate identity, alignment, sensor, and label invariants."""

    required = {
        "RecordingId",
        "Subject",
        "Time",
        *SENSOR_COLUMNS,
        *TARGET_COLUMNS,
        ANY_FOG_COLUMN,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Benchmark input is missing columns: {missing}")
    if data[["RecordingId", "Subject"]].isna().any().any():
        raise ValueError("Recording and subject identities must be complete")
    if data.groupby("RecordingId", observed=True)["Subject"].nunique().max() != 1:
        raise ValueError("Every recording must map to exactly one subject")

    data["Subject"] = data["Subject"].astype(str).astype("category")
    for column in ("Time", *SENSOR_COLUMNS, *TARGET_COLUMNS, ANY_FOG_COLUMN):
        values = data[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain only finite values")

    time_values = data["Time"].to_numpy(dtype=np.float64)
    if not np.equal(time_values, np.round(time_values)).all():
        raise ValueError("Time must contain integer sample indices")
    if data.duplicated(["RecordingId", "Time"]).any():
        raise ValueError("Each (RecordingId, Time) pair must be unique")
    differences = data.groupby(
        "RecordingId", sort=False, observed=True
    )["Time"].diff()
    if (differences.dropna() <= 0).any():
        raise ValueError("Time must increase strictly within recordings")

    targets = data.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.int8)
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("Event-type labels must be binary 0/1")
    expected_any_fog = targets.max(axis=1)
    supplied_any_fog = data[ANY_FOG_COLUMN].to_numpy(dtype=np.int8)
    if not np.array_equal(expected_any_fog, supplied_any_fog):
        raise ValueError("AnyFoG must equal the union of event-type labels")
    return data


def add_contiguous_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Reset temporal state at recording boundaries and missing native samples."""

    ordered = frame.sort_values(["RecordingId", "Time"], kind="stable").copy()
    difference = ordered.groupby(
        "RecordingId", sort=False, observed=True
    )["Time"].diff()
    starts_segment = difference.ne(1)
    ordered[SEGMENT_COLUMN] = (
        starts_segment.groupby(ordered["RecordingId"], observed=True).cumsum() - 1
    ).astype(np.int32)
    return ordered


def causal_downsample_recordings(
    data: pd.DataFrame,
    settings: BenchmarkSettings,
) -> pd.DataFrame:
    """Aggregate preceding native samples onto a causal near-25 Hz endpoint grid.

    Sensor values are means of samples at or before the retained endpoint. Each
    label is the label at that endpoint, so the target remains whether FoG is
    occurring now. Variable-width bins preserve an average target rate when 25
    does not divide the native rate (for example, TDCS at 128 Hz).
    """

    settings.validate()
    validate_benchmark_data(data)
    segmented = add_contiguous_segments(data)
    position = segmented.groupby(
        ["RecordingId", SEGMENT_COLUMN], sort=False, observed=True
    ).cumcount()
    segmented["_BenchmarkBin"] = np.floor(
        position.to_numpy(dtype=np.float64)
        * settings.benchmark_sampling_rate_hz
        / settings.native_sampling_rate_hz
    ).astype(np.int64)

    group_columns = ["RecordingId", SEGMENT_COLUMN, "_BenchmarkBin"]
    aggregation: dict[str, str] = {
        "Subject": "first",
        "Time": "last",
        **{sensor: "mean" for sensor in SENSOR_COLUMNS},
        **{target: "last" for target in TARGET_COLUMNS},
    }
    binned = (
        segmented.groupby(group_columns, sort=False, observed=True)
        .agg(aggregation)
        .reset_index()
    )
    binned[ANY_FOG_COLUMN] = (
        binned.loc[:, TARGET_COLUMNS].max(axis=1).astype(np.int8)
    )
    binned["RecordingId"] = binned["RecordingId"].astype("category")
    binned["Subject"] = binned["Subject"].astype(str).astype("category")
    binned = binned.drop(columns="_BenchmarkBin")
    validate_benchmark_data(binned)
    if binned.duplicated(["RecordingId", SEGMENT_COLUMN, "Time"]).any():
        raise AssertionError("Downsampled endpoint keys must be unique")
    return binned


def assign_subject_outer_folds(
    data: pd.DataFrame,
    settings: BenchmarkSettings,
) -> tuple[pd.DataFrame, str]:
    """Assign every subject to one deterministic, burden-aware outer fold."""

    settings.validate()
    manifest = (
        data.groupby("Subject", observed=True)
        .agg(
            rows=("Time", "size"),
            recordings=("RecordingId", "nunique"),
            any_fog_rows=(ANY_FOG_COLUMN, "sum"),
        )
        .reset_index()
    )
    manifest["Subject"] = manifest["Subject"].astype(str)
    manifest = manifest.sort_values("Subject", kind="stable").reset_index(drop=True)
    manifest["any_fog_rate"] = manifest["any_fog_rows"] / manifest["rows"]
    if len(manifest) < settings.n_outer_folds:
        raise ValueError(
            f"Need at least {settings.n_outer_folds} subjects; found {len(manifest)}"
        )

    stratum_count = min(
        settings.n_outer_folds,
        max(2, len(manifest) // settings.n_outer_folds),
    )
    manifest["stratum"] = pd.qcut(
        manifest["any_fog_rate"].rank(method="first"),
        q=stratum_count,
        labels=False,
        duplicates="drop",
    )
    stratum_sizes = manifest["stratum"].value_counts()
    can_stratify = (
        len(stratum_sizes) > 1
        and int(stratum_sizes.min()) >= settings.n_outer_folds
    )
    splitter_class = StratifiedKFold if can_stratify else KFold
    splitter = splitter_class(
        n_splits=settings.n_outer_folds,
        shuffle=True,
        random_state=settings.random_state,
    )
    split_target = manifest["stratum"] if can_stratify else None
    manifest["OuterFold"] = -1
    for fold, (_, test_indices) in enumerate(splitter.split(manifest, split_target)):
        manifest.loc[test_indices, "OuterFold"] = fold
    if (manifest["OuterFold"] < 0).any():
        raise AssertionError("Every subject must receive one outer fold")

    fold_by_subject = manifest.set_index("Subject")["OuterFold"]
    data["OuterFold"] = data["Subject"].astype(str).map(fold_by_subject).astype(int)
    if data.groupby("Subject", observed=True)["OuterFold"].nunique().max() != 1:
        raise AssertionError("A subject appears in more than one outer fold")
    if data.groupby("RecordingId", observed=True)["OuterFold"].nunique().max() != 1:
        raise AssertionError("A recording appears in more than one outer fold")

    strategy = "burden-stratified" if can_stratify else "shuffled subject K-fold"
    return manifest, strategy


def subject_partitions_for_fold(
    manifest: pd.DataFrame,
    outer_fold: int,
    settings: BenchmarkSettings,
) -> dict[str, set[str]]:
    """Create common train, early-stop, calibration, and test subject sets."""

    outer_fold = int(outer_fold)
    development_fold = (outer_fold + 1) % settings.n_outer_folds
    ordered_development = (
        manifest.loc[manifest["OuterFold"] == development_fold]
        .sort_values(["any_fog_rate", "Subject"], kind="stable")["Subject"]
        .astype(str)
        .tolist()
    )
    if len(ordered_development) < 2:
        raise ValueError(
            "Each development fold needs at least two subjects so early stopping "
            "and decoder calibration remain disjoint"
        )

    partitions = {
        "train": set(
            manifest.loc[
                ~manifest["OuterFold"].isin([outer_fold, development_fold]),
                "Subject",
            ].astype(str)
        ),
        "early_stop": set(ordered_development[::2]),
        "calibration": set(ordered_development[1::2]),
        "test": set(
            manifest.loc[manifest["OuterFold"] == outer_fold, "Subject"].astype(str)
        ),
    }
    partition_values = list(partitions.values())
    for index, left in enumerate(partition_values):
        for right in partition_values[index + 1 :]:
            if left & right:
                raise AssertionError("Benchmark subject partitions overlap")
    if set().union(*partition_values) != set(manifest["Subject"].astype(str)):
        raise AssertionError("Benchmark partitions do not cover every subject")
    return partitions


def _frame_for_subjects(data: pd.DataFrame, subjects: set[str]) -> pd.DataFrame:
    return data.loc[data["Subject"].astype(str).isin(subjects)].copy()


def _contains_both_any_fog_classes(frame: pd.DataFrame) -> bool:
    positives = int(frame[ANY_FOG_COLUMN].sum())
    return 0 < positives < len(frame)


def raw_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    sensors = frame.loc[:, SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    magnitude = np.sqrt(np.square(sensors).sum(axis=1, keepdims=True))
    features = np.concatenate([sensors, magnitude], axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Model features must be finite")
    return features


def fit_training_scaler(training_frame: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    for _, recording in training_frame.groupby(
        "RecordingId", sort=False, observed=True
    ):
        scaler.partial_fit(raw_feature_matrix(recording))
    return scaler


def build_segments(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    *,
    include_alignment: bool,
) -> list[dict[str, object]]:
    """Build recording-bounded arrays; the first window-1 bins stay unscored."""

    ordered = frame.sort_values(
        ["RecordingId", SEGMENT_COLUMN, "Time"], kind="stable"
    )
    segments: list[dict[str, object]] = []
    for _, segment in ordered.groupby(
        ["RecordingId", SEGMENT_COLUMN], sort=False, observed=True
    ):
        features = scaler.transform(raw_feature_matrix(segment)).astype(np.float32)
        event_types = segment.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
        any_fog = event_types.max(axis=1, keepdims=True).astype(np.float32)
        alignment = None
        if include_alignment:
            alignment = segment.loc[
                :, ["RecordingId", "Subject", "Time", SEGMENT_COLUMN]
            ].reset_index(drop=True)
        segments.append(
            {
                "x": features,
                "y_fog": any_fog,
                "y_type": event_types,
                "alignment": alignment,
                "subject": str(segment["Subject"].iloc[0]),
            }
        )
    if not segments:
        raise ValueError("No contiguous segments were produced")
    return segments


def make_window_references(
    segments: Sequence[Mapping[str, object]],
    *,
    window_samples: int,
    stride: int,
) -> tuple[tuple[int, int], ...]:
    references = tuple(
        (segment_index, endpoint)
        for segment_index, segment in enumerate(segments)
        for endpoint in range(
            window_samples - 1,
            len(np.asarray(segment["x"])),
            stride,
        )
    )
    if not references:
        raise ValueError("No complete endpoint windows were available")
    return references


def make_subject_weights(
    segments: Sequence[Mapping[str, object]],
    references: Sequence[tuple[int, int]],
    *,
    subject_balanced: bool,
) -> dict[str, float]:
    counts: dict[str, int] = {}
    for segment_index, _ in references:
        subject = str(segments[segment_index]["subject"])
        counts[subject] = counts.get(subject, 0) + 1
    if not subject_balanced:
        return {subject: 1.0 for subject in counts}
    total = float(sum(counts.values()))
    subject_count = float(len(counts))
    return {
        subject: total / (subject_count * float(count))
        for subject, count in counts.items()
    }


def class_weights_from_references(
    segments: Sequence[Mapping[str, object]],
    references: Sequence[tuple[int, int]],
    subject_weights: Mapping[str, float],
    settings: BenchmarkSettings,
) -> tuple[float, np.ndarray]:
    row_weight = 0.0
    fog_positive = 0.0
    type_positive = np.zeros(len(TARGET_COLUMNS), dtype=float)
    for segment_index, endpoint in references:
        segment = segments[segment_index]
        weight = float(subject_weights[str(segment["subject"])])
        y_fog = float(np.asarray(segment["y_fog"])[endpoint, 0])
        y_type = np.asarray(segment["y_type"])[endpoint]
        row_weight += weight
        fog_positive += weight * y_fog
        type_positive += weight * y_type
    fog_negative = row_weight - fog_positive
    type_negative = row_weight - type_positive
    fog_weight = float(
        np.clip(
            (fog_negative / max(fog_positive, 1.0)) ** settings.class_weight_power,
            1.0,
            settings.max_class_weight,
        )
    )
    type_weights = np.clip(
        (type_negative / np.maximum(type_positive, 1.0))
        ** settings.class_weight_power,
        1.0,
        settings.max_class_weight,
    ).astype(np.float32)
    return fog_weight, type_weights


class EndpointWindowSequence(tf.keras.utils.Sequence):
    """Stream fixed causal windows without materialising overlapping tensors."""

    def __init__(
        self,
        segments: Sequence[Mapping[str, object]],
        references: Sequence[tuple[int, int]],
        *,
        window_samples: int,
        batch_size: int,
        shuffle: bool,
        seed: int,
        include_targets: bool,
        subject_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(
            workers=1,
            use_multiprocessing=False,
            max_queue_size=2,
        )
        self.segments = segments
        self.references = tuple(references)
        self.window_samples = int(window_samples)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.include_targets = bool(include_targets)
        self.subject_weights = dict(subject_weights or {})
        self.rng = np.random.default_rng(seed)
        if not self.references:
            raise ValueError("An endpoint sequence needs at least one reference")
        self.order = np.arange(len(self.references), dtype=np.int64)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.references) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __getitem__(self, batch_index: int):
        order_slice = self.order[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        batch_length = len(order_slice)
        x_batch = np.empty(
            (batch_length, self.window_samples, len(MODEL_FEATURES)),
            dtype=np.float32,
        )
        if self.include_targets:
            fog_batch = np.empty((batch_length, 1), dtype=np.float32)
            type_batch = np.empty(
                (batch_length, len(TARGET_COLUMNS)), dtype=np.float32
            )
            weight_batch = np.empty(batch_length, dtype=np.float32)

        for row, reference_index in enumerate(order_slice):
            segment_index, endpoint = self.references[int(reference_index)]
            segment = self.segments[segment_index]
            window_start = endpoint - self.window_samples + 1
            x_batch[row] = np.asarray(segment["x"])[window_start : endpoint + 1]
            if self.include_targets:
                fog_batch[row] = np.asarray(segment["y_fog"])[endpoint]
                type_batch[row] = np.asarray(segment["y_type"])[endpoint]
                weight_batch[row] = float(
                    self.subject_weights.get(str(segment["subject"]), 1.0)
                )

        if not self.include_targets:
            return x_batch
        return (
            x_batch,
            {"fog": fog_batch, "event_type": type_batch},
            {"fog": weight_batch, "event_type": weight_batch},
        )


def make_weighted_binary_crossentropy(positive_weights):
    positive_weights = tf.constant(positive_weights, dtype=tf.float32)

    def weighted_binary_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon(),
        )
        element_loss = -(
            positive_weights * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(element_loss, axis=-1)

    return weighted_binary_crossentropy


def _model_outputs(shared):
    fog_output = tf.keras.layers.Dense(1, activation="sigmoid", name="fog")(shared)
    event_type_output = tf.keras.layers.Dense(
        len(TARGET_COLUMNS), activation="sigmoid", name="event_type"
    )(shared)
    return {"fog": fog_output, "event_type": event_type_output}


def build_benchmark_model(
    architecture: str,
    fog_positive_weight: float,
    type_positive_weights: np.ndarray,
    settings: BenchmarkSettings,
    *,
    compile_model: bool = True,
) -> tf.keras.Model:
    """Build a near-capacity-matched model on the identical 50 by 4 input."""

    architecture = str(architecture).upper()
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    inputs = tf.keras.layers.Input(
        shape=(settings.window_samples, len(MODEL_FEATURES)),
        name="accelerometer_window",
    )

    if architecture == "MLP":
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(
            settings.mlp_hidden_units[0], activation="swish"
        )(x)
        x = tf.keras.layers.Dropout(settings.model_dropout)(x)
        shared = tf.keras.layers.Dense(
            settings.mlp_hidden_units[1], activation="swish"
        )(x)
    elif architecture == "LSTM":
        x = tf.keras.layers.LayerNormalization()(inputs)
        x = tf.keras.layers.LSTM(
            settings.lstm_units,
            dropout=settings.model_dropout,
            recurrent_dropout=0.0,
        )(x)
        shared = tf.keras.layers.Dense(
            settings.shared_dense_units, activation="swish"
        )(x)
    else:
        x = tf.keras.layers.Conv1D(
            settings.tcn_filters, kernel_size=1
        )(inputs)
        for dilation in settings.tcn_dilations:
            residual = x
            y = tf.keras.layers.LayerNormalization()(x)
            y = tf.keras.layers.Activation("swish")(y)
            y = tf.keras.layers.Conv1D(
                settings.tcn_filters,
                kernel_size=3,
                padding="causal",
                dilation_rate=dilation,
            )(y)
            y = tf.keras.layers.SpatialDropout1D(settings.model_dropout)(y)
            y = tf.keras.layers.LayerNormalization()(y)
            y = tf.keras.layers.Activation("swish")(y)
            y = tf.keras.layers.Conv1D(
                settings.tcn_filters,
                kernel_size=3,
                padding="causal",
                dilation_rate=dilation,
            )(y)
            x = tf.keras.layers.Add()([residual, y])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation("swish")(x)
        x = tf.keras.layers.Lambda(lambda values: values[:, -1, :])(x)
        shared = tf.keras.layers.Dense(
            settings.shared_dense_units, activation="swish"
        )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=_model_outputs(shared),
        name=f"fair_{architecture.lower()}",
    )
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=settings.learning_rate
            ),
            loss={
                "fog": make_weighted_binary_crossentropy(fog_positive_weight),
                "event_type": make_weighted_binary_crossentropy(
                    type_positive_weights
                ),
            },
            loss_weights={
                "fog": 1.0,
                "event_type": settings.auxiliary_type_loss_weight,
            },
            weighted_metrics={
                "fog": [
                    tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                ]
            },
        )
    return model


def predict_reference_scores(
    model: tf.keras.Model,
    segments: Sequence[Mapping[str, object]],
    references: Sequence[tuple[int, int]],
    settings: BenchmarkSettings,
) -> np.ndarray:
    sequence = EndpointWindowSequence(
        segments,
        references,
        window_samples=settings.window_samples,
        batch_size=settings.prediction_batch_size,
        shuffle=False,
        seed=settings.random_state,
        include_targets=False,
    )
    fog_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("fog").output,
    )
    scores = fog_model.predict(sequence, verbose=0).reshape(-1).astype(np.float32)
    if len(scores) != len(references):
        raise AssertionError("Prediction rows are not aligned with endpoint references")
    if not np.isfinite(scores).all():
        raise AssertionError("Model probabilities must be finite")
    del fog_model, sequence
    return scores


def endpoint_frame(
    segments: Sequence[Mapping[str, object]],
    references: Sequence[tuple[int, int]],
    scores: np.ndarray,
) -> pd.DataFrame:
    if len(references) != len(scores):
        raise ValueError("References and scores must have equal lengths")
    # References are emitted segment-by-segment. Slice each alignment frame once
    # rather than constructing millions of individual pandas Series objects.
    pieces: list[pd.DataFrame] = []
    reference_offset = 0
    while reference_offset < len(references):
        segment_index = int(references[reference_offset][0])
        endpoint_values: list[int] = []
        score_start = reference_offset
        while (
            reference_offset < len(references)
            and int(references[reference_offset][0]) == segment_index
        ):
            endpoint_values.append(int(references[reference_offset][1]))
            reference_offset += 1
        segment = segments[segment_index]
        alignment = segment["alignment"]
        if alignment is None:
            raise ValueError("Evaluation segments require alignment data")
        piece = alignment.iloc[endpoint_values].copy().reset_index(drop=True)
        piece["AnyFoGTrue"] = (
            np.asarray(segment["y_fog"])[endpoint_values, 0].astype(np.int8)
        )
        piece["AnyFoGScore"] = np.asarray(
            scores[score_start:reference_offset], dtype=np.float32
        )
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True)
    if frame.duplicated(["RecordingId", SEGMENT_COLUMN, "Time"]).any():
        raise AssertionError("Every evaluation endpoint must be unique")
    return frame


def hysteresis_decode(
    scores: np.ndarray,
    *,
    on_threshold: float,
    off_threshold: float,
    on_confirm_samples: int,
    off_confirm_samples: int,
) -> np.ndarray:
    predictions = np.zeros(len(scores), dtype=np.int8)
    state_on = False
    high_count = 0
    low_count = 0
    for index, score in enumerate(scores):
        if not state_on:
            high_count = high_count + 1 if score >= on_threshold else 0
            if high_count >= on_confirm_samples:
                state_on = True
                high_count = 0
                low_count = 0
        else:
            low_count = low_count + 1 if score < off_threshold else 0
            if low_count >= off_confirm_samples:
                state_on = False
                low_count = 0
                high_count = 0
        predictions[index] = int(state_on)
    return predictions


def decode_endpoint_frame(
    frame: pd.DataFrame,
    on_threshold: float,
    settings: BenchmarkSettings,
) -> np.ndarray:
    off_threshold = max(0.01, on_threshold - settings.decoder_hysteresis_gap)
    predictions = np.zeros(len(frame), dtype=np.int8)
    grouped = frame.groupby(
        ["RecordingId", SEGMENT_COLUMN], sort=False, observed=True
    )
    for _, indices in grouped.groups.items():
        ordered_indices = frame.loc[indices].sort_values(
            "Time", kind="stable"
        ).index.to_numpy(dtype=np.int64)
        predictions[ordered_indices] = hysteresis_decode(
            frame.loc[ordered_indices, "AnyFoGScore"].to_numpy(dtype=float),
            on_threshold=on_threshold,
            off_threshold=off_threshold,
            on_confirm_samples=settings.decoder_on_confirm_samples,
            off_confirm_samples=settings.decoder_off_confirm_samples,
        )
    return predictions


def evaluate_decoded_frame(frame: pd.DataFrame, settings: BenchmarkSettings):
    return evaluate_episode_predictions(
        frame.loc[:, ["RecordingId", "Subject", "Time", SEGMENT_COLUMN]],
        frame["AnyFoGTrue"].to_numpy(dtype=np.int8),
        frame["AnyFoGPredicted"].to_numpy(dtype=np.float32),
        threshold=0.5,
        sampling_rate_hz=settings.native_sampling_rate_hz,
        minimum_iou=settings.episode_iou_threshold,
    )


def subject_episode_metrics(
    frame: pd.DataFrame,
    settings: BenchmarkSettings,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows = []
    for subject, subject_frame in frame.groupby(
        "Subject", sort=False, observed=True
    ):
        evaluation = evaluate_decoded_frame(subject_frame, settings)
        rows.append(
            {
                "Subject": str(subject),
                "true_episodes": evaluation.true_count,
                "predicted_episodes": evaluation.predicted_count,
                "matched_episodes": evaluation.true_positive_count,
                "episode_precision": evaluation.precision,
                "episode_recall": evaluation.recall,
                "episode_f1": evaluation.f1,
                "evaluation_minutes": evaluation.evaluation_minutes,
                "false_alarms_per_minute": evaluation.false_alarms_per_minute,
            }
        )
    result = pd.DataFrame(rows)
    positive_mask = result["true_episodes"] > 0
    positive_count = int(positive_mask.sum())
    macro: dict[str, float | int] = {}
    for source, output in (
        ("episode_precision", "macro_episode_precision"),
        ("episode_recall", "macro_episode_recall"),
        ("episode_f1", "macro_episode_f1"),
    ):
        values = result.loc[positive_mask, source].fillna(0.0)
        macro[output] = float(values.mean()) if positive_count else float("nan")
        macro[f"{output}_n_subjects"] = positive_count
    alarm_values = result["false_alarms_per_minute"]
    macro["macro_false_alarms_per_minute"] = float(alarm_values.mean(skipna=True))
    macro["macro_false_alarms_per_minute_n_subjects"] = int(
        alarm_values.notna().sum()
    )
    return result, macro


def select_validation_decoders(
    frame: pd.DataFrame,
    settings: BenchmarkSettings,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Choose both the primary F1 point and a fixed false-alarm-budget point."""

    rows = []
    for on_threshold in settings.decoder_on_thresholds:
        candidate = frame.copy()
        candidate["AnyFoGPredicted"] = decode_endpoint_frame(
            candidate, float(on_threshold), settings
        )
        evaluation = evaluate_decoded_frame(candidate, settings)
        _, macro = subject_episode_metrics(candidate, settings)
        rows.append(
            {
                "on_threshold": float(on_threshold),
                "off_threshold": float(
                    max(0.01, on_threshold - settings.decoder_hysteresis_gap)
                ),
                "macro_episode_f1": macro["macro_episode_f1"],
                "macro_episode_f1_n_subjects": macro[
                    "macro_episode_f1_n_subjects"
                ],
                "macro_episode_recall": macro["macro_episode_recall"],
                "macro_false_alarms_per_minute": macro[
                    "macro_false_alarms_per_minute"
                ],
                "episode_precision": evaluation.precision,
                "episode_recall": evaluation.recall,
                "episode_f1": evaluation.f1,
                "false_alarms_per_minute": evaluation.false_alarms_per_minute,
                "mean_matched_iou": evaluation.mean_duration_iou,
            }
        )
    table = pd.DataFrame(rows)
    primary_ranked = table.assign(
        _macro_f1=table["macro_episode_f1"].fillna(-np.inf),
        _macro_false_alarms=table[
            "macro_false_alarms_per_minute"
        ].fillna(np.inf),
        _macro_recall=table["macro_episode_recall"].fillna(-np.inf),
        _mean_iou=table["mean_matched_iou"].fillna(-np.inf),
    ).sort_values(
        [
            "_macro_f1",
            "_macro_false_alarms",
            "_macro_recall",
            "_mean_iou",
            "on_threshold",
        ],
        ascending=[False, True, False, False, False],
        kind="stable",
    )
    primary_threshold = float(primary_ranked.iloc[0]["on_threshold"])

    feasible = table.loc[
        table["macro_false_alarms_per_minute"]
        <= settings.false_alarm_budget_per_minute + 1e-12
    ]
    budget_was_feasible = not feasible.empty
    if feasible.empty:
        # Retain a complete, paired comparison even when the requested budget
        # cannot be met. Use the least-alarming calibrated candidate and mark
        # the fallback explicitly instead of silently dropping this model/fold.
        budget_ranked = table.assign(
            _macro_false_alarms=table[
                "macro_false_alarms_per_minute"
            ].fillna(np.inf),
            _macro_recall=table["macro_episode_recall"].fillna(-np.inf),
            _macro_f1=table["macro_episode_f1"].fillna(-np.inf),
        ).sort_values(
            [
                "_macro_false_alarms",
                "_macro_recall",
                "_macro_f1",
                "on_threshold",
            ],
            ascending=[True, False, False, False],
            kind="stable",
        )
        budget_threshold = float(budget_ranked.iloc[0]["on_threshold"])
    else:
        budget_ranked = feasible.assign(
            _macro_recall=feasible["macro_episode_recall"].fillna(-np.inf),
            _macro_f1=feasible["macro_episode_f1"].fillna(-np.inf),
            _mean_iou=feasible["mean_matched_iou"].fillna(-np.inf),
        ).sort_values(
            ["_macro_recall", "_macro_f1", "_mean_iou", "on_threshold"],
            ascending=[False, False, False, False],
            kind="stable",
        )
        budget_threshold = float(budget_ranked.iloc[0]["on_threshold"])

    table["selected_macro_f1"] = table["on_threshold"].eq(primary_threshold)
    table["selected_alarm_budget"] = table["on_threshold"].eq(budget_threshold)
    table["alarm_budget_was_feasible"] = budget_was_feasible
    return {
        PRIMARY_OPERATING_POINT: primary_threshold,
        ALARM_BUDGET_OPERATING_POINT: budget_threshold,
    }, table


def _safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(average_precision_score(y_true, scores))
        if np.any(y_true == 1)
        else float("nan")
    )


def _weighted_fog_loss(
    y_true: np.ndarray,
    scores: np.ndarray,
    positive_weight: float,
) -> float:
    clipped = np.clip(
        np.asarray(scores, dtype=np.float64),
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    truth = np.asarray(y_true, dtype=np.float64)
    losses = -(
        positive_weight * truth * np.log(clipped)
        + (1.0 - truth) * np.log(1.0 - clipped)
    )
    return float(losses.mean())


def _compact_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_points: int,
) -> pd.DataFrame:
    if not np.any(y_true == 1):
        return pd.DataFrame(columns=["recall", "precision"])
    precision, recall, _ = precision_recall_curve(y_true, scores)
    if len(precision) > max_points:
        indices = np.unique(
            np.linspace(0, len(precision) - 1, max_points, dtype=np.int64)
        )
        precision = precision[indices]
        recall = recall[indices]
    return pd.DataFrame({"recall": recall, "precision": precision})


def _summarise_overall_results(
    fold_results: pd.DataFrame,
    subject_results: pd.DataFrame,
    matches: pd.DataFrame,
    model_parameters: Mapping[str, int],
) -> pd.DataFrame:
    rows = []
    group_columns = ["model", "operating_point"]
    for (architecture, operating_point), model_folds in fold_results.groupby(
        group_columns, sort=False, observed=True
    ):
        true_episodes = int(model_folds["true_episodes"].sum())
        predicted_episodes = int(model_folds["predicted_episodes"].sum())
        matched_episodes = int(model_folds["matched_episodes"].sum())
        false_episodes = predicted_episodes - matched_episodes
        missed_episodes = true_episodes - matched_episodes
        episode_precision = (
            matched_episodes / predicted_episodes
            if predicted_episodes
            else float("nan")
        )
        episode_recall = (
            matched_episodes / true_episodes if true_episodes else float("nan")
        )
        episode_f1 = (
            2.0 * matched_episodes / (true_episodes + predicted_episodes)
            if true_episodes + predicted_episodes
            else float("nan")
        )
        evaluation_minutes = float(model_folds["evaluation_minutes"].sum())
        false_alarms_per_minute = (
            false_episodes / evaluation_minutes
            if evaluation_minutes > 0
            else float("nan")
        )

        tn = int(model_folds["tn"].sum())
        fp = int(model_folds["fp"].sum())
        fn = int(model_folds["fn"].sum())
        tp = int(model_folds["tp"].sum())
        timestep_precision = tp / (tp + fp) if tp + fp else float("nan")
        timestep_recall = tp / (tp + fn) if tp + fn else float("nan")
        timestep_specificity = tn / (tn + fp) if tn + fp else float("nan")
        timestep_f1 = (
            2.0 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn
            else float("nan")
        )

        model_subjects = subject_results.loc[
            (subject_results["model"] == architecture)
            & (subject_results["operating_point"] == operating_point)
        ]
        positive_subjects = model_subjects.loc[
            model_subjects["true_episodes"] > 0
        ]
        model_matches = matches.loc[
            (matches["model"] == architecture)
            & (matches["operating_point"] == operating_point)
        ]
        rows.append(
            {
                "model": architecture,
                "operating_point": operating_point,
                "parameters": int(model_parameters[architecture]),
                "evaluated_outer_folds": int(model_folds["outer_fold"].nunique()),
                "evaluated_subjects": int(model_subjects["Subject"].nunique()),
                "mean_fold_average_precision": model_folds[
                    "average_precision"
                ].mean(),
                "std_fold_average_precision": model_folds[
                    "average_precision"
                ].std(ddof=1),
                "mean_fold_test_weighted_fog_loss": model_folds[
                    "test_weighted_fog_loss"
                ].mean(),
                "std_fold_test_weighted_fog_loss": model_folds[
                    "test_weighted_fog_loss"
                ].std(ddof=1),
                "timestep_accuracy": (tn + tp) / (tn + fp + fn + tp),
                "timestep_balanced_accuracy": (
                    timestep_recall + timestep_specificity
                )
                / 2,
                "timestep_precision": timestep_precision,
                "timestep_recall": timestep_recall,
                "timestep_f1": timestep_f1,
                "true_episodes": true_episodes,
                "predicted_episodes": predicted_episodes,
                "matched_episodes": matched_episodes,
                "false_predicted_episodes": false_episodes,
                "missed_episodes": missed_episodes,
                "episode_precision": episode_precision,
                "episode_recall": episode_recall,
                "episode_f1": episode_f1,
                "false_alarms_per_minute": false_alarms_per_minute,
                "macro_episode_precision": positive_subjects[
                    "episode_precision"
                ].fillna(0.0).mean(),
                "macro_episode_recall": positive_subjects[
                    "episode_recall"
                ].fillna(0.0).mean(),
                "macro_episode_f1": positive_subjects[
                    "episode_f1"
                ].fillna(0.0).mean(),
                "macro_false_alarms_per_minute": model_subjects[
                    "false_alarms_per_minute"
                ].mean(skipna=True),
                "median_onset_delay_seconds": model_matches[
                    "onset_delay_seconds"
                ].median(),
                "mean_absolute_onset_error_seconds": model_matches[
                    "absolute_onset_error_seconds"
                ].mean(),
                "mean_matched_iou": model_matches["iou"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index(["model", "operating_point"])


def _assert_subject_denominators(subject_frame: pd.DataFrame) -> None:
    grouped = subject_frame.groupby(
        ["outer_fold", "Subject"], observed=True
    )
    if grouped["true_episodes"].nunique().max() != 1:
        raise AssertionError("Models received different true episode denominators")
    minute_ranges = grouped["evaluation_minutes"].agg(lambda x: x.max() - x.min())
    if (minute_ranges > 1e-9).any():
        raise AssertionError("Models received different evaluation durations")


def run_fair_benchmark(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    settings: BenchmarkSettings,
    *,
    architectures: Iterable[str] = ARCHITECTURES,
    verbose: int = 2,
) -> BenchmarkResults:
    """Train and test every model under one frozen subject-fold protocol."""

    settings.validate()
    requested_architectures = tuple(str(name).upper() for name in architectures)
    if requested_architectures != ARCHITECTURES:
        raise ValueError(
            f"For a complete fair comparison use architectures={ARCHITECTURES}"
        )

    histories: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    decoder_frames: list[pd.DataFrame] = []
    subject_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, object]] = []
    pr_frames: list[pd.DataFrame] = []
    match_frames: list[pd.DataFrame] = []
    partition_rows: list[dict[str, object]] = []
    model_parameters: dict[str, int] = {}

    for outer_fold in settings.folds_to_run:
        outer_fold = int(outer_fold)
        partitions = subject_partitions_for_fold(manifest, outer_fold, settings)
        training_frame = _frame_for_subjects(data, partitions["train"])
        stopping_frame = _frame_for_subjects(data, partitions["early_stop"])
        calibration_frame = _frame_for_subjects(data, partitions["calibration"])
        for split_name, split_frame in (
            ("train", training_frame),
            ("early_stop", stopping_frame),
            ("calibration", calibration_frame),
        ):
            if not _contains_both_any_fog_classes(split_frame):
                raise ValueError(
                    f"Outer fold {outer_fold} {split_name} does not contain "
                    "both AnyFoG classes"
                )
            partition_rows.append(
                {
                    "outer_fold": outer_fold,
                    "partition": split_name,
                    "subjects": len(partitions[split_name]),
                    "recordings": split_frame["RecordingId"].nunique(),
                    "rows": len(split_frame),
                    "any_fog_rate": split_frame[ANY_FOG_COLUMN].mean(),
                }
            )

        print(
            f"Outer fold {outer_fold}: "
            f"train={len(partitions['train'])}, "
            f"early-stop={len(partitions['early_stop'])}, "
            f"calibration={len(partitions['calibration'])}, "
            f"test={len(partitions['test'])} subjects"
        )

        scaler = fit_training_scaler(training_frame)
        training_segments = build_segments(
            training_frame, scaler, include_alignment=False
        )
        stopping_segments = build_segments(
            stopping_frame, scaler, include_alignment=False
        )
        calibration_segments = build_segments(
            calibration_frame, scaler, include_alignment=True
        )
        training_references = make_window_references(
            training_segments,
            window_samples=settings.window_samples,
            stride=settings.training_stride_bins,
        )
        stopping_references = make_window_references(
            stopping_segments,
            window_samples=settings.window_samples,
            stride=settings.early_stop_stride_bins,
        )
        calibration_references = make_window_references(
            calibration_segments,
            window_samples=settings.window_samples,
            stride=1,
        )
        subject_weights = make_subject_weights(
            training_segments,
            training_references,
            subject_balanced=settings.subject_balanced_training,
        )
        fog_positive_weight, type_positive_weights = class_weights_from_references(
            training_segments,
            training_references,
            subject_weights,
            settings,
        )
        del training_frame, stopping_frame, calibration_frame
        gc.collect()

        artifacts: dict[str, dict[str, object]] = {}
        for architecture in ARCHITECTURES:
            print(f"Training {architecture} for outer fold {outer_fold}")
            calibration_score_sum = np.zeros(
                len(calibration_references), dtype=np.float64
            )
            seed_weights: list[list[np.ndarray]] = []
            for seed in settings.ensemble_seeds:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(int(seed))
                training_sequence = EndpointWindowSequence(
                    training_segments,
                    training_references,
                    window_samples=settings.window_samples,
                    batch_size=settings.batch_size,
                    shuffle=True,
                    seed=int(seed),
                    include_targets=True,
                    subject_weights=subject_weights,
                )
                stopping_sequence = EndpointWindowSequence(
                    stopping_segments,
                    stopping_references,
                    window_samples=settings.window_samples,
                    batch_size=settings.prediction_batch_size,
                    shuffle=False,
                    seed=settings.random_state,
                    include_targets=True,
                )
                model = build_benchmark_model(
                    architecture,
                    fog_positive_weight,
                    type_positive_weights,
                    settings,
                )
                model_parameters.setdefault(architecture, int(model.count_params()))
                history = model.fit(
                    training_sequence,
                    validation_data=stopping_sequence,
                    epochs=settings.epochs,
                    # The Sequence owns a seeded sample permutation. Prevent
                    # Keras from adding a second batch-level shuffle.
                    shuffle=False,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_fog_pr_auc",
                            mode="max",
                            min_delta=1e-4,
                            patience=settings.early_stopping_patience,
                            restore_best_weights=True,
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor="val_fog_pr_auc",
                            mode="max",
                            factor=0.5,
                            patience=max(
                                2, settings.early_stopping_patience // 2
                            ),
                            min_lr=1e-5,
                        ),
                    ],
                    verbose=verbose,
                )
                history_frame = pd.DataFrame(history.history)
                history_frame["model"] = architecture
                history_frame["outer_fold"] = outer_fold
                history_frame["seed"] = int(seed)
                history_frame["epoch"] = np.arange(len(history_frame))
                histories.append(history_frame)

                calibration_score_sum += predict_reference_scores(
                    model,
                    calibration_segments,
                    calibration_references,
                    settings,
                )
                seed_weights.append(model.get_weights())
                del model, training_sequence, stopping_sequence
                gc.collect()

            calibration_scores = (
                calibration_score_sum / len(settings.ensemble_seeds)
            ).astype(np.float32)
            calibration_output = endpoint_frame(
                calibration_segments,
                calibration_references,
                calibration_scores,
            )
            thresholds, decoder_table = select_validation_decoders(
                calibration_output, settings
            )
            decoder_table.insert(0, "outer_fold", outer_fold)
            decoder_table.insert(0, "model", architecture)
            decoder_frames.append(decoder_table)
            print(
                f"{architecture} fold {outer_fold}: macro-F1 threshold="
                f"{thresholds[PRIMARY_OPERATING_POINT]:.2f}; "
                f"alarm-budget threshold="
                f"{thresholds[ALARM_BUDGET_OPERATING_POINT]:.2f}"
            )
            artifacts[architecture] = {
                "seed_weights": seed_weights,
                "thresholds": thresholds,
            }
            del calibration_score_sum, calibration_scores, calibration_output

        # Every model and decoder is frozen before test predictions or metrics
        # are computed. Test labels participate only in the prespecified fold
        # stratification until this point.
        test_frame = _frame_for_subjects(data, partitions["test"])
        if not _contains_both_any_fog_classes(test_frame):
            raise ValueError(
                f"Outer fold {outer_fold} test partition lacks both classes"
            )
        partition_rows.append(
            {
                "outer_fold": outer_fold,
                "partition": "test",
                "subjects": len(partitions["test"]),
                "recordings": test_frame["RecordingId"].nunique(),
                "rows": len(test_frame),
                "any_fog_rate": test_frame[ANY_FOG_COLUMN].mean(),
            }
        )
        test_segments = build_segments(test_frame, scaler, include_alignment=True)
        test_references = make_window_references(
            test_segments,
            window_samples=settings.window_samples,
            stride=1,
        )
        reference_output = endpoint_frame(
            test_segments,
            test_references,
            np.zeros(len(test_references), dtype=np.float32),
        )
        reference_keys = reference_output.loc[
            :, ["RecordingId", "Subject", SEGMENT_COLUMN, "Time"]
        ].astype({"RecordingId": str, "Subject": str})
        reference_truth = reference_output["AnyFoGTrue"].to_numpy(dtype=np.int8)
        reference_output["AnyFoGPredicted"] = 0
        reference_evaluation = evaluate_decoded_frame(reference_output, settings)
        expected_true_episodes = reference_evaluation.true_count
        expected_minutes = reference_evaluation.evaluation_minutes
        del test_frame, reference_output, reference_evaluation
        gc.collect()

        for architecture in ARCHITECTURES:
            artifact = artifacts[architecture]
            test_score_sum = np.zeros(len(test_references), dtype=np.float64)
            for weights in artifact["seed_weights"]:
                tf.keras.backend.clear_session()
                inference_model = build_benchmark_model(
                    architecture,
                    fog_positive_weight,
                    type_positive_weights,
                    settings,
                    compile_model=False,
                )
                inference_model.set_weights(weights)
                test_score_sum += predict_reference_scores(
                    inference_model,
                    test_segments,
                    test_references,
                    settings,
                )
                del inference_model
                gc.collect()
            test_scores = (
                test_score_sum / len(settings.ensemble_seeds)
            ).astype(np.float32)
            base_output = endpoint_frame(
                test_segments, test_references, test_scores
            )
            observed_keys = base_output.loc[
                :, ["RecordingId", "Subject", SEGMENT_COLUMN, "Time"]
            ].astype({"RecordingId": str, "Subject": str})
            if not observed_keys.equals(reference_keys):
                raise AssertionError(
                    f"{architecture} did not score the common test endpoints"
                )
            if not np.array_equal(
                base_output["AnyFoGTrue"].to_numpy(dtype=np.int8),
                reference_truth,
            ):
                raise AssertionError(
                    f"{architecture} received different held-out truth labels"
                )
            y_true = reference_truth
            y_score = test_scores
            average_precision = _safe_average_precision(y_true, y_score)
            test_weighted_fog_loss = _weighted_fog_loss(
                y_true, y_score, fog_positive_weight
            )
            curve = _compact_pr_curve(
                y_true, y_score, settings.max_pr_curve_points
            )
            curve.insert(0, "outer_fold", outer_fold)
            curve.insert(0, "model", architecture)
            pr_frames.append(curve)

            for operating_point, threshold in artifact["thresholds"].items():
                if not np.isfinite(threshold):
                    raise AssertionError("Every operating point needs a threshold")
                test_output = base_output.copy()
                test_output["AnyFoGPredicted"] = decode_endpoint_frame(
                    test_output, float(threshold), settings
                )
                y_pred = test_output["AnyFoGPredicted"].to_numpy(dtype=np.int8)
                tn, fp, fn, tp = confusion_matrix(
                    y_true, y_pred, labels=[0, 1]
                ).ravel()
                evaluation = evaluate_decoded_frame(test_output, settings)
                if evaluation.true_count != expected_true_episodes:
                    raise AssertionError("Episode truth denominator changed by model")
                if abs(evaluation.evaluation_minutes - expected_minutes) > 1e-9:
                    raise AssertionError("Evaluation duration changed by model")
                per_subject, macro = subject_episode_metrics(test_output, settings)
                per_subject.insert(0, "operating_point", operating_point)
                per_subject.insert(0, "outer_fold", outer_fold)
                per_subject.insert(0, "model", architecture)
                subject_frames.append(per_subject)

                fold_rows.append(
                    {
                        "model": architecture,
                        "operating_point": operating_point,
                        "outer_fold": outer_fold,
                        "test_subjects": len(partitions["test"]),
                        "selected_on_threshold": float(threshold),
                        "average_precision": average_precision,
                        "test_weighted_fog_loss": test_weighted_fog_loss,
                        "timestep_accuracy": float((tn + tp) / len(y_true)),
                        "timestep_balanced_accuracy": (
                            float(balanced_accuracy_score(y_true, y_pred))
                            if np.unique(y_true).size == 2
                            else float("nan")
                        ),
                        "timestep_precision": float(
                            precision_score(y_true, y_pred, zero_division=0)
                        ),
                        "timestep_recall": float(
                            recall_score(y_true, y_pred, zero_division=0)
                        ),
                        "timestep_f1": float(
                            f1_score(y_true, y_pred, zero_division=0)
                        ),
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                        "true_episodes": evaluation.true_count,
                        "predicted_episodes": evaluation.predicted_count,
                        "matched_episodes": evaluation.true_positive_count,
                        "episode_precision": evaluation.precision,
                        "episode_recall": evaluation.recall,
                        "episode_f1": evaluation.f1,
                        "false_alarms_per_minute": (
                            evaluation.false_alarms_per_minute
                        ),
                        "evaluation_minutes": evaluation.evaluation_minutes,
                        "mean_absolute_onset_error_seconds": (
                            evaluation.mean_absolute_onset_error_seconds
                        ),
                        "mean_matched_iou": evaluation.mean_duration_iou,
                        **macro,
                    }
                )
                confusion_rows.append(
                    {
                        "model": architecture,
                        "operating_point": operating_point,
                        "outer_fold": outer_fold,
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    }
                )
                if not evaluation.matches.empty:
                    matched = evaluation.matches.copy()
                    matched.insert(0, "operating_point", operating_point)
                    matched.insert(0, "outer_fold", outer_fold)
                    matched.insert(0, "model", architecture)
                    match_frames.append(matched)
                del test_output, y_pred
                gc.collect()

            del test_score_sum, test_scores, base_output, y_score
            gc.collect()

        del (
            training_segments,
            stopping_segments,
            calibration_segments,
            test_segments,
            training_references,
            stopping_references,
            calibration_references,
            test_references,
            reference_keys,
            reference_truth,
            artifacts,
            scaler,
            subject_weights,
        )
        tf.keras.backend.clear_session()
        gc.collect()

    history_frame = pd.concat(histories, ignore_index=True)
    fold_frame = pd.DataFrame(fold_rows)
    expected_fold_rows = (
        len(settings.folds_to_run) * len(ARCHITECTURES) * 2
    )
    if len(fold_frame) != expected_fold_rows:
        raise AssertionError(
            "Every model/fold must have both complete operating-point results"
        )
    decoder_frame = pd.concat(decoder_frames, ignore_index=True)
    subject_frame = pd.concat(subject_frames, ignore_index=True)
    _assert_subject_denominators(subject_frame)
    confusion_frame = pd.DataFrame(confusion_rows)
    pr_frame = pd.concat(pr_frames, ignore_index=True)
    if match_frames:
        match_frame = pd.concat(match_frames, ignore_index=True)
    else:
        match_frame = pd.DataFrame(
            columns=[
                "model",
                "outer_fold",
                "operating_point",
                "onset_delay_seconds",
                "absolute_onset_error_seconds",
                "iou",
            ]
        )
    parameter_frame = pd.DataFrame(
        {
            "model": list(model_parameters),
            "parameters": list(model_parameters.values()),
        }
    ).set_index("model")
    overall_frame = _summarise_overall_results(
        fold_frame, subject_frame, match_frame, model_parameters
    )
    partition_frame = pd.DataFrame(partition_rows)
    return BenchmarkResults(
        histories=history_frame,
        fold_results=fold_frame,
        overall_results=overall_frame,
        decoder_tables=decoder_frame,
        subject_results=subject_frame,
        confusion_counts=confusion_frame,
        pr_curves=pr_frame,
        model_parameters=parameter_frame,
        fold_partitions=partition_frame,
        matches=match_frame,
    )
