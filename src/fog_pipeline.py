"""Leakage-safe data preparation for the Parkinson's FoG notebooks.

The key invariant is that subjects are assigned to exactly one of train,
validation, or test before any learned preprocessing is fitted.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


TARGET_COLUMNS = ("StartHesitation", "Turn", "Walking")
ANY_FOG_COLUMN = "AnyFoG"
SENSOR_COLUMNS = ("AccV", "AccML", "AccAP")
SEGMENT_COLUMN = "ContiguousSegment"
ALIGNMENT_COLUMNS = ("RecordingId", "Subject", "Time", SEGMENT_COLUMN)
ROLLING_FEATURE_COLUMNS = tuple(
    f"Rolling_{sensor}_{stat}"
    for sensor in SENSOR_COLUMNS
    for stat in ("Mean", "Std")
)
FEATURE_COLUMNS = SENSOR_COLUMNS + ROLLING_FEATURE_COLUMNS + ("Acc_Magnitude",)


@dataclass(frozen=True)
class SubjectSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    random_state: int | None = None


@dataclass(frozen=True)
class AlignmentSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class PreparedTabularData:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    preprocessor: Pipeline
    engineered_splits: SubjectSplits
    alignment_splits: AlignmentSplits | None = None

    @property
    def y_any_fog_train(self) -> np.ndarray:
        return collapse_targets_to_any_fog(self.y_train)

    @property
    def y_any_fog_validation(self) -> np.ndarray:
        return collapse_targets_to_any_fog(self.y_validation)

    @property
    def y_any_fog_test(self) -> np.ndarray:
        return collapse_targets_to_any_fog(self.y_test)


@dataclass(frozen=True)
class PreparedSequenceData:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    preprocessor: Pipeline
    engineered_splits: SubjectSplits
    alignment_splits: AlignmentSplits | None = None

    @property
    def y_any_fog_train(self) -> np.ndarray:
        return collapse_targets_to_any_fog(self.y_train)

    @property
    def y_any_fog_validation(self) -> np.ndarray:
        return collapse_targets_to_any_fog(self.y_validation)

    @property
    def y_any_fog_test(self) -> np.ndarray:
        return collapse_targets_to_any_fog(self.y_test)


@dataclass(frozen=True)
class EpisodeEvaluation:
    true_episodes: pd.DataFrame
    predicted_episodes: pd.DataFrame
    matches: pd.DataFrame
    per_recording: pd.DataFrame
    true_count: int
    predicted_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    precision: float
    recall: float
    f1: float
    evaluation_minutes: float
    false_alarms_per_minute: float
    mean_onset_delay_seconds: float
    median_onset_delay_seconds: float
    mean_absolute_onset_error_seconds: float
    mean_duration_iou: float


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def collapse_targets_to_any_fog(labels: np.ndarray) -> np.ndarray:
    """Collapse the three event-type labels into one any-FoG target."""

    values = np.asarray(labels)
    if values.ndim != 2 or values.shape[1] != len(TARGET_COLUMNS):
        raise ValueError(
            "Event labels must have shape "
            f"(n_samples, {len(TARGET_COLUMNS)}); received {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Event labels must be finite")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Event labels must contain binary 0/1 values")
    return np.any(values == 1, axis=1, keepdims=True).astype(np.float32)


def add_any_fog_target(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with AnyFoG equal to the union of all event types."""

    _require_columns(data, TARGET_COLUMNS, "event-labelled data")
    frame = data.copy()
    expected = frame.loc[:, TARGET_COLUMNS].max(axis=1).astype(np.int8)
    if ANY_FOG_COLUMN in frame.columns:
        supplied = pd.to_numeric(frame[ANY_FOG_COLUMN], errors="coerce")
        if supplied.isna().any() or not np.array_equal(
            supplied.to_numpy(dtype=np.int8), expected.to_numpy(dtype=np.int8)
        ):
            raise ValueError(
                f"{ANY_FOG_COLUMN} must equal the union of {list(TARGET_COLUMNS)}"
            )
    frame[ANY_FOG_COLUMN] = expected
    return frame


def load_recordings(
    recordings_dir: str | Path,
    metadata_path: str | Path | None,
    *,
    limit_recordings: int | None = None,
    filter_valid_task: bool = True,
    assume_recording_is_subject: bool = False,
) -> pd.DataFrame:
    """Load recordings while retaining both recording and subject identity.

    Metadata should map each recording Id to a Subject. If a dataset genuinely
    contains exactly one subject per file, callers may explicitly set
    assume_recording_is_subject=True and use the filename stem as the group.
    """

    recordings_dir = Path(recordings_dir)

    if not recordings_dir.is_dir():
        raise FileNotFoundError(f"Recording directory not found: {recordings_dir}")
    recording_paths = sorted(recordings_dir.glob("*.csv"))
    if limit_recordings is not None:
        if limit_recordings < 1:
            raise ValueError("limit_recordings must be a positive integer or None")
        recording_paths = recording_paths[:limit_recordings]
    if not recording_paths:
        raise FileNotFoundError(f"No CSV recordings found in {recordings_dir}")

    if metadata_path is None:
        if not assume_recording_is_subject:
            raise FileNotFoundError(
                "No subject metadata was supplied. Set assume_recording_is_subject=True "
                "only when each file is known to contain a different subject."
            )
        subject_by_id = pd.Series(
            {path.stem: path.stem for path in recording_paths},
            name="Subject",
            dtype="object",
        )
        metadata_label = "filename-derived subject mapping"
    else:
        metadata_path = Path(metadata_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Subject metadata not found: {metadata_path}. "
                "Add metadata containing Id and Subject, or explicitly enable "
                "the one-subject-per-file assumption."
            )

        metadata = pd.read_csv(metadata_path)
        _require_columns(metadata, ("Id", "Subject"), str(metadata_path))
        metadata = metadata[["Id", "Subject"]].drop_duplicates()

        duplicated_ids = metadata[metadata["Id"].duplicated(keep=False)]["Id"].unique()
        if len(duplicated_ids):
            raise ValueError(
                "Each recording Id must map to one subject; duplicated metadata IDs: "
                f"{duplicated_ids[:10].tolist()}"
            )
        subject_by_id = metadata.set_index("Id")["Subject"]
        metadata_label = str(metadata_path)

    frames: list[pd.DataFrame] = []
    unmapped_ids: list[str] = []
    required_signal_columns = ("Time",) + SENSOR_COLUMNS + TARGET_COLUMNS

    for path in recording_paths:
        recording_id = path.stem
        if recording_id not in subject_by_id.index:
            unmapped_ids.append(recording_id)
            continue

        frame = pd.read_csv(path)
        _require_columns(frame, required_signal_columns, str(path))

        if filter_valid_task and {"Valid", "Task"}.issubset(frame.columns):
            frame = frame.loc[frame["Valid"].astype(bool) & frame["Task"].astype(bool)]

        target_frame = frame.loc[:, TARGET_COLUMNS]
        if target_frame.isna().any().any():
            missing_counts = target_frame.isna().sum()
            missing_counts = missing_counts[missing_counts > 0].to_dict()
            raise ValueError(
                f"Target labels must not be imputed; {path} contains missing labels: "
                f"{missing_counts}"
            )
        invalid_target_values = {
            target: sorted(target_frame.loc[~target_frame[target].isin((0, 1)), target].unique())
            for target in TARGET_COLUMNS
            if (~target_frame[target].isin((0, 1))).any()
        }
        if invalid_target_values:
            raise ValueError(
                f"Target labels must be binary 0/1 in {path}: {invalid_target_values}"
            )

        frame = frame.copy()
        frame["Time"] = pd.to_numeric(frame["Time"], downcast="integer")
        for sensor in SENSOR_COLUMNS:
            frame[sensor] = pd.to_numeric(frame[sensor], downcast="float")
        for target in TARGET_COLUMNS:
            frame[target] = pd.to_numeric(frame[target], downcast="integer")
        frame[ANY_FOG_COLUMN] = frame.loc[:, TARGET_COLUMNS].max(axis=1).astype(np.int8)
        frame["RecordingId"] = recording_id
        frame["Subject"] = subject_by_id.loc[recording_id]
        frames.append(frame)

    if unmapped_ids:
        preview = ", ".join(unmapped_ids[:10])
        raise ValueError(
            f"{len(unmapped_ids)} recordings have no subject mapping in "
            f"{metadata_label}: {preview}"
        )
    if not frames:
        raise ValueError("No rows remained after loading and validity filtering")

    combined = pd.concat(frames, ignore_index=True)
    combined["RecordingId"] = combined["RecordingId"].astype("category")
    combined["Subject"] = combined["Subject"].astype("category")
    return combined.sort_values(["RecordingId", "Time"], kind="stable").reset_index(drop=True)


def split_by_subject(
    data: pd.DataFrame,
    *,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
    require_all_targets: bool = True,
    max_split_attempts: int = 1000,
) -> SubjectSplits:
    """Create subject-disjoint train, validation, and test frames.

    validation_size is the fraction of the complete dataset, not the fraction
    of the post-test development partition.
    """

    _require_columns(data, ("Subject", "RecordingId"), "input data")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < validation_size < 1 - test_size:
        raise ValueError("validation_size must be positive and leave training data")
    if data["Subject"].nunique() < 3:
        raise ValueError("At least three subjects are required for a three-way split")

    if max_split_attempts < 1:
        raise ValueError("max_split_attempts must be positive")
    if require_all_targets:
        _require_columns(data, TARGET_COLUMNS, "split input")

    subject_values = pd.Series(data["Subject"].astype("object").unique())
    subject_frame = pd.DataFrame({"Subject": subject_values})
    relative_validation_size = validation_size / (1.0 - test_size)

    if require_all_targets:
        grouped = data.groupby("Subject", observed=True)
        summary = grouped[list(TARGET_COLUMNS)].sum()
        summary.insert(0, "_row_count", grouped.size())

        def has_all_outcomes(subjects: set) -> bool:
            totals = summary.loc[list(subjects)].sum()
            row_count = int(totals["_row_count"])
            return all(
                0 < float(totals[target]) < row_count for target in TARGET_COLUMNS
            )
    else:
        def has_all_outcomes(subjects: set) -> bool:
            return True

    selected_subjects = None
    selected_seed = None
    for attempt in range(max_split_attempts):
        candidate_seed = random_state + attempt
        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=candidate_seed,
        )
        development_index, test_index = next(
            outer.split(subject_frame, groups=subject_frame["Subject"])
        )
        development_subject_frame = subject_frame.iloc[development_index]
        test_subjects = set(subject_frame.iloc[test_index]["Subject"])

        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_validation_size,
            random_state=candidate_seed + 1,
        )
        train_index, validation_index = next(
            inner.split(
                development_subject_frame,
                groups=development_subject_frame["Subject"],
            )
        )
        train_subjects = set(
            development_subject_frame.iloc[train_index]["Subject"]
        )
        validation_subjects = set(
            development_subject_frame.iloc[validation_index]["Subject"]
        )

        if all(
            has_all_outcomes(subjects)
            for subjects in (train_subjects, validation_subjects, test_subjects)
        ):
            selected_subjects = (
                train_subjects,
                validation_subjects,
                test_subjects,
            )
            selected_seed = candidate_seed
            break

    if selected_subjects is None:
        raise ValueError(
            "Could not create subject-disjoint splits containing both classes for "
            f"every target after {max_split_attempts} deterministic attempts. "
            "Use more subjects or set require_all_targets=False only for a smoke test."
        )

    train_subjects, validation_subjects, test_subjects = selected_subjects
    splits = SubjectSplits(
        train=data.loc[data["Subject"].isin(train_subjects)].copy(),
        validation=data.loc[data["Subject"].isin(validation_subjects)].copy(),
        test=data.loc[data["Subject"].isin(test_subjects)].copy(),
        random_state=selected_seed,
    )
    assert_subject_disjoint(splits)
    return splits


def assert_subject_disjoint(splits: SubjectSplits) -> None:
    """Raise if any subject or recording occurs in more than one split."""

    named_frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    for name, frame in named_frames.items():
        _require_columns(frame, ("Subject", "RecordingId"), f"{name} split")

    for left_name, right_name in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        left = named_frames[left_name]
        right = named_frames[right_name]
        subject_overlap = set(left["Subject"]).intersection(right["Subject"])
        recording_overlap = set(left["RecordingId"]).intersection(right["RecordingId"])
        if subject_overlap or recording_overlap:
            raise AssertionError(
                f"Leakage between {left_name} and {right_name}: "
                f"subjects={sorted(subject_overlap)}, "
                f"recordings={sorted(recording_overlap)}"
            )


def _add_contiguous_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Number uninterrupted Time runs separately inside every recording."""

    _require_columns(data, ("RecordingId", "Time"), "segmentation input")
    frame = data.sort_values(["RecordingId", "Time"], kind="stable").copy()
    time_difference = frame.groupby(
        "RecordingId", sort=False, observed=True
    )["Time"].diff()
    new_segment = time_difference.ne(1)
    frame[SEGMENT_COLUMN] = (
        new_segment.groupby(frame["RecordingId"], observed=True).cumsum() - 1
    ).astype(np.int32)
    return frame


def add_sensor_features(data: pd.DataFrame, *, window_size: int = 5) -> pd.DataFrame:
    """Create causal sensor features, resetting at recordings and Time gaps."""

    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    _require_columns(
        data,
        ("RecordingId", "Time") + SENSOR_COLUMNS + TARGET_COLUMNS,
        "feature input",
    )

    frame = add_any_fog_target(_add_contiguous_segments(data))
    grouped = frame.groupby(
        ["RecordingId", SEGMENT_COLUMN], sort=False, observed=True
    )

    for sensor in SENSOR_COLUMNS:
        frame[f"Rolling_{sensor}_Mean"] = grouped[sensor].transform(
            lambda values: values.rolling(window_size, min_periods=window_size).mean()
        )
        frame[f"Rolling_{sensor}_Std"] = grouped[sensor].transform(
            lambda values: values.rolling(window_size, min_periods=window_size).std()
        )

    frame["Acc_Magnitude"] = np.sqrt(
        frame["AccV"].pow(2) + frame["AccML"].pow(2) + frame["AccAP"].pow(2)
    )
    frame.loc[:, FEATURE_COLUMNS] = frame.loc[:, FEATURE_COLUMNS].astype(np.float32)
    return frame


def build_preprocessor() -> Pipeline:
    """Return preprocessing whose learned state must be fitted on train only."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )


def prepare_tabular_splits(
    splits: SubjectSplits,
    *,
    window_size: int = 5,
) -> PreparedTabularData:
    """Engineer each split separately and fit preprocessing on train only."""

    assert_subject_disjoint(splits)
    if set(TARGET_COLUMNS).intersection(FEATURE_COLUMNS):
        raise AssertionError("Target columns must never be model features")

    engineered = SubjectSplits(
        train=add_sensor_features(splits.train, window_size=window_size),
        validation=add_sensor_features(splits.validation, window_size=window_size),
        test=add_sensor_features(splits.test, window_size=window_size),
        random_state=splits.random_state,
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(engineered.train.loc[:, FEATURE_COLUMNS])
    X_validation = preprocessor.transform(
        engineered.validation.loc[:, FEATURE_COLUMNS]
    )
    X_test = preprocessor.transform(engineered.test.loc[:, FEATURE_COLUMNS])

    return PreparedTabularData(
        X_train=np.asarray(X_train, dtype=np.float32),
        X_validation=np.asarray(X_validation, dtype=np.float32),
        X_test=np.asarray(X_test, dtype=np.float32),
        y_train=engineered.train.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32),
        y_validation=engineered.validation.loc[:, TARGET_COLUMNS].to_numpy(
            dtype=np.float32
        ),
        y_test=engineered.test.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32),
        preprocessor=preprocessor,
        engineered_splits=engineered,
        alignment_splits=AlignmentSplits(
            train=engineered.train.loc[:, ALIGNMENT_COLUMNS].reset_index(drop=True),
            validation=engineered.validation.loc[:, ALIGNMENT_COLUMNS].reset_index(
                drop=True
            ),
            test=engineered.test.loc[:, ALIGNMENT_COLUMNS].reset_index(drop=True),
        ),
    )


def _make_windows(
    frame: pd.DataFrame,
    preprocessor: Pipeline,
    *,
    timesteps: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    alignment_frames: list[pd.DataFrame] = []

    for _, recording in frame.groupby(
        ["RecordingId", SEGMENT_COLUMN], sort=False, observed=True
    ):
        recording = recording.sort_values("Time", kind="stable")
        X_recording = preprocessor.transform(recording.loc[:, FEATURE_COLUMNS])
        y_recording = recording.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)

        starts = np.arange(0, len(recording) - timesteps + 1, stride)
        if len(starts):
            window_indices = starts[:, None] + np.arange(timesteps)[None, :]
            X_windows.append(
                np.asarray(X_recording[window_indices], dtype=np.float32)
            )
            endpoint_indices = starts + timesteps - 1
            y_windows.append(y_recording[endpoint_indices])
            endpoints = recording.iloc[endpoint_indices].loc[:, ALIGNMENT_COLUMNS].copy()
            endpoints["WindowStartTime"] = recording.iloc[starts]["Time"].to_numpy()
            endpoints["WindowEndTime"] = endpoints["Time"].to_numpy()
            alignment_frames.append(endpoints.reset_index(drop=True))

    if not X_windows:
        return (
            np.empty((0, timesteps, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0, len(TARGET_COLUMNS)), dtype=np.float32),
            pd.DataFrame(
                columns=(*ALIGNMENT_COLUMNS, "WindowStartTime", "WindowEndTime")
            ),
        )
    return (
        np.concatenate(X_windows, axis=0),
        np.concatenate(y_windows, axis=0),
        pd.concat(alignment_frames, ignore_index=True),
    )


def prepare_sequence_splits(
    splits: SubjectSplits,
    *,
    timesteps: int = 5,
    stride: int = 5,
    evaluation_stride: int | None = None,
    feature_window_size: int = 5,
) -> PreparedSequenceData:
    """Create chronological, recording-bounded LSTM windows."""

    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")
    if evaluation_stride is None:
        evaluation_stride = stride
    if evaluation_stride < 1:
        raise ValueError("evaluation_stride must be positive")
    assert_subject_disjoint(splits)

    engineered = SubjectSplits(
        train=add_sensor_features(splits.train, window_size=feature_window_size),
        validation=add_sensor_features(
            splits.validation, window_size=feature_window_size
        ),
        test=add_sensor_features(splits.test, window_size=feature_window_size),
        random_state=splits.random_state,
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(engineered.train.loc[:, FEATURE_COLUMNS])

    X_train, y_train, train_alignment = _make_windows(
        engineered.train,
        preprocessor,
        timesteps=timesteps,
        stride=stride,
    )
    X_validation, y_validation, validation_alignment = _make_windows(
        engineered.validation,
        preprocessor,
        timesteps=timesteps,
        stride=evaluation_stride,
    )
    X_test, y_test, test_alignment = _make_windows(
        engineered.test,
        preprocessor,
        timesteps=timesteps,
        stride=evaluation_stride,
    )

    return PreparedSequenceData(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        preprocessor=preprocessor,
        engineered_splits=engineered,
        alignment_splits=AlignmentSplits(
            train=train_alignment,
            validation=validation_alignment,
            test=test_alignment,
        ),
    )


def split_summary(splits: SubjectSplits) -> pd.DataFrame:
    """Summarise split sizes and event prevalence without exposing test to training."""

    rows = []
    for name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        row = {
            "split": name,
            "subjects": int(frame["Subject"].nunique()),
            "recordings": int(frame["RecordingId"].nunique()),
            "rows": int(len(frame)),
        }
        for target in TARGET_COLUMNS:
            row[f"{target}_rate"] = float(frame[target].mean())
        row[f"{ANY_FOG_COLUMN}_rate"] = float(
            frame.loc[:, TARGET_COLUMNS].max(axis=1).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def _as_binary_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.isin(vector, (0, 1)).all():
        raise ValueError(f"{name} must contain binary 0/1 values")
    return vector.astype(np.int8)


def _sampling_rate_for_recording(
    recording_id: object,
    sampling_rate_hz: float | Mapping[str, float],
) -> float:
    if isinstance(sampling_rate_hz, Mapping):
        key = str(recording_id)
        if key not in sampling_rate_hz:
            raise ValueError(f"No sampling rate supplied for recording {key}")
        rate = float(sampling_rate_hz[key])
    else:
        rate = float(sampling_rate_hz)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sampling_rate_hz must be a positive finite value")
    return rate


def align_binary_predictions(
    alignment: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    sampling_rate_hz: float | Mapping[str, float],
) -> pd.DataFrame:
    """Align any-FoG labels and probabilities to evaluable recording times."""

    _require_columns(alignment, ("RecordingId", "Subject", "Time"), "alignment")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    truth = _as_binary_vector(y_true, name="y_true")
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if len(alignment) != len(truth) or len(alignment) != len(scores):
        raise ValueError("alignment, y_true, and y_score must have equal lengths")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("y_score must contain finite probabilities between 0 and 1")

    frame = alignment.reset_index(drop=True).copy()
    frame["Time"] = pd.to_numeric(frame["Time"], errors="raise")
    if frame.duplicated(["RecordingId", "Time"]).any():
        raise ValueError("alignment contains duplicate RecordingId/Time pairs")
    frame["AnyFoGTrue"] = truth
    frame["AnyFoGScore"] = scores
    frame["AnyFoGPredicted"] = (scores >= threshold).astype(np.int8)

    aligned_recordings: list[pd.DataFrame] = []
    for recording_id, recording in frame.groupby(
        "RecordingId", sort=False, observed=True
    ):
        recording = recording.sort_values("Time", kind="stable").copy()
        rate = _sampling_rate_for_recording(recording_id, sampling_rate_hz)
        time_differences = recording["Time"].diff()
        if (time_differences.dropna() <= 0).any():
            raise ValueError("Time must increase strictly within each recording")

        if SEGMENT_COLUMN not in recording.columns:
            recording[SEGMENT_COLUMN] = time_differences.ne(1).cumsum() - 1

        recording["TimeSeconds"] = recording["Time"] / rate
        recording["ExpectedStepSeconds"] = np.nan
        recording["CoverageSeconds"] = np.nan

        for _, segment in recording.groupby(
            SEGMENT_COLUMN, sort=False, observed=True
        ):
            segment_differences = segment["Time"].diff().dropna().to_numpy(dtype=float)
            positive_differences = segment_differences[segment_differences > 0]
            expected_ticks = (
                float(np.median(positive_differences))
                if len(positive_differences)
                else 1.0
            )
            expected_step_seconds = expected_ticks / rate
            next_difference_seconds = segment["Time"].shift(-1).sub(
                segment["Time"]
            ) / rate
            coverage_seconds = next_difference_seconds.clip(
                upper=expected_step_seconds * 1.5
            ).fillna(expected_step_seconds)
            recording.loc[segment.index, "ExpectedStepSeconds"] = expected_step_seconds
            recording.loc[segment.index, "CoverageSeconds"] = coverage_seconds

        aligned_recordings.append(recording)

    return pd.concat(aligned_recordings, ignore_index=True)


EPISODE_COLUMNS = (
    "episode_id",
    "RecordingId",
    "Subject",
    SEGMENT_COLUMN,
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "mean_score",
    "max_score",
)


def reconstruct_episodes(
    aligned: pd.DataFrame,
    *,
    label_column: str,
    score_column: str | None = None,
    merge_gap_seconds: float = 0.0,
    minimum_duration_seconds: float = 0.0,
) -> pd.DataFrame:
    """Convert positive time rows into recording-bounded half-open episodes."""

    required = (
        "RecordingId",
        "Subject",
        "TimeSeconds",
        "ExpectedStepSeconds",
        SEGMENT_COLUMN,
        label_column,
    )
    _require_columns(aligned, required, "aligned predictions")
    if score_column is not None:
        _require_columns(aligned, (score_column,), "aligned predictions")
    if merge_gap_seconds < 0 or minimum_duration_seconds < 0:
        raise ValueError("episode gap and duration parameters must be non-negative")

    label_values = _as_binary_vector(aligned[label_column], name=label_column)
    frame = aligned.copy()
    frame[label_column] = label_values
    frame = frame.sort_values(
        ["RecordingId", SEGMENT_COLUMN, "TimeSeconds"], kind="stable"
    ).reset_index(drop=True)
    positive = frame[label_column].to_numpy(dtype=np.int8) == 1
    if not positive.any():
        return pd.DataFrame(columns=EPISODE_COLUMNS)

    same_group_as_previous = (
        frame["RecordingId"].eq(frame["RecordingId"].shift())
        & frame[SEGMENT_COLUMN].eq(frame[SEGMENT_COLUMN].shift())
    ).to_numpy()
    previous_positive = np.r_[False, positive[:-1]]
    starts_positive_run = positive & (
        ~previous_positive | ~same_group_as_previous
    )
    run_ids = np.cumsum(starts_positive_run)
    positive_rows = frame.loc[positive].copy()
    positive_rows["_run_id"] = run_ids[positive]
    positive_rows["_end_seconds"] = (
        positive_rows["TimeSeconds"] + positive_rows["ExpectedStepSeconds"]
    )
    if score_column is None:
        positive_rows["_score_value"] = 0.0
        positive_rows["_score_count"] = 0
    else:
        positive_rows["_score_value"] = positive_rows[score_column].astype(float)
        positive_rows["_score_count"] = 1

    episodes = positive_rows.groupby("_run_id", sort=False, observed=True).agg(
        RecordingId=("RecordingId", "first"),
        Subject=("Subject", "first"),
        **{
            SEGMENT_COLUMN: (SEGMENT_COLUMN, "first"),
            "start_seconds": ("TimeSeconds", "first"),
            "end_seconds": ("_end_seconds", "last"),
            "_score_sum": ("_score_value", "sum"),
            "_score_count": ("_score_count", "sum"),
            "max_score": ("_score_value", "max"),
        },
    ).reset_index(drop=True)
    if score_column is None:
        episodes["mean_score"] = np.nan
        episodes["max_score"] = np.nan
    else:
        episodes["mean_score"] = (
            episodes["_score_sum"] / episodes["_score_count"]
        )

    if merge_gap_seconds > 0 and len(episodes) > 1:
        same_episode_group = (
            episodes["RecordingId"].eq(episodes["RecordingId"].shift())
            & episodes[SEGMENT_COLUMN].eq(episodes[SEGMENT_COLUMN].shift())
        )
        gaps = episodes["start_seconds"] - episodes["end_seconds"].shift()
        starts_merged_episode = ~same_episode_group | (
            gaps > merge_gap_seconds + 1e-12
        )
        episodes["_merged_id"] = starts_merged_episode.cumsum()
        episodes = episodes.groupby(
            "_merged_id", sort=False, observed=True
        ).agg(
            RecordingId=("RecordingId", "first"),
            Subject=("Subject", "first"),
            **{
                SEGMENT_COLUMN: (SEGMENT_COLUMN, "first"),
                "start_seconds": ("start_seconds", "first"),
                "end_seconds": ("end_seconds", "last"),
                "_score_sum": ("_score_sum", "sum"),
                "_score_count": ("_score_count", "sum"),
                "max_score": ("max_score", "max"),
            },
        ).reset_index(drop=True)
        episodes["mean_score"] = np.where(
            episodes["_score_count"] > 0,
            episodes["_score_sum"] / episodes["_score_count"],
            np.nan,
        )

    episodes["duration_seconds"] = (
        episodes["end_seconds"] - episodes["start_seconds"]
    )
    result = episodes.loc[
        episodes["duration_seconds"] + 1e-12 >= minimum_duration_seconds
    ].copy()
    if result.empty:
        return pd.DataFrame(columns=EPISODE_COLUMNS)
    result.insert(0, "episode_id", np.arange(len(result), dtype=int))
    return result.loc[:, EPISODE_COLUMNS].reset_index(drop=True)


MATCH_COLUMNS = (
    "RecordingId",
    "true_episode_id",
    "predicted_episode_id",
    "overlap_seconds",
    "iou",
    "onset_delay_seconds",
    "absolute_onset_error_seconds",
    "duration_error_seconds",
)


def match_episodes(
    true_episodes: pd.DataFrame,
    predicted_episodes: pd.DataFrame,
    *,
    minimum_iou: float = 0.25,
) -> pd.DataFrame:
    """Find cardinality-first, one-to-one temporal matches within recordings."""

    if not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be between 0 and 1")
    if true_episodes.empty or predicted_episodes.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    matches: list[dict] = []
    recording_ids = sorted(
        set(true_episodes["RecordingId"]).intersection(
            predicted_episodes["RecordingId"]
        ),
        key=str,
    )
    for recording_id in recording_ids:
        true_rows = true_episodes.loc[
            true_episodes["RecordingId"] == recording_id
        ].sort_values("start_seconds", kind="stable")
        predicted_rows = predicted_episodes.loc[
            predicted_episodes["RecordingId"] == recording_id
        ].sort_values("start_seconds", kind="stable")
        true_start = true_rows["start_seconds"].to_numpy(dtype=float)
        true_end = true_rows["end_seconds"].to_numpy(dtype=float)
        predicted_start = predicted_rows["start_seconds"].to_numpy(dtype=float)
        predicted_end = predicted_rows["end_seconds"].to_numpy(dtype=float)

        # Both episode sets contain non-overlapping, chronologically sorted
        # intervals. A two-pointer sweep therefore finds every overlap in
        # O(true + predicted + overlaps), avoiding a full Cartesian matrix.
        candidate_details: dict[tuple[int, int], tuple[float, float, float]] = {}
        true_to_predicted: dict[int, set[int]] = defaultdict(set)
        predicted_to_true: dict[int, set[int]] = defaultdict(set)
        true_index = 0
        predicted_index = 0
        while true_index < len(true_rows) and predicted_index < len(predicted_rows):
            if true_end[true_index] <= predicted_start[predicted_index]:
                true_index += 1
                continue
            if predicted_end[predicted_index] <= true_start[true_index]:
                predicted_index += 1
                continue

            overlap = max(
                0.0,
                min(true_end[true_index], predicted_end[predicted_index])
                - max(true_start[true_index], predicted_start[predicted_index]),
            )
            union = max(true_end[true_index], predicted_end[predicted_index]) - min(
                true_start[true_index], predicted_start[predicted_index]
            )
            iou = overlap / union if union > 0 else 0.0
            onset_delay = (
                predicted_start[predicted_index] - true_start[true_index]
            )
            if overlap > 0 and iou + 1e-12 >= minimum_iou:
                pair = (true_index, predicted_index)
                candidate_details[pair] = (overlap, iou, onset_delay)
                true_to_predicted[true_index].add(predicted_index)
                predicted_to_true[predicted_index].add(true_index)

            if true_end[true_index] < predicted_end[predicted_index]:
                true_index += 1
            elif predicted_end[predicted_index] < true_end[true_index]:
                predicted_index += 1
            else:
                true_index += 1
                predicted_index += 1

        # Solve each connected overlap component independently. This preserves
        # cardinality-first optimal assignment without allocating huge sparse
        # matrices for unrelated episodes in the same long recording.
        unvisited_true = set(true_to_predicted)
        while unvisited_true:
            component_true: set[int] = set()
            component_predicted: set[int] = set()
            queue = deque([("true", min(unvisited_true))])
            while queue:
                node_type, node_index = queue.popleft()
                if node_type == "true":
                    if node_index in component_true:
                        continue
                    component_true.add(node_index)
                    unvisited_true.discard(node_index)
                    queue.extend(
                        ("predicted", index)
                        for index in sorted(true_to_predicted[node_index])
                    )
                else:
                    if node_index in component_predicted:
                        continue
                    component_predicted.add(node_index)
                    queue.extend(
                        ("true", index)
                        for index in sorted(predicted_to_true[node_index])
                    )

            true_component = sorted(component_true)
            predicted_component = sorted(component_predicted)
            score_matrix = np.zeros(
                (len(true_component), len(predicted_component)), dtype=float
            )
            valid_matrix = np.zeros_like(score_matrix, dtype=bool)
            for component_true_index, original_true_index in enumerate(
                true_component
            ):
                for component_predicted_index, original_predicted_index in enumerate(
                    predicted_component
                ):
                    pair = (original_true_index, original_predicted_index)
                    if pair not in candidate_details:
                        continue
                    _, iou, onset_delay = candidate_details[pair]
                    valid_matrix[
                        component_true_index, component_predicted_index
                    ] = True
                    score_matrix[
                        component_true_index, component_predicted_index
                    ] = (
                        1_000_000.0
                        + iou * 1_000.0
                        - abs(onset_delay) * 1e-3
                        - (
                            original_true_index * len(predicted_rows)
                            + original_predicted_index
                        )
                        * 1e-9
                    )

            true_assignment, predicted_assignment = linear_sum_assignment(
                -score_matrix
            )
            for assigned_true, assigned_predicted in zip(
                true_assignment, predicted_assignment
            ):
                if not valid_matrix[assigned_true, assigned_predicted]:
                    continue
                original_true_index = true_component[assigned_true]
                original_predicted_index = predicted_component[assigned_predicted]
                truth = true_rows.iloc[original_true_index]
                prediction = predicted_rows.iloc[original_predicted_index]
                overlap, iou, onset_delay = candidate_details[
                    (original_true_index, original_predicted_index)
                ]
                matches.append(
                    {
                        "RecordingId": recording_id,
                        "true_episode_id": int(truth["episode_id"]),
                        "predicted_episode_id": int(prediction["episode_id"]),
                        "overlap_seconds": overlap,
                        "iou": iou,
                        "onset_delay_seconds": onset_delay,
                        "absolute_onset_error_seconds": abs(onset_delay),
                        "duration_error_seconds": (
                            prediction["duration_seconds"]
                            - truth["duration_seconds"]
                        ),
                    }
                )

    return pd.DataFrame(matches, columns=MATCH_COLUMNS).sort_values(
        ["RecordingId", "true_episode_id"], kind="stable", ignore_index=True
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _episode_f1(true_positive: int, predicted_count: int, true_count: int) -> float:
    if predicted_count == 0 and true_count == 0:
        return float("nan")
    if true_positive == 0:
        return 0.0
    precision = true_positive / predicted_count
    recall = true_positive / true_count
    return float(2 * precision * recall / (precision + recall))


def evaluate_episode_predictions(
    alignment: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    sampling_rate_hz: float | Mapping[str, float],
    minimum_iou: float = 0.25,
    merge_gap_seconds: float = 0.0,
    minimum_predicted_duration_seconds: float = 0.0,
    _aligned: pd.DataFrame | None = None,
    _true_episodes: pd.DataFrame | None = None,
) -> EpisodeEvaluation:
    """Evaluate a frozen any-FoG detector at the episode level."""

    if _aligned is None:
        aligned = align_binary_predictions(
            alignment,
            y_true,
            y_score,
            threshold=threshold,
            sampling_rate_hz=sampling_rate_hz,
        )
    else:
        aligned = _aligned.copy()
        aligned["AnyFoGPredicted"] = (
            aligned["AnyFoGScore"].to_numpy(dtype=float) >= threshold
        ).astype(np.int8)
    true_episodes = (
        reconstruct_episodes(aligned, label_column="AnyFoGTrue")
        if _true_episodes is None
        else _true_episodes
    )
    predicted_episodes = reconstruct_episodes(
        aligned,
        label_column="AnyFoGPredicted",
        score_column="AnyFoGScore",
        merge_gap_seconds=merge_gap_seconds,
        minimum_duration_seconds=minimum_predicted_duration_seconds,
    )
    matches = match_episodes(
        true_episodes,
        predicted_episodes,
        minimum_iou=minimum_iou,
    )

    true_count = int(len(true_episodes))
    predicted_count = int(len(predicted_episodes))
    true_positive_count = int(len(matches))
    false_positive_count = predicted_count - true_positive_count
    false_negative_count = true_count - true_positive_count
    evaluation_minutes = float(aligned["CoverageSeconds"].sum() / 60.0)
    precision = _safe_ratio(true_positive_count, predicted_count)
    recall = _safe_ratio(true_positive_count, true_count)
    f1 = _episode_f1(true_positive_count, predicted_count, true_count)
    false_alarms_per_minute = _safe_ratio(
        false_positive_count, evaluation_minutes
    )

    if matches.empty:
        mean_onset_delay = float("nan")
        median_onset_delay = float("nan")
        mean_absolute_onset_error = float("nan")
        mean_duration_iou = float("nan")
    else:
        mean_onset_delay = float(matches["onset_delay_seconds"].mean())
        median_onset_delay = float(matches["onset_delay_seconds"].median())
        mean_absolute_onset_error = float(
            matches["absolute_onset_error_seconds"].mean()
        )
        mean_duration_iou = float(matches["iou"].mean())

    per_recording_rows = []
    for recording_id, recording in aligned.groupby(
        "RecordingId", sort=False, observed=True
    ):
        recording_true = true_episodes.loc[
            true_episodes["RecordingId"] == recording_id
        ]
        recording_predicted = predicted_episodes.loc[
            predicted_episodes["RecordingId"] == recording_id
        ]
        recording_matches = matches.loc[matches["RecordingId"] == recording_id]
        recording_minutes = float(recording["CoverageSeconds"].sum() / 60.0)
        recording_tp = int(len(recording_matches))
        recording_fp = int(len(recording_predicted) - recording_tp)
        recording_true_count = int(len(recording_true))
        recording_predicted_count = int(len(recording_predicted))
        per_recording_rows.append(
            {
                "RecordingId": recording_id,
                "true_episodes": recording_true_count,
                "predicted_episodes": recording_predicted_count,
                "matched_episodes": recording_tp,
                "episode_precision": _safe_ratio(
                    recording_tp, recording_predicted_count
                ),
                "episode_recall": _safe_ratio(recording_tp, recording_true_count),
                "episode_f1": _episode_f1(
                    recording_tp, recording_predicted_count, recording_true_count
                ),
                "evaluation_minutes": recording_minutes,
                "false_alarms_per_minute": _safe_ratio(
                    recording_fp, recording_minutes
                ),
            }
        )

    return EpisodeEvaluation(
        true_episodes=true_episodes,
        predicted_episodes=predicted_episodes,
        matches=matches,
        per_recording=pd.DataFrame(per_recording_rows),
        true_count=true_count,
        predicted_count=predicted_count,
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        precision=precision,
        recall=recall,
        f1=f1,
        evaluation_minutes=evaluation_minutes,
        false_alarms_per_minute=false_alarms_per_minute,
        mean_onset_delay_seconds=mean_onset_delay,
        median_onset_delay_seconds=median_onset_delay,
        mean_absolute_onset_error_seconds=mean_absolute_onset_error,
        mean_duration_iou=mean_duration_iou,
    )


def select_episode_threshold(
    alignment: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    sampling_rate_hz: float | Mapping[str, float],
    thresholds: Sequence[float] | None = None,
    minimum_iou: float = 0.25,
    merge_gap_seconds: float = 0.0,
    minimum_predicted_duration_seconds: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    """Choose a probability threshold using validation episode F1 only."""

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    candidate_thresholds = np.asarray(list(thresholds), dtype=float)
    if not len(candidate_thresholds):
        raise ValueError("At least one threshold is required")
    if (
        not np.isfinite(candidate_thresholds).all()
        or ((candidate_thresholds < 0) | (candidate_thresholds > 1)).any()
    ):
        raise ValueError("All thresholds must be between 0 and 1")

    rows = []
    aligned = align_binary_predictions(
        alignment,
        y_true,
        y_score,
        threshold=0.0,
        sampling_rate_hz=sampling_rate_hz,
    )
    true_episodes = reconstruct_episodes(aligned, label_column="AnyFoGTrue")
    for threshold in candidate_thresholds:
        evaluation = evaluate_episode_predictions(
            alignment,
            y_true,
            y_score,
            threshold=float(threshold),
            sampling_rate_hz=sampling_rate_hz,
            minimum_iou=minimum_iou,
            merge_gap_seconds=merge_gap_seconds,
            minimum_predicted_duration_seconds=minimum_predicted_duration_seconds,
            _aligned=aligned,
            _true_episodes=true_episodes,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "episode_precision": evaluation.precision,
                "episode_recall": evaluation.recall,
                "episode_f1": evaluation.f1,
                "false_alarms_per_minute": evaluation.false_alarms_per_minute,
                "mean_duration_iou": evaluation.mean_duration_iou,
                "true_episodes": evaluation.true_count,
                "predicted_episodes": evaluation.predicted_count,
                "matched_episodes": evaluation.true_positive_count,
            }
        )

    results = pd.DataFrame(rows).sort_values("threshold", ignore_index=True)
    ranked = results.assign(
        _f1=results["episode_f1"].fillna(-np.inf),
        _false_alarms=results["false_alarms_per_minute"].fillna(np.inf),
        _recall=results["episode_recall"].fillna(-np.inf),
        _iou=results["mean_duration_iou"].fillna(-np.inf),
    ).sort_values(
        ["_f1", "_false_alarms", "_recall", "_iou", "threshold"],
        ascending=[False, True, False, False, False],
        kind="stable",
    )
    return float(ranked.iloc[0]["threshold"]), results
