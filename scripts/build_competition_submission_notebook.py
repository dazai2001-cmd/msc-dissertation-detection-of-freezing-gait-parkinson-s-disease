"""Build the self-contained Kaggle FoG competition submission notebook.

The generated notebook deliberately contains its complete data and modelling
pipeline so it can be uploaded to Kaggle without the local ``src`` package.
"""

from build_notebooks import code, markdown, save_notebook


def build_notebook() -> None:
    cells = [
        markdown(
            r"""
            # Parkinson's Freezing of Gait — competition submission

            ## Goal

            Train leakage-safe models for the Kaggle competition and create the
            required `submission.csv` containing continuous confidence scores for
            `StartHesitation`, `Turn`, and `Walking` at every original test timestep.

            This is a separate competition workflow. It does not modify the dissertation
            comparison notebooks or their saved results.

            ### What is upgraded here

            - subjects, not rows or filenames, define cross-validation folds;
            - any recording or subject found in test is excluded from training;
            - DeFOG labels contribute to loss and validation only where both `Valid`
              and `Task` are true, while the complete acceleration stream remains input
              context;
            - DeFOG (100 Hz) and TDCS FoG (128 Hz) use separate scalers and models;
            - causal acceleration, jerk, dynamic-motion, RMS, and frequency-power
              features are learned over four seconds of context;
            - a compact CNN–LSTM and causal TCN are ensembled using subject-disjoint
              out-of-fold (OOF) predictions;
            - training uses focal class weighting, hard-negative-aware sampling,
              AdamW, noise/dropout regularisation, exact-AP early stopping, and GPU
              mixed precision;
            - only OOF-selected continuous smoothing is allowed—there is no episode
              threshold or binary decoder because the competition scores confidence
              rankings;
            - OOF Platt scaling aligns TDCS and DeFOG score scales before the pooled
              competition metric is calculated;
            - predictions are restored to every native timestep and reindexed to the
              competition sample IDs before strict submission checks.

            The default `competition` profile is designed for Kaggle GPU execution
            under the nine-hour limit. Use `smoke` only to test the mechanics; never
            submit its output as a serious entry.
            """
        ),
        markdown("## Setup"),
        code(
            r"""
            from pathlib import Path
            from dataclasses import dataclass
            import gc
            import math
            import os
            import random
            import time
            import warnings

            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import display
            from scipy.signal import butter, lfilter, sosfilt
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import average_precision_score, precision_recall_curve
            from sklearn.preprocessing import StandardScaler
            import tensorflow as tf

            warnings.filterwarnings("ignore", category=FutureWarning)
            pd.set_option("display.max_columns", 50)

            TARGET_COLUMNS = ["StartHesitation", "Turn", "Walking"]
            SENSOR_COLUMNS = ["AccV", "AccML", "AccAP"]
            DATASET_SPECS = {
                "defog": {"native_hz": 100.0, "metadata": "defog_metadata.csv"},
                "tdcsfog": {"native_hz": 128.0, "metadata": "tdcsfog_metadata.csv"},
            }
            RANDOM_SEED = 42
            RUN_PROFILE = "competition"  # change to "smoke" only for a quick mechanics test

            PROFILES = {
                "competition": {
                    "target_hz": 25.0,
                    "window_seconds": 4.0,
                    "n_folds": 3,
                    "architectures": ("cnn_lstm", "tcn"),
                    "epochs": 28,
                    "early_stopping_patience": 5,
                    "batch_size": 512,
                    "max_train_windows": 450_000,
                    "positive_window_fraction": 0.70,
                    "max_monitor_windows": 100_000,
                    "validation_stride": 2,
                    "negative_to_positive_ratio": 4.0,
                    "hard_negative_fraction": 0.50,
                    "negative_windows_for_no_fog_recording": 2_000,
                    "focal_gamma": 1.5,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-4,
                    "max_files_per_dataset": None,
                },
                "smoke": {
                    "target_hz": 25.0,
                    "window_seconds": 4.0,
                    "n_folds": 2,
                    "architectures": ("cnn_lstm",),
                    "epochs": 2,
                    "early_stopping_patience": 1,
                    "batch_size": 256,
                    "max_train_windows": 12_000,
                    "positive_window_fraction": 0.70,
                    "max_monitor_windows": 4_000,
                    "validation_stride": 4,
                    "negative_to_positive_ratio": 3.0,
                    "hard_negative_fraction": 0.50,
                    "negative_windows_for_no_fog_recording": 400,
                    "focal_gamma": 1.5,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-4,
                    "max_files_per_dataset": 20,
                },
            }
            SETTINGS = PROFILES[RUN_PROFILE]
            WINDOW_SAMPLES = int(round(
                SETTINGS["window_seconds"] * SETTINGS["target_hz"]
            ))

            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            tf.keras.utils.set_random_seed(RANDOM_SEED)
            try:
                tf.config.experimental.enable_op_determinism()
            except (AttributeError, RuntimeError):
                pass

            GPU_DEVICES = tf.config.list_physical_devices("GPU")
            for device in GPU_DEVICES:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except RuntimeError:
                    pass
            if GPU_DEVICES:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("GPU enabled:", GPU_DEVICES)
                print("Mixed-precision policy:", tf.keras.mixed_precision.global_policy())
            else:
                print("WARNING: no GPU detected; the competition profile may exceed 9 hours.")

            print("TensorFlow:", tf.__version__)
            print("Run profile:", RUN_PROFILE)
            print("Window:", WINDOW_SAMPLES, "samples at", SETTINGS["target_hz"], "Hz")
            """
        ),
        markdown("### 1. Locate Kaggle or local inputs"),
        code(
            r"""
            @dataclass(frozen=True)
            class DataLayout:
                input_root: Path
                train_root: Path
                test_root: Path
                metadata_root: Path
                sample_path: Path
                output_path: Path
                work_root: Path
                environment: str


            def resolve_data_layout() -> DataLayout:
                # The first path exactly matches the competition mount shown in Kaggle.
                kaggle_candidates = [
                    Path("/kaggle/input/competitions/tlvmc-parkinsons-freezing-gait-prediction"),
                    Path("/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction"),
                ]
                for root in kaggle_candidates:
                    if (
                        (root / "train" / "defog").exists()
                        and (root / "test" / "tdcsfog").exists()
                        and (root / "sample_submission.csv").exists()
                    ):
                        return DataLayout(
                            input_root=root,
                            train_root=root / "train",
                            test_root=root / "test",
                            metadata_root=root,
                            sample_path=root / "sample_submission.csv",
                            output_path=Path("/kaggle/working/submission.csv"),
                            work_root=Path("/kaggle/working/fog_competition_work"),
                            environment="kaggle",
                        )

                for candidate in (Path.cwd(), *Path.cwd().parents):
                    raw_root = candidate / "data" / "raw"
                    if (
                        (raw_root / "train" / "defog").exists()
                        and (raw_root / "test" / "tdcsfog").exists()
                    ):
                        return DataLayout(
                            input_root=candidate,
                            train_root=raw_root / "train",
                            test_root=raw_root / "test",
                            metadata_root=candidate / "data" / "metadata",
                            sample_path=(
                                candidate / "data" / "submissions" / "sample_submission.csv"
                            ),
                            output_path=(
                                candidate / "data" / "submissions" / "submission.csv"
                            ),
                            work_root=candidate / "results" / "competition_submission",
                            environment="local",
                        )
                raise FileNotFoundError(
                    "Competition data was not found in either Kaggle mount or the local project."
                )


            LAYOUT = resolve_data_layout()
            LAYOUT.work_root.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_ROOT = LAYOUT.work_root / "checkpoints"
            CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

            expected_sample_columns = ["Id", *TARGET_COLUMNS]
            sample_columns = pd.read_csv(LAYOUT.sample_path, nrows=0).columns.tolist()
            if sample_columns != expected_sample_columns:
                raise ValueError(
                    f"Unexpected sample-submission columns: {sample_columns}"
                )

            # SECURITY / LEAKAGE GUARD: only Id is loaded. Sample target values are ignored.
            sample_ids = pd.read_csv(
                LAYOUT.sample_path, usecols=["Id"], dtype={"Id": "string"}
            )
            if not sample_ids["Id"].is_unique:
                raise ValueError("sample_submission.csv contains duplicate Id values")

            inventory_rows = []
            for dataset_name in DATASET_SPECS:
                train_files = sorted((LAYOUT.train_root / dataset_name).glob("*.csv"))
                test_files = sorted((LAYOUT.test_root / dataset_name).glob("*.csv"))
                inventory_rows.append(
                    {
                        "dataset": dataset_name,
                        "train_files": len(train_files),
                        "test_files": len(test_files),
                    }
                )
            display(pd.DataFrame(inventory_rows))
            print("Environment:", LAYOUT.environment)
            print("Input root:", LAYOUT.input_root)
            print("Submission will be written to:", LAYOUT.output_path)
            print("Sample rows (Id only):", f"{len(sample_ids):,}")
            """
        ),
        markdown("## Data preparation"),
        markdown(
            r"""
            The two datasets are resampled independently to 25 Hz for tractable GPU
            training. Predictions are later interpolated back to every original `Time`
            value before submission. Sensor features are causal; the final interpolation
            is an offline competition step.

            DeFOG `Valid`/`Task` values are **masks**, not negative labels. Complete
            recordings remain available as input context, while masked endpoints never
            enter loss or OOF scoring. No subject identifier is supplied to the model.
            """
        ),
        code(
            r"""
            FEATURE_NAMES = [
                "AccV", "AccML", "AccAP", "AccMagnitude",
                "JerkV", "JerkML", "JerkAP", "JerkMagnitude",
                "DynamicV", "DynamicML", "DynamicAP", "DynamicRMS",
                "FreezeBandPower", "FreezeToLocomotorRatio", "MedicationOn",
            ]
            FREEZE_RATIO_FEATURE_INDEX = FEATURE_NAMES.index("FreezeToLocomotorRatio")


            def clean_sensor_values(frame: pd.DataFrame) -> np.ndarray:
                values = (
                    frame.loc[:, SENSOR_COLUMNS]
                    .replace([np.inf, -np.inf], np.nan)
                    .ffill()
                    .bfill()
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
                return values


            def causal_block_downsample(values: np.ndarray, native_hz: float, target_hz: float):
                if len(values) == 0:
                    raise ValueError("An empty recording cannot be downsampled")
                positions = np.arange(0.0, len(values), native_hz / target_hz)
                endpoint_indices = np.unique(
                    np.clip(np.rint(positions).astype(np.int64), 0, len(values) - 1)
                )
                if endpoint_indices[-1] != len(values) - 1:
                    endpoint_indices = np.append(endpoint_indices, len(values) - 1)

                starts = np.r_[0, endpoint_indices[:-1] + 1]
                cumulative = np.vstack(
                    [np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)]
                )
                counts = (endpoint_indices - starts + 1).astype(np.float64)[:, None]
                block_means = (
                    cumulative[endpoint_indices + 1] - cumulative[starts]
                ) / counts
                return endpoint_indices, block_means


            def causal_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
                return (
                    pd.DataFrame(values)
                    .rolling(window=max(1, int(window)), min_periods=1)
                    .mean()
                    .to_numpy(dtype=np.float64)
                )


            def engineer_features(
                downsampled_axes: np.ndarray,
                target_hz: float,
                medication_on: float,
            ) -> np.ndarray:
                axes = downsampled_axes.astype(np.float64, copy=False)
                magnitude = np.linalg.norm(axes, axis=1)

                jerk_axes = np.vstack([np.zeros((1, 3)), np.diff(axes, axis=0)]) * target_hz
                jerk_magnitude = np.linalg.norm(jerk_axes, axis=1)

                slow_axes = causal_rolling_mean(axes, round(target_hz))
                dynamic_axes = axes - slow_axes
                dynamic_magnitude = np.linalg.norm(dynamic_axes, axis=1)
                dynamic_rms = np.sqrt(
                    causal_rolling_mean(
                        np.square(dynamic_magnitude)[:, None], round(target_hz)
                    )[:, 0]
                )

                freeze_sos = butter(
                    4, [3.0, 8.0], btype="bandpass", fs=target_hz, output="sos"
                )
                locomotor_sos = butter(
                    4, [0.5, 3.0], btype="bandpass", fs=target_hz, output="sos"
                )
                freeze_signal = sosfilt(freeze_sos, dynamic_magnitude)
                locomotor_signal = sosfilt(locomotor_sos, dynamic_magnitude)
                power_window = max(2, int(round(target_hz)))
                freeze_power = causal_rolling_mean(
                    np.square(freeze_signal)[:, None], power_window
                )[:, 0]
                locomotor_power = causal_rolling_mean(
                    np.square(locomotor_signal)[:, None], power_window
                )[:, 0]
                freeze_ratio = np.log1p(
                    freeze_power / np.maximum(locomotor_power, 1e-8)
                )

                metadata_feature = np.full((len(axes), 1), medication_on)
                features = np.column_stack(
                    [
                        axes,
                        magnitude,
                        jerk_axes,
                        jerk_magnitude,
                        dynamic_axes,
                        dynamic_rms,
                        freeze_power,
                        freeze_ratio,
                        metadata_feature,
                    ]
                ).astype(np.float32)
                if features.shape[1] != len(FEATURE_NAMES):
                    raise AssertionError((features.shape, FEATURE_NAMES))
                if not np.isfinite(features).all():
                    raise ValueError("Feature engineering produced non-finite values")
                return features


            def medication_value(metadata_row: pd.Series) -> float:
                value = str(metadata_row.get("Medication", "unknown")).strip().lower()
                if value == "on":
                    return 1.0
                if value == "off":
                    return 0.0
                return 0.5


            def read_recording(
                path: Path,
                dataset_name: str,
                native_hz: float,
                metadata_row: pd.Series,
                is_train: bool,
            ) -> dict:
                frame = pd.read_csv(path)
                required = ["Time", *SENSOR_COLUMNS]
                if is_train:
                    required += TARGET_COLUMNS
                    if dataset_name == "defog":
                        required += ["Valid", "Task"]
                missing = sorted(set(required) - set(frame.columns))
                if missing:
                    raise ValueError(f"{path.name} is missing columns {missing}")
                if frame["Time"].duplicated().any():
                    raise ValueError(f"{path.name} contains duplicate Time values")
                if not frame["Time"].is_monotonic_increasing:
                    raise ValueError(f"{path.name} is not chronological")

                native_time = frame["Time"].to_numpy(dtype=np.int64)
                sensor_values = clean_sensor_values(frame)
                endpoints, downsampled_axes = causal_block_downsample(
                    sensor_values, native_hz, SETTINGS["target_hz"]
                )
                features = engineer_features(
                    downsampled_axes,
                    SETTINGS["target_hz"],
                    medication_value(metadata_row),
                )

                if is_train:
                    native_y = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.int8)
                    if dataset_name == "defog":
                        native_mask = (
                            frame["Valid"].astype(bool).to_numpy()
                            & frame["Task"].astype(bool).to_numpy()
                        )
                    else:
                        native_mask = np.ones(len(frame), dtype=bool)
                    down_y = native_y[endpoints]
                    down_mask = native_mask[endpoints]
                else:
                    native_y = None
                    native_mask = np.ones(len(frame), dtype=bool)
                    down_y = None
                    down_mask = np.ones(len(endpoints), dtype=bool)

                return {
                    "dataset": dataset_name,
                    "recording_id": path.stem,
                    "subject": str(metadata_row["Subject"]),
                    "features": features,
                    "down_time": native_time[endpoints],
                    "down_y": down_y,
                    "down_mask": down_mask,
                    "native_time": native_time,
                    "native_y": native_y,
                    "native_mask": native_mask,
                }


            def load_dataset(dataset_name: str):
                spec = DATASET_SPECS[dataset_name]
                metadata_path = LAYOUT.metadata_root / spec["metadata"]
                metadata = pd.read_csv(metadata_path, dtype={"Id": "string", "Subject": "string"})
                if metadata["Id"].duplicated().any():
                    raise ValueError(f"Duplicate recording IDs in {metadata_path.name}")
                metadata_by_id = metadata.set_index("Id", drop=False)

                train_paths = sorted((LAYOUT.train_root / dataset_name).glob("*.csv"))
                test_paths = sorted((LAYOUT.test_root / dataset_name).glob("*.csv"))
                if not train_paths or not test_paths:
                    raise FileNotFoundError(f"Missing {dataset_name} train or test CSV files")
                missing_metadata = [
                    path.stem for path in [*train_paths, *test_paths]
                    if path.stem not in metadata_by_id.index
                ]
                if missing_metadata:
                    raise ValueError(
                        f"{dataset_name} recordings missing authoritative metadata: "
                        f"{missing_metadata[:10]}"
                    )

                test_ids = {path.stem for path in test_paths}
                test_subjects = {
                    str(metadata_by_id.loc[recording_id, "Subject"])
                    for recording_id in test_ids
                }
                safe_train_paths = []
                excluded = []
                for path in train_paths:
                    subject = str(metadata_by_id.loc[path.stem, "Subject"])
                    if path.stem in test_ids or subject in test_subjects:
                        excluded.append((path.stem, subject))
                    else:
                        safe_train_paths.append(path)

                max_files = SETTINGS["max_files_per_dataset"]
                if max_files is not None:
                    # Deterministic smoke subset, chosen only after leakage exclusions.
                    safe_train_paths = safe_train_paths[:max_files]

                print(
                    f"{dataset_name}: {len(safe_train_paths)} safe train files, "
                    f"{len(test_paths)} test files"
                )
                if excluded:
                    print(
                        f"LEAKAGE GUARD: excluded {len(excluded)} train recordings "
                        "sharing a test recording ID or subject. First entries:",
                        excluded[:5],
                    )

                def load_paths(paths, is_train):
                    records = []
                    for number, path in enumerate(paths, start=1):
                        records.append(
                            read_recording(
                                path,
                                dataset_name,
                                spec["native_hz"],
                                metadata_by_id.loc[path.stem],
                                is_train,
                            )
                        )
                        if number % 50 == 0 or number == len(paths):
                            print(f"  loaded {number}/{len(paths)} {'train' if is_train else 'test'} files")
                    return records

                train_records = load_paths(safe_train_paths, True)
                test_records = load_paths(test_paths, False)
                return train_records, test_records, excluded


            def dataset_summary(records: list[dict]) -> pd.DataFrame:
                valid_y = np.concatenate(
                    [record["native_y"][record["native_mask"]] for record in records],
                    axis=0,
                )
                return pd.DataFrame(
                    {
                        "recordings": [len(records)],
                        "subjects": [len({record["subject"] for record in records})],
                        "native_rows": [sum(len(record["native_time"]) for record in records)],
                        "scored_rows": [len(valid_y)],
                        **{
                            f"{target}_rate": [float(valid_y[:, index].mean())]
                            for index, target in enumerate(TARGET_COLUMNS)
                        },
                    }
                )
            """
        ),
        markdown("## Subject folds and streaming windows"),
        code(
            r"""
            def make_subject_folds(records: list[dict], n_folds: int, seed: int):
                subject_vectors = {}
                for record in records:
                    valid_y = record["native_y"][record["native_mask"]]
                    vector = np.r_[len(valid_y), valid_y.sum(axis=0)].astype(np.float64)
                    subject_vectors.setdefault(record["subject"], np.zeros(4, dtype=np.float64))
                    subject_vectors[record["subject"]] += vector

                subjects = list(subject_vectors)
                if len(subjects) < n_folds:
                    raise ValueError(
                        f"Need at least {n_folds} subjects, found {len(subjects)}"
                    )
                rng = np.random.default_rng(seed)
                rng.shuffle(subjects)
                global_total = np.sum([subject_vectors[s] for s in subjects], axis=0)
                target = np.maximum(global_total / n_folds, 1.0)
                subjects.sort(
                    key=lambda s: np.max(subject_vectors[s] / target), reverse=True
                )

                fold_totals = np.zeros((n_folds, 4), dtype=np.float64)
                fold_subject_counts = np.zeros(n_folds, dtype=np.int64)
                assignment = {}
                target_subject_count = len(subjects) / n_folds
                for subject in subjects:
                    costs = []
                    for fold in range(n_folds):
                        candidate_totals = fold_totals.copy()
                        candidate_counts = fold_subject_counts.copy()
                        candidate_totals[fold] += subject_vectors[subject]
                        candidate_counts[fold] += 1
                        balance_cost = np.square(
                            (candidate_totals - target) / target
                        ).sum()
                        count_cost = 0.15 * np.square(
                            (candidate_counts - target_subject_count)
                            / max(target_subject_count, 1.0)
                        ).sum()
                        costs.append(balance_cost + count_cost)
                    chosen = int(np.argmin(costs))
                    assignment[subject] = chosen
                    fold_totals[chosen] += subject_vectors[subject]
                    fold_subject_counts[chosen] += 1

                rows = []
                for fold in range(n_folds):
                    fold_subjects = {s for s, f in assignment.items() if f == fold}
                    fold_records = [r for r in records if r["subject"] in fold_subjects]
                    valid_y = np.concatenate(
                        [r["native_y"][r["native_mask"]] for r in fold_records], axis=0
                    )
                    rows.append(
                        {
                            "fold": fold,
                            "subjects": len(fold_subjects),
                            "recordings": len(fold_records),
                            "scored_rows": len(valid_y),
                            **{
                                f"{target}_positives": int(valid_y[:, c].sum())
                                for c, target in enumerate(TARGET_COLUMNS)
                            },
                        }
                    )
                fold_table = pd.DataFrame(rows)
                if (fold_table[[f"{c}_positives" for c in TARGET_COLUMNS]] == 0).any().any():
                    print("WARNING: at least one fold has no positives for one event class.")
                return assignment, fold_table


            def fit_feature_scaler(records: list[dict], record_indices: list[int]):
                scaler = StandardScaler()
                rng = np.random.default_rng(RANDOM_SEED)
                for record_index in record_indices:
                    features = records[record_index]["features"]
                    max_rows = 50_000
                    if len(features) > max_rows:
                        chosen = np.sort(rng.choice(len(features), max_rows, replace=False))
                        features = features[chosen]
                    scaler.partial_fit(features)
                return scaler


            def transform_records(records: list[dict], scaler: StandardScaler):
                transformed = []
                for record in records:
                    copy = dict(record)
                    copy["scaled_features"] = scaler.transform(
                        record["features"]
                    ).astype(np.float32)
                    transformed.append(copy)
                return transformed


            def gather_example_targets(records: list[dict], examples: np.ndarray):
                targets = np.empty((len(examples), len(TARGET_COLUMNS)), dtype=np.float32)
                for record_index in np.unique(examples[:, 0]):
                    positions = np.flatnonzero(examples[:, 0] == record_index)
                    endpoints = examples[positions, 1]
                    targets[positions] = records[int(record_index)]["down_y"][endpoints]
                return targets


            def sample_training_examples(
                records: list[dict], record_indices: list[int], seed: int
            ) -> np.ndarray:
                rng = np.random.default_rng(seed)
                positive_parts = []
                negative_parts = []
                for record_index in record_indices:
                    record = records[record_index]
                    eligible = np.flatnonzero(record["down_mask"])
                    if not len(eligible):
                        continue
                    labels = record["down_y"][eligible]
                    positive = eligible[labels.any(axis=1)]
                    negative = eligible[~labels.any(axis=1)]
                    positive_parts.append(
                        np.column_stack(
                            [np.full(len(positive), record_index), positive]
                        ).astype(np.int64)
                    )

                    if len(positive):
                        negative_target = min(
                            len(negative),
                            int(math.ceil(
                                len(positive) * SETTINGS["negative_to_positive_ratio"]
                            )),
                        )
                    else:
                        negative_target = min(
                            len(negative),
                            SETTINGS["negative_windows_for_no_fog_recording"],
                        )
                    if negative_target == 0:
                        continue

                    hard_count = min(
                        negative_target,
                        int(round(
                            negative_target * SETTINGS["hard_negative_fraction"]
                        )),
                    )
                    hard_order = np.argsort(
                        record["features"][negative, FREEZE_RATIO_FEATURE_INDEX]
                    )[::-1]
                    hard_negative = negative[hard_order[:hard_count]]
                    remaining_pool = np.setdiff1d(
                        negative, hard_negative, assume_unique=False
                    )
                    random_count = min(
                        negative_target - len(hard_negative), len(remaining_pool)
                    )
                    random_negative = (
                        rng.choice(remaining_pool, random_count, replace=False)
                        if random_count
                        else np.empty(0, dtype=np.int64)
                    )
                    chosen_negative = np.r_[hard_negative, random_negative]
                    negative_parts.append(
                        np.column_stack(
                            [np.full(len(chosen_negative), record_index), chosen_negative]
                        ).astype(np.int64)
                    )

                positive_examples = (
                    np.concatenate(positive_parts) if positive_parts
                    else np.empty((0, 2), dtype=np.int64)
                )
                negative_examples = (
                    np.concatenate(negative_parts) if negative_parts
                    else np.empty((0, 2), dtype=np.int64)
                )
                maximum = SETTINGS["max_train_windows"]
                positive_budget = min(
                    len(positive_examples),
                    int(round(maximum * SETTINGS["positive_window_fraction"])),
                )
                if len(positive_examples) > positive_budget:
                    labels = gather_example_targets(records, positive_examples)
                    class_counts = np.maximum(labels.sum(axis=0), 1.0)
                    rarity_weight = (labels / class_counts).sum(axis=1)
                    rarity_weight /= rarity_weight.sum()
                    keep = rng.choice(
                        len(positive_examples), positive_budget,
                        replace=False, p=rarity_weight
                    )
                    positive_examples = positive_examples[keep]
                negative_budget = min(
                    len(negative_examples), maximum - len(positive_examples)
                )
                if negative_budget < len(negative_examples):
                    keep = rng.choice(
                        len(negative_examples), negative_budget, replace=False
                    )
                    negative_examples = negative_examples[keep]
                examples = np.concatenate([positive_examples, negative_examples])
                rng.shuffle(examples)
                return examples


            def validation_examples(records: list[dict], record_indices: list[int]):
                parts = []
                stride = SETTINGS["validation_stride"]
                for record_index in record_indices:
                    eligible = np.flatnonzero(records[record_index]["down_mask"])[::stride]
                    parts.append(
                        np.column_stack(
                            [np.full(len(eligible), record_index), eligible]
                        ).astype(np.int64)
                    )
                examples = np.concatenate(parts)
                maximum = SETTINGS["max_monitor_windows"]
                if len(examples) > maximum:
                    rng = np.random.default_rng(RANDOM_SEED)
                    examples = examples[
                        np.sort(rng.choice(len(examples), maximum, replace=False))
                    ]
                return examples


            def all_examples(records: list[dict]):
                return np.concatenate(
                    [
                        np.column_stack(
                            [np.full(len(record["features"]), index),
                             np.arange(len(record["features"]))]
                        ).astype(np.int64)
                        for index, record in enumerate(records)
                    ]
                )


            class WindowSequence(tf.keras.utils.Sequence):
                def __init__(
                    self,
                    records: list[dict],
                    examples: np.ndarray,
                    window_samples: int,
                    batch_size: int,
                    include_targets: bool,
                    shuffle: bool,
                    seed: int,
                ):
                    super().__init__()
                    self.records = records
                    self.examples = np.asarray(examples, dtype=np.int64).copy()
                    self.window_samples = int(window_samples)
                    self.batch_size = int(batch_size)
                    self.include_targets = bool(include_targets)
                    self.shuffle = bool(shuffle)
                    self.rng = np.random.default_rng(seed)
                    self.window_offsets = np.arange(self.window_samples, dtype=np.int64)
                    self.padded = [
                        np.pad(
                            record["scaled_features"],
                            ((self.window_samples - 1, 0), (0, 0)),
                            mode="edge",
                        )
                        for record in records
                    ]
                    if self.shuffle:
                        self.rng.shuffle(self.examples)

                def __len__(self):
                    return math.ceil(len(self.examples) / self.batch_size)

                def __getitem__(self, batch_index):
                    batch_examples = self.examples[
                        batch_index * self.batch_size:
                        (batch_index + 1) * self.batch_size
                    ]
                    features = np.empty(
                        (
                            len(batch_examples),
                            self.window_samples,
                            len(FEATURE_NAMES),
                        ),
                        dtype=np.float32,
                    )
                    targets = np.empty(
                        (len(batch_examples), len(TARGET_COLUMNS)), dtype=np.float32
                    )
                    for record_index in np.unique(batch_examples[:, 0]):
                        positions = np.flatnonzero(batch_examples[:, 0] == record_index)
                        endpoints = batch_examples[positions, 1]
                        gather_indices = endpoints[:, None] + self.window_offsets[None, :]
                        features[positions] = self.padded[int(record_index)][gather_indices]
                        if self.include_targets:
                            targets[positions] = self.records[int(record_index)]["down_y"][endpoints]
                    return (features, targets) if self.include_targets else features

                def on_epoch_end(self):
                    if self.shuffle:
                        self.rng.shuffle(self.examples)

                def target_array(self):
                    if not self.include_targets:
                        raise ValueError("This sequence has no targets")
                    return gather_example_targets(self.records, self.examples)
            """
        ),
        markdown("## Competition models"),
        code(
            r"""
            def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
                y_true = np.asarray(y_true)
                if len(y_true) == 0 or np.unique(y_true).size < 2:
                    return float("nan")
                return float(average_precision_score(y_true, y_score))


            def competition_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
                scores = {
                    target: safe_average_precision(y_true[:, index], y_score[:, index])
                    for index, target in enumerate(TARGET_COLUMNS)
                }
                scores["mean_average_precision"] = float(
                    np.nanmean([scores[target] for target in TARGET_COLUMNS])
                )
                return scores


            def make_balanced_focal_loss(positive_weights: np.ndarray, gamma: float):
                weights = tf.constant(positive_weights, dtype=tf.float32)

                def balanced_focal_loss(y_true, y_pred):
                    y_true = tf.cast(y_true, tf.float32)
                    y_pred = tf.cast(y_pred, tf.float32)
                    epsilon = tf.keras.backend.epsilon()
                    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
                    cross_entropy = -(
                        y_true * tf.math.log(y_pred)
                        + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
                    )
                    probability_true = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
                    class_weight = y_true * weights + (1.0 - y_true)
                    focal_weight = tf.pow(1.0 - probability_true, gamma)
                    return tf.reduce_mean(
                        class_weight * focal_weight * cross_entropy, axis=-1
                    )

                return balanced_focal_loss


            def tcn_residual_block(inputs, filters: int, dilation: int, dropout: float):
                residual = inputs
                x = tf.keras.layers.Conv1D(
                    filters, 3, padding="causal", dilation_rate=dilation,
                    kernel_initializer="he_normal"
                )(inputs)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation("swish")(x)
                x = tf.keras.layers.SpatialDropout1D(dropout)(x)
                x = tf.keras.layers.Conv1D(
                    filters, 3, padding="causal", dilation_rate=dilation,
                    kernel_initializer="he_normal"
                )(x)
                x = tf.keras.layers.BatchNormalization()(x)
                if residual.shape[-1] != filters:
                    residual = tf.keras.layers.Conv1D(filters, 1)(residual)
                return tf.keras.layers.Activation("swish")(
                    tf.keras.layers.Add()([residual, x])
                )


            def build_competition_model(
                architecture: str,
                positive_weights: np.ndarray,
            ) -> tf.keras.Model:
                inputs = tf.keras.layers.Input(
                    shape=(WINDOW_SAMPLES, len(FEATURE_NAMES)), name="sensor_window"
                )
                x = tf.keras.layers.GaussianNoise(0.015)(inputs)

                if architecture == "cnn_lstm":
                    x = tf.keras.layers.Conv1D(
                        32, 5, padding="causal", kernel_initializer="he_normal"
                    )(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Activation("swish")(x)
                    x = tf.keras.layers.SpatialDropout1D(0.15)(x)
                    x = tf.keras.layers.Conv1D(
                        48, 5, padding="causal", dilation_rate=2,
                        kernel_initializer="he_normal"
                    )(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Activation("swish")(x)
                    x = tf.keras.layers.LSTM(64, dropout=0.0, recurrent_dropout=0.0)(x)
                elif architecture == "tcn":
                    x = tf.keras.layers.Conv1D(48, 1, padding="same")(x)
                    for dilation in (1, 2, 4, 8, 16):
                        x = tcn_residual_block(x, 48, dilation, 0.15)
                    final_state = tf.keras.layers.Lambda(lambda z: z[:, -1, :])(x)
                    pooled_state = tf.keras.layers.GlobalAveragePooling1D()(x)
                    x = tf.keras.layers.Concatenate()([final_state, pooled_state])
                else:
                    raise ValueError(f"Unknown architecture: {architecture}")

                x = tf.keras.layers.Dense(64, activation="swish")(x)
                x = tf.keras.layers.Dropout(0.25)(x)
                outputs = tf.keras.layers.Dense(
                    len(TARGET_COLUMNS), activation="sigmoid", dtype="float32",
                    name="event_probabilities"
                )(x)
                model = tf.keras.Model(inputs, outputs, name=architecture)

                try:
                    optimizer = tf.keras.optimizers.AdamW(
                        learning_rate=SETTINGS["learning_rate"],
                        weight_decay=SETTINGS["weight_decay"],
                        clipnorm=1.0,
                    )
                except AttributeError:
                    optimizer = tf.keras.optimizers.Adam(
                        learning_rate=SETTINGS["learning_rate"], clipnorm=1.0
                    )
                model.compile(
                    optimizer=optimizer,
                    loss=make_balanced_focal_loss(
                        positive_weights, SETTINGS["focal_gamma"]
                    ),
                    metrics=[
                        tf.keras.metrics.AUC(
                            curve="PR", multi_label=True,
                            num_labels=len(TARGET_COLUMNS), name="mean_pr_auc"
                        )
                    ],
                )
                return model


            class ExactMAPCallback(tf.keras.callbacks.Callback):
                def __init__(self, validation_records: list[dict]):
                    super().__init__()
                    self.validation_records = validation_records
                    self.validation_sequence = WindowSequence(
                        validation_records,
                        all_examples(validation_records),
                        WINDOW_SAMPLES,
                        SETTINGS["batch_size"],
                        include_targets=False,
                        shuffle=False,
                        seed=RANDOM_SEED,
                    )
                    self.values = []

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs if logs is not None else {}
                    flat_predictions = self.model.predict(
                        self.validation_sequence, verbose=0
                    )
                    down_predictions = []
                    offset = 0
                    for record in self.validation_records:
                        length = len(record["features"])
                        down_predictions.append(
                            flat_predictions[offset:offset + length]
                        )
                        offset += length
                    targets, predictions = native_scored_arrays(
                        self.validation_records, down_predictions
                    )
                    value = competition_scores(
                        targets, predictions
                    )["mean_average_precision"]
                    self.values.append(value)
                    logs["val_exact_map"] = value
                    print(f" — val_exact_mAP: {value:.5f}")


            def positive_class_weights(records: list[dict], record_indices: list[int]):
                labels = np.concatenate(
                    [
                        records[index]["down_y"][records[index]["down_mask"]]
                        for index in record_indices
                    ],
                    axis=0,
                ).astype(np.float64)
                positives = labels.sum(axis=0)
                negatives = len(labels) - positives
                weights = np.sqrt(negatives / np.maximum(positives, 1.0))
                return np.clip(weights, 1.0, 12.0).astype(np.float32)


            def predict_record_list(model, records: list[dict]) -> list[np.ndarray]:
                sequence = WindowSequence(
                    records,
                    all_examples(records),
                    WINDOW_SAMPLES,
                    SETTINGS["batch_size"],
                    include_targets=False,
                    shuffle=False,
                    seed=RANDOM_SEED,
                )
                flat_predictions = model.predict(sequence, verbose=0).astype(np.float32)
                predictions = []
                offset = 0
                for record in records:
                    length = len(record["features"])
                    predictions.append(flat_predictions[offset:offset + length])
                    offset += length
                if offset != len(flat_predictions):
                    raise AssertionError("Prediction splitting failed")
                return predictions


            def interpolate_prediction(
                native_time: np.ndarray,
                down_time: np.ndarray,
                down_prediction: np.ndarray,
            ) -> np.ndarray:
                return np.interp(
                    native_time.astype(np.float64),
                    down_time.astype(np.float64),
                    down_prediction.astype(np.float64),
                ).astype(np.float32)


            def native_scored_arrays(
                records: list[dict], down_predictions: list[np.ndarray]
            ):
                labels = []
                predictions = []
                for record, prediction in zip(records, down_predictions):
                    native_prediction = np.column_stack(
                        [
                            interpolate_prediction(
                                record["native_time"], record["down_time"], prediction[:, c]
                            )
                            for c in range(len(TARGET_COLUMNS))
                        ]
                    )
                    mask = record["native_mask"]
                    labels.append(record["native_y"][mask])
                    predictions.append(native_prediction[mask])
                return np.concatenate(labels), np.concatenate(predictions)
            """
        ),
        markdown("## Train, validate, and ensemble each dataset"),
        code(
            r"""
            def train_fold_model(
                dataset_name: str,
                fold: int,
                architecture: str,
                scaled_train_records: list[dict],
                scaled_test_records: list[dict],
                train_indices: list[int],
                validation_indices: list[int],
                positive_weights: np.ndarray,
            ):
                train_examples = sample_training_examples(
                    scaled_train_records,
                    train_indices,
                    RANDOM_SEED + 1000 * fold + 17 * len(architecture),
                )
                monitor_examples = validation_examples(
                    scaled_train_records, validation_indices
                )
                train_sequence = WindowSequence(
                    scaled_train_records,
                    train_examples,
                    WINDOW_SAMPLES,
                    SETTINGS["batch_size"],
                    include_targets=True,
                    shuffle=True,
                    seed=RANDOM_SEED + fold,
                )
                monitor_sequence = WindowSequence(
                    scaled_train_records,
                    monitor_examples,
                    WINDOW_SAMPLES,
                    SETTINGS["batch_size"],
                    include_targets=True,
                    shuffle=False,
                    seed=RANDOM_SEED,
                )

                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(RANDOM_SEED + fold)
                model = build_competition_model(architecture, positive_weights)
                validation_records = [
                    scaled_train_records[index] for index in validation_indices
                ]
                exact_callback = ExactMAPCallback(validation_records)
                checkpoint_path = (
                    CHECKPOINT_ROOT
                    / f"{dataset_name}_fold{fold}_{architecture}.weights.h5"
                )
                callbacks = [
                    exact_callback,
                    tf.keras.callbacks.ModelCheckpoint(
                        checkpoint_path,
                        monitor="val_exact_map",
                        mode="max",
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=0,
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_exact_map",
                        mode="max",
                        patience=SETTINGS["early_stopping_patience"],
                        min_delta=1e-4,
                        restore_best_weights=True,
                        verbose=1,
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_exact_map",
                        mode="max",
                        factor=0.5,
                        patience=max(2, SETTINGS["early_stopping_patience"] // 2),
                        min_lr=2e-6,
                        verbose=1,
                    ),
                    tf.keras.callbacks.TerminateOnNaN(),
                ]
                print(
                    f"\n{dataset_name} fold {fold} {architecture}: "
                    f"{len(train_examples):,} train windows, "
                    f"{len(monitor_examples):,} monitor windows"
                )
                history = model.fit(
                    train_sequence,
                    validation_data=monitor_sequence,
                    epochs=SETTINGS["epochs"],
                    callbacks=callbacks,
                    verbose=2,
                )
                if checkpoint_path.exists():
                    model.load_weights(checkpoint_path)

                history_frame = pd.DataFrame(history.history)
                history_frame["val_exact_map"] = exact_callback.values
                history_frame["epoch"] = np.arange(1, len(history_frame) + 1)
                history_frame["dataset"] = dataset_name
                history_frame["fold"] = fold
                history_frame["architecture"] = architecture
                return model, history_frame


            def causal_ema(values: np.ndarray, span: int) -> np.ndarray:
                if span <= 1 or len(values) <= 1:
                    return values.astype(np.float32, copy=True)
                alpha = 2.0 / (span + 1.0)
                filtered, _ = lfilter(
                    [alpha], [1.0, -(1.0 - alpha)], values,
                    zi=[(1.0 - alpha) * float(values[0])],
                )
                return filtered.astype(np.float32)


            def blend_architectures_oof(
                records: list[dict],
                model_oof: dict[str, list[np.ndarray]],
                model_test: dict[str, list[np.ndarray]],
            ):
                architectures = list(model_oof)
                if len(architectures) == 1:
                    weights = np.ones((len(TARGET_COLUMNS), 1), dtype=np.float32)
                    return (
                        [p.copy() for p in model_oof[architectures[0]]],
                        [p.copy() for p in model_test[architectures[0]]],
                        pd.DataFrame(
                            {
                                "target": TARGET_COLUMNS,
                                architectures[0]: 1.0,
                                "oof_ap": np.nan,
                            }
                        ),
                    )
                if len(architectures) != 2:
                    raise ValueError("The current OOF blender supports one or two models")

                first, second = architectures
                labels, first_native = native_scored_arrays(records, model_oof[first])
                _, second_native = native_scored_arrays(records, model_oof[second])
                selected_first_weights = []
                rows = []
                for class_index, target in enumerate(TARGET_COLUMNS):
                    candidates = []
                    for first_weight in np.linspace(0.0, 1.0, 11):
                        prediction = (
                            first_weight * first_native[:, class_index]
                            + (1.0 - first_weight) * second_native[:, class_index]
                        )
                        score = safe_average_precision(
                            labels[:, class_index], prediction
                        )
                        candidates.append((score, -abs(first_weight - 0.5), first_weight))
                    best_score, _, best_weight = max(candidates)
                    selected_first_weights.append(best_weight)
                    rows.append(
                        {
                            "target": target,
                            f"{first}_weight": best_weight,
                            f"{second}_weight": 1.0 - best_weight,
                            "oof_ap": best_score,
                        }
                    )

                def apply_blend(first_predictions, second_predictions):
                    blended = []
                    for first_prediction, second_prediction in zip(
                        first_predictions, second_predictions
                    ):
                        output = np.empty_like(first_prediction, dtype=np.float32)
                        for class_index, first_weight in enumerate(selected_first_weights):
                            output[:, class_index] = (
                                first_weight * first_prediction[:, class_index]
                                + (1.0 - first_weight) * second_prediction[:, class_index]
                            )
                        blended.append(output)
                    return blended

                return (
                    apply_blend(model_oof[first], model_oof[second]),
                    apply_blend(model_test[first], model_test[second]),
                    pd.DataFrame(rows),
                )


            def select_continuous_smoothing(
                records: list[dict],
                oof_down: list[np.ndarray],
                test_down: list[np.ndarray],
            ):
                smoothing_candidates = [(0.0, 0.0)]
                smoothing_candidates += [
                    (seconds, blend)
                    for seconds in (0.25, 0.50, 1.00)
                    for blend in (0.25, 0.50, 0.75, 1.00)
                ]
                selected = []
                rows = []
                for class_index, target in enumerate(TARGET_COLUMNS):
                    candidate_rows = []
                    for seconds, blend in smoothing_candidates:
                        span = max(1, int(round(seconds * SETTINGS["target_hz"])))
                        labels_parts = []
                        prediction_parts = []
                        for record, prediction in zip(records, oof_down):
                            raw = prediction[:, class_index]
                            smoothed = causal_ema(raw, span)
                            candidate = (1.0 - blend) * raw + blend * smoothed
                            native_prediction = interpolate_prediction(
                                record["native_time"], record["down_time"], candidate
                            )
                            mask = record["native_mask"]
                            labels_parts.append(record["native_y"][mask, class_index])
                            prediction_parts.append(native_prediction[mask])
                        score = safe_average_precision(
                            np.concatenate(labels_parts), np.concatenate(prediction_parts)
                        )
                        candidate_rows.append((score, -seconds, -blend, seconds, blend))
                    best_score, _, _, best_seconds, best_blend = max(candidate_rows)
                    selected.append((best_seconds, best_blend))
                    rows.append(
                        {
                            "target": target,
                            "ema_seconds": best_seconds,
                            "ema_blend": best_blend,
                            "native_oof_ap": best_score,
                        }
                    )

                def apply_smoothing(predictions):
                    outputs = []
                    for prediction in predictions:
                        output = prediction.copy().astype(np.float32)
                        for class_index, (seconds, blend) in enumerate(selected):
                            span = max(1, int(round(seconds * SETTINGS["target_hz"])))
                            smoothed = causal_ema(prediction[:, class_index], span)
                            output[:, class_index] = (
                                (1.0 - blend) * prediction[:, class_index]
                                + blend * smoothed
                            )
                        outputs.append(output)
                    return outputs

                return (
                    apply_smoothing(oof_down),
                    apply_smoothing(test_down),
                    pd.DataFrame(rows),
                )


            def run_dataset_experiment(dataset_name: str):
                start_time = time.time()
                records, test_records, excluded = load_dataset(dataset_name)
                display(dataset_summary(records).assign(dataset=dataset_name))

                fold_assignment, fold_table = make_subject_folds(
                    records, SETTINGS["n_folds"], RANDOM_SEED
                )
                display(fold_table)
                for fold in range(SETTINGS["n_folds"]):
                    train_subjects = {
                        subject for subject, assigned_fold in fold_assignment.items()
                        if assigned_fold != fold
                    }
                    validation_subjects = {
                        subject for subject, assigned_fold in fold_assignment.items()
                        if assigned_fold == fold
                    }
                    if train_subjects & validation_subjects:
                        raise AssertionError("Subject leakage between train and validation")

                model_oof = {
                    architecture: [
                        np.full((len(record["features"]), len(TARGET_COLUMNS)), np.nan,
                                dtype=np.float32)
                        for record in records
                    ]
                    for architecture in SETTINGS["architectures"]
                }
                model_test_sum = {
                    architecture: [
                        np.zeros((len(record["features"]), len(TARGET_COLUMNS)),
                                 dtype=np.float32)
                        for record in test_records
                    ]
                    for architecture in SETTINGS["architectures"]
                }
                histories = []
                fold_metric_rows = []

                for fold in range(SETTINGS["n_folds"]):
                    train_indices = [
                        index for index, record in enumerate(records)
                        if fold_assignment[record["subject"]] != fold
                    ]
                    validation_indices = [
                        index for index, record in enumerate(records)
                        if fold_assignment[record["subject"]] == fold
                    ]
                    scaler = fit_feature_scaler(records, train_indices)
                    scaled_records = transform_records(records, scaler)
                    scaled_test_records = transform_records(test_records, scaler)
                    weights = positive_class_weights(records, train_indices)
                    print("Positive class weights:", dict(zip(TARGET_COLUMNS, weights.round(3))))

                    for architecture in SETTINGS["architectures"]:
                        model, history = train_fold_model(
                            dataset_name,
                            fold,
                            architecture,
                            scaled_records,
                            scaled_test_records,
                            train_indices,
                            validation_indices,
                            weights,
                        )
                        histories.append(history)

                        validation_subset = [scaled_records[i] for i in validation_indices]
                        validation_predictions = predict_record_list(
                            model, validation_subset
                        )
                        for original_index, prediction in zip(
                            validation_indices, validation_predictions
                        ):
                            model_oof[architecture][original_index] = prediction
                        validation_y, validation_p = native_scored_arrays(
                            validation_subset, validation_predictions
                        )
                        metrics = competition_scores(validation_y, validation_p)
                        fold_metric_rows.append(
                            {
                                "dataset": dataset_name,
                                "fold": fold,
                                "architecture": architecture,
                                **metrics,
                            }
                        )

                        test_predictions = predict_record_list(
                            model, scaled_test_records
                        )
                        for index, prediction in enumerate(test_predictions):
                            model_test_sum[architecture][index] += (
                                prediction / SETTINGS["n_folds"]
                            )
                        del model, validation_predictions, test_predictions
                        tf.keras.backend.clear_session()
                        gc.collect()

                    del scaled_records, scaled_test_records, scaler
                    gc.collect()

                for architecture, predictions in model_oof.items():
                    if any(np.isnan(prediction).any() for prediction in predictions):
                        raise AssertionError(
                            f"Incomplete OOF predictions for {dataset_name}/{architecture}"
                        )

                blended_oof, blended_test, blend_table = blend_architectures_oof(
                    records, model_oof, model_test_sum
                )
                final_oof_down, final_test_down, smoothing_table = (
                    select_continuous_smoothing(records, blended_oof, blended_test)
                )
                oof_y, oof_prediction = native_scored_arrays(
                    records, final_oof_down
                )
                elapsed_minutes = (time.time() - start_time) / 60.0
                print(f"{dataset_name} completed in {elapsed_minutes:.1f} minutes")
                print("Native OOF scores:", competition_scores(oof_y, oof_prediction))
                display(blend_table)
                display(smoothing_table)

                return {
                    "dataset": dataset_name,
                    "records": records,
                    "test_records": test_records,
                    "test_down_predictions": final_test_down,
                    "oof_y": oof_y.astype(np.int8),
                    "oof_prediction": oof_prediction.astype(np.float32),
                    "fold_table": fold_table,
                    "fold_metrics": pd.DataFrame(fold_metric_rows),
                    "histories": pd.concat(histories, ignore_index=True),
                    "blend_table": blend_table,
                    "smoothing_table": smoothing_table,
                    "excluded_train_overlaps": excluded,
                }
            """
        ),
        markdown("### 2. Run both dataset-specific experiments"),
        code(
            r"""
            dataset_results = {}
            for dataset_name in ("defog", "tdcsfog"):
                print("\n" + "=" * 80)
                print("RUNNING", dataset_name.upper())
                print("=" * 80)
                dataset_results[dataset_name] = run_dataset_experiment(dataset_name)

            fold_metrics = pd.concat(
                [result["fold_metrics"] for result in dataset_results.values()],
                ignore_index=True,
            )
            display(fold_metrics)
            """
        ),
        markdown("## OOF scale alignment and exact pooled metric"),
        markdown(
            r"""
            The leaderboard pools scored rows from both datasets before calculating
            class AP. A monotonic transform preserves ranking inside one dataset but can
            change the ordering between datasets, so a separate OOF-only Platt transform
            is fitted for every dataset/class. Test labels and sample target values never
            enter this step.
            """
        ),
        code(
            r"""
            def clipped_logit(probability: np.ndarray) -> np.ndarray:
                probability = np.clip(probability.astype(np.float64), 1e-5, 1.0 - 1e-5)
                return np.log(probability / (1.0 - probability))


            def sigmoid(values: np.ndarray) -> np.ndarray:
                values = np.clip(values, -40.0, 40.0)
                return 1.0 / (1.0 + np.exp(-values))


            def fit_platt_parameters(
                y_true: np.ndarray,
                probability: np.ndarray,
                seed: int,
                maximum_rows: int = 300_000,
            ):
                if np.unique(y_true).size < 2:
                    return 1.0, 0.0
                rng = np.random.default_rng(seed)
                if len(y_true) > maximum_rows:
                    chosen = rng.choice(len(y_true), maximum_rows, replace=False)
                    y_fit = y_true[chosen]
                    p_fit = probability[chosen]
                else:
                    y_fit = y_true
                    p_fit = probability
                calibrator = LogisticRegression(
                    C=10.0, solver="lbfgs", max_iter=500, random_state=seed
                )
                calibrator.fit(clipped_logit(p_fit).reshape(-1, 1), y_fit)
                slope = float(np.clip(calibrator.coef_[0, 0], 0.05, 10.0))
                intercept = float(np.clip(calibrator.intercept_[0], -15.0, 15.0))
                return slope, intercept


            calibration_rows = []
            calibrated_oof_parts = []
            pooled_label_parts = []
            for dataset_index, (dataset_name, result) in enumerate(dataset_results.items()):
                calibrated = np.empty_like(result["oof_prediction"], dtype=np.float32)
                parameters = []
                for class_index, target in enumerate(TARGET_COLUMNS):
                    slope, intercept = fit_platt_parameters(
                        result["oof_y"][:, class_index],
                        result["oof_prediction"][:, class_index],
                        RANDOM_SEED + dataset_index * 100 + class_index,
                    )
                    calibrated[:, class_index] = sigmoid(
                        slope * clipped_logit(result["oof_prediction"][:, class_index])
                        + intercept
                    )
                    parameters.append((slope, intercept))
                    calibration_rows.append(
                        {
                            "dataset": dataset_name,
                            "target": target,
                            "logit_slope": slope,
                            "logit_intercept": intercept,
                        }
                    )
                result["calibration_parameters"] = parameters
                result["calibrated_oof_prediction"] = calibrated
                calibrated_oof_parts.append(calibrated)
                pooled_label_parts.append(result["oof_y"])

            pooled_oof_y = np.concatenate(pooled_label_parts)
            pooled_oof_prediction = np.concatenate(calibrated_oof_parts)
            pooled_scores = competition_scores(pooled_oof_y, pooled_oof_prediction)
            display(pd.DataFrame(calibration_rows))
            display(
                pd.DataFrame(
                    [pooled_scores],
                    index=["post-processing-tuned subject-disjoint OOF"],
                )
            )
            """
        ),
        markdown("## Results"),
        code(
            r"""
            histories = pd.concat(
                [result["histories"] for result in dataset_results.values()],
                ignore_index=True,
            )
            figure, axes = plt.subplots(1, 2, figsize=(16, 5))
            for (dataset_name, fold, architecture), history in histories.groupby(
                ["dataset", "fold", "architecture"], observed=True
            ):
                label = f"{dataset_name} f{fold} {architecture}"
                axes[0].plot(history["epoch"], history["loss"], alpha=0.65, label=label)
                if "val_loss" in history:
                    axes[0].plot(
                        history["epoch"], history["val_loss"], alpha=0.65,
                        linestyle="--"
                    )
                axes[1].plot(
                    history["epoch"], history["val_exact_map"], alpha=0.75, label=label
                )
            axes[0].set_title("Training loss (solid) and validation loss (dashed)")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Focal loss")
            axes[1].set_title("Exact validation mean average precision")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("mAP")
            axes[1].set_ylim(0, 1)
            handles, labels = axes[1].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
            figure.tight_layout(rect=(0, 0.12, 1, 1))
            plt.show()

            figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
            for class_index, target in enumerate(TARGET_COLUMNS):
                precision, recall, _ = precision_recall_curve(
                    pooled_oof_y[:, class_index], pooled_oof_prediction[:, class_index]
                )
                axes[class_index].plot(recall, precision, color="#3366aa")
                axes[class_index].set_title(
                    f"{target} OOF PR curve\nAP={pooled_scores[target]:.4f}"
                )
                axes[class_index].set_xlabel("Recall")
                axes[class_index].set_ylabel("Precision")
                axes[class_index].set_xlim(0, 1)
                axes[class_index].set_ylim(0, 1)
            figure.suptitle("Pooled subject-disjoint native-timestep validation")
            figure.tight_layout()
            plt.show()

            print(
                "Binary accuracy is intentionally omitted: it is dominated by the many "
                "non-FoG rows and is not the competition metric."
            )
            """
        ),
        markdown("## Create and validate `submission.csv`"),
        code(
            r"""
            prediction_frames = []
            total_test_rows = 0
            for dataset_name, result in dataset_results.items():
                parameters = result["calibration_parameters"]
                for record, down_prediction in zip(
                    result["test_records"], result["test_down_predictions"]
                ):
                    native_prediction = np.column_stack(
                        [
                            interpolate_prediction(
                                record["native_time"],
                                record["down_time"],
                                down_prediction[:, class_index],
                            )
                            for class_index in range(len(TARGET_COLUMNS))
                        ]
                    )
                    for class_index, (slope, intercept) in enumerate(parameters):
                        native_prediction[:, class_index] = sigmoid(
                            slope * clipped_logit(native_prediction[:, class_index])
                            + intercept
                        )
                    native_prediction = np.clip(native_prediction, 0.0, 1.0)
                    ids = [
                        f"{record['recording_id']}_{int(time_value)}"
                        for time_value in record["native_time"]
                    ]
                    frame = pd.DataFrame(native_prediction, columns=TARGET_COLUMNS)
                    frame.insert(0, "Id", ids)
                    prediction_frames.append(frame)
                    total_test_rows += len(frame)

            predictions_by_id = pd.concat(prediction_frames, ignore_index=True)
            if not predictions_by_id["Id"].is_unique:
                duplicates = predictions_by_id.loc[
                    predictions_by_id["Id"].duplicated(), "Id"
                ].head().tolist()
                raise ValueError(f"Duplicate generated prediction IDs: {duplicates}")

            # Read only Id again. Existing sample target values are never copied.
            submission_ids = pd.read_csv(
                LAYOUT.sample_path, usecols=["Id"], dtype={"Id": "string"}
            )
            submission = submission_ids.merge(
                predictions_by_id, on="Id", how="left", validate="one_to_one"
            )

            required_columns = ["Id", *TARGET_COLUMNS]
            if submission.columns.tolist() != required_columns:
                raise AssertionError(submission.columns.tolist())
            if len(submission) != total_test_rows:
                raise AssertionError(
                    f"Submission has {len(submission):,} rows; test has {total_test_rows:,}"
                )
            if submission["Id"].tolist() != submission_ids["Id"].tolist():
                raise AssertionError("Submission Id order differs from sample submission")
            if not submission["Id"].is_unique:
                raise AssertionError("Submission Id values are not unique")
            values = submission.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                missing_ids = submission.loc[
                    ~np.isfinite(values).all(axis=1), "Id"
                ].head().tolist()
                raise AssertionError(f"Missing/non-finite predictions for {missing_ids}")
            if not ((values >= 0.0) & (values <= 1.0)).all():
                raise AssertionError("Predictions fall outside [0, 1]")

            LAYOUT.output_path.parent.mkdir(parents=True, exist_ok=True)
            submission.to_csv(LAYOUT.output_path, index=False, float_format="%.8f")
            print("PASS: exact columns, row count, Id order, unique IDs, finite probabilities")
            print("Saved:", LAYOUT.output_path)
            print("Rows:", f"{len(submission):,}")
            display(submission.head())
            display(submission.loc[:, TARGET_COLUMNS].describe())
            """
        ),
        markdown(
            r"""
            ## Checks and next steps

            A valid full run ends with `PASS` and writes:

            - Kaggle: `/kaggle/working/submission.csv`
            - local project: `data/submissions/submission.csv`

            On Kaggle, select a GPU accelerator, keep internet disabled, choose
            `RUN_PROFILE = "competition"`, and run all cells before saving a version.
            The three output columns must remain continuous scores. Do not convert them
            to binary episodes before submitting.

            This notebook reports exact AP at the original timestep resolution from
            subject-disjoint model predictions. Because architecture weights, smoothing,
            and calibration are selected on those OOF predictions, the displayed pooled
            number is labelled **post-processing-tuned OOF**; it is useful for selection
            but is not presented as an untouched final generalisation estimate. The
            relevant competition comparison is still AP—not binary accuracy and not the
            event-decoder metrics used by the dissertation's episode notebooks.
            """
        ),
    ]

    save_notebook(
        "kaggle_fog_competition_submission.ipynb",
        cells,
        kernel_name="python3",
        kernel_display_name="Python 3",
    )


if __name__ == "__main__":
    build_notebook()
    print("Built notebooks/kaggle_fog_competition_submission.ipynb")
