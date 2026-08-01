import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fog_pipeline import (
    ANY_FOG_COLUMN,
    FEATURE_COLUMNS,
    SEGMENT_COLUMN,
    TARGET_COLUMNS,
    SubjectSplits,
    add_any_fog_target,
    add_sensor_features,
    align_binary_predictions,
    assert_subject_disjoint,
    collapse_targets_to_any_fog,
    evaluate_episode_predictions,
    load_recordings,
    match_episodes,
    prepare_sequence_splits,
    prepare_tabular_splits,
    reconstruct_episodes,
    select_episode_threshold,
    split_by_subject,
)


def make_recording(subject: str, recording: str, offset: float) -> pd.DataFrame:
    rows = 12
    values = np.arange(rows, dtype=float) + offset
    return pd.DataFrame(
        {
            "Time": np.arange(rows),
            "AccV": values,
            "AccML": values + 0.25,
            "AccAP": values - 0.25,
            "StartHesitation": (np.arange(rows) % 7 == 0).astype(int),
            "Turn": (np.arange(rows) % 5 == 0).astype(int),
            "Walking": (np.arange(rows) % 3 == 0).astype(int),
            "Valid": True,
            "Task": True,
            "RecordingId": recording,
            "Subject": subject,
        }
    )


class FogPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        frames = []
        for subject_number in range(10):
            subject = f"subject-{subject_number}"
            frames.append(
                make_recording(
                    subject,
                    f"recording-{subject_number}",
                    float(subject_number * 10),
                )
            )
        self.data = pd.concat(frames, ignore_index=True)

    def test_subject_split_has_no_overlap(self) -> None:
        splits = split_by_subject(self.data, random_state=7)
        assert_subject_disjoint(splits)
        self.assertEqual(
            len(splits.train) + len(splits.validation) + len(splits.test),
            len(self.data),
        )

    def test_features_exclude_targets_and_reset_at_recording_boundaries(self) -> None:
        engineered = add_sensor_features(self.data, window_size=5)
        self.assertTrue(set(TARGET_COLUMNS).isdisjoint(FEATURE_COLUMNS))
        first_rows = engineered.groupby("RecordingId", observed=True).head(1)
        self.assertTrue(first_rows["Rolling_AccV_Mean"].isna().all())

    def test_any_fog_is_union_of_event_types(self) -> None:
        labels = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=float)
        collapsed = collapse_targets_to_any_fog(labels)
        np.testing.assert_array_equal(collapsed.ravel(), [0, 1, 1])

        with_target = add_any_fog_target(
            pd.DataFrame(labels, columns=TARGET_COLUMNS)
        )
        np.testing.assert_array_equal(with_target[ANY_FOG_COLUMN], [0, 1, 1])

    def test_preprocessor_is_fitted_only_on_training_data(self) -> None:
        train = make_recording("train-subject", "train-recording", 0.0)
        validation = make_recording("validation-subject", "validation-recording", 100.0)
        test = make_recording("test-subject", "test-recording", 200.0)
        prepared = prepare_tabular_splits(
            SubjectSplits(train=train, validation=validation, test=test)
        )
        self.assertGreater(float(np.nanmax(prepared.X_validation)), 1.0)
        self.assertGreater(float(np.nanmax(prepared.X_test)), 1.0)
        self.assertEqual(prepared.y_any_fog_train.shape, (len(train), 1))
        self.assertEqual(len(prepared.alignment_splits.train), len(prepared.X_train))
        np.testing.assert_array_equal(
            prepared.alignment_splits.train["Time"],
            prepared.engineered_splits.train["Time"],
        )

    def test_sequences_stay_inside_recordings(self) -> None:
        splits = split_by_subject(self.data, random_state=11)
        prepared = prepare_sequence_splits(splits, timesteps=5, stride=5)
        self.assertEqual(prepared.X_train.shape[1:], (5, len(FEATURE_COLUMNS)))
        self.assertEqual(prepared.y_train.shape[1], len(TARGET_COLUMNS))
        self.assertEqual(prepared.y_any_fog_train.shape[1], 1)
        self.assertGreater(len(prepared.X_train), 0)
        self.assertEqual(
            len(prepared.alignment_splits.train), len(prepared.X_train)
        )
        self.assertTrue(
            (
                prepared.alignment_splits.train["Time"]
                - prepared.alignment_splits.train["WindowStartTime"]
                == 4
            ).all()
        )

    def test_features_and_sequences_reset_at_time_gaps(self) -> None:
        gapped = make_recording("subject", "recording", 0.0).drop(index=[5])
        engineered = add_sensor_features(gapped, window_size=3)
        after_gap = engineered.loc[engineered["Time"] == 6].iloc[0]
        self.assertEqual(int(after_gap[SEGMENT_COLUMN]), 1)
        self.assertTrue(np.isnan(after_gap["Rolling_AccV_Mean"]))

        prepared = prepare_sequence_splits(
            SubjectSplits(
                train=gapped,
                validation=make_recording("validation", "validation-recording", 20.0),
                test=make_recording("test", "test-recording", 40.0),
            ),
            timesteps=3,
            stride=1,
            evaluation_stride=1,
            feature_window_size=3,
        )
        train_alignment = prepared.alignment_splits.train
        self.assertFalse(
            (
                (train_alignment["WindowStartTime"] < 5)
                & (train_alignment["WindowEndTime"] > 5)
            ).any()
        )

    def test_filename_can_be_explicit_subject_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            recording = make_recording("ignored", "abc123", 0.0).drop(
                columns=["RecordingId", "Subject"]
            )
            recording.to_csv(directory / "abc123.csv", index=False)

            loaded = load_recordings(
                directory,
                metadata_path=None,
                assume_recording_is_subject=True,
            )
            self.assertEqual(set(loaded["Subject"]), {"abc123"})
            self.assertEqual(set(loaded["RecordingId"]), {"abc123"})

    def test_missing_targets_are_not_relabelled_as_no_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            recording = make_recording("ignored", "abc123", 0.0).drop(
                columns=["RecordingId", "Subject"]
            )
            recording.loc[0, "StartHesitation"] = np.nan
            recording.to_csv(directory / "abc123.csv", index=False)

            with self.assertRaisesRegex(ValueError, "must not be imputed"):
                load_recordings(
                    directory,
                    metadata_path=None,
                    assume_recording_is_subject=True,
                )

    def test_episode_reconstruction_respects_recordings_and_time_gaps(self) -> None:
        alignment = pd.DataFrame(
            {
                "RecordingId": ["a", "a", "b", "b", "b", "b"],
                "Subject": ["s1", "s1", "s2", "s2", "s2", "s2"],
                "Time": [0, 1, 0, 1, 4, 5],
            }
        )
        truth = np.ones((6, 1), dtype=np.float32)
        scores = np.full((6, 1), 0.9, dtype=np.float32)
        aligned = align_binary_predictions(
            alignment,
            truth,
            scores,
            threshold=0.5,
            sampling_rate_hz=10,
        )
        episodes = reconstruct_episodes(
            aligned,
            label_column="AnyFoGTrue",
        )
        self.assertEqual(len(episodes), 3)
        self.assertEqual(episodes.groupby("RecordingId").size().to_dict(), {"a": 1, "b": 2})

    def test_episode_metrics_known_case(self) -> None:
        sample_count = 100
        alignment = pd.DataFrame(
            {
                "RecordingId": ["recording"] * sample_count,
                "Subject": ["subject"] * sample_count,
                "Time": np.arange(sample_count),
            }
        )
        truth = np.zeros(sample_count, dtype=np.int8)
        truth[10:20] = 1
        truth[50:60] = 1
        scores = np.full(sample_count, 0.1, dtype=float)
        scores[12:22] = 0.9
        scores[70:75] = 0.9

        evaluation = evaluate_episode_predictions(
            alignment,
            truth,
            scores,
            threshold=0.5,
            sampling_rate_hz=10,
            minimum_iou=0.25,
        )
        self.assertEqual(evaluation.true_positive_count, 1)
        self.assertEqual(evaluation.false_positive_count, 1)
        self.assertEqual(evaluation.false_negative_count, 1)
        self.assertAlmostEqual(evaluation.precision, 0.5)
        self.assertAlmostEqual(evaluation.recall, 0.5)
        self.assertAlmostEqual(evaluation.f1, 0.5)
        self.assertAlmostEqual(evaluation.false_alarms_per_minute, 6.0)
        self.assertAlmostEqual(evaluation.mean_onset_delay_seconds, 0.2)
        self.assertAlmostEqual(evaluation.mean_duration_iou, 2 / 3)

    def test_episode_matching_is_one_to_one(self) -> None:
        true_episodes = pd.DataFrame(
            {
                "episode_id": [0, 1],
                "RecordingId": ["r", "r"],
                "Subject": ["s", "s"],
                SEGMENT_COLUMN: [0, 0],
                "start_seconds": [1.0, 3.0],
                "end_seconds": [2.0, 4.0],
                "duration_seconds": [1.0, 1.0],
                "mean_score": [np.nan, np.nan],
                "max_score": [np.nan, np.nan],
            }
        )
        predicted_episodes = pd.DataFrame(
            {
                "episode_id": [0],
                "RecordingId": ["r"],
                "Subject": ["s"],
                SEGMENT_COLUMN: [0],
                "start_seconds": [1.0],
                "end_seconds": [4.0],
                "duration_seconds": [3.0],
                "mean_score": [0.8],
                "max_score": [0.9],
            }
        )
        matches = match_episodes(
            true_episodes,
            predicted_episodes,
            minimum_iou=0.25,
        )
        self.assertEqual(len(matches), 1)

    def test_threshold_is_selected_from_episode_f1(self) -> None:
        alignment = pd.DataFrame(
            {
                "RecordingId": ["r"] * 20,
                "Subject": ["s"] * 20,
                "Time": np.arange(20),
            }
        )
        truth = np.zeros(20, dtype=int)
        truth[5:10] = 1
        scores = np.full(20, 0.2)
        scores[5:10] = 0.8
        threshold, table = select_episode_threshold(
            alignment,
            truth,
            scores,
            sampling_rate_hz=10,
            thresholds=[0.3, 0.9],
        )
        self.assertEqual(threshold, 0.3)
        self.assertEqual(len(table), 2)


if __name__ == "__main__":
    unittest.main()
