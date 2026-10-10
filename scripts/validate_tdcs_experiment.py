"""Independent native-row AP and fold checks for the separate TDCS experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("StartHesitation", "Turn", "Walking")


def independent_ap(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Average precision from ranked threshold groups, including exact score ties."""
    order = np.argsort(-probabilities, kind="stable")
    ranked_y = labels[order]
    ranked_p = probabilities[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(ranked_p) != 0), len(ranked_p) - 1]
    true_positives = np.cumsum(ranked_y, dtype=np.int64)[threshold_ends]
    new_positives = np.diff(np.r_[0, true_positives])
    precision = true_positives / (threshold_ends + 1)
    return float(np.sum(new_positives * precision) / labels.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Native-row OOF directory; defaults to the original sensor experiment")
    args = parser.parse_args()
    if args.run_dir:
        run_dir = args.run_dir.resolve()
    else:
        latest = json.loads((ROOT / "results" / "tdcs_multiscale_signal" / "latest_run.json").read_text())
        run_dir = Path(latest["run_dir"])
    manifest = pd.read_csv(run_dir / "frozen_tdcs_manifest.csv", dtype={"recording_id": str, "subject": str})
    metadata = pd.read_csv(ROOT / "data" / "metadata" / "tdcsfog_metadata.csv", dtype={"Id": str, "Subject": str})
    check = manifest.merge(metadata[["Id", "Subject"]], left_on="recording_id", right_on="Id", validate="one_to_one")
    assert len(check) == len(manifest) == 815
    assert (check.subject == check.Subject).all()
    assert check.groupby("subject").fold.nunique().max() == 1
    assert "4dc2f8" not in set(check.subject)
    prediction_ids = {path.stem for path in (run_dir / "oof").glob("*.npz")}
    assert prediction_ids == set(manifest.recording_id)
    label_parts, prediction_parts = [], []
    fold_positive_rows = {fold: np.zeros(3, np.int64) for fold in range(5)}
    for row in manifest.itertuples(index=False):
        y = pd.read_csv(row.path, usecols=list(TARGETS)).loc[:, list(TARGETS)].to_numpy(np.uint8)
        with np.load(run_dir / "oof" / f"{row.recording_id}.npz") as payload:
            p = payload["prediction"]
        assert y.shape == p.shape and np.isfinite(p).all()
        assert np.isin(y, [0, 1]).all() and ((p >= 0) & (p <= 1)).all()
        fold_positive_rows[row.fold] += y.sum(axis=0, dtype=np.int64)
        label_parts.append(y)
        prediction_parts.append(p)
    labels = np.concatenate(label_parts)
    predictions = np.concatenate(prediction_parts)
    assert len(labels) == 6959762
    independent = {target: independent_ap(labels[:, index], predictions[:, index])
                   for index, target in enumerate(TARGETS)}
    independent["mAP"] = float(np.mean(list(independent.values())))
    reported = pd.read_csv(run_dir / "native_oof_metrics.csv").iloc[0]
    for target in [*TARGETS, "mAP"]:
        assert abs(independent[target] - reported[target]) < 1e-10, (target, independent[target], reported[target])
    validation = {"status": "passed", "native_rows": len(labels), "recordings": len(manifest),
                  "subjects": int(manifest.subject.nunique()), "prediction_ids_exact_match": True,
                  "subject_disjoint": True, "independent_rank_based_AP": independent,
                  "fold_positive_rows": {fold: dict(zip(TARGETS, counts.tolist()))
                                         for fold, counts in fold_positive_rows.items()}}
    (run_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
