"""Data contracts for the two comprehensive TDCS suites; never fit models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

import tdcs_record_rank_gpu as frozen
import tdcs_metadata_context_gpu as recording_metadata

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("StartHesitation", "Walking")
AXES = ("AccV", "AccML", "AccAP")
CLINICAL_NUMERIC = ("Age", "YearsSinceDx", "UPDRSIII_On", "UPDRSIII_Off", "NFOGQ")
CLINICAL_NAMES = tuple(name for field in CLINICAL_NUMERIC for name in ("clinical_" + field, "clinical_" + field + "_missing")) + (
    "clinical_Sex_female", "clinical_Sex_male", "clinical_Sex_unknown")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def array_digest(value):
    return frozen.array_digest(value)


def csv_path(name):
    if Path("/kaggle/input").exists():
        _, metadata = frozen.shared.common.locate_data()
        path = Path(metadata).parent / name
    else:
        fresh = ROOT / "data/metadata/kaggle_20261119" / name
        path = fresh if fresh.is_file() else ROOT / "data/metadata" / name
    if not path.is_file():
        raise FileNotFoundError(f"Required competition input missing: {name}")
    return path


def clinical_features(ids, recording_frame, subjects):
    """Exact visit first; unique subject-level (blank Visit) fallback, never nearest visit."""
    required = {"Subject", "Visit", "Sex", *CLINICAL_NUMERIC}
    if not required <= set(subjects) or not {"Id", "Subject", "Visit"} <= set(recording_frame):
        raise ValueError("Unexpected clinical/recording metadata schema")
    table = subjects.copy()
    table["Subject"] = table.Subject.astype(str).str.strip()
    table["Visit"] = pd.to_numeric(table.Visit, errors="coerce")
    specific, general = {}, {}
    for row in table.to_dict("records"):
        key = row["Subject"] if pd.isna(row["Visit"]) else (row["Subject"], float(row["Visit"]))
        lookup = general if pd.isna(row["Visit"]) else specific
        if key in lookup:
            raise ValueError("Clinical key is not unique; refusing a many-to-many join")
        lookup[key] = row
    records = recording_frame.copy()
    if records.Id.astype(str).duplicated().any():
        raise ValueError("Duplicate recording metadata keys")
    records["Id"] = records.Id.astype(str)
    ordered = records.set_index("Id").reindex(list(map(str, ids)))
    chosen, kinds = [], []
    for row in ordered.to_dict("records"):
        subject = str(row.get("Subject", "")).strip()
        visit = pd.to_numeric(row.get("Visit"), errors="coerce")
        value = specific.get((subject, float(visit))) if pd.notna(visit) else None
        kind = "subject_visit"
        if value is None:
            value, kind = general.get(subject), "subject_level"
        chosen.append(value or {})
        kinds.append(kind if value is not None else "missing")
    selected = pd.DataFrame(chosen).reindex(columns=["Sex", *CLINICAL_NUMERIC])
    columns = []
    for field in CLINICAL_NUMERIC:
        values = pd.to_numeric(selected[field], errors="coerce").to_numpy(float)
        missing = ~np.isfinite(values)
        columns.extend((np.where(missing, 0., values), missing.astype(float)))
    sex = selected.Sex.astype("string").str.strip().str.lower().fillna("")
    female, male = sex.isin(["f", "female"]).to_numpy(), sex.isin(["m", "male"]).to_numpy()
    columns.extend((female, male, ~(female | male)))
    x = np.column_stack(columns).astype(np.float32)
    assert x.shape == (len(ids), 13) and np.isfinite(x).all()
    return x, {"rows": len(ids), "subject_visit_matches": kinds.count("subject_visit"),
        "subject_level_matches": kinds.count("subject_level"), "missing": kinds.count("missing"),
        "rule": "Exact subject/visit or unique blank-Visit subject-level row; no arbitrary visit fallback"}


def kinetic_labels(times, labels, events):
    """Times/boundaries are seconds. Auxiliary FOG target only, never an input."""
    times, labels = np.asarray(times, float), np.asarray(labels)
    values, mask = np.zeros(len(times), np.float32), np.zeros(len(times), np.float32)
    conflict = np.zeros(len(times), bool)
    for row in events.itertuples(index=False):
        value = pd.to_numeric(getattr(row, "Kinetic"), errors="coerce")
        begin = pd.to_numeric(getattr(row, "Init"), errors="coerce")
        end = pd.to_numeric(getattr(row, "Completion"), errors="coerce")
        if value not in (0., 1.) or not np.isfinite(begin) or not np.isfinite(end) or end <= begin:
            continue
        event_type = getattr(row, 'Type', None)
        column = {'StartHesitation': 0, 'Turn': 1, 'Walking': 2}.get(event_type)
        fog = labels[:, column] > 0 if column is not None else labels.any(1)
        selected = (times >= begin) & (times < end) & fog
        conflict |= selected & (mask > 0) & (values != value)
        values[selected], mask[selected] = value, 1.
    mask[conflict] = 0.
    return np.column_stack([values, mask]).astype(np.float32)


def synthetic_context():
    arrays, manifest, native, gate = frozen.synthetic_inputs()
    rng = np.random.default_rng(1607)
    ids = manifest.recording_id.to_numpy(str)
    frame = pd.DataFrame({"Id": ids, "Subject": manifest.subject, "Visit": np.arange(5) + 1,
        "Test": [1, 2, 3, 1, 2], "Medication": ["on", "off", "on", "off", "on"]})
    clinical = pd.DataFrame({"Subject": manifest.subject, "Visit": [np.nan] * 5,
        "Age": np.arange(5) + 60, "Sex": ["M", "F", "M", "F", "M"],
        "YearsSinceDx": 5., "UPDRSIII_On": 20., "UPDRSIII_Off": 30., "NFOGQ": 10.})
    m = recording_metadata.metadata_matrix(ids, frame, require_all=True)
    c, audit = clinical_features(ids, frame, clinical)
    k = np.column_stack([np.arange(len(arrays["y"])) % 2, arrays["y"].any(1)]).astype(np.float32)
    u = {"x": rng.normal(size=(256, 325)).astype(np.float32),
        "subject": np.repeat(np.r_[manifest.subject.to_numpy(str), ["external"]], [40, 40, 40, 40, 40, 56]),
        "record": np.repeat(np.arange(6), [40, 40, 40, 40, 40, 56]).astype(np.int32)}
    return {"arrays": arrays, "manifest": manifest, "native_y": np.concatenate(native),
        "record_metadata": m, "clinical": c, "clinical_audit": audit, "frame": frame,
        "kinetic": k, "gates": frozen.gate_scores(gate, manifest), "gate_payload": gate,
        "native_offsets": np.r_[0, np.cumsum(arrays["lengths"])], "external": u, "smoke_only": True,
        "source_inputs": {}, "parent": "synthetic"}


def load_context():
    parent, config, arrays, manifest = frozen.gate_helper.load_parent()
    ids = manifest.recording_id.to_numpy(str)
    paths = {name: csv_path(name) for name in ("tdcsfog_metadata.csv", "subjects.csv", "events.csv", "daily_metadata.csv")}
    frame = pd.read_csv(paths["tdcsfog_metadata.csv"], dtype={"Id": str, "Subject": str})
    np.testing.assert_array_equal(frame.set_index("Id").Subject.reindex(ids).to_numpy(str), manifest.subject.to_numpy(str))
    meta = recording_metadata.metadata_matrix(ids, frame, require_all=True)
    clinical, audit = clinical_features(ids, frame, pd.read_csv(paths["subjects.csv"], dtype={"Subject": str}))
    if audit["missing"]:
        raise ValueError("Some TDCS subjects lack clinical metadata; review the explicit join audit")
    events = pd.read_csv(paths["events.csv"], dtype={"Id": str})
    groups = dict(tuple(events.groupby("Id", sort=False)))
    y, kinetic = [], np.zeros((len(arrays["y"]), 2), np.float32)
    for i, row in enumerate(manifest.itertuples(index=False)):
        raw = pd.read_csv(row.path, usecols=["Time", "StartHesitation", "Turn", "Walking"])
        labels = raw[["StartHesitation", "Turn", "Walking"]].to_numpy(np.uint8)
        assert np.isin(labels, [0, 1]).all() and (labels.sum(1) <= 1).all()
        chosen = arrays["record"] == i
        grid = arrays["grid"][chosen]
        np.testing.assert_array_equal(labels[grid][:, [0, 2]], arrays["y"][chosen])
        y.append(labels[:, [0, 2]])
        if row.recording_id in groups:
            np.testing.assert_array_equal(raw.Time.to_numpy(), np.arange(len(raw)))
            kinetic[chosen] = kinetic_labels(raw.Time.to_numpy()[grid] / 128., labels[grid], groups[row.recording_id])
        if (i + 1) % 200 == 0:
            print(f"Audited native labels and auxiliary annotations: {i + 1}/815", flush=True)
    gate = frozen.gate_helper.gate_payload()
    return {"arrays": arrays, "manifest": manifest, "native_y": np.concatenate(y),
        "record_metadata": meta, "clinical": clinical, "clinical_audit": audit, "frame": frame,
        "kinetic": kinetic, "gates": frozen.gate_scores(gate, manifest), "gate_payload": gate,
        "native_offsets": np.r_[0, np.cumsum(arrays["lengths"])], "external": None, "smoke_only": False,
        "source_inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()},
        "parent": str(parent), "parent_config": config}


def input_features(context, encoding):
    a = context["arrays"]
    if encoding == "rank_metadata":
        x, _, _ = frozen.transform_features(a, "rank_replace")
        return np.column_stack([x, context["record_metadata"][a["record"]]]).astype(np.float32)
    if encoding in ("local_metadata", "local_rank_metadata"):
        x = a["x"][:, :213] if encoding == "local_metadata" else frozen.record_percentiles(a["x"], a["record"])
        return np.column_stack([x, context["record_metadata"][a["record"]]]).astype(np.float32)
    fields = {"record_metadata": context["record_metadata"],
        "clinical_metadata": np.column_stack([context["record_metadata"], context["clinical"]]),
        "metadata_only": np.column_stack([context["record_metadata"], context["clinical"]]),
        "visit_only": context["record_metadata"][:, :2], "test_only": context["record_metadata"][:, 2:6],
        "medication_only": context["record_metadata"][:, 6:9]}
    if encoding == "sensor":
        return a["x"]
    if encoding not in fields:
        raise ValueError("Unknown fixed feature encoding")
    values = fields[encoding][a["record"]]
    return values.astype(np.float32) if encoding == "metadata_only" else np.column_stack([a["x"], values]).astype(np.float32)


def external_pool(context, output):
    """Bounded label-free competition daily records, with known subject mapping."""
    if context["external"] is not None:
        return context["external"]
    cache = Path(output) / "unlabeled_feature_pool.npz"
    if cache.is_file():
        provenance = json.loads((Path(output) / "unlabeled_pool_provenance.json").read_text())
        if provenance["metadata_sha256"] != digest(csv_path("daily_metadata.csv")):
            raise ValueError("Cached external metadata changed")
        if provenance["cache_sha256"] != digest(cache):
            raise ValueError("Cached external feature pool changed")
        with np.load(cache, allow_pickle=False) as saved:
            context["external"] = {key: saved[key] for key in saved.files}
        return context["external"]
    daily_path = csv_path("daily_metadata.csv")
    daily = pd.read_csv(daily_path, dtype={"Id": str, "Subject": str})
    if daily.Id.duplicated().any() or daily.Subject.isna().any() or daily.Subject.str.strip().eq('').any():
        raise ValueError("Unlabeled records need unique IDs and known subject mappings")
    roots = [daily_path.parent / "unlabeled"]
    paths = {path.stem: path for root in roots if root.exists() for path in root.rglob("*.csv")}
    eligible = daily[daily.Id.isin(paths)].sort_values("Id").head(64)
    if eligible.empty:
        raise FileNotFoundError("No mapped competition unlabeled CSVs are attached; external-data tests cannot run")
    xs, ss, rr = [], [], []
    rows = []
    for i, row in enumerate(eligible.itertuples(index=False)):
        raw = pd.read_csv(paths[row.Id], nrows=32768, usecols=["Time", *AXES])
        prefix_sha = array_digest(raw[["Time", *AXES]].to_numpy(np.float64))
        signal = daily_signal(raw)
        _, x, _ = frozen.shared.features.recording_features(signal)
        xs.append(x)
        ss.append(np.full(len(x), row.Subject))
        rr.append(np.full(len(x), i, np.int32))
        rows.append({"Id": row.Id, "Subject": row.Subject, "source_bytes": paths[row.Id].stat().st_size,
            "used_prefix_sha256": prefix_sha, "source_rows_read": len(raw),
            "documented_input_rate_hz": 100., "output_rate_hz": 128.,
            "acceleration_conversion": "g * 9.80665 to m/s^2",
            "resampled_native_rows": len(signal), "feature_rows": len(x)})
        if (i + 1) % 8 == 0:
            print(f"Unlabeled pool: {i + 1}/{len(eligible)} bounded recordings", flush=True)
    pool = {"x": np.concatenate(xs), "subject": np.concatenate(ss), "record": np.concatenate(rr)}
    np.savez_compressed(cache, **pool)
    frozen.shared.common.write_json(Path(output) / "unlabeled_pool_provenance.json", {
        "metadata_sha256": digest(daily_path), "cache_sha256": digest(cache),
        "records": rows, "bounded_prefix_sampling": True,
        "no_unlabeled_targets_read": True, "heldout_subjects_excluded_separately_per_fold": True})
    context["external"] = pool
    return pool


def daily_signal(raw):
    """Official daily 100-Hz sample index/g units -> TDCS 128-Hz m/s^2.

    The Time column is an integer timestep, not seconds. Do not infer 1 Hz from
    adjacent sample indices. Reject unexpected spacing rather than guess units.
    """
    times = raw.Time.to_numpy(np.float64)
    if len(times) < 1024 or not np.isfinite(times).all() or not np.equal(times, np.floor(times)).all() or not np.equal(np.diff(times), 1).all():
        raise ValueError('Daily Time must be consecutive finite integer sample indices')
    signal = raw[list(AXES)].to_numpy(np.float64)
    if not np.isfinite(signal).all():
        raise ValueError('Nonfinite daily accelerometer values')
    return resample_poly(signal * 9.80665, 32, 25, axis=0)[:16384]


def external_training_indices(pool, heldout_subjects):
    if len(pool["x"]) != len(pool["subject"]) or pd.isna(pool["subject"]).any():
        raise ValueError("External rows require aligned known subjects")
    for record in np.unique(pool['record']):
        if len(np.unique(pool['subject'][pool['record'] == record])) != 1:
            raise ValueError('An external recording must map to exactly one subject')
    result = np.flatnonzero(~np.isin(pool["subject"].astype(str), list(map(str, heldout_subjects))))
    assert set(pool["subject"][result]).isdisjoint(set(heldout_subjects))
    if not len(result):
        raise ValueError("No external rows remain after held-out-subject exclusion")
    return result
