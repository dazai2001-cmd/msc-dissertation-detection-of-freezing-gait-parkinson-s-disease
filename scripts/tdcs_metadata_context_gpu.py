"""Matched sensor-only versus inference-available TDCS metadata GPU trials.

Event annotations and task intervals are not features. The only added fields
are Visit, Test and Medication; IDs/subjects are join/split keys only.
"""
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

import tdcs_gpu_regularized_context as parent

shared = parent.shared
ROOT = Path(__file__).resolve().parents[1]
TARGETS = parent.TARGETS
VARIANTS = ('sensor_only', 'sensor_metadata')
PARAMETERS = parent.PARAMETERS['regularized_depth6']
FIELDS = ('Visit', 'Test', 'Medication')
METADATA_NAMES = ('metadata_Visit', 'metadata_Visit_missing', 'metadata_Test_1',
    'metadata_Test_2', 'metadata_Test_3', 'metadata_Test_unknown',
    'metadata_Medication_off', 'metadata_Medication_on', 'metadata_Medication_unknown')
LOCAL_METADATA = ROOT / 'data/metadata/kaggle_20261107/tdcsfog_metadata.csv'


@dataclass(frozen=True)
class Config:
    name: str = 'tdcs_metadata_context_gpu_v11'
    targets: tuple = TARGETS
    folds: tuple = (0, 1, 2, 3, 4)
    negative_divisor: int = 4
    seed: int = 5107
    smoke_rounds: int = 4
    max_seconds: int = 21600


CFG = Config()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {'source_sha256': globals().get('SOURCE_SHA256') or digest(__file__),
        'parent_sources': parent.source_hashes()}


def check_config(cfg):
    if tuple(cfg.targets) != TARGETS or tuple(cfg.folds) != tuple(range(5)):
        raise ValueError('Both difficult targets and all five frozen folds required')
    if min(cfg.negative_divisor, cfg.smoke_rounds, cfg.max_seconds) < 1:
        raise ValueError('Positive sampling and budget required')


def locate_metadata():
    if Path('/kaggle/input').exists():
        return shared.common.locate_data()[1]
    if not LOCAL_METADATA.is_file():
        raise FileNotFoundError('Download the TDCS metadata into data/metadata/kaggle_20261107')
    return LOCAL_METADATA


def metadata_matrix(recording_ids, frame, require_all=False):
    """Fixed, label-free encoding; unknown IDs/conditions never need labels."""
    required = {'Id', *FIELDS}
    if not required <= set(frame):
        raise ValueError('Need Id, Visit, Test and Medication columns')
    data = frame.loc[:, ['Id', *FIELDS]].copy()
    data['Id'] = data.Id.astype(str)
    if data.Id.duplicated().any():
        raise ValueError('Duplicate metadata recording IDs')
    ids = np.asarray(recording_ids, str)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique recording IDs required')
    if require_all and not set(ids) <= set(data.Id):
        raise ValueError('Training/evaluation recording metadata is missing')
    ordered = data.set_index('Id').reindex(ids)
    visit = pd.to_numeric(ordered.Visit, errors='coerce').to_numpy(np.float64)
    valid_visit = np.isfinite(visit) & (visit >= 0)
    test = pd.to_numeric(ordered.Test, errors='coerce').to_numpy(np.float64)
    known_test = np.isin(test, [1., 2., 3.])
    medication = ordered.Medication.astype('string').str.strip().str.lower().fillna('unknown').to_numpy(str)
    known_medication = np.isin(medication, ['off', 'on'])
    x = np.column_stack([np.where(valid_visit, visit, -1.), ~valid_visit,
        test == 1, test == 2, test == 3, ~known_test,
        medication == 'off', medication == 'on', ~known_medication]).astype(np.float32)
    assert x.shape == (len(ids), len(METADATA_NAMES)) and np.isfinite(x).all()
    return x


def metadata_digest(x, recording_ids):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes() +
        json.dumps({'ids': list(map(str, recording_ids)), 'names': METADATA_NAMES}, sort_keys=True).encode()).hexdigest()


def model_features(arrays, metadata, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown metadata comparison')
    x = arrays['x']
    if variant == 'sensor_metadata':
        x = np.column_stack([x, metadata[arrays['record']]]).astype(np.float32)
    assert x.shape == (len(arrays['y']), 334 if variant == 'sensor_metadata' else 325)
    assert np.isfinite(x).all()
    return x


def synthetic_inputs():
    rng = np.random.default_rng(1)
    folds = np.repeat(np.arange(5), 64).astype(np.int8)
    record = folds.astype(np.int16)
    y = np.zeros((320, 2), np.uint8)
    y[:, 0] = (np.arange(320) % 13 == 0).astype(np.uint8)
    y[:, 1] = (np.arange(320) % 11 == 1).astype(np.uint8)
    y[folds == 4] = 0
    manifest = pd.DataFrame({'recording_id': [f'smoke_{i}' for i in range(5)],
        'subject': [f'subject_{i}' for i in range(5)], 'fold': range(5)})
    arrays = {'x': rng.normal(size=(320, 325)).astype(np.float32), 'y': y, 'fold': folds,
        'subject': manifest.subject.to_numpy(str)[record], 'record': record,
        'grid': np.tile(np.arange(64, dtype=np.int32), 5), 'lengths': np.full(5, 64),
        'recording_ids': manifest.recording_id.to_numpy(str)}
    metadata = pd.DataFrame({'Id': manifest.recording_id, 'Visit': [2, 3, 4, 5, 13],
        'Test': [1, 2, 3, 1, 2], 'Medication': ['on', 'off', 'on', 'off', 'on']})
    return arrays, manifest, [y[folds == i].copy() for i in range(5)], metadata


def run_experiment(variant, cfg=CFG, output_root=None, synthetic=None):
    check_config(cfg)
    if variant not in VARIANTS:
        raise ValueError('Unknown metadata comparison')
    smoke = synthetic is not None
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError('Real-data metadata fitting requires Kaggle CUDA; no CPU fallback')
    output_root = Path(output_root or ROOT / 'results/tdcs_metadata_context_gpu' / variant)
    output_root.mkdir(parents=True, exist_ok=True)
    if smoke:
        arrays, manifest, native_labels, metadata_frame = synthetic
        parent_run, parent_config = Path('synthetic'), {}
        metadata_bytes = metadata_frame.to_csv(index=False, lineterminator='\n').encode()
    else:
        parent_run, parent_config, arrays, manifest = parent.parent_loader.load_parent()
        path = locate_metadata()
        metadata_bytes = path.read_bytes()
        metadata_frame = pd.read_csv(path, dtype={'Id': str})
        native_labels = None
        if 'Subject' not in metadata_frame:
            raise ValueError('Need subject metadata to audit split alignment')
        subjects = metadata_frame.set_index('Id').Subject.reindex(manifest.recording_id).to_numpy(str)
        np.testing.assert_array_equal(subjects, manifest.subject.to_numpy(str))
    metadata = metadata_matrix(manifest.recording_id.to_numpy(str), metadata_frame, require_all=True)
    x = model_features(arrays, metadata, variant)
    population = ('synthetic wiring only; not TDCS performance' if smoke else
        manifest[['recording_id', 'subject', 'fold', 'source_bytes']].to_csv(index=False, lineterminator='\n'))
    config = {'configuration': asdict(cfg), 'variant': variant, 'targets': TARGETS,
        'parameters': PARAMETERS, 'sources': source_hashes(), 'smoke_only': smoke,
        'feature_count': x.shape[1], 'metadata_used_as_features': variant == 'sensor_metadata',
        'metadata_fields': FIELDS, 'metadata_feature_names': METADATA_NAMES,
        'metadata_csv_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'metadata_array_sha256': metadata_digest(metadata, arrays['recording_ids']),
        'metadata_snapshot_file': 'metadata_snapshot.csv', 'metadata_transform_fitted': False,
        'parent_fingerprint': parent_run.name, 'parent_feature_array_sha256': parent_config.get('feature_array_sha256'),
        'parent_cache_sha256': parent_config.get('feature_cache_sha256'),
        'population_sha256': hashlib.sha256(population.encode()).hexdigest(),
        'runtime': {name: version(name) for name in ('catboost', 'numpy', 'torch')},
        'used_gpu_indices': [] if smoke else [0],
        'prediction_backend': 'CPU saved-model scoring followed by within-record native interpolation',
        'sampling': 'All training files eligible; all positives plus one-quarter negatives with inverse correction',
        'subject_weights': 'Inverse full-fitting-row counts per training subject, normalized',
        'file_gate_applied': False, 'early_stopping': False,
        'event_boundaries_used_as_features': False, 'task_intervals_used_as_features': False,
        'IDs_or_subjects_used_as_features': False, 'base_OOF_predictions_used_as_features': False,
        'outer_labels_used_for_fitting_or_stopping': False,
        'retained_heads_overwritten': False, 'competition_submission_created': False,
        'confidence_calibrated': False, 'independent_generalization_established': False}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    run = output_root / fingerprint
    run.mkdir(parents=True, exist_ok=True)
    shared.common.write_json(run / 'configuration.json', config)
    (run / 'metadata_snapshot.csv').write_bytes(metadata_bytes)
    np.savez_compressed(run / 'metadata_features.npz', x=metadata,
        recording_ids=arrays['recording_ids'], feature_names=np.asarray(METADATA_NAMES))
    manifest.to_csv(run / 'frozen_tdcs_manifest.csv', index=False)
    (run / 'oof').mkdir(exist_ok=True)
    scores = np.full(arrays['y'].shape, np.nan, np.float32)
    deadline = time.monotonic() + cfg.max_seconds
    for fold in cfg.folds:
        heldout = arrays['fold'] == fold
        training_subjects = sorted(set(arrays['subject'][~heldout]))
        heldout_subjects = sorted(set(arrays['subject'][heldout]))
        assert set(training_subjects).isdisjoint(heldout_subjects)
        directory = run / f'fold_{fold}'
        directory.mkdir(exist_ok=True)
        for index, target in enumerate(TARGETS):
            if time.monotonic() >= deadline:
                raise TimeoutError('Metadata experiment safety budget reached')
            seed = cfg.seed + 1009 * fold + index
            chosen, weights, probability = shared.training_plan(arrays['y'][:, index], arrays['fold'],
                arrays['subject'], fold, seed, cfg.negative_divisor)
            sample_path = directory / (target + '_training_sample.npz')
            model_path, receipt_path = directory / (target + '.cbm'), directory / (target + '_training_receipt.json')
            existing = [p.is_file() for p in (sample_path, model_path, receipt_path)]
            if any(existing):
                if not all(existing):
                    raise ValueError('Incomplete model/sample/receipt; preserve artifacts and choose a fresh root')
                receipt = json.loads(receipt_path.read_text())
                assert receipt['fingerprint'] == fingerprint and receipt['target'] == target
                assert receipt['variant'] == variant and receipt['seed'] == seed
                assert receipt['parameters'] == PARAMETERS and receipt['smoke_only'] == smoke
                assert receipt['trees'] == (cfg.smoke_rounds if smoke else PARAMETERS['iterations'])
                assert receipt['actual_training_backend'] == ('CPU' if smoke else 'GPU')
                assert receipt['sample_sha256'] == digest(sample_path)
                with np.load(sample_path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['indices'], chosen)
                    np.testing.assert_array_equal(saved['weights'], weights)
                model = shared.load_model('catboost', model_path)
            else:
                np.savez_compressed(sample_path, indices=chosen, weights=weights)
                print(f'{variant} fold {fold} {target}: {len(chosen):,} rows, {x.shape[1]} features', flush=True)
                model, backend = parent.fit_model('regularized_depth6', x[chosen], arrays['y'][chosen, index],
                    weights, seed, gpu=not smoke, rounds=cfg.smoke_rounds if smoke else None)
                shared.save_model(model, 'catboost', model_path)
                shared.common.write_json(receipt_path, {'fingerprint': fingerprint, 'variant': variant,
                    'target': target, 'parameters': PARAMETERS, 'seed': seed, 'feature_count': x.shape[1],
                    'training_subjects': training_subjects, 'held_out_subjects': heldout_subjects,
                    'training_grid_rows_before_sampling': int((~heldout).sum()), 'sampled_grid_rows': len(chosen),
                    'negative_sampling_probability': probability, 'all_positive_training_rows_kept': True,
                    'inverse_sampling_correction': True, 'subject_weights_from_training_rows_only': True,
                    'outer_labels_used_for_fitting_or_stopping': False, 'base_OOF_predictions_used_as_features': False,
                    'event_boundaries_used_as_features': False, 'task_intervals_used_as_features': False,
                    'IDs_or_subjects_used_as_features': False, 'early_stopping': False,
                    'sample_sha256': digest(sample_path), 'smoke_only': smoke, **backend})
            scores[heldout, index] = shared.predict_model(model, 'catboost', x[heldout])
            del model
    assert np.isfinite(scores).all()
    np.savez_compressed(run / 'grid_scores.npz', prediction=scores, targets=np.asarray(TARGETS))
    ys, ps = [], []
    for i, row in enumerate(manifest.itertuples(index=False)):
        selected = arrays['record'] == i
        p = shared.expand_native(arrays['grid'][selected], scores[selected], int(arrays['lengths'][i]))
        y = (native_labels[i] if smoke else
            pd.read_csv(row.path, usecols=list(TARGETS)).loc[:, list(TARGETS)].to_numpy(np.uint8))
        assert p.shape == y.shape and np.isin(y, [0, 1]).all()
        np.savez_compressed(run / 'oof' / (row.recording_id + '.npz'), prediction=p, targets=np.asarray(TARGETS))
        ys.append(y); ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    values = shared.common.average_precision(y, p)
    pd.DataFrame([values]).to_csv(run / 'native_oof_metrics.csv', index=False)
    result = {'status': 'complete', 'run_dir': str(run), 'variant': variant, 'targets': TARGETS,
        **values, 'native_rows': len(y), 'smoke_only': smoke, 'retained_heads_unchanged': True,
        'confidence_calibrated': False, 'independent_generalization_established': False,
        'competition_submission_created': False}
    shared.common.write_json(run / 'decision.json', result)
    shared.common.write_json(output_root / 'latest_run.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def smoke_experiment(variant, output_root):
    return run_experiment(variant, replace(CFG, smoke_rounds=4), output_root, synthetic_inputs())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    (smoke_experiment if args.smoke else run_experiment)(args.variant, output_root=args.output_root)
