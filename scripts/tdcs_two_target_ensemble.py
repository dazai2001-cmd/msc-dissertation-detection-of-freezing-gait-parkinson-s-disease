"""TDCS StartHesitation/Walking ensembles with honest confidence diagnostics.

Recombine preserved subject-held-out predictions. Turn is neither an output nor
a metric; its rows are NOT filtered out. Existing neural members were originally
trained with three heads, which is disclosed rather than called a new two-head fit.
No learned OOF calibrator, leaderboard calls, or changes to the three-class winner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

import tdcs_multiscale_signal_experiment as base
import tdcs_dense_context_experiment as dense_experiment


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / 'results' / 'tdcs_two_target'
TARGETS = ('StartHesitation', 'Walking')
SOURCE_INDICES = (0, 2)
MEMBERS = ('neural', 'multiscale_hgb', 'dense_context_hgb')
FIXED_CANDIDATES = {
    'current_model_two_target_reference': {
        'StartHesitation': [0.25, 0.25, 0.50], 'Walking': [0.25, 0.25, 0.50]},
    'hesitation_dense_80_walking_retained': {
        'StartHesitation': [0.10, 0.10, 0.80], 'Walking': [0.25, 0.25, 0.50]},
    'hesitation_dense_90_walking_retained': {
        'StartHesitation': [0.05, 0.05, 0.90], 'Walking': [0.25, 0.25, 0.50]},
}
LABELS = {
    'current_model_two_target_reference': 'Current model: two-target reference',
    'hesitation_dense_80_walking_retained': 'Hesitation: 80% dense / Walking: retained',
    'hesitation_dense_90_walking_retained': 'Hesitation: 90% dense / Walking: retained',
    'dense_context_hgb_only_reference': 'Dense sensor model only (reference)',
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def check_scores(scores: np.ndarray, n_rows: int | None = None) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != len(TARGETS):
        raise ValueError('Require two columns in StartHesitation, Walking order; no Turn column')
    if n_rows is not None and len(scores) != n_rows:
        raise ValueError('Native row counts differ')
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Scores must be finite and between zero and one')
    return scores


def combine_members(predictions: dict, weights: dict) -> np.ndarray:
    """Independent per-target weighted scores, NOT softmax/exclusive confidence."""
    if set(weights) != set(TARGETS) or set(predictions) != set(MEMBERS):
        raise ValueError('Exactly the two targets and three declared members are required')
    n_rows = len(predictions[MEMBERS[0]])
    components = [check_scores(predictions[member], n_rows) for member in MEMBERS]
    result = np.empty((n_rows, len(TARGETS)), np.float32)
    for column, target in enumerate(TARGETS):
        coefficients = np.asarray(weights[target], np.float64)
        if coefficients.shape != (len(MEMBERS),) or not np.isfinite(coefficients).all():
            raise ValueError('Invalid ensemble coefficients')
        if (coefficients < 0).any() or abs(coefficients.sum() - 1) > 1e-9:
            raise ValueError('Each target needs nonnegative weights summing to one')
        # The neural contribution internally averages patch-BiLSTM and BiGRU.
        result[:, column] = sum(np.float32(w) * p[:, column] for w, p in zip(coefficients, components))
    return check_scores(result, n_rows)


def score_targets(y: np.ndarray, p: np.ndarray) -> dict:
    check_scores(p, len(y))
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid binary two-target labels')
    result = {t: float(average_precision_score(y[:, i], p[:, i])) for i, t in enumerate(TARGETS)}
    result['two_target_mAP'] = float(np.mean(list(result.values())))
    return result


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Native-row bins, with counts and exact endpoint coverage, including p=1."""
    y, p = np.asarray(y), np.asarray(p, np.float64)
    if y.shape != p.shape or y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Need nonempty aligned binary labels and scalar scores')
    if n_bins < 1 or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid probability bins')
    bin_index = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(bin_index, minlength=n_bins)
    totals = np.bincount(bin_index, weights=p, minlength=n_bins)
    positives = np.bincount(bin_index, weights=y, minlength=n_bins)
    return pd.DataFrame({
        'bin': np.arange(n_bins), 'lower': np.arange(n_bins) / n_bins,
        'upper': (np.arange(n_bins) + 1) / n_bins, 'native_rows': counts,
        'positive_rows': positives.astype(np.int64),
        'mean_score': np.divide(totals, counts, out=np.full(n_bins, np.nan), where=counts > 0),
        'observed_fraction': np.divide(positives, counts, out=np.full(n_bins, np.nan), where=counts > 0),
    })


def confidence_metrics(y: np.ndarray, p: np.ndarray, method: str) -> tuple[list[dict], list[pd.DataFrame]]:
    rows, bin_tables = [], []
    for index, target in enumerate(TARGETS):
        labels, scores = y[:, index].astype(np.float64), p[:, index].astype(np.float64)
        bins = reliability_bins(labels, scores)
        occupied = bins.native_rows > 0
        ece = float(np.sum(bins.loc[occupied, 'native_rows'] *
                          np.abs(bins.loc[occupied, 'mean_score'] - bins.loc[occupied, 'observed_fraction'])) / len(labels))
        clipped = np.clip(scores, 1e-7, 1 - 1e-7)
        prevalence = float(labels.mean())
        q = np.clip(prevalence, 1e-7, 1 - 1e-7)
        rows.append({'method': method, 'target': target, 'prevalence': prevalence,
                     'mean_confidence_score': float(scores.mean()),
                     'brier_loss': float(np.mean((scores - labels) ** 2)),
                     'log_loss': float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
                     'ece_10_uniform_bins': ece,
                     'constant_prevalence_brier': prevalence * (1 - prevalence),
                     'constant_prevalence_log_loss': float(-(q * np.log(q) + (1 - q) * np.log1p(-q))),
                     'confidence_status': 'uncalibrated model score; not clinical certainty'})
        bins.insert(0, 'target', target)
        bins.insert(0, 'method', method)
        bin_tables.append(bins)
    return rows, bin_tables


def locate_sources() -> tuple[Path, Path, pd.DataFrame]:
    sensor = Path(json.loads((ROOT / 'results/tdcs_multiscale_signal/latest_run.json').read_text())['run_dir'])
    dense = Path(json.loads((ROOT / 'results/tdcs_dense_context/latest_run.json').read_text())['run_dir'])
    for path in [sensor, dense / 'selected_candidate']:
        if json.loads((path / 'validation.json').read_text())['status'] != 'passed':
            raise ValueError('A source experiment is not independently validated')
    manifest = pd.read_csv(dense / 'frozen_tdcs_manifest.csv', dtype={'recording_id': str, 'subject': str})
    raw_root, metadata, _ = base.locate_data()
    fresh_manifest = base.build_manifest(raw_root, metadata)
    columns = ['recording_id', 'subject', 'fold', 'source_bytes', 'source_mtime_ns']
    pd.testing.assert_frame_equal(manifest[columns], fresh_manifest[columns], check_dtype=False)
    sensor_manifest = pd.read_csv(sensor / 'frozen_tdcs_manifest.csv', dtype={'recording_id': str, 'subject': str})
    pd.testing.assert_frame_equal(manifest[columns], sensor_manifest[columns], check_dtype=False)
    return sensor, dense, manifest


def load_predictions(sensor: Path, dense: Path, manifest: pd.DataFrame) -> tuple:
    neural_root = ROOT / 'kaggle-output/fog-evidence-weighted/competition/neural/tdcsfog'
    # Deliberately do NOT load the TDCS Turn specialist. Its two retained-target weights were zero.
    neural = {}
    expected = set(manifest.recording_id)
    for member in ['patch_bilstm', 'engineered_bigru']:
        paths = list((neural_root / member).glob('fold_*/seed_42/oof/*.npz'))
        lookup = {p.stem: p for p in paths}
        if len(paths) != len(lookup) or set(lookup) != expected:
            raise ValueError('Neural OOF ID coverage changed')
        neural[member] = lookup
    for directory in [sensor / 'oof', dense / 'dense_hgb/oof', dense / 'selected_candidate/oof']:
        if {p.stem for p in directory.glob('*.npz')} != expected:
            raise ValueError('Sensor OOF coverage changed')
    labels, folds, row_counts = [], [], []
    parts = {member: [] for member in MEMBERS}
    retained = []
    for row in manifest.itertuples(index=False):
        y = pd.read_csv(row.path, usecols=list(TARGETS)).loc[:, list(TARGETS)].to_numpy()
        if not np.isin(y, [0, 1]).all():
            raise ValueError('Raw labels are not binary')
        scores = {}
        for branch in neural:
            path = neural[branch][row.recording_id]
            if f'fold_{row.fold}' not in path.parts:
                raise ValueError('Wrong held-out neural fold')
            with np.load(path, allow_pickle=False) as payload:
                scores[branch] = check_scores(payload['prediction'][:, SOURCE_INDICES], len(y))
        parts['neural'].append(np.float32(0.5) * scores['patch_bilstm'] + np.float32(0.5) * scores['engineered_bigru'])
        for member, directory in [('multiscale_hgb', sensor / 'oof'),
                                  ('dense_context_hgb', dense / 'dense_hgb/oof')]:
            with np.load(directory / f'{row.recording_id}.npz', allow_pickle=False) as payload:
                parts[member].append(check_scores(payload['prediction'][:, SOURCE_INDICES], len(y)))
        with np.load(dense / 'selected_candidate/oof' / f'{row.recording_id}.npz', allow_pickle=False) as payload:
            retained.append(check_scores(payload['prediction'][:, SOURCE_INDICES], len(y)))
        labels.append(y.astype(np.uint8))
        folds.append(np.full(len(y), row.fold, np.int8))
        row_counts.append(len(y))
    return (np.concatenate(labels), np.concatenate(folds), np.asarray(row_counts, np.int64),
            {member: np.concatenate(p) for member, p in parts.items()}, np.concatenate(retained))


def make_figures(metrics: pd.DataFrame, bins: pd.DataFrame, selected: str, run_dir: Path) -> list[Path]:
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    comparison = metrics.set_index('method')
    order = ['current_model_two_target_reference', selected]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    x = np.arange(2)
    for j, (method, label, color, hatch) in enumerate(zip(order, ['Two-target reference', 'Target-specific ensemble'],
                                                        ['#9299a2', '#2463a0'], ['', '//'])):
        bars = ax.bar(x + (j - 0.5) * 0.34, comparison.loc[method, list(TARGETS)].to_numpy(float), 0.34,
                      label=label, color=color, hatch=hatch, edgecolor='#333333', linewidth=0.5)
        ax.bar_label(bars, fmt='%.4f', padding=5)
    ax.set_xticks(x, ['Start hesitation', 'Walking'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Pooled native-row average precision')
    ax.set_title('TDCS two-target comparison: 61 subjects, same five held-out folds')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.2)
    ap_path = run_dir / 'two_target_average_precision.png'
    fig.savefig(ap_path, dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained', sharex='col')
    for column, target in enumerate(TARGETS):
        table = bins.loc[(bins.method == selected) & (bins.target == target)]
        occupied = table.native_rows > 0
        ax = axes[0, column]
        ax.plot([0, 1], [0, 1], '--', color='#333333', label='Ideal calibration')
        ax.plot(table.loc[occupied, 'mean_score'], table.loc[occupied, 'observed_fraction'],
                'o-', color='#2463a0', label='Observed OOF reliability')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Observed positive fraction')
        ax.set_title(target)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
        ax = axes[1, column]
        shares = table.native_rows / table.native_rows.sum()
        ax.bar((table.lower + table.upper) / 2, shares, width=0.09, color='#2463a0')
        ax.set_xlabel('Uncalibrated confidence score')
        ax.set_ylabel('Fraction of native rows')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle('Confidence checks, not proof of certainty: 6,959,762 TDCS native rows')
    calibration_path = run_dir / 'two_target_confidence_reliability.png'
    fig.savefig(calibration_path, dpi=150)
    plt.close(fig)
    return [ap_path, calibration_path]


def predict_sensor_members(raw: np.ndarray, sensor: Path, dense: Path, folds=(0, 1, 2, 3, 4)) -> dict:
    """Predict only SH/Walking with preserved tree models; never load a Turn model."""
    if not folds or any(f not in range(5) for f in folds) or len(set(folds)) != len(folds):
        raise ValueError('Need unique valid model folds')
    grids, features = {}, {}
    grids['multiscale_hgb'], features['multiscale_hgb'], _ = base.sensor_features(raw)
    grids['dense_context_hgb'], features['dense_context_hgb'], _ = dense_experiment.expanded_features(raw)
    result = {member: np.zeros((len(raw), 2), np.float32) for member in grids}
    with threadpool_limits(limits=4):
        for fold in folds:
            for member, directory in [('multiscale_hgb', sensor / f'fold_{fold}'),
                                      ('dense_context_hgb', dense / 'dense_hgb' / f'fold_{fold}')]:
                for index, target in enumerate(TARGETS):
                    model = joblib.load(directory / f'{target}.joblib')
                    sparse = model.predict_proba(features[member])[:, 1]
                    result[member][:, index] += (np.interp(np.arange(len(raw)), grids[member], sparse) / len(folds)).astype(np.float32)
    return result


def run_experiment() -> dict:
    sensor, dense, manifest = locate_sources()
    configuration = {
        'targets': TARGETS, 'members': MEMBERS, 'fixed_candidates': FIXED_CANDIDATES,
        'sensor_run': str(sensor), 'dense_run': str(dense),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_files': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                         [('three_class_registry', ROOT / 'results/tdcs_current_best.json'),
                          ('frozen_manifest', dense / 'frozen_tdcs_manifest.csv')]},
        'confidence_output': 'two independent uncalibrated 0-1 model scores, no softmax',
        'calibrator_fit': False,
        'selection_scope': 'Exploratory fixed class-specific blends; repeatedly reused development folds',
        'historical_neural_training': 'Existing BiGRU and patch BiLSTM originally learned three outputs; projected to SH/Walking',
        'turn_specialist_loaded': False, 'turn_only_rows_dropped': False,
        'held_out_subject_labels_used_for_fitting': False,
    }
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:20]
    run_dir = OUTPUT_ROOT / fingerprint
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'configuration.json', configuration)
    manifest.to_csv(run_dir / 'frozen_tdcs_manifest.csv', index=False)
    print(f'Two-target artifacts: {run_dir}', flush=True)
    y, folds, row_counts, predictions, retained = load_predictions(sensor, dense, manifest)
    if int(row_counts.sum()) != 6959762:
        raise ValueError('The complete TDCS population must remain unchanged')
    reference = combine_members(predictions, FIXED_CANDIDATES['current_model_two_target_reference'])
    if not np.allclose(reference, retained, atol=1.2e-7, rtol=0):
        raise ValueError('The retained three-class projection did not reproduce')
    metrics, fold_metrics = [], []
    for method in [*FIXED_CANDIDATES, 'dense_context_hgb_only_reference']:
        p = combine_members(predictions, FIXED_CANDIDATES[method]) if method in FIXED_CANDIDATES else predictions['dense_context_hgb']
        metrics.append({'method': method, **score_targets(y, p)})
        for fold in range(5):
            mask = folds == fold
            fold_metrics.append({'method': method, 'fold': fold, **score_targets(y[mask], p[mask])})
        print(f"{method}: {metrics[-1]['two_target_mAP']:.6f}", flush=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(run_dir / 'candidate_metrics.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(run_dir / 'candidate_fold_metrics.csv', index=False)
    # Single-member dense scores are a transparent diagnostic, not eligible as an ensemble.
    selected = metrics.loc[metrics.method.isin(FIXED_CANDIDATES)].sort_values('two_target_mAP', ascending=False).iloc[0]
    selected_method = str(selected.method)
    p = combine_members(predictions, FIXED_CANDIDATES[selected_method])
    confidence, bins = [], []
    for method, scores in [('current_model_two_target_reference', reference), (selected_method, p)]:
        if method == selected_method and selected_method == 'current_model_two_target_reference' and confidence:
            continue
        rows, tables = confidence_metrics(y, scores, method)
        confidence.extend(rows)
        bins.extend(tables)
    pd.DataFrame(confidence).to_csv(run_dir / 'confidence_metrics.csv', index=False)
    bins = pd.concat(bins, ignore_index=True)
    bins.to_csv(run_dir / 'reliability_bins.csv', index=False)
    selected_dir = run_dir / 'selected_ensemble'
    (selected_dir / 'oof').mkdir(parents=True, exist_ok=True)
    manifest.to_csv(selected_dir / 'frozen_tdcs_manifest.csv', index=False)
    pd.DataFrame([selected]).to_csv(selected_dir / 'native_oof_metrics.csv', index=False)
    offset = 0
    for row, n_rows in zip(manifest.itertuples(index=False), row_counts):
        np.savez_compressed(selected_dir / 'oof' / f'{row.recording_id}.npz',
                            prediction=p[offset:offset + n_rows], targets=np.asarray(TARGETS),
                            confidence_status=np.asarray('uncalibrated'))
        offset += n_rows
    first_record = manifest.iloc[0]
    preview = pd.DataFrame({'Id': [f'{first_record.recording_id}_{i}' for i in range(min(128, row_counts[0]))],
                            'StartHesitation_confidence_score': p[:min(128, row_counts[0]), 0],
                            'Walking_confidence_score': p[:min(128, row_counts[0]), 1]})
    preview.to_csv(run_dir / 'confidence_output_example.csv', index=False)
    make_figures(metrics, bins, selected_method, run_dir)
    reference_map = float(metrics.loc[metrics.method.eq('current_model_two_target_reference'), 'two_target_mAP'].iloc[0])
    decision = {
        'run_dir': str(run_dir), 'selected_candidate_dir': str(selected_dir),
        'selected_method': selected_method, 'target_weights': FIXED_CANDIDATES[selected_method],
        'two_target_mAP': float(selected.two_target_mAP), 'two_target_reference_mAP': reference_map,
        'delta_two_target_mAP': float(selected.two_target_mAP - reference_map),
        'class_average_precision': {target: float(selected[target]) for target in TARGETS},
        'confidence_status': 'Uncalibrated model scores; calibration diagnostics reported, no certainty claim',
        'old_three_target_mAP_is_not_comparable': True, 'old_three_class_model_unchanged': True,
        'retained_defog_reported_oof_map': 0.817,
        'competition_submission_created': False, 'leaderboard_scoring_used': False,
        'independent_generalization_established': False,
    }
    write_json(run_dir / 'decision.json', decision)
    write_json(OUTPUT_ROOT / 'latest_run.json', decision)
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    return decision


if __name__ == '__main__':
    run_experiment()
