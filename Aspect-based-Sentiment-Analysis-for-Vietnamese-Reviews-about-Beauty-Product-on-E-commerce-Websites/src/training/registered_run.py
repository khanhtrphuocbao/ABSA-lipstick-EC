"""Explicit TRAIN-04 execution for one authorized registered candidate/seed.

There is no default fit command: callers must select `run`, provide the exact
run ID, and use a manifest with a separate TRAIN-03 authorization record.
This module never loads or evaluates the official test CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .checkpoint import save_checkpoint, serialize_estimator, training_code_sha256
from .dataset import load_train_validation
from .decoding import decode_probabilities
from .prediction_export import write_predictions
from .run_manifest import (collect_observation, validate_run_manifest,
                           verify_authorization)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def official_fit_paths(data_dir):
    """Return only the official train and validation CSV paths."""
    root = Path(data_dir).expanduser().resolve()
    return {'train': root / 'data_train.csv', 'val': root / 'data_val.csv'}


def require_run_authorization(manifest_path, supplied_run_id):
    """Return the manifest only when a write-once authorization matches its bytes."""
    verify_authorization(manifest_path, supplied_run_id)
    return json.loads(Path(manifest_path).read_text(encoding='utf-8'))


def validate_startup_environment(seed, environment=None):
    environment = os.environ if environment is None else environment
    if environment.get('PYTHONHASHSEED') != str(seed):
        raise RuntimeError(f'Launch a fresh process with PYTHONHASHSEED={seed} before running this seed')
    if environment.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Start the process with CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing/initializing CUDA')


def _resolve_manifest_path(root, stored_path):
    path = Path(stored_path).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _validate_authorized_run(root, manifest_path, run_id, assets_dir=None, data_dir=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = require_run_authorization(manifest_path, run_id)
    if manifest.get('state') != 'PREPARED' or manifest.get('run_id') != run_id:
        raise ValueError('Run manifest is not in PREPARED state or the run_id differs')
    selected_data_dir = (Path(data_dir).expanduser().resolve() if data_dir else
                         _resolve_manifest_path(root, manifest['data_policy']['dataset_directory']))
    observation = collect_observation(root, manifest, assets_dir, selected_data_dir)
    report = validate_run_manifest(manifest, observation)
    if report['status'] != 'PASS':
        raise RuntimeError('TRAIN-04 launch gate BLOCKED: ' + '; '.join(report['blockers']))
    for split, path in official_fit_paths(selected_data_dir).items():
        expected = manifest['integrity']['dataset_sha256'][split]['expected_sha256']
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f'{split} CSV does not match the authorized manifest')
    run_directory = _resolve_manifest_path(root, manifest['checkpoint_policy']['run_directory'])
    checkpoint_directory = _resolve_manifest_path(root, manifest['checkpoint_policy']['artifact_directory'])
    if checkpoint_directory.exists() or (run_directory / 'run_result.json').exists():
        raise FileExistsError('This run already has artifacts; resume/overwrite is not registered')
    return root, manifest, selected_data_dir, run_directory, checkpoint_directory


def _canonical_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                        allow_nan=False) + '\n').encode('utf-8')


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite run artifact: {path}')
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(_canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


def _record_event(run_directory, state, **details):
    stamp = datetime.now(timezone.utc)
    event = {'state': state, 'timestamp_utc': stamp.isoformat(), **details}
    name = f"run-event-{stamp.strftime('%Y%m%dT%H%M%S%fZ')}-{state.lower()}.json"
    return _write_once(Path(run_directory) / name, event)


def _checkpoint_metadata(root, manifest, validation_metrics, threshold, epoch):
    protocol = json.loads((Path(root) / 'protocols/evaluation_protocol_v1.yaml').read_text(encoding='utf-8'))
    registration_path = Path(root) / 'protocols/experiments/experiment_registration_v1.json'
    return {
        'candidate_id': manifest['candidate_id'], 'seed': manifest['seed'], 'epoch': int(epoch),
        'threshold': float(threshold), 'validation_metrics': validation_metrics,
        'preprocessing': {'family': manifest['family'], 'candidate_config': manifest['candidate_config'],
                          'asset_revisions': manifest['assets']},
        'dataset_version': protocol['dataset_version'], 'protocol_id': protocol['protocol_id'],
        'protocol_sha256': _sha256(Path(root) / 'protocols/evaluation_protocol_v1.yaml'),
        'registration_id': 'absa_model_comparison_registration_v1',
        'registration_sha256': _sha256(registration_path),
        'code_sha256': training_code_sha256(root), 'model_family': manifest['family'],
    }


def _score_metrics(metrics):
    joint = metrics['joint']
    return {'val_joint_macro_f1': float(joint['macro_f1']),
            'val_joint_micro_f1': float(joint['micro']['f1'])}


def _write_validation_predictions(path, rows, model, threshold, device=None, loader=None):
    import numpy as np
    if loader is None:
        aspect_probabilities, sentiment_probabilities = model.predict_probabilities(
            [row['data'] for row in rows])
    else:
        import torch
        aspect_batches, sentiment_batches, ordered_rows = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
                output = model(**inputs)
                aspect_batches.append(torch.sigmoid(output.aspect_logits).cpu().numpy())
                sentiment_batches.append(torch.softmax(output.sentiment_logits, dim=-1).cpu().numpy())
                ordered_rows.extend(batch['gold_rows'])
        if ordered_rows != rows:
            raise ValueError('Validation loader changed official row order or identity')
        aspect_probabilities = np.concatenate(aspect_batches, axis=0)
        sentiment_probabilities = np.concatenate(sentiment_batches, axis=0)
    predictions = [decode_probabilities(aspect_probabilities[i], sentiment_probabilities[i], threshold,
                                        'val', rows[i]['source_row_index'])
                   for i in range(len(rows))]
    return write_predictions(path, rows, predictions, 'val'), predictions


def _run_tfidf(root, manifest, train_rows, validation_rows, checkpoint_directory, run_directory):
    from .trainer import fit_tfidf_validation
    from .checkpoint import serialize_estimator
    def progress(record):
        print(json.dumps({'run_id': manifest['run_id'], **record}, ensure_ascii=False), flush=True)
    fitted = fit_tfidf_validation(train_rows, validation_rows, manifest['candidate_id'],
                                  seed=manifest['seed'], progress_callback=progress)
    metrics = fitted['validation_metrics']
    checkpoint = checkpoint_directory / f"{manifest['candidate_id']}-seed{manifest['seed']}.json"
    metadata = _checkpoint_metadata(root, manifest, _score_metrics(metrics), fitted['threshold'], 1)
    checkpoint_sha = save_checkpoint(checkpoint, serialize_estimator(fitted['model']), metadata)
    prediction_path, _ = _write_validation_predictions(run_directory / 'validation_predictions.jsonl',
                                                         validation_rows, fitted['model'], fitted['threshold'])
    return {'epoch': 1, 'threshold': float(fitted['threshold']), 'validation_metrics': metrics,
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
            'prediction_path': str(prediction_path), 'history': []}


def _run_neural(root, manifest, train_rows, validation_rows, checkpoint_directory,
                run_directory, assets_dir, progress):
    import torch
    from .preprocessing import recurrent_tokens, segment_vietnamese
    from .trainer import (build_neural_optimizer, fit_neural_candidate,
                          make_phobert_loader, make_recurrent_loader)

    progress({'phase': 'segment_train_started', 'rows': len(train_rows)})
    train_segmented = segment_vietnamese([row['data'] for row in train_rows], assets_dir, root)
    progress({'phase': 'segment_train_completed', 'rows': len(train_segmented)})
    progress({'phase': 'segment_val_started', 'rows': len(validation_rows)})
    val_segmented = segment_vietnamese([row['data'] for row in validation_rows], assets_dir, root)
    progress({'phase': 'segment_val_completed', 'rows': len(val_segmented)})
    candidate_id, seed = manifest['candidate_id'], manifest['seed']
    if manifest['family'] == 'bigru_conv1d':
        from .models.factory import build_recurrent
        from .vocabulary import Vocabulary
        train_tokens = [recurrent_tokens(text) for text in train_segmented]
        val_tokens = [recurrent_tokens(text) for text in val_segmented]
        vocabulary = Vocabulary(max_size=30000, min_frequency=1).fit(train_tokens, source_split='train')
        train_loader = make_recurrent_loader(train_rows, train_tokens, vocabulary, 32, seed, True)
        val_loader = make_recurrent_loader(validation_rows, val_tokens, vocabulary, 32, seed, False)
        model = build_recurrent(candidate_id, len(vocabulary.token_to_id), root)
    elif manifest['family'] == 'phobert':
        from transformers import RobertaModel
        from .models.factory import build_phobert
        from .preprocessing import load_phobert_tokenizer
        tokenizer = load_phobert_tokenizer(assets_dir)
        train_loader = make_phobert_loader(train_rows, train_segmented, tokenizer, 16, seed, True)
        val_loader = make_phobert_loader(validation_rows, val_segmented, tokenizer, 16, seed, False)
        revision = manifest['candidate_config'].get('revision')
        if not revision:
            revision = json.loads((root / 'protocols/experiments/experiment_registration_v1.json').read_text())['families']['phobert']['initialization']['revision']
        encoder = RobertaModel.from_pretrained('vinai/phobert-base', revision=revision,
                                               add_pooling_layer=False)
        model = build_phobert(encoder, candidate_id, root)
    else:
        raise ValueError(f'Unsupported neural family: {manifest["family"]}')
    registration = json.loads((root / 'protocols/experiments/experiment_registration_v1.json').read_text())
    family_key = 'bigru_conv1d' if manifest['family'] == 'bigru_conv1d' else 'phobert'
    family_config = registration['families'][family_key]
    accumulation = family_config['gradient_accumulation']
    maximum_epochs = family_config['max_epochs']
    optimizer_steps = math.ceil(len(train_loader) / accumulation) * maximum_epochs
    optimizer, scheduler, _ = build_neural_optimizer(model, candidate_id, optimizer_steps)
    metadata = _checkpoint_metadata(root, manifest, {}, .5, 1)
    outcome = fit_neural_candidate(model, train_loader, val_loader, optimizer, scheduler,
                                   candidate_id, seed, checkpoint_directory, metadata,
                                   device='cuda', on_epoch_end=progress)
    best = outcome['best']
    prediction_path, _ = _write_validation_predictions(run_directory / 'validation_predictions.jsonl',
                                                        validation_rows, model, best['threshold'],
                                                        device='cuda', loader=val_loader)
    return {'epoch': best['epoch'], 'threshold': float(best['threshold']),
            'validation_metrics': best['validation_metrics'],
            'checkpoint_path': best['checkpoint_path'],
            'checkpoint_sha256': best['checkpoint_sha256'],
            'prediction_path': str(prediction_path), 'history': outcome['history']}


def execute_registered_run(root, manifest_path, run_id, assets_dir=None, data_dir=None,
                           progress_callback=None):
    """Execute exactly one authorized manifest. Intended only for explicit CLI use."""
    authorized_manifest = require_run_authorization(manifest_path, run_id)
    seed = authorized_manifest['seed']
    validate_startup_environment(seed)
    root, manifest, selected_data_dir, run_directory, checkpoint_directory = _validate_authorized_run(
        root, manifest_path, run_id, assets_dir, data_dir)
    from .reproducibility import seed_everything
    seed_report = seed_everything(seed, require_hash_seed=True)
    paths = official_fit_paths(selected_data_dir)
    train_rows, validation_rows = load_train_validation(paths['train'], paths['val'])
    if any(row['source_split'] != 'train' for row in train_rows):
        raise ValueError('Only train rows may be passed to a fitting primitive')
    if any(row['source_split'] != 'val' for row in validation_rows):
        raise ValueError('Only validation rows may be used for checkpoint/threshold selection')
    _record_event(run_directory, 'RUNNING', run_id=run_id, candidate_id=manifest['candidate_id'],
                  seed=seed, source_split_policy={'fit': 'train', 'selection': 'val', 'test_loaded': False})
    print(json.dumps({'run_id': run_id, 'state': 'RUNNING', 'candidate_id': manifest['candidate_id'],
                      'seed': seed, 'train_rows': len(train_rows),
                      'validation_rows': len(validation_rows), 'test_loaded': False}), flush=True)

    def progress(record):
        event = {'run_id': run_id, **record}
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
        if progress_callback is not None:
            progress_callback(event)

    try:
        checkpoint_directory.mkdir(parents=True, exist_ok=False)
        if manifest['family'] == 'tfidf_logistic':
            fitted = _run_tfidf(root, manifest, train_rows, validation_rows,
                                checkpoint_directory, run_directory)
        else:
            fitted = _run_neural(root, manifest, train_rows, validation_rows, checkpoint_directory,
                                 run_directory, assets_dir, progress)
        metrics = _score_metrics(fitted['validation_metrics'])
        result = {'status': 'COMPLETED', 'run_id': run_id,
                  'candidate_id': manifest['candidate_id'], 'family': manifest['family'], 'seed': seed,
                  'best_epoch': int(fitted['epoch']), 'threshold': float(fitted['threshold']),
                  **metrics, 'validation_metrics': fitted['validation_metrics'],
                  'validation_predictions_path': fitted['prediction_path'],
                  'validation_predictions_sha256': _sha256(fitted['prediction_path']),
                  'checkpoint_path': fitted['checkpoint_path'],
                  'checkpoint_sha256': fitted['checkpoint_sha256'],
                  'run_manifest_sha256': _sha256(manifest_path),
                  'training_source_sha256': manifest['source']['training_source_sha256'],
                  'seed_setup': seed_report, 'history': fitted['history'],
                  'test_evaluated': False, 'finished_utc': datetime.now(timezone.utc).isoformat()}
        result_path = _write_once(run_directory / 'run_result.json', result)
        _record_event(run_directory, 'COMPLETED', run_result_sha256=_sha256(result_path))
        return result
    except Exception as exc:
        _record_event(run_directory, 'FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


def aggregate_completed_runs(family, results):
    """Select a registered family candidate from its complete three-seed val runs."""
    for item in results:
        if item.get('status') != 'COMPLETED':
            raise ValueError('Only completed candidate/seed runs may enter family selection')
    try:
        from .trainer import aggregate_candidate_results
    except ImportError:
        from training.trainer import aggregate_candidate_results
    return aggregate_candidate_results(results, family)


def _load_family_results(root, family, artifact_root):
    root = Path(root).resolve()
    registration = json.loads((root / 'protocols/experiments/experiment_registration_v1.json').read_text())
    candidate_to_family = {candidate['id']: family_name
                           for family_name, config in registration['families'].items()
                           for candidate in config['candidates']}
    directory = Path(artifact_root).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    results = []
    for path in sorted(directory.glob('*/run_result.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        if candidate_to_family.get(result.get('candidate_id')) != family:
            continue
        run_directory = path.parent
        manifest_path = run_directory / 'run_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        checkpoint = Path(result['checkpoint_path'])
        if not checkpoint.is_absolute():
            checkpoint = (root / checkpoint).resolve()
        if result.get('run_manifest_sha256') != _sha256(manifest_path):
            raise ValueError(f'Run result is not bound to its manifest: {path}')
        if (manifest.get('candidate_id'), manifest.get('seed')) != (
                result.get('candidate_id'), result.get('seed')):
            raise ValueError(f'Run result candidate/seed differs from its manifest: {path}')
        if (not checkpoint.is_file() or _sha256(checkpoint) != result.get('checkpoint_sha256')
                or result.get('test_evaluated') is not False):
            raise ValueError(f'Run result checkpoint or test-use record is invalid: {path}')
        results.append(result)
    return results


def _cli():
    parser = argparse.ArgumentParser(description='Explicit TRAIN-04 runner; no default command launches training.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    subparsers = parser.add_subparsers(dest='command', required=True)
    run = subparsers.add_parser('run', help='fit exactly one explicitly authorized candidate/seed')
    run.add_argument('--manifest', required=True)
    run.add_argument('--run-id', required=True)
    run.add_argument('--assets-dir')
    run.add_argument('--data-dir')
    summary = subparsers.add_parser('aggregate', help='select a family candidate from completed validation runs')
    summary.add_argument('--family', required=True,
                         choices=('tfidf_logistic', 'bigru_conv1d', 'phobert'))
    summary.add_argument('--artifact-root', default='artifacts/runs')
    summary.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == 'run':
        result = execute_registered_run(root, args.manifest, args.run_id,
                                        args.assets_dir, args.data_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    results = _load_family_results(root, args.family, args.artifact_root)
    selected, per_candidate = aggregate_completed_runs(args.family, results)
    output = {'status': 'FAMILY_SELECTED_ON_VALIDATION', 'family': args.family,
              'selected_candidate_id': selected, 'candidate_validation_summary': per_candidate,
              'test_evaluated': False, 'created_utc': datetime.now(timezone.utc).isoformat()}
    default_output = Path(args.artifact_root).expanduser()
    if not default_output.is_absolute():
        default_output = root / default_output
    output_path = Path(args.output).expanduser() if args.output else default_output / 'family_selection' / f'{args.family}.json'
    if not output_path.is_absolute():
        output_path = root / output_path
    _write_once(output_path, output)
    print(json.dumps({'result': output, 'path': str(output_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli())
