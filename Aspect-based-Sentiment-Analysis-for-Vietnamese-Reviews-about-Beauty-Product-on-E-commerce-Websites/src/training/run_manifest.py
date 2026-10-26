"""Fail-closed TRAIN-03 manifest and launch authorization gates.

This module prepares and validates run metadata only. It never loads a dataset,
constructs a model, downloads weights, or calls a fit function.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .assets import verify_assets
from .checkpoint import training_code_sha256


V2_REGISTRATION = 'protocols/experiments/experiment_registration_v2.json'
V2_LOCK = 'protocols/experiments/experiment_registration_lock_v2.json'
V1_REGISTRATION = 'protocols/experiments/experiment_registration_v1.json'
V1_LOCK = 'protocols/experiments/experiment_registration_lock_v1.json'
FREEZE_LOCK = 'protocols/freeze_lock_v1.json'
EVALUATION_PROTOCOL = 'protocols/evaluation_protocol_v1.yaml'
PACKAGE_MAP = {'scikit-learn': 'scikit-learn', 'torch': 'torch', 'transformers': 'transformers'}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(root: Path, relative: str):
    return json.loads((root / relative).read_text(encoding='utf-8'))


def _runtime_expected(registration):
    modeling = registration['runtime']['modeling']
    return {'python': modeling['python'], 'unicode': modeling['unicode'],
            'packages': modeling['package_versions']}


def _runtime_observed():
    packages = {}
    for package in PACKAGE_MAP:
        try:
            packages[package] = importlib.metadata.version(PACKAGE_MAP[package])
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'python': platform.python_version(), 'unicode': unicodedata.unidata_version,
            'packages': packages}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(root), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _git_observation(root: Path):
    try:
        commit = _git(root, 'rev-parse', 'HEAD')
        status = _git(root, 'status', '--porcelain')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': None, 'dirty': True, 'dirty_sha256': None, 'error': 'git metadata unavailable'}
    dirty = bool(status)
    return {'commit': commit, 'dirty': dirty,
            'dirty_sha256': hashlib.sha256(status.encode('utf-8')).hexdigest() if dirty else None}


def _manifest_code_sha256(root: Path):
    files = [*(root / 'src/training').rglob('*.py'),
             root / 'src/absa_protocol.py', root / 'src/evaluate_predictions.py',
             root / 'src/modeling_preflight.py', root / 'src/VnCoreNLPBatch.java']
    digest = hashlib.sha256()
    names = []
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        names.append(relative)
        digest.update(relative.encode('utf-8') + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest(), names


def _artifact_integrity(root: Path, data_dir=None):
    registration_lock = _json(root, V2_LOCK)
    artifacts = {}
    for relative, expected in sorted(registration_lock['artifacts'].items()):
        path = root / relative
        artifacts[relative] = {'expected_sha256': expected,
                               'observed_sha256': _sha256(path) if path.is_file() else None}
    freeze_path = root / FREEZE_LOCK
    protocol_path = root / EVALUATION_PROTOCOL
    registration_path = root / V2_REGISTRATION
    return {
        'artifacts': artifacts,
        'freeze_lock_sha256': _sha256(freeze_path),
        'protocol_sha256': _sha256(protocol_path),
        'registration_v1_sha256': _sha256(root / V1_REGISTRATION),
        'registration_v2_sha256': _sha256(registration_path),
        'registration_lock_v1_sha256': _sha256(root / V1_LOCK),
        'registration_lock_v2_sha256': _sha256(root / V2_LOCK),
        'exact_runtime_gate_receipt_sha256': _sha256(root / 'reports/modeling_preflight/reproducibility_gate_v1.json'),
        'dataset_sha256': _dataset_hashes(root, protocol_path, data_dir),
    }


def _dataset_hashes(root: Path, protocol_path: Path, data_dir=None):
    # The frozen evaluation protocol is JSON-compatible YAML. Hash bytes only;
    # never parse rows here or expose test labels to training code.
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    expected = protocol['source_files']
    result = {}
    for split, item in expected.items():
        path = (Path(data_dir) / Path(item['path']).name) if data_dir else root / item['path']
        result[split] = {'path': item['path'], 'expected_sha256': item['sha256'],
                         'observed_sha256': _sha256(path) if path.is_file() else None}
    return result


def _find_candidate(registration, candidate_id):
    for family, config in registration['families'].items():
        for candidate in config['candidates']:
            if candidate['id'] == candidate_id:
                return family, candidate
    raise ValueError(f'{candidate_id!r} is not a registered candidate')


def build_run_manifest(root, candidate_id, seed, run_id, artifact_root='artifacts/runs', data_dir=None):
    """Build a PREPARED manifest. It has no authorization and cannot start fit."""
    root = Path(root).resolve()
    registration = _json(root, V2_REGISTRATION)
    family, candidate = _find_candidate(registration, candidate_id)
    if type(seed) is not int or seed not in registration['seeds']:
        raise ValueError(f'{seed!r} is not a registered seed')
    if (not run_id or run_id in {'.', '..'}
            or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for char in run_id)):
        raise ValueError('run_id must contain only letters, digits, dot, underscore, or hyphen')
    artifact_root = Path(artifact_root)
    artifact_root_value = str(artifact_root if artifact_root.is_absolute()
                              else artifact_root.as_posix())
    run_directory = str(artifact_root / run_id)
    dataset_directory = Path(data_dir).expanduser().resolve() if data_dir else root / 'data'
    integrity = _artifact_integrity(root, dataset_directory)
    registration_lock = _json(root, V2_LOCK)
    freeze_lock = _json(root, FREEZE_LOCK)
    gate_receipt = _json(root, 'reports/modeling_preflight/reproducibility_gate_v1.json')
    if gate_receipt.get('status') != 'PASS':
        raise ValueError('Historical exact-runtime data gate receipt is not PASS')
    if (gate_receipt.get('python'), gate_receipt.get('unicode')) != (
            freeze_lock.get('python'), freeze_lock.get('unicode_version')):
        raise ValueError('Historical exact-runtime receipt does not match the frozen data runtime')
    if gate_receipt.get('freeze_lock_sha256_after') != integrity['freeze_lock_sha256']:
        raise ValueError('Historical exact-runtime receipt does not confirm the current freeze lock')
    source_sha, source_files = _manifest_code_sha256(root)
    selection = registration['selection']
    return {
        'schema': 'absa-training-run-manifest-v1',
        'state': 'PREPARED',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': run_id,
        'candidate_id': candidate_id,
        'family': family,
        'candidate_config': candidate,
        'seed': seed,
        'registration': {'registration_id': registration['registration_id'],
                         'registration_sha256': integrity['registration_v2_sha256'],
                         'base_registration_sha256': integrity['registration_v1_sha256'],
                         'registration_lock_sha256': integrity['registration_lock_v2_sha256'],
                         'protocol_sha256': registration['protocol_sha256'],
                         'source_train_sha256': registration['source_train_sha256'],
                         'original_freeze_lock_sha256': registration_lock['original_freeze_lock_sha256']},
        'source': {'git_commit': _git_observation(root)['commit'],
                   'training_code_sha256': training_code_sha256(root),
                   'training_source_sha256': source_sha,
                   'training_source_files': source_files},
        'integrity': integrity,
        'runtime': {'expected': _runtime_expected(registration), 'observed_at_prepare': _runtime_observed(),
                    'exact_data_gate_receipt': 'reports/modeling_preflight/reproducibility_gate_v1.json',
                    'exact_data_gate_receipt_sha256': _sha256(root / 'reports/modeling_preflight/reproducibility_gate_v1.json')},
        'assets': {'required': family != 'tfidf_logistic',
                   'phobert_revision': registration['families']['phobert']['initialization']['revision'],
                   'vncorenlp_revision': registration['families']['phobert']['preprocessing']['segmentation']['revision'],
                   'java_release': registration['families']['bigru_conv1d']['preprocessing']['segmentation']['java_release']},
        'data_policy': {'fit_splits': ['train'], 'selection_split': 'val',
                        'test_used_for_selection': False, 'dataset_hashes_are_byte_only': True,
                        'dataset_directory': str(dataset_directory)},
        'evaluation_policy': {'primary_metric': selection['primary'],
                              'checkpoint_selection': selection['checkpoint_selection'],
                              'threshold_grid': selection['aspect_threshold_grid'],
                              'threshold_tie_break': selection['threshold_tie_break'],
                              'final_test_policy': selection['test']},
        'checkpoint_policy': {'new_run_only': True, 'overwrite_existing': False,
                              'optimizer_resume_supported': False,
                              'resume_status': 'NOT_IMPLEMENTED',
                              'registry_directory': artifact_root_value,
                              'run_directory': run_directory,
                              'artifact_directory': str(Path(run_directory) / 'checkpoints')},
        'launch_authorization': None,
    }


def validate_run_manifest(manifest, observed):
    """Compare a manifest with current observations; return PASS/BLOCKED, never fit."""
    blockers = []
    if manifest.get('schema') != 'absa-training-run-manifest-v1' or manifest.get('state') != 'PREPARED':
        blockers.append('manifest schema/state is not a prepared TRAIN-03 manifest')
    git = observed.get('git', {})
    if not git.get('commit') or git.get('commit') != manifest.get('source', {}).get('git_commit'):
        blockers.append('Git commit differs from manifest or is unavailable')
    if git.get('dirty'):
        blockers.append('Git worktree is dirty; record the trainer in a committed checkout before validation')
    runtime_expected = manifest.get('runtime', {}).get('expected', {})
    registration = observed.get('registration_contract', {})
    if manifest.get('registration', {}).get('registration_id') != registration.get('registration_id'):
        blockers.append('manifest registration ID is not the active registered version')
    if manifest.get('registration', {}).get('registration_sha256') != registration.get('registration_sha256'):
        blockers.append('manifest registration SHA-256 is not the active registered file')
    if runtime_expected != registration.get('runtime_expected'):
        blockers.append('manifest runtime pins differ from the registered v2 runtime')
    if manifest.get('candidate_id') not in registration.get('registered_seeds_by_candidate', {}):
        blockers.append('candidate is absent from the active registration')
    else:
        registered = registration['registered_seeds_by_candidate'][manifest['candidate_id']]
        if manifest.get('seed') not in registered['seeds']:
            blockers.append('seed is not registered for this run')
        if manifest.get('family') != registered['family'] or manifest.get('candidate_config') != registered['config']:
            blockers.append('candidate family/config differs from the registered candidate')
    data_policy = manifest.get('data_policy', {})
    if (data_policy.get('fit_splits') != ['train'] or data_policy.get('selection_split') != 'val'
            or data_policy.get('test_used_for_selection') is not False):
        blockers.append('manifest violates train-only fit / validation-only selection policy')
    runtime = observed.get('runtime', {})
    for name in ('python', 'unicode'):
        if runtime.get(name) != runtime_expected.get(name):
            blockers.append(f'{name} runtime mismatch: expected {runtime_expected.get(name)}, got {runtime.get(name)}')
    for package, expected in runtime_expected.get('packages', {}).items():
        if runtime.get('packages', {}).get(package) != expected:
            blockers.append(f'{package} version mismatch: expected {expected}, got {runtime.get("packages", {}).get(package)}')
    if observed.get('artifacts') != manifest.get('integrity', {}).get('artifacts'):
        blockers.append('registered immutable artifact hashes differ from the manifest')
    for name, record in manifest.get('integrity', {}).get('artifacts', {}).items():
        if record.get('expected_sha256') != record.get('observed_sha256'):
            blockers.append(f'pinned artifact hash mismatch: {name}')
    if observed.get('integrity') != manifest.get('integrity'):
        blockers.append('frozen protocol, registration, lock, or source CSV hashes changed since manifest creation')
    if manifest.get('integrity', {}).get('freeze_lock_sha256') != manifest.get('registration', {}).get('original_freeze_lock_sha256'):
        blockers.append('freeze lock does not match the v2 registration lock reference')
    if manifest.get('integrity', {}).get('protocol_sha256') != manifest.get('registration', {}).get('protocol_sha256'):
        blockers.append('evaluation protocol hash does not match the v2 registration reference')
    train_record = manifest.get('integrity', {}).get('dataset_sha256', {}).get('train', {})
    if train_record.get('expected_sha256') != manifest.get('registration', {}).get('source_train_sha256'):
        blockers.append('train CSV hash does not match the v2 registration source hash')
    for split, record in manifest.get('integrity', {}).get('dataset_sha256', {}).items():
        current = record.get('observed_sha256')
        if not current or current != record.get('expected_sha256'):
            blockers.append(f'{split} CSV hash does not match the frozen evaluation protocol')
    if manifest.get('assets', {}).get('required'):
        assets = observed.get('assets', {})
        if assets.get('status') != 'PASS' or not assets.get('checked_files'):
            blockers.append('pinned model assets are missing or were not fingerprint-verified')
        if assets.get('phobert_revision') != manifest['assets'].get('phobert_revision'):
            blockers.append('PhoBERT asset revision differs from the registered revision')
        if assets.get('vncorenlp_revision') != manifest['assets'].get('vncorenlp_revision'):
            blockers.append('VnCoreNLP asset revision differs from the registered revision')
        if assets.get('java_release') != manifest['assets'].get('java_release'):
            blockers.append('Java runtime release differs from the registered release')
    if manifest.get('family') != 'tfidf_logistic':
        hardware = observed.get('hardware', {})
        if not hardware.get('cuda_available') or not hardware.get('gpu_name'):
            blockers.append('registered neural run requires a CUDA GPU')
        if (hardware.get('vram_gib') or 0) < 16:
            blockers.append('registered neural run requires a single GPU with at least 16 GiB VRAM')
    if observed.get('checkpoint_path_exists'):
        blockers.append('run artifact/checkpoint path already exists; overwrite is forbidden')
    inventory = observed.get('registered_run_inventory', {})
    if inventory.get('status') != 'PASS':
        blockers.extend(inventory.get('conflicts') or ['registered run inventory is unavailable'])
    if observed.get('training_source_sha256') != manifest.get('source', {}).get('training_source_sha256'):
        blockers.append('training/evaluation source code SHA-256 differs from the manifest')
    return {'status': 'BLOCKED' if blockers else 'PASS', 'run_id': manifest.get('run_id'),
            'candidate_id': manifest.get('candidate_id'), 'seed': manifest.get('seed'),
            'blockers': blockers, 'fit_started': False}


def _canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                        allow_nan=False) + '\n').encode('utf-8')


def _write_once(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite: {path}')
    descriptor, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def verify_authorization(manifest_path, supplied_run_id):
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    authorization_path = manifest_path.parent / 'run_authorization.json'
    if not authorization_path.is_file():
        raise ValueError('Explicit run authorization is missing')
    authorization = json.loads(authorization_path.read_text(encoding='utf-8'))
    if supplied_run_id != manifest.get('run_id') or authorization.get('run_id') != manifest.get('run_id'):
        raise ValueError('Supplied run_id does not match manifest authorization')
    if authorization.get('manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('Manifest changed after authorization')
    return authorization


def _registered_run_inventory(root: Path, current_manifest):
    registration = _json(root, V2_REGISTRATION)
    registry = Path(current_manifest['checkpoint_policy']['registry_directory'])
    if not registry.is_absolute():
        registry = root / registry
    manifests = []
    for path in sorted(registry.glob('*/run_manifest.json')):
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return {'status': 'BLOCKED', 'count': None,
                    'conflicts': [f'malformed manifest: {path}']}
        if item.get('run_id') != current_manifest.get('run_id'):
            authorized = (path.parent / 'run_authorization.json').is_file()
            if item.get('fit_started') is True or authorized:
                manifests.append(item)
    duplicate = [item.get('run_id') for item in manifests
                 if item.get('candidate_id') == current_manifest.get('candidate_id')
                 and item.get('seed') == current_manifest.get('seed')]
    maximum = registration['budget']['maximum_main_training_runs']
    conflicts = [f'candidate/seed already has a manifest: {run_id}' for run_id in duplicate]
    if len(manifests) >= maximum:
        conflicts.append(f'registered main-fit budget of {maximum} manifests is exhausted')
    return {'status': 'BLOCKED' if conflicts else 'PASS', 'count': len(manifests),
            'maximum': maximum, 'conflicts': conflicts}


def collect_observation(root, manifest, assets_dir=None, data_dir=None):
    root = Path(root).resolve()
    selected_data_dir = (Path(data_dir).expanduser().resolve() if data_dir else
                         Path(manifest.get('data_policy', {}).get('dataset_directory', root / 'data')))
    asset_status = {'status': 'NOT_REQUIRED', 'checked_files': []}
    if manifest['assets']['required']:
        if assets_dir is None:
            asset_status = {'status': 'BLOCKED', 'checked_files': []}
        else:
            try:
                report = verify_assets(assets_dir, root, require_java_archive=False)
                asset_status = {'status': 'PASS', 'checked_files': report['checked_files'],
                                'phobert_revision': report['phobert_revision'],
                                'vncorenlp_revision': report['vncorenlp_revision'],
                                'java_release': report['java_release'],
                                'java_sha256': report['java_sha256']}
            except (OSError, ValueError) as exc:
                asset_status = {'status': 'BLOCKED', 'checked_files': [], 'reason': str(exc)}
    integrity = _artifact_integrity(root, selected_data_dir)
    registration_data = _json(root, V2_REGISTRATION)
    candidate_map = {}
    for family, family_config in registration_data['families'].items():
        for candidate in family_config['candidates']:
            candidate_map[candidate['id']] = {'family': family, 'config': candidate,
                                              'seeds': registration_data['seeds']}
    hardware = {'cuda_available': False, 'gpu_name': None, 'vram_gib': 0.0}
    try:
        import torch
        hardware['cuda_available'] = torch.cuda.is_available()
        if hardware['cuda_available']:
            device = torch.cuda.get_device_properties(0)
            hardware.update(gpu_name=device.name, vram_gib=device.total_memory / (1024 ** 3),
                            cuda_version=torch.version.cuda)
    except ImportError:
        pass
    code_sha, _ = _manifest_code_sha256(root)
    return {'git': _git_observation(root), 'runtime': _runtime_observed(),
            'artifacts': integrity['artifacts'], 'assets': asset_status,
            'integrity': integrity, 'training_source_sha256': code_sha,
            'registration_contract': {'registration_id': registration_data['registration_id'],
                                      'registration_sha256': integrity['registration_v2_sha256'],
                                      'runtime_expected': _runtime_expected(registration_data),
                                      'registered_seeds_by_candidate': candidate_map},
            'registered_run_inventory': _registered_run_inventory(root, manifest),
            'hardware': hardware,
            'checkpoint_path_exists': (root / manifest['checkpoint_policy']['artifact_directory']).exists()}


def _cli():
    parser = argparse.ArgumentParser(description='TRAIN-03 run manifest gates only; never starts model fitting.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare', help='write a new un-authorized run manifest')
    prepare.add_argument('--candidate', required=True)
    prepare.add_argument('--seed', type=int, required=True)
    prepare.add_argument('--run-id', required=True)
    prepare.add_argument('--artifact-root', default='artifacts/runs')
    prepare.add_argument('--data-dir', help='directory holding all four byte-identical frozen CSVs')
    validate = subparsers.add_parser('validate', help='check manifest gates without fitting')
    validate.add_argument('--manifest', required=True)
    validate.add_argument('--assets-dir')
    validate.add_argument('--data-dir')
    authorize = subparsers.add_parser('authorize', help='record a separate explicit gate authorization')
    authorize.add_argument('--manifest', required=True)
    authorize.add_argument('--confirm-run-id', required=True)
    authorize.add_argument('--assets-dir')
    authorize.add_argument('--data-dir')
    launch = subparsers.add_parser('launch-check', help='verify the gate only; this command never fits')
    launch.add_argument('--manifest', required=True)
    launch.add_argument('--run-id', required=True)
    launch.add_argument('--assets-dir')
    launch.add_argument('--data-dir')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == 'prepare':
        manifest = build_run_manifest(root, args.candidate, args.seed, args.run_id,
                                      args.artifact_root, args.data_dir)
        output = (Path(manifest['checkpoint_policy']['run_directory']) / 'run_manifest.json')
        if not output.is_absolute():
            output = root / output
        output = output.resolve()
        _write_once(output, _canonical_bytes(manifest))
        print(json.dumps({'status': 'PREPARED', 'manifest': str(output), 'fit_started': False}, indent=2))
        return 0
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    observation = collect_observation(root, manifest, getattr(args, 'assets_dir', None),
                                      getattr(args, 'data_dir', None))
    report = validate_run_manifest(manifest, observation)
    if args.command == 'validate':
        report['validated_utc'] = datetime.now(timezone.utc).isoformat()
        report['observations'] = observation
        report_path = manifest_path.parent / f"validation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        _write_once(report_path, _canonical_bytes(report))
        print(json.dumps({'report': str(report_path), **report}, indent=2, ensure_ascii=False))
        return 0 if report['status'] == 'PASS' else 2
    if report['status'] != 'PASS':
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    if args.command == 'authorize':
        if args.confirm_run_id != manifest['run_id']:
            raise SystemExit('--confirm-run-id must exactly match the manifest run_id')
        payload = {'run_id': manifest['run_id'], 'manifest_sha256': _sha256(manifest_path),
                   'authorized_utc': datetime.now(timezone.utc).isoformat(),
                   'authorization_scope': 'one registered candidate/seed; a separate explicit TRAIN-04 run command is required'}
        _write_once(manifest_path.parent / 'run_authorization.json', _canonical_bytes(payload))
        print(json.dumps({'status': 'AUTHORIZED_FOR_FUTURE_TRAIN04_ONLY', 'fit_started': False}, indent=2))
        return 0
    verify_authorization(manifest_path, args.run_id)
    print('Gate PASS; this check did not start training. Use the separate TRAIN-04 run command only when intended.')
    return 3


if __name__ == '__main__':
    raise SystemExit(_cli())
