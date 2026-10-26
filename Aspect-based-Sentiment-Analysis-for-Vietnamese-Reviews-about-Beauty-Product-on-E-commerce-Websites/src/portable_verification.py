"""Colab-safe immutable hash verification; does not rerun Unicode normalization preflight."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

BASE_LOCK_SHA256 = '16e9b76b6a71530d08704cadc2d1ddeddf54ca57615c35f8e650ccc57ac86ee8'
REGISTRATION_LOCK_SHA256 = '7279166cb41c3676e835e27bfd3bc53e9330889145fc73a4e91b96fea16a83e4'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_portable(root, expected_commit=None, check_git=True, assets_dir=None):
    root = Path(root).resolve()
    def load(relative):
        return json.loads((root / relative).read_text(encoding='utf-8'))

    freeze_path = root / 'protocols/freeze_lock_v1.json'
    if sha256(freeze_path) != BASE_LOCK_SHA256:
        raise ValueError('Frozen data/evaluator lock SHA-256 mismatch')
    freeze = load('protocols/freeze_lock_v1.json')
    for relative, expected in freeze['artifacts'].items():
        target = root / relative
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f'Frozen artifact hash mismatch: {relative}')

    protocol_path = root / 'protocols/evaluation_protocol_v1.yaml'
    protocol = load('protocols/evaluation_protocol_v1.yaml')
    source_hashes = {}
    for split, source in protocol['source_files'].items():
        path = root / source['path']
        if not path.is_file() or sha256(path) != source['sha256']:
            raise ValueError(f'Official CSV hash mismatch: {source["path"]}')
        source_hashes[split] = source['sha256']

    registration_path = root / 'protocols/experiments/experiment_registration_v1.json'
    registration_lock_path = root / 'protocols/experiments/experiment_registration_lock_v1.json'
    registration = load('protocols/experiments/experiment_registration_v1.json')
    if registration['protocol_sha256'] != sha256(protocol_path) or registration['original_freeze_lock_sha256'] != BASE_LOCK_SHA256:
        raise ValueError('Experiment registration is not bound to frozen data/evaluator v1')
    if sha256(registration_lock_path) != REGISTRATION_LOCK_SHA256:
        raise ValueError('Experiment registration lock SHA-256 mismatch')
    registration_lock = load('protocols/experiments/experiment_registration_lock_v1.json')
    if registration_lock['original_freeze_lock_sha256'] != BASE_LOCK_SHA256:
        raise ValueError('Registration lock points to an unexpected data freeze')
    for relative, expected in registration_lock['artifacts'].items():
        target = root / relative
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f'Registered experiment artifact hash mismatch: {relative}')

    assets = load('reports/modeling_preflight/assets_v1.json')
    audit = load('reports/modeling_preflight/phobert_tokenizer_audit_v1.json')
    if assets['phobert']['revision'] != registration['families']['phobert']['initialization']['revision']:
        raise ValueError('PhoBERT asset revision differs from registered model')
    if assets['vncorenlp']['revision'] != registration['families']['bigru_conv1d']['preprocessing']['segmentation']['revision']:
        raise ValueError('VnCoreNLP asset revision differs from registration')
    if audit['status'] != 'PASS' or audit['registered_max_length'] != registration['families']['phobert']['preprocessing']['max_length']:
        raise ValueError('PhoBERT tokenizer audit no longer matches the registered configuration')
    asset_files_checked = []
    if assets_dir is not None:
        try:
            from training.assets import verify_assets
        except ImportError:
            from .training.assets import verify_assets
        verified = verify_assets(assets_dir, root)
        asset_files_checked = verified['checked_files']

    gate_receipt = load('reports/modeling_preflight/reproducibility_gate_v1.json')
    if gate_receipt.get('status') != 'PASS' or gate_receipt.get('python') != '3.15.1' or gate_receipt.get('unicode') != '17.0.0':
        raise ValueError('Historical exact-runtime reproducibility receipt is missing or invalid')

    commit = None
    if check_git:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        if not expected_commit or not re.fullmatch(r'[0-9a-fA-F]{40}', expected_commit):
            raise ValueError('A full 40-character COMMIT_SHA is required')
        if commit.lower() != expected_commit.lower():
            raise ValueError(f'Git HEAD {commit} does not match requested COMMIT_SHA')
    elif expected_commit is not None and not re.fullmatch(r'[0-9a-fA-F]{40}', expected_commit):
        raise ValueError('COMMIT_SHA must be a full 40-character SHA')

    return {'status': 'PASS', 'portable_immutable_hashes': 'PASS',
            'exact_runtime_reproduced': False,
            'note': 'Portable byte hashes only; this does not reproduce Python 3.15.1 / Unicode 17 normalization.',
            'gate_runtime': {'python': '3.15.1', 'unicode': '17.0.0',
                             'historical_receipt': 'reports/modeling_preflight/reproducibility_gate_v1.json'},
            'modeling_runtime': {'python': registration['runtime']['modeling']['python'],
                                 'unicode': registration['runtime']['modeling']['unicode']},
            'freeze_lock_sha256': BASE_LOCK_SHA256,
            'registration_lock_sha256': REGISTRATION_LOCK_SHA256,
            'source_csv_sha256': source_hashes, 'asset_revisions': {
                'phobert': assets['phobert']['revision'], 'vncorenlp': assets['vncorenlp']['revision']},
            'asset_files_checked': asset_files_checked, 'git_head': commit,
            'training_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--commit-sha', required=True)
    parser.add_argument('--assets-dir', type=Path)
    args = parser.parse_args()
    result = verify_portable(args.root, args.commit_sha, check_git=True, assets_dir=args.assets_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
