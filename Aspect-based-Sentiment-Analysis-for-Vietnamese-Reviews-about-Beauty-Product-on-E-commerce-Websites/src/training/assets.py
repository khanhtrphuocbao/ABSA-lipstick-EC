"""Read-only access to the pinned tokenizer/segmenter receipt."""
import hashlib
import json
from pathlib import Path


def load_asset_receipt(repo_root=None):
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    return json.loads((root / 'reports/modeling_preflight/assets_v1.json').read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(assets_dir, repo_root=None):
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    receipt = load_asset_receipt(root)
    directory = Path(assets_dir)
    checked = []
    for group in ('phobert', 'vncorenlp'):
        for item in receipt[group]['files']:
            path = directory / item['relative_path']
            if not path.is_file() or sha256(path) != item['sha256']:
                raise ValueError(f'Pinned model asset mismatch: {item["relative_path"]}')
            checked.append(item['relative_path'])
    item = receipt['java']['file']
    archive = directory / item['relative_path']
    if not archive.is_file() or sha256(archive) != item['sha256']:
        raise ValueError('Pinned Java archive fingerprint mismatch')
    checked.append(item['relative_path'])
    return {'phobert_revision': receipt['phobert']['revision'],
            'vncorenlp_revision': receipt['vncorenlp']['revision'], 'checked_files': checked}
