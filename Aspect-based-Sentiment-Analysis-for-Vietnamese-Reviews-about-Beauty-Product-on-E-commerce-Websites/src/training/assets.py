"""Read-only access to the pinned tokenizer/segmenter receipt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def load_asset_receipt(repo_root=None):
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    return json.loads((root / 'reports/modeling_preflight/assets_v1.json').read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_java_runtime(assets_dir, repo_root=None):
    """Find the registered JDK release, allowing a Linux Colab JDK installation."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    receipt = load_asset_receipt(root)
    directory = Path(assets_dir)
    expected_release = receipt['java']['release'].removeprefix('jdk-')
    candidates = []
    if os.environ.get('JAVA_HOME'):
        candidates.append(Path(os.environ['JAVA_HOME']) / 'bin/java')
    on_path = shutil.which('java')
    if on_path:
        candidates.append(Path(on_path))
    candidates.append(directory / receipt['java']['java_relative_path'])
    seen = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        result = subprocess.run([str(candidate), '-version'], capture_output=True, text=True, timeout=10)
        version_output = result.stdout + result.stderr
        if result.returncode == 0 and expected_release in version_output:
            return {'path': str(candidate), 'release': receipt['java']['release'],
                    'sha256': sha256(candidate), 'version_output': version_output.strip()}
    raise ValueError(f'Registered Java {receipt["java"]["release"]} not found via JAVA_HOME, PATH, or pinned asset')


def verify_assets(assets_dir, repo_root=None, require_java_archive=True):
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
    if require_java_archive:
        item = receipt['java']['file']
        archive = directory / item['relative_path']
        if not archive.is_file() or sha256(archive) != item['sha256']:
            raise ValueError('Pinned Java archive fingerprint mismatch')
        checked.append(item['relative_path'])
    java = resolve_java_runtime(directory, root)
    return {'phobert_revision': receipt['phobert']['revision'],
            'vncorenlp_revision': receipt['vncorenlp']['revision'], 'checked_files': checked,
            'java_release': java['release'], 'java_executable': java['path'],
            'java_sha256': java['sha256'], 'java_version_output': java['version_output']}
