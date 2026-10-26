"""Restore only fingerprint-pinned tokenizer and VnCoreNLP runtime assets.

This utility never downloads pretrained PhoBERT model weights or a JDK. Network
access occurs only when the explicit CLI is invoked; imports are side-effect free.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[2]


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _fetch_bytes(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'ABSA-runtime-assets/1.0'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _safe_target(destination, relative_path):
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe pinned asset path: {relative_path}')
    root = Path(destination).resolve()
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Pinned asset path escapes destination: {relative_path}')
    return target


def restore_asset_items(destination, items, fetch_bytes=_fetch_bytes):
    """Restore supplied pinned files without overwriting existing files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for item in items:
        target = _safe_target(destination, item['relative_path'])
        expected = item['sha256']
        if target.exists():
            if not target.is_file() or _sha256(target.read_bytes()) != expected:
                raise ValueError(f'Existing asset checksum mismatch; refusing overwrite: {target}')
            written.append(str(target))
            continue

        payload = fetch_bytes(item['url'])
        if _sha256(payload) != expected:
            raise ValueError(f'Downloaded asset checksum mismatch: {item["relative_path"]}')

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.asset-', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # A hard link gives write-once semantics: it fails if another
                # process created the destination in the meantime.
                os.link(temporary, target)
            except FileExistsError:
                if not target.is_file() or _sha256(target.read_bytes()) != expected:
                    raise ValueError(f'Concurrent asset checksum mismatch; refusing overwrite: {target}')
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        written.append(str(target))
    return written


def restore(destination, repo_root=None, fetch_bytes=_fetch_bytes):
    root = Path(repo_root) if repo_root else ROOT
    receipt_path = root / 'reports/modeling_preflight/assets_v1.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    items = [item for group in ('phobert', 'vncorenlp') for item in receipt[group]['files']]
    return restore_asset_items(destination, items, fetch_bytes=fetch_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True,
                        help='Destination directory for fingerprint-pinned tokenizer/segmenter assets')
    args = parser.parse_args(argv)
    paths = restore(args.assets.resolve())
    print(json.dumps({'status': 'PASS', 'assets_written_or_verified': paths,
                      'pretrained_weights_downloaded': False, 'jdk_downloaded': False}, indent=2))


if __name__ == '__main__':
    main()
