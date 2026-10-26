"""Download only the asset URLs and SHA-256 values already pinned in assets_v1.json."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def restore(destination):
    receipt=json.loads((ROOT/'reports/modeling_preflight/assets_v1.json').read_text())
    items=[r for key in ['phobert','vncorenlp'] for r in receipt[key]['files']]+[receipt['java']['file']]
    for item in items:
        target=destination/item['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            request=urllib.request.Request(item['url'],headers={'User-Agent':'ABSA-modeling-preflight/1.0'})
            with urllib.request.urlopen(request,timeout=180) as response:
                data=response.read()
            if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Remote asset checksum mismatch; do not refresh pins')
            target.write_bytes(data)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Local asset drifted: {target}; refusing silent replacement')
    java=destination/receipt['java']['java_relative_path']
    if not java.exists():
        folder=destination/'jdk';folder.mkdir(exist_ok=True)
        with tarfile.open(destination/receipt['java']['file']['relative_path']) as archive:
            archive.extractall(folder,filter='data')
    if not java.is_file():raise ValueError('Pinned Java executable missing after extraction')
    print('Pinned assets ready:',destination)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--assets',type=Path,required=True)
    restore(parser.parse_args().assets.resolve())
