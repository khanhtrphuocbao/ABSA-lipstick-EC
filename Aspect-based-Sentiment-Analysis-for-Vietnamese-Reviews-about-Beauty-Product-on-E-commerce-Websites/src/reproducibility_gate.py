"""Verify the original frozen protocol on its exact runtime; never rebuild or reseal."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
EXPECTED_LOCK_SHA256='16e9b76b6a71530d08704cadc2d1ddeddf54ca57615c35f8e650ccc57ac86ee8'

def run_gate():
    if platform.python_version()!='3.15.1' or unicodedata.unidata_version!='17.0.0':
        raise ValueError('Required gate runtime: Python 3.15.1 / Unicode 17.0.0')
    lock=ROOT/'protocols/freeze_lock_v1.json'
    before=hashlib.sha256(lock.read_bytes()).hexdigest()
    if before!=EXPECTED_LOCK_SHA256:raise ValueError('Original v1 lock changed; refusing checksum refresh')
    commands=[[sys.executable,'src/preflight_validation.py','--check'],
              [sys.executable,'-m','unittest','discover','-s','tests','-v']]
    results=[]
    for cmd in commands:
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=180)
        results.append({'command':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    after=hashlib.sha256(lock.read_bytes()).hexdigest()
    if before!=after:raise ValueError('Lock changed during gate')
    return {'status':'PASS','python':platform.python_version(),'unicode':unicodedata.unidata_version,
            'freeze_lock_sha256_before':before,'freeze_lock_sha256_after':after,
            'source_and_frozen_artifact_checksums_unchanged':True,'commands':results}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=run_gate()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS exact-runtime gate; v1 lock and checksums unchanged')
