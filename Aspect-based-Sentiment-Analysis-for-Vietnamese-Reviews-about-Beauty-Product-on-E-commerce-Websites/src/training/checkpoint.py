"""Write-once checkpoint envelope with model payload and integrity hashes."""
import base64
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
import json

REQUIRED_METADATA = ('candidate_id', 'seed', 'epoch', 'threshold', 'validation_metrics',
                     'preprocessing', 'dataset_version', 'protocol_id', 'protocol_sha256',
                     'registration_id', 'registration_sha256', 'code_sha256', 'model_family')


def _registered_seeds():
    root = Path(__file__).resolve().parents[2]
    return tuple(json.loads((root / 'protocols/experiments/experiment_registration_v1.json').read_text())['seeds'])


def training_code_sha256(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / 'src/training').rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode('utf-8') + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def serialize_torch_state(state_dict):
    import torch
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def restore_torch_state(payload, map_location='cpu'):
    import torch
    return torch.load(io.BytesIO(payload), map_location=map_location, weights_only=True)


def serialize_estimator(estimator):
    """Serialize a locally produced sklearn estimator; never load untrusted pickle payloads."""
    import joblib
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer, compress=0)
    return buffer.getvalue()


def restore_estimator(payload):
    import joblib
    return joblib.load(io.BytesIO(payload))


def save_checkpoint(path, model_payload, metadata):
    path = Path(path)
    missing = set(REQUIRED_METADATA) - set(metadata)
    if missing:
        raise ValueError(f'Checkpoint metadata missing: {", ".join(sorted(missing))}')
    if not isinstance(model_payload, (bytes, bytearray)):
        raise TypeError('Serialize model/estimator state to bytes before checkpointing')
    if metadata['seed'] not in _registered_seeds() or not 0 <= metadata['threshold'] <= 1:
        raise ValueError('Checkpoint seed or threshold is outside the registered contract')
    if type(metadata['epoch']) is not int or metadata['epoch'] < 1:
        raise ValueError('Checkpoint epoch must be a positive integer')
    for name in ('protocol_sha256', 'registration_sha256', 'code_sha256'):
        if not isinstance(metadata[name], str) or not re.fullmatch(r'[0-9a-f]{64}', metadata[name]):
            raise ValueError(f'Checkpoint {name} must be a lowercase SHA-256')
    root = Path(__file__).resolve().parents[2]
    protocol_sha = hashlib.sha256((root / 'protocols/evaluation_protocol_v1.yaml').read_bytes()).hexdigest()
    registration_sha = hashlib.sha256((root / 'protocols/experiments/experiment_registration_v1.json').read_bytes()).hexdigest()
    if metadata['protocol_sha256'] != protocol_sha or metadata['registration_sha256'] != registration_sha:
        raise ValueError('Checkpoint protocol/registration reference does not match the frozen files')
    if metadata['code_sha256'] != training_code_sha256(root):
        raise ValueError('Checkpoint trainer code SHA-256 does not match current training sources')
    payload = bytes(model_payload)
    envelope = {'format': 'absa-checkpoint-v1', 'metadata': metadata,
                'model_payload_sha256': hashlib.sha256(payload).hexdigest(),
                'model_payload_base64': base64.b64encode(payload).decode('ascii')}
    encoded = (json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                          allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
    fd, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_name, path)
    except FileExistsError:
        raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint(path, expected_sha256=None):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is None:
        raise ValueError('An expected checkpoint SHA-256 is required for restoration')
    if actual != expected_sha256:
        raise ValueError('Checkpoint SHA-256 checksum mismatch')
    try:
        envelope = json.loads(raw)
        payload = base64.b64decode(envelope['model_payload_base64'], validate=True)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError('Checkpoint envelope is malformed') from exc
    if envelope.get('format') != 'absa-checkpoint-v1' or hashlib.sha256(payload).hexdigest() != envelope.get('model_payload_sha256'):
        raise ValueError('Checkpoint model payload checksum mismatch')
    return payload, envelope['metadata']
