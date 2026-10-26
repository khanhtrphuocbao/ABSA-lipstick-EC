"""Stable evaluator-compatible JSONL export; output files are write-once."""
import json
import os
import tempfile
from pathlib import Path

try:
    from ..absa_protocol import ASPECTS, CORE_ASPECTS, evaluate_rows
except ImportError:
    from absa_protocol import ASPECTS, CORE_ASPECTS, evaluate_rows


def write_predictions(path, gold_rows, predictions, split):
    if split not in ('val', 'test'):
        raise ValueError('Predictions may only be exported for val or test')
    if len(gold_rows) != len(predictions):
        raise ValueError('Prediction/gold length mismatch')
    expected = [(split, int(row.get('source_row_index', i))) for i, row in enumerate(gold_rows)]
    actual = [(row.get('source_split'), row.get('source_row_index')) for row in predictions]
    if actual != expected:
        raise ValueError('Predictions must match official split and row identity in source order')
    evaluate_rows(gold_rows, predictions)
    canonical = []
    for row in predictions:
        canonical.append({'source_split': split, 'source_row_index': int(row['source_row_index']),
                          'aspects': [a for a in ASPECTS if a in row['aspects']],
                          'conditional_sentiments': {a: row['conditional_sentiments'][a] for a in CORE_ASPECTS}})
    payload = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in canonical).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite prediction file: {path}')
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    except FileExistsError:
        raise FileExistsError(f'Refusing to overwrite prediction file: {path}')
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
    return path
