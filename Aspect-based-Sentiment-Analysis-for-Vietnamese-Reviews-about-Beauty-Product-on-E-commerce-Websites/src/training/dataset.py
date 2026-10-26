"""Strict CSV loading with source-row identity kept outside model features."""
import csv
from pathlib import Path

try:
    from ..absa_protocol import ASPECTS, encode_targets
except ImportError:  # direct `training.*` imports with src/ on sys.path
    from absa_protocol import ASPECTS, encode_targets


def load_split(path, split):
    if split not in ('train', 'val', 'test'):
        raise ValueError('split must be train, val, or test')
    path = Path(path)
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != ['data', *ASPECTS]:
            raise ValueError('Unexpected dataset schema/order')
        for index, row in enumerate(reader):
            if any(value is None for value in row.values()):
                raise ValueError(f'Malformed CSV row: {split}:{index}')
            encode_targets(row)
            rows.append({**row, 'source_split': split, 'source_row_index': index})
    return rows


def load_train_validation(train_path, validation_path):
    """Load only the explicitly supplied train/validation files; never reads test."""
    train = load_split(train_path, 'train')
    validation = load_split(validation_path, 'val')
    if not train:
        raise ValueError('Training split is empty')
    if not validation:
        raise ValueError('Validation split is empty')
    return train, validation
