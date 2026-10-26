"""Evaluate complete, row-keyed JSONL predictions after the data protocol preflight."""
import argparse
import csv
import json
from pathlib import Path

from absa_protocol import evaluate_rows
from preflight_validation import ROOT, MANIFEST, LOCK, CONFIG, check, json_bytes, require, sha256


def align_predictions(manifest, predictions, split):
    gold = [r for r in manifest if r['source_split'] == split]
    expected = {(split, int(r['source_row_index'])) for r in gold}
    indexed = {}
    for p in predictions:
        require(isinstance(p, dict), 'Prediction record must be an object')
        require(type(p.get('source_row_index')) is int and isinstance(p.get('source_split'), str), 'Invalid prediction row key')
        key = (p['source_split'], p['source_row_index'])
        require(key in expected, f'Unexpected prediction key: {key}')
        require(key not in indexed, f'Duplicate prediction key: {key}')
        indexed[key] = p
    require(set(indexed) == expected, 'Predictions do not cover the complete requested split')
    return gold, [indexed[(split, int(r['source_row_index']))] for r in gold]


def evaluate_file(path, split, root=ROOT):
    check(root)
    with (root / MANIFEST).open(encoding='utf-8', newline='') as handle:
        manifest = list(csv.DictReader(handle))
    raw = path.read_bytes()
    predictions = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    gold, ordered = align_predictions(manifest, predictions, split)
    names = ['original_test', 'test_nonoverlap_l1', 'test_nonoverlap_l2',
             'test_without_placeholder', 'test_placeholder_only'] if split == 'test' else [f'original_{split}']
    reports = {}
    for name in names:
        indices = [i for i, r in enumerate(gold) if name.startswith('original_') or r[name] == 'True']
        reports[name] = evaluate_rows([gold[i] for i in indices], [ordered[i] for i in indices])
    return {'dataset_version': 'original_shopee_absa_v1',
            'protocol_id': 'absa_data_evaluation_v1',
            'protocol_sha256': sha256((root / CONFIG).read_bytes()),
            'freeze_lock_sha256': sha256((root / LOCK).read_bytes()),
            'prediction_file_sha256': sha256(raw), 'split': split,
            'note': 'Scores apply to supplied predictions; checkpoint/preprocessing/run-registration provenance is recorded by the experiment.',
            'reports': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_file(args.predictions, args.split)
    require(not args.output.exists(), 'Refusing to overwrite a previous metric report')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(report))
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
