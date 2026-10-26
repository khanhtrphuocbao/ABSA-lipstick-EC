"""Build once, seal once, then check without writing. Python standard library only."""
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unicodedata

from absa_protocol import (ASPECTS, CORE_ASPECTS, JOINT_LABELS, SENTIMENTS,
                           PLACEHOLDER_TOKENS, audit_key, placeholder_candidate, encode_targets)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'protocols/evaluation_protocol_v1.yaml'
MANIFEST = 'protocols/data_audit_manifest.csv'
SUMMARY = 'reports/protocol_freeze/audit_summary_v1.json'
REPORT = 'reports/protocol_freeze/NAN_normalization_audit_v1.md'
LOCK = 'protocols/freeze_lock_v1.json'
LEVELS = ('L0', 'L1', 'L2')
SPLITS = ('train', 'val', 'test')
SEALED_FILES = (CONFIG, MANIFEST, SUMMARY, REPORT, 'protocols/data_protocol_v1.md',
                'src/absa_protocol.py', 'src/preflight_validation.py', 'src/evaluate_predictions.py',
                'tests/test_protocol.py', 'tests/test_preflight.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def read_protocol(root=ROOT):
    config = json.loads((root / CONFIG).read_text(encoding='utf-8'))
    target = config['targets']
    require(target['aspect_detection_labels'] == list(ASPECTS), '8 aspect order mismatch')
    require(target['core_aspects'] == list(CORE_ASPECTS), '7 core aspect order mismatch')
    require(target['joint_core_labels'] == list(JOINT_LABELS), '21 joint labels mismatch')
    require(target['sentiments'] == list(SENTIMENTS), 'Sentiment order mismatch')
    require(config['audit']['placeholder_casefold_trim_tokens'] == sorted(PLACEHOLDER_TOKENS), 'Placeholder lexicon mismatch')
    require(config['evaluation']['f1_zero_division'] == 0, 'F1 zero division must be 0')
    return config


def read_sources(root, config):
    result = {}
    for split, meta in config['source_files'].items():
        raw = (root / meta['path']).read_bytes()
        require(sha256(raw) == meta['sha256'], f'Source hash mismatch: {split}')
        lines = raw.splitlines(keepends=True)
        reader = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
        require(next(reader) == ['data', *ASPECTS], f'Schema mismatch: {split}')
        previous = reader.line_num
        rows = []
        for index, values in enumerate(reader):
            start, end = previous + 1, reader.line_num
            previous = end
            require(len(values) == 9, f'Malformed CSV record: {split}:{index}')
            row = dict(zip(['data', *ASPECTS], values))
            encode_targets(row)
            record = b''.join(lines[start-1:end])
            # Reparse the raw record to prove physical line provenance (including multiline CSV).
            require(list(csv.reader(io.StringIO(record.decode('utf-8'), newline=''), strict=True)) == [values],
                    f'Raw CSV provenance mismatch: {split}:{index}')
            row.update(source_split=split, source_row_index=index, source_file=meta['path'],
                       source_csv_line_start=start, source_csv_line_end=end, raw_record_sha256=sha256(record))
            rows.append(row)
        require(len(rows) == meta['rows'], f'Row count mismatch: {split}')
        result[split] = rows
    require(sum(len(result[s]) for s in SPLITS) == len(result['full']) == 16227, 'Total count mismatch')
    vector = lambda r: tuple(r[k] for k in ('data', *ASPECTS))
    require(Counter(map(vector, result['full'])) == Counter(vector(r) for s in SPLITS for r in result[s]),
            'Full/split row multisets differ')
    return result


def quantile(values, q):
    """Linear interpolation, matching pandas/numpy default percentile on this integer vector."""
    values = sorted(values)
    if not values:
        return None
    pos = (len(values)-1)*q
    low = int(pos)
    high = min(low+1, len(values)-1)
    return values[low] + (values[high]-values[low])*(pos-low)


def data_stats(rows):
    vectors = [tuple(r[k] for k in ('data', *ASPECTS)) for r in rows]
    cardinality = [sum(bool(r[a]) for a in ASPECTS) for r in rows]
    words = [len(r['data'].split()) for r in rows]
    chars = [len(r['data']) for r in rows]
    return {'reviews': len(rows), 'duplicate_row_extra': len(rows)-len(set(vectors)),
            'duplicate_text_extra': {l: len(rows)-len({audit_key(r['data'], l) for r in rows}) for l in LEVELS},
            'others_count': sum(bool(r['others']) for r in rows),
            'multi_aspect_count': sum(c >= 2 for c in cardinality),
            'multi_aspect_rate': sum(c >= 2 for c in cardinality)/len(rows) if rows else None,
            'word_p50': quantile(words, .5), 'word_p95': quantile(words, .95),
            'char_p50': quantile(chars, .5), 'char_p95': quantile(chars, .95),
            'word_mean': sum(words)/len(words) if words else None}


def compute(root=ROOT):
    config = read_protocol(root)
    sources = read_sources(root, config)
    canonical = [r for s in SPLITS for r in sources[s]]
    group = {l: defaultdict(list) for l in LEVELS}
    for r in canonical:
        for level in LEVELS:
            group[level][audit_key(r['data'], level)].append(r)
    # Full file mirrors the split corpus. Map equal row occurrences bijectively; no original ID is inferred.
    mapping = defaultdict(deque)
    for r in sources['full']:
        mapping[tuple(r[k] for k in ('data', *ASPECTS))].append(r)
    metadata = {l: {} for l in LEVELS}
    for level in LEVELS:
        for key, rows in group[level].items():
            variants = {tuple(r[a] for a in ASPECTS) for r in rows}
            metadata[level][key] = {'size': len(rows), 'split_counts': Counter(r['source_split'] for r in rows),
                                    'conflict': len(variants) > 1,
                                    'polarity_conflict': any(len({r[a] for r in rows if r[a]}) > 1 for a in ASPECTS)}
    manifest = []
    for row in canonical:
        full = mapping[tuple(row[k] for k in ('data', *ASPECTS))].popleft()
        r = {k: row[k] for k in ('source_split', 'source_row_index', 'source_file',
                                 'source_csv_line_start', 'source_csv_line_end', 'raw_record_sha256')}
        r.update(source_file_sha256=config['source_files'][row['source_split']]['sha256'],
                 full_source_file='data/full_data.csv', full_source_row_index=full['source_row_index'],
                 full_csv_line_start=full['source_csv_line_start'], full_csv_line_end=full['source_csv_line_end'],
                 full_raw_record_sha256=full['raw_record_sha256'], raw_text=row['data'])
        r.update({a: row[a] for a in ASPECTS})
        r.update(literal_NAN=row['data'] == 'NAN', placeholder_candidate=placeholder_candidate(row['data']),
                 audit_status='placeholder_origin_unresolved' if placeholder_candidate(row['data']) else 'automated_audit_only',
                 l3_status='incomplete_candidate_screening_not_exhaustive')
        for level in LEVELS:
            key = audit_key(row['data'], level)
            meta = metadata[level][key]
            suffix = level.lower()
            r['exact_text_key' if level == 'L0' else f'normalized_key_{suffix}'] = key
            r[f'group_id_{suffix}'] = level + ':' + sha256(key.encode('utf-8'))
            r[f'group_size_{suffix}'] = meta['size']
            r[f'same_split_group_size_{suffix}'] = meta['split_counts'][row['source_split']]
            r[f'has_label_conflict_{suffix}'] = meta['conflict']
            r[f'has_polarity_conflict_{suffix}'] = meta['polarity_conflict']
            r[f'empty_key_{suffix}'] = not bool(key)
            for split in SPLITS:
                # Excludes the same split: repeated occurrences within train are not train->train overlap.
                r[f'overlap_{split}_{suffix}'] = bool(key) and split != row['source_split'] and meta['split_counts'][split] > 0
        r['duplicate_group_id'] = r['group_id_l1']
        r['has_label_conflict'] = r['has_label_conflict_l1']
        r['overlap_train'] = r['overlap_train_l1']
        r['overlap_validation'] = r['overlap_val_l1']
        is_test = row['source_split'] == 'test'
        for level in ('l1', 'l2'):
            r[f'test_nonoverlap_{level}'] = is_test and not (r[f'overlap_train_{level}'] or r[f'overlap_val_{level}'])
        r['test_without_placeholder'] = is_test and not r['placeholder_candidate']
        r['test_placeholder_only'] = is_test and r['placeholder_candidate']
        manifest.append(r)
    require(len(manifest) == 16227, 'Manifest row count mismatch')
    require({r['full_source_row_index'] for r in manifest} == set(range(16227)), 'Manifest full-row mapping not bijective')
    require(all(not q for q in mapping.values()), 'Unmapped full records')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(manifest[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(manifest)
    overlap = []
    for level in LEVELS:
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            c1 = Counter(audit_key(r['data'], level) for r in sources[left])
            c2 = Counter(audit_key(r['data'], level) for r in sources[right])
            shared = (set(c1) & set(c2)) - {''}
            overlap.append({'level': level, 'left': left, 'right': right, 'shared_keys': len(shared),
                            'rows_left': sum(c1[k] for k in shared), 'rows_right': sum(c2[k] for k in shared),
                            'cross_pairs': sum(c1[k]*c2[k] for k in shared)})
    original = {s: data_stats(rows) for s, rows in sources.items()}
    without_nan = {s: data_stats([r for r in rows if r['data'] != 'NAN']) for s, rows in sources.items()}
    require(original['full']['duplicate_text_extra'] == config['audit']['expected_full_duplicate_text_extra'], 'L0/L1/L2 counts drifted')
    for split, rows in sources.items():
        nan = [r for r in rows if r['data'] == 'NAN']
        require(len(nan) == config['audit']['expected_literal_NAN'][split], f'NAN count drifted: {split}')
        require(all(r['others'] == 'neutral' and not any(r[a] for a in CORE_ASPECTS) for r in nan), 'Unexpected NAN labels')
        require(len({tuple(r[k] for k in ('data', *ASPECTS)) for r in nan}) == 1, 'NAN rows are not exact-row identical')
    diagnostic = {}
    for name in ('original_test', 'test_nonoverlap_l1', 'test_nonoverlap_l2', 'test_without_placeholder', 'test_placeholder_only'):
        selected = [r for r in manifest if r['source_split'] == 'test' and (name == 'original_test' or r[name])]
        diagnostic[name] = {'reviews': len(selected), 'support_21': {label: sum(r[a] == s for r in selected)
                           for label in JOINT_LABELS for a, s in [label.split(':')]}}
    placeholder_variants = {s: dict(sorted(Counter(r['data'] for r in rows if placeholder_candidate(r['data'])).items()))
                            for s, rows in sources.items()}
    summary = {'dataset_version': config['dataset_version'], 'original': original, 'excluding_literal_NAN': without_nan,
               'placeholder_variants': placeholder_variants, 'overlap': overlap, 'diagnostic_subsets': diagnostic,
               'literal_NAN_raw_csv_proof': {s: {'records_checked': config['audit']['expected_literal_NAN'][s],
                                                'exact_text_and_only_others_neutral': True,
                                                'raw_records_reparsed': True} for s in sources},
               'l2_empty_keys': {s: sum(not audit_key(r['data'], 'L2') for r in rows) for s, rows in sources.items()},
               'conflicts': {l: {'keys': sum(m['conflict'] for m in metadata[l].values()),
                                 'rows': sum(m['size'] for m in metadata[l].values() if m['conflict']),
                                 'polarity_conflict_keys': sum(m['polarity_conflict'] for m in metadata[l].values())} for l in LEVELS}}
    return {MANIFEST: buffer.getvalue().encode('utf-8'), SUMMARY: json_bytes(summary),
            REPORT: audit_report(summary).encode('utf-8')}, summary


def audit_report(s):
    orig, filtered = s['original'], s['excluding_literal_NAN']
    lines = ['# NAN & normalization audit v1', '',
             'Đây là sensitivity analysis; benchmark `original_shopee_absa_v1` giữ nguyên tất cả dòng/nhãn/splits.', '',
             '## NAN: xác nhận và chưa xác nhận', '',
             '- Raw CSV có đúng 105 field text `NAN`: train 84, val 13, test 8. Tất cả chỉ có OTHERS:Neutral.',
             '- Mỗi record gốc đã được reparse; manifest có source record index, physical line start/end và raw-record SHA-256.',
             '- 105 row vectors giống nhau hoàn toàn: đóng góp 104 duplicate-row-extra trên full. Giữ `NAN` là string; không ép thành missing.',
             '- Không thấy biến thể khác trong lexicon casefold+trim: NAN/nan/NaN, NULL, None, N/A, NA, <NA>, nil.',
             '- Chưa xác minh nguồn gốc: paper dùng “nan” để nói aspect absent, không giải thích literal review `NAN`. Không đánh đồng hai điều này.',
             '- Đã đọc paper §3–4 và README/notebook hiện có; chưa tìm được giải thích tác giả về việc tạo chuỗi này. Đây là giới hạn tìm kiếm, không chứng minh tài liệu đó không tồn tại.', '',
             '## Original vs excluding literal NAN (đếm riêng, không cộng full với splits)', '',
             '| Split | Rows original → excl | Duplicate row extra | OTHERS count | Multi-aspect % | Word P50/P95 | Char P50/P95 |',
             '|---|---:|---:|---:|---:|---|---|']
    for split in ('full', *SPLITS):
        a,b = orig[split], filtered[split]
        lines.append(f'| {split} | {a["reviews"]} → {b["reviews"]} | {a["duplicate_row_extra"]} → {b["duplicate_row_extra"]} | '
                     f'{a["others_count"]} → {b["others_count"]} | {100*a["multi_aspect_rate"]:.3f} → {100*b["multi_aspect_rate"]:.3f} | '
                     f'{a["word_p50"]:g}/{a["word_p95"]:g} → {b["word_p50"]:g}/{b["word_p95"]:g} | '
                     f'{a["char_p50"]:g}/{a["char_p95"]:g} → {b["char_p50"]:g}/{b["char_p95"]:g} |')
    lines += ['', '## Normalization / overlap', '',
              f'Full text duplicate extras L0/L1/L2 = {orig["full"]["duplicate_text_extra"]}. L2 là screening, không phải thêm review-copy đã xác nhận.',
              f'L2 empty-key rows: {s["l2_empty_keys"]}. Giữ collisions trong duplicate census; loại key rỗng khỏi overlap, gắn flag riêng.', '',
              '| Level | Pair | Shared keys | Rows left | Rows right | Cross-pairs |', '|---|---|---:|---:|---:|---:|']
    for r in s['overlap']:
        lines.append(f'| {r["level"]} | {r["left"]} ↔ {r["right"]} | {r["shared_keys"]} | {r["rows_left"]} | {r["rows_right"]} | {r["cross_pairs"]} |')
    lines += ['', '## Diagnostic test subsets', '', '| Subset | Reviews | PRICE-Neutral support |', '|---|---:|---:|']
    for name,r in s['diagnostic_subsets'].items():
        lines.append(f'| {name} | {r["reviews"]} | {r["support_21"]["price:neutral"]} |')
    lines += ['', '## Quyết định reporting', '',
              '1. L1 là contamination chính; L0/L2 là sensitivity. L3 vẫn incomplete candidate screening, không được coi là exhaustive.',
              '2. Cùng checkpoint: report original test, without-placeholder và nonoverlap diagnostics, kèm support/zero-support. Không retune từ các subset test.',
              '3. Training-ablation bỏ placeholder chỉ trong train là run riêng, evaluation subsets cố định, không thay main benchmark.',
              '4. Core joint metric không gồm OTHERS. Placeholder-only test có 0 core gold positives: recall N/A; FP vẫn phải báo nếu model hallucinate core aspects.',
              '5. Chưa có predictions, nên chưa đo score inflation, hiệu quả loại NAN hoặc model superiority.', '',
              '## Nguồn đã đối chiếu', '',
              '- [Paper PACLIC 2022](https://aclanthology.org/2022.paclic-1.84.pdf)',
              '- [VinAI PhoBERT: segmented input](https://github.com/VinAIResearch/PhoBERT)',
              '- [scikit-learn F1: fixed labels / zero_division](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)', '']
    return '\n'.join(lines)


def verify_bytes(root, outputs):
    for relative, expected in outputs.items():
        path = root / relative
        require(path.is_file() and path.read_bytes() == expected, f'Artifact missing/drifted: {relative}')


def build(root=ROOT):
    outputs, summary = compute(root)
    # Existing artifacts are never silently updated to a new frozen version.
    for relative, data in outputs.items():
        path = root / relative
        if path.exists():
            require(path.read_bytes() == data, f'Refusing to overwrite drifted artifact: {relative}; create v2')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return summary


def seal(root=ROOT):
    outputs, _ = compute(root)
    verify_bytes(root, outputs)
    lock = {'protocol_id': 'absa_data_evaluation_v1', 'python': platform.python_version(),
            'unicode_version': unicodedata.unidata_version,
            'artifacts': {f: sha256((root / f).read_bytes()) for f in SEALED_FILES},
            'notebook_evidence': {str(f.relative_to(root)): sha256(f.read_bytes())
                                  for f in (root / 'src/eda.ipynb', root / 'src/eda_nlp.ipynb')}}
    target = root / LOCK
    data = json_bytes(lock)
    require(not target.exists() or target.read_bytes() == data, 'Freeze lock already exists with different content; create v2')
    target.write_bytes(data)


def check(root=ROOT):
    lock = json.loads((root / LOCK).read_text(encoding='utf-8'))
    require(unicodedata.unidata_version == lock['unicode_version'], 'Unicode runtime drift: use frozen runtime or version protocol')
    require(set(lock['artifacts']) == set(SEALED_FILES), 'Lock artifact list drift')
    for relative, expected in lock['artifacts'].items():
        require(sha256((root / relative).read_bytes()) == expected, f'Frozen artifact hash mismatch: {relative}')
    outputs, summary = compute(root)
    verify_bytes(root, outputs)
    print('PASS source SHA-256, schemas, 16,227 rows and official 12,981/1,623/1,623 splits')
    print('PASS full/split multisets; 7 core / 8 aspect / 21 joint labels; canonical labels')
    print('PASS manifest covers all 16,227 split rows and bijectively maps all full records; raw CSV provenance')
    print('PASS deterministic L0/L1/L2, conflict/overlap/placeholder flags, subsets and audit summaries')
    print('PASS protocol/evaluator/artifact fingerprints and Unicode runtime')
    print('DATA + EVALUATOR FREEZE READY; model-specific tokenizer audit / run registration still required')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--build', action='store_true')
    group.add_argument('--seal', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        if args.build:
            build(); print('Built deterministic audit artifacts (not sealed yet).')
        elif args.seal:
            seal(); print('Sealed protocol and evaluator fingerprints.')
        else:
            check()
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f'FAIL {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
