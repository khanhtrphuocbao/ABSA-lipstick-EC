"""Frozen target semantics and metrics. Standard library only; no text cleaning for models."""
from collections import Counter
import re
import unicodedata

ASPECTS = ('stayingpower', 'texture', 'smell', 'price', 'others', 'colour', 'shipping', 'packing')
CORE_ASPECTS = tuple(a for a in ASPECTS if a != 'others')
SENTIMENTS = ('positive', 'neutral', 'negative')
JOINT_LABELS = tuple(f'{a}:{s}' for a in CORE_ASPECTS for s in SENTIMENTS)
PLACEHOLDER_TOKENS = frozenset({'nan', 'null', 'none', 'n/a', 'na', '<na>', 'nil'})


def audit_key(text, level):
    if not isinstance(text, str):
        raise ValueError('Audit input must be raw text, not missing/NaN coerced to a string')
    if level == 'L0':
        return text
    if level == 'L1':
        return re.sub(r'\s+', ' ', text.lower().strip())
    if level == 'L2':
        text = unicodedata.normalize('NFC', text).lower()
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        return re.sub(r'[\s_]+', ' ', text).strip()
    raise ValueError(f'Unknown audit level: {level}')


def placeholder_candidate(text):
    return text.strip().casefold() in PLACEHOLDER_TOKENS


def encode_targets(row):
    if any(row.get(a) not in ('', *SENTIMENTS) for a in ASPECTS):
        raise ValueError('Missing/noncanonical label; label normalization v1 is strict identity')
    return {'aspects': {a for a in ASPECTS if row[a]},
            'joint': {f'{a}:{row[a]}' for a in CORE_ASPECTS if row[a]}}


def _counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'support': tp + fn,
            'predicted_positive': tp + fp, 'precision': precision, 'recall': recall,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def score_sets(gold, predicted, labels):
    """Fixed-label binary PRF. None serializes to JSON null, not NaN."""
    labels = tuple(labels)
    if len(set(labels)) != len(labels) or not labels:
        raise ValueError('Need a nonempty, unique label space')
    if len(gold) != len(predicted):
        raise ValueError('Prediction/gold length mismatch')
    allowed = set(labels)
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, predicted):
        if not isinstance(g, (set, frozenset)) or not isinstance(p, (set, frozenset)):
            raise ValueError('Labels must be explicit sets, not strings/lists with silent duplicates')
        if not (g | p) <= allowed:
            raise ValueError('Unknown gold or predicted label')
        tp.update(g & p); fp.update(p - g); fn.update(g - p)
    per = {label: _counts(tp[label], fp[label], fn[label]) for label in labels}
    support = sum(r['support'] for r in per.values())
    observed = [r for r in per.values() if r['support'] > 0]
    weighted = {metric: sum((r[metric] or 0.0) * r['support'] for r in per.values()) / support
                if support else 0.0 for metric in ('precision', 'recall', 'f1')}
    macro = {metric: sum(r[metric] or 0.0 for r in per.values()) / len(labels)
             for metric in ('precision', 'recall', 'f1')}
    empty = not gold
    return {'status': 'empty_subset' if empty else 'evaluated', 'reviews': len(gold),
            'labels': list(labels), 'per_label': per,
            'macro_f1': None if empty else macro['f1'],
            'macro': {k: None if empty else v for k, v in macro.items()},
            'micro': {k: None if empty else v for k, v in _counts(sum(tp.values()), sum(fp.values()), sum(fn.values())).items()},
            'weighted': {k: None if empty else v for k, v in weighted.items()},
            'observed_support_macro_f1': sum(r['f1'] for r in observed) / len(observed) if observed else None,
            'zero_gold_support_labels': [k for k, v in per.items() if not v['support']]}


def _validate_prediction(row):
    if not isinstance(row, dict) or not isinstance(row.get('aspects'), list):
        raise ValueError('Each prediction needs an aspects list')
    aspects = row['aspects']
    if any(not isinstance(a, str) for a in aspects) or len(set(aspects)) != len(aspects) or not set(aspects) <= set(ASPECTS):
        raise ValueError('Unknown or repeated predicted aspect')
    cond = row.get('conditional_sentiments')
    if not isinstance(cond, dict) or set(cond) != set(CORE_ASPECTS) or any(v not in SENTIMENTS for v in cond.values()):
        raise ValueError('Need a 3-class conditional prediction for ALL 7 core aspects, independent of detection')


def evaluate_rows(gold_rows, prediction_rows):
    if len(gold_rows) != len(prediction_rows):
        raise ValueError('Prediction/gold length mismatch')
    gold_targets = [encode_targets(r) for r in gold_rows]
    for p in prediction_rows:
        _validate_prediction(p)
    predicted_aspects = [set(r['aspects']) for r in prediction_rows]
    predicted_joint = [{f'{a}:{r["conditional_sentiments"][a]}' for a in r['aspects'] if a in CORE_ASPECTS}
                       for r in prediction_rows]
    joint = score_sets([t['joint'] for t in gold_targets], predicted_joint, JOINT_LABELS)
    aspects = score_sets([t['aspects'] for t in gold_targets], predicted_aspects, ASPECTS)
    by_aspect = {a: score_sets([g['joint'] & {f'{a}:{s}' for s in SENTIMENTS} for g in gold_targets],
                              [p & {f'{a}:{s}' for s in SENTIMENTS} for p in predicted_joint],
                              [f'{a}:{s}' for s in SENTIMENTS]) for a in CORE_ASPECTS}
    by_sentiment = {}
    for s in SENTIMENTS:
        records = [joint['per_label'][f'{a}:{s}'] for a in CORE_ASPECTS]
        by_sentiment[s] = _counts(*(sum(r[k] for r in records) for k in ('tp', 'fp', 'fn')))
    conditional = {}
    for a in CORE_ASPECTS:
        indices = [i for i, row in enumerate(gold_rows) if row[a]]
        conditional[a] = score_sets([{gold_rows[i][a]} for i in indices],
                                    [{prediction_rows[i]['conditional_sentiments'][a]} for i in indices], SENTIMENTS)
    return {'joint': joint, 'aspect_detection': aspects, 'joint_per_aspect': by_aspect,
            'joint_pooled_per_sentiment': by_sentiment, 'negative_complaints': by_sentiment['negative'],
            'others': aspects['per_label']['others'],
            'conditional': {'unit': 'gold-present core aspects; no predicted-aspect gating',
                            'macro_f1_fixed7_aspects_fixed3_classes': sum(r['macro_f1'] or 0.0 for r in conditional.values()) / 7
                            if gold_rows else None,
                            'unevaluable_aspects': [a for a, r in conditional.items() if r['status'] == 'empty_subset'],
                            'per_aspect': conditional}}
