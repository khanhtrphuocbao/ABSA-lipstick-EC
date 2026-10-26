"""Prediction decoding and registered validation threshold selection."""
import json
from pathlib import Path

try:
    from ..absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS, evaluate_rows
except ImportError:
    from absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS, evaluate_rows

def decode_probabilities(aspect_probabilities, conditional_probabilities, threshold,
                         source_split, source_row_index):
    import numpy as np
    aspects = np.asarray(aspect_probabilities, dtype=float)
    conditional = np.asarray(conditional_probabilities, dtype=float)
    if aspects.shape != (len(ASPECTS),) or conditional.shape != (len(CORE_ASPECTS), len(SENTIMENTS)):
        raise ValueError('Expected aspect [8] and conditional sentiment [7,3] probabilities')
    if not 0 <= threshold <= 1 or not np.isfinite(aspects).all() or not np.isfinite(conditional).all():
        raise ValueError('Threshold/probabilities are invalid or non-finite')
    if np.any((aspects < 0) | (aspects > 1)) or np.any((conditional < 0) | (conditional > 1)):
        raise ValueError('Probabilities must lie in [0,1]')
    chosen = [a for i, a in enumerate(ASPECTS) if aspects[i] >= threshold]
    sentiments = {a: SENTIMENTS[int(np.argmax(conditional[i]))] for i, a in enumerate(CORE_ASPECTS)}
    return {'source_split': source_split, 'source_row_index': int(source_row_index),
            'aspects': chosen, 'conditional_sentiments': sentiments}


def select_global_threshold(gold_rows, aspect_probabilities, conditional_probabilities, split):
    import numpy as np
    aspect = np.asarray(aspect_probabilities, dtype=float)
    conditional = np.asarray(conditional_probabilities, dtype=float)
    if aspect.shape != (len(gold_rows), len(ASPECTS)) or conditional.shape != (len(gold_rows), len(CORE_ASPECTS), len(SENTIMENTS)):
        raise ValueError('Validation probability arrays do not align with rows/registered labels')
    if not np.isfinite(aspect).all() or not np.isfinite(conditional).all():
        raise ValueError('Validation probabilities contain NaN/Inf')
    predictions_by_threshold = {}
    threshold_grid = registered_configs()['selection']['aspect_threshold_grid']
    if threshold_grid != [0.3, 0.4, 0.5, 0.6, 0.7]:
        raise ValueError('Registered global aspect threshold grid drifted')
    for threshold in threshold_grid:
        predictions = [decode_probabilities(aspect[i], conditional[i], threshold, split,
                                             row.get('source_row_index', i))
                       for i, row in enumerate(gold_rows)]
        metrics = evaluate_rows(gold_rows, predictions)
        predictions_by_threshold[threshold] = {
            'threshold': threshold,
            'joint_macro_f1': metrics['joint']['macro_f1'],
            'joint_micro_f1': metrics['joint']['micro']['f1']}
    return max(predictions_by_threshold.values(), key=lambda item: (
        item['joint_macro_f1'], item['joint_micro_f1'], -abs(item['threshold'] - 0.5), -item['threshold']))


def registered_configs(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    return json.loads((root / 'protocols/experiments/experiment_registration_v1.json').read_text(encoding='utf-8'))
