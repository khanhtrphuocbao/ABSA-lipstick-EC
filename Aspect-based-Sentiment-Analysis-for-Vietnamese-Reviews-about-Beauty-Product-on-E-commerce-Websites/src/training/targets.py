"""Array encoding from the frozen target constants; blank cells remain masked."""
try:
    from ..absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS, encode_targets as validate_targets
except ImportError:
    from absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS, encode_targets as validate_targets


def encode_batch(rows):
    import numpy as np
    aspect_targets = np.zeros((len(rows), len(ASPECTS)), dtype=np.float32)
    sentiment_targets = np.zeros((len(rows), len(CORE_ASPECTS)), dtype=np.int64)
    sentiment_mask = np.zeros((len(rows), len(CORE_ASPECTS)), dtype=bool)
    for i, row in enumerate(rows):
        validate_targets(row)
        for j, aspect in enumerate(ASPECTS):
            aspect_targets[i, j] = float(bool(row[aspect]))
        for j, aspect in enumerate(CORE_ASPECTS):
            if row[aspect]:
                sentiment_mask[i, j] = True
                sentiment_targets[i, j] = SENTIMENTS.index(row[aspect])
    return {'aspect_targets': aspect_targets, 'sentiment_targets': sentiment_targets,
            'sentiment_mask': sentiment_mask}
