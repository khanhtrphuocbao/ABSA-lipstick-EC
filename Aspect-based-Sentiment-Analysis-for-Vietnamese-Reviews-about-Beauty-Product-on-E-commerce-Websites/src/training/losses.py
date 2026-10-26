"""Registered multitask loss only; no weighting, focal term, or label smoothing."""

try:
    from ..absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS
except ImportError:
    from absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS


def compute_multitask_loss(aspect_logits, sentiment_logits, aspect_targets,
                           sentiment_targets, sentiment_mask):
    import torch
    import torch.nn.functional as functional
    if aspect_logits.ndim != 2 or aspect_logits.shape[1] != len(ASPECTS):
        raise ValueError(f'aspect_logits must have shape [batch, {len(ASPECTS)}]')
    if sentiment_logits.ndim != 3 or tuple(sentiment_logits.shape[1:]) != (len(CORE_ASPECTS), len(SENTIMENTS)):
        raise ValueError(f'sentiment_logits must have shape [batch, {len(CORE_ASPECTS)}, {len(SENTIMENTS)}]')
    batch = aspect_logits.shape[0]
    if (aspect_targets.shape != (batch, len(ASPECTS)) or
            sentiment_targets.shape != (batch, len(CORE_ASPECTS)) or
            sentiment_mask.shape != (batch, len(CORE_ASPECTS))):
        raise ValueError('Target/mask shapes do not match model outputs')
    if not (torch.isfinite(aspect_logits).all() and torch.isfinite(sentiment_logits).all()):
        raise ValueError('Model logits contain NaN/Inf')
    aspect = functional.binary_cross_entropy_with_logits(aspect_logits, aspect_targets.to(aspect_logits.dtype), reduction='mean')
    mask = sentiment_mask.to(dtype=torch.bool, device=sentiment_logits.device)
    if mask.any():
        labels = sentiment_targets.to(device=sentiment_logits.device, dtype=torch.long)
        sentiment = functional.cross_entropy(sentiment_logits[mask], labels[mask], reduction='sum') / mask.sum()
    else:
        sentiment = sentiment_logits.sum() * 0.0
    total = aspect + sentiment
    return total, {'aspect': aspect, 'sentiment': sentiment, 'total': total}
