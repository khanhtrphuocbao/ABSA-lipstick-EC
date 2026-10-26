"""PhoBERT-compatible adapter; encoder is injected to prevent implicit weight downloads."""
try:
    from ...absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS
    from .bigru import ModelOutput, nn
except ImportError:
    from absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS
    from training.models.bigru import ModelOutput, nn


class PhoBERTABSA(nn.Module):
    def __init__(self, encoder, dropout=.1):
        super().__init__()
        if not hasattr(encoder, 'config') or not hasattr(encoder.config, 'hidden_size'):
            raise ValueError('Inject a RobertaModel-compatible encoder with hidden_size config')
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.aspect_head = nn.Linear(hidden_size, len(ASPECTS))
        self.sentiment_head = nn.Linear(hidden_size, len(CORE_ASPECTS) * len(SENTIMENTS))

    def forward(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError('PhoBERT input_ids and attention_mask must align as [batch, sequence]')
        if input_ids.shape[1] > 128 or torch.any((attention_mask != 0) & (attention_mask != 1)):
            raise ValueError('Registered PhoBERT input is capped at 128 with a binary attention mask')
        lengths = attention_mask.sum(dim=1).long()
        expected_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
        if torch.any(lengths < 2) or not torch.equal(attention_mask.bool(), expected_mask):
            raise ValueError('PhoBERT attention mask must represent right padding and preserve BOS/EOS')
        bos_id = getattr(self.encoder.config, 'bos_token_id', None)
        eos_id = getattr(self.encoder.config, 'eos_token_id', None)
        if bos_id is not None and torch.any(input_ids[:, 0] != bos_id):
            raise ValueError('PhoBERT BOS must be at position zero')
        if eos_id is not None and torch.any(input_ids[torch.arange(len(input_ids), device=input_ids.device), lengths.to(input_ids.device) - 1] != eos_id):
            raise ValueError('PhoBERT EOS must be the final unmasked token')
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoded.last_hidden_state if hasattr(encoded, 'last_hidden_state') else encoded[0]
        if hidden.shape[:2] != input_ids.shape:
            raise ValueError('Encoder returned an incompatible sequence shape')
        bos = self.dropout(hidden[:, 0, :])
        return ModelOutput(self.aspect_head(bos),
                           self.sentiment_head(bos).reshape(-1, len(CORE_ASPECTS), len(SENTIMENTS)))


# `torch` comes from the same torch module imported by bigru; this adapter never resolves a model id.
try:
    import torch
except ImportError as exc:
    raise ImportError('PhoBERT adapter requires torch==2.9.0 from the registered modeling lock') from exc
