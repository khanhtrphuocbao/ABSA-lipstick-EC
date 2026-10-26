"""Registered random-embedding BiGRU and parallel Conv1D variants."""
from typing import NamedTuple

try:
    from ...absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS
except ImportError:
    from absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ImportError as exc:  # produce an actionable optional-dependency diagnostic
    raise ImportError('Neural ABSA adapters require torch==2.9.0 from the registered modeling lock') from exc


class ModelOutput(NamedTuple):
    aspect_logits: torch.Tensor
    sentiment_logits: torch.Tensor


class RecurrentABSA(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, num_layers=1,
                 dropout=.3, use_conv=False, conv_filters=128, conv_kernel=3, padding_idx=0):
        super().__init__()
        if vocab_size <= 2 or padding_idx != 0:
            raise ValueError('Vocabulary must include PAD=0 and UNK=1')
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.use_conv = bool(use_conv)
        if self.use_conv:
            if conv_kernel != 3:
                raise ValueError('Registered Conv1D kernel is exactly 3')
            self.conv = nn.Conv1d(embedding_dim, conv_filters, kernel_size=conv_kernel,
                                  stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        representation = hidden_size * 4 + (conv_filters if self.use_conv else 0)
        self.aspect_head = nn.Linear(representation, len(ASPECTS))
        self.sentiment_head = nn.Linear(representation, len(CORE_ASPECTS) * len(SENTIMENTS))

    def forward(self, input_ids, lengths=None):
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError('input_ids must be a nonempty [batch, sequence] tensor')
        if lengths is None:
            lengths = input_ids.ne(self.embedding.padding_idx).sum(dim=1)
        lengths = torch.as_tensor(lengths, device='cpu', dtype=torch.long)
        if lengths.shape != (input_ids.shape[0],) or torch.any(lengths <= 0) or torch.any(lengths > input_ids.shape[1]):
            raise ValueError('Invalid non-padding sequence lengths')
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_states, _ = self.gru(packed)
        states, _ = pad_packed_sequence(packed_states, batch_first=True,
                                        total_length=input_ids.shape[1])
        mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < lengths.to(input_ids.device)[:, None]
        expanded = mask.unsqueeze(-1)
        maximum = states.masked_fill(~expanded, torch.finfo(states.dtype).min).amax(dim=1)
        mean = (states * expanded).sum(dim=1) / lengths.to(states.device, states.dtype).unsqueeze(1)
        features = [maximum, mean]
        if self.use_conv:
            convolution = torch.relu(self.conv(embedded.transpose(1, 2)))
            valid = mask.unsqueeze(1)
            convolution = convolution.masked_fill(~valid, torch.finfo(convolution.dtype).min)
            features.append(convolution.amax(dim=2))
        representation = self.dropout(torch.cat(features, dim=1))
        return ModelOutput(self.aspect_head(representation),
                           self.sentiment_head(representation).reshape(-1, len(CORE_ASPECTS), len(SENTIMENTS)))
