"""Train-fitted word vocabulary for the registered recurrent baselines."""
from collections import Counter


class Vocabulary:
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, max_size=30000, min_frequency=1):
        if max_size < 3 or min_frequency < 1:
            raise ValueError('Vocabulary size/frequency is invalid')
        self.max_size, self.min_frequency = max_size, min_frequency
        self.token_to_id = {self.PAD: 0, self.UNK: 1}

    def fit(self, token_sequences, source_split=None):
        if source_split != 'train':
            raise ValueError('Vocabulary may be fit on official train only')
        counts = Counter(token for sequence in token_sequences for token in sequence)
        ordered = sorted((token for token, count in counts.items() if count >= self.min_frequency and
                          token not in (self.PAD, self.UNK)), key=lambda token: (-counts[token], token))
        capacity = self.max_size - 2
        self.token_to_id = {self.PAD: 0, self.UNK: 1}
        self.token_to_id.update({token: index + 2 for index, token in enumerate(ordered[:capacity])})
        self.fitted = True
        return self

    def encode(self, tokens, max_length=128):
        if not getattr(self, 'fitted', False):
            raise ValueError('Vocabulary has not been fit')
        if max_length != 128:
            raise ValueError('Registered recurrent maximum length is 128')
        values = [self.token_to_id.get(token, 1) for token in tokens[:max_length]]
        return values or [1]
