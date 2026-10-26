"""Factories bind model architecture to the immutable experiment registration."""
import json
from pathlib import Path

try:
    from .bigru import RecurrentABSA
    from .phobert import PhoBERTABSA
except ImportError:
    from training.models.bigru import RecurrentABSA
    from training.models.phobert import PhoBERTABSA


def _registration(root=None):
    base = Path(root) if root else Path(__file__).resolve().parents[3]
    return json.loads((base / 'protocols/experiments/experiment_registration_v1.json').read_text())


def build_recurrent(candidate_id, vocab_size, root=None):
    registration = _registration(root)['families']['bigru_conv1d']
    candidate = next((item for item in registration['candidates'] if item['id'] == candidate_id), None)
    if candidate is None:
        raise ValueError('Unknown registered recurrent candidate')
    architecture = registration['architecture']
    use_conv = candidate['id'] == 'BIGRU-02'
    return RecurrentABSA(vocab_size=vocab_size,
                         embedding_dim=architecture['embedding_dim'],
                         hidden_size=architecture['bidirectional_GRU_hidden_per_direction'],
                         num_layers=architecture['GRU_layers'], dropout=architecture['dropout'],
                         use_conv=use_conv, conv_filters=128, conv_kernel=3, padding_idx=0)


def build_phobert(encoder, candidate_id, root=None):
    registration = _registration(root)['families']['phobert']
    if candidate_id not in {item['id'] for item in registration['candidates']}:
        raise ValueError('Unknown registered PhoBERT candidate')
    expected = registration['initialization']
    if encoder.__class__.__name__ != 'RobertaModel':
        raise ValueError(f'Registration requires transformers.RobertaModel, got {encoder.__class__.__name__}')
    if expected['pooler'] != 'not used' or expected['representation'] != 'last_hidden_state[:,0,:] (BOS/CLS)':
        raise ValueError('PhoBERT registration semantics drifted')
    return PhoBERTABSA(encoder, dropout=registration['heads']['dropout'])
