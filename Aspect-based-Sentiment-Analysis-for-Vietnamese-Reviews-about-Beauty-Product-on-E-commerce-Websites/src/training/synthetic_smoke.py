"""CPU-only artificial-data smoke. This does not load official CSVs or pretrained weights."""
import argparse

try:
    from .models.bigru import RecurrentABSA
    from .models.phobert import PhoBERTABSA
    from .models.tfidf_lr import TfidfLogisticABSA
    from .reproducibility import seed_everything
    from .targets import encode_batch
    from .trainer import train_one_epoch
except ImportError:
    from training.models.bigru import RecurrentABSA
    from training.models.phobert import PhoBERTABSA
    from training.models.tfidf_lr import TfidfLogisticABSA
    from training.reproducibility import seed_everything
    from training.targets import encode_batch
    from training.trainer import train_one_epoch


def synthetic_rows():
    try:
        from ..absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS
    except ImportError:
        from absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS
    rows = []
    for i, sentiment in enumerate(SENTIMENTS):
        labels = {aspect: sentiment for aspect in CORE_ASPECTS}
        labels['others'] = 'neutral' if i == 1 else ''
        rows.append({'data': f'son dep mau {sentiment} synthetic {i}',
                     **{aspect: labels.get(aspect, '') for aspect in ASPECTS},
                     'source_split': 'train', 'source_row_index': i})
    for i in range(3):
        rows.append({'data': f'khong co aspect synthetic {i}',
                     **{aspect: '' for aspect in ASPECTS},
                     'source_split': 'train', 'source_row_index': len(rows)})
    return rows * 4


def run_smoke():
    import io
    import torch
    from torch.nn.utils.rnn import pad_sequence
    try:
        from transformers import RobertaConfig, RobertaModel
    except ImportError as exc:
        raise RuntimeError('Synthetic PhoBERT adapter smoke requires transformers==4.57.1 and torch==2.9.0') from exc
    seed_everything(42, deterministic_cuda=False)
    rows = synthetic_rows()
    fitted = TfidfLogisticABSA('TFIDF-01').fit(rows)
    tfidf_aspects, tfidf_sentiments = fitted.predict_probabilities([r['data'] for r in rows[:2]])
    if tfidf_aspects.shape != (2, 8) or tfidf_sentiments.shape != (2, 7, 3):
        raise AssertionError('TF-IDF interface smoke failed')

    results = {}
    encoded = encode_batch(rows[:2])
    ids = pad_sequence([torch.tensor([2, 4, 5]), torch.tensor([2, 6])], batch_first=True, padding_value=0)
    for use_conv in (False, True):
        model = RecurrentABSA(vocab_size=16, use_conv=use_conv)
        batch = {'inputs': {'input_ids': ids}, 'targets': {
            key: torch.as_tensor(value) for key, value in encoded.items()}}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = train_one_epoch(model, [batch], optimizer, device='cpu')
        results['bigru_conv' if use_conv else 'bigru'] = result

    config = RobertaConfig(vocab_size=32, hidden_size=24, num_hidden_layers=1,
                           num_attention_heads=4, intermediate_size=32,
                           max_position_embeddings=16, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    model = PhoBERTABSA(RobertaModel(config))
    ids_bert = torch.tensor([[0, 4, 2, 1], [0, 5, 6, 2]])
    output = model(ids_bert, ids_bert.ne(1).long())
    if tuple(output.aspect_logits.shape) != (2, 8) or tuple(output.sentiment_logits.shape) != (2, 7, 3):
        raise AssertionError('Random-config PhoBERT interface smoke failed')
    return {'status': 'PASS', 'data': 'generated synthetic rows only', 'official_csvs_read': False,
            'pretrained_weights_loaded': False, 'gpu_used': False,
            'tfidf_probability_shapes': [list(tfidf_aspects.shape), list(tfidf_sentiments.shape)],
            'neural_cpu_optimizer_steps': results, 'random_phobert_shapes':
            [list(output.aspect_logits.shape), list(output.sentiment_logits.shape)],
            'actual_dataset_training': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-only', action='store_true', default=True)
    args = parser.parse_args()
    result = run_smoke()
    import json
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
