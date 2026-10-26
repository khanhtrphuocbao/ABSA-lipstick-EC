import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError as exc:
    raise ImportError('BLOCKED: install protocols/experiments/modeling_requirements_v1.lock in the registered modeling runtime') from exc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS
from training.models.bigru import RecurrentABSA
from training.models.phobert import PhoBERTABSA
from training.models.tfidf_lr import TfidfLogisticABSA
from training.checkpoint import (save_checkpoint, load_checkpoint, serialize_torch_state,
                                 restore_torch_state, serialize_estimator, restore_estimator,
                                 training_code_sha256)
from training.prediction_export import write_predictions
from training.reproducibility import seed_everything
from training.vocabulary import Vocabulary
from training.preprocessing import encode_phobert_batch
from training.models.factory import build_recurrent, build_phobert
from training.losses import compute_multitask_loss
from training.trainer import (make_recurrent_loader, make_phobert_loader,
                              build_neural_optimizer, fit_neural_candidate)


def tiny_rows():
    rows = []
    for i in range(3):
        labels = {a: SENTIMENTS[(i + j) % 3] for j, a in enumerate(CORE_ASPECTS)}
        labels['others'] = 'neutral' if i == 1 else ''
        rows.append({'data': f'son dep mau synthetic review mau {i}', **{a: labels.get(a, '') for a in ASPECTS}})
    rows.extend([{'data': f'khong co aspect synthetic {i}', **{a: '' for a in ASPECTS}}
                 for i in range(3)])
    rows *= 4
    for index, row in enumerate(rows):
        row.update(source_split='train', source_row_index=index)
    return rows


class ModelTests(unittest.TestCase):
    def test_vocabulary_is_train_only_deterministic_and_capped_including_specials(self):
        vocab = Vocabulary(max_size=5).fit([['b', 'a', 'b', 'c']], source_split='train')
        self.assertEqual(vocab.token_to_id, {'<PAD>': 0, '<UNK>': 1, 'b': 2, 'a': 3, 'c': 4})
        self.assertEqual(vocab.encode(['missing', 'b'], max_length=128), [1, 2])
        with self.assertRaisesRegex(ValueError, 'train only'):
            Vocabulary().fit([['leak']], source_split='val')

    def test_recurrent_loader_keeps_ids_out_of_features_and_only_shuffles_train(self):
        train = [{'data': 'secret row id is not text', **{a: '' for a in ASPECTS},
                  'source_split': 'train', 'source_row_index': 999}]
        vocab = Vocabulary().fit([['hello']], source_split='train')
        loader = make_recurrent_loader(train, [['hello']], vocab, batch_size=32, seed=42, shuffle=True)
        batch = next(iter(loader))
        self.assertEqual(set(batch['inputs']), {'input_ids'})
        self.assertEqual(batch['inputs']['input_ids'].tolist(), [[2]])
        validation = [{**train[0], 'source_split': 'val', 'source_row_index': 7}]
        with self.assertRaisesRegex(ValueError, 'Shuffle is enabled'):
            make_recurrent_loader(validation, [['hello']], vocab, batch_size=32, seed=42, shuffle=True)
        valid_loader = make_recurrent_loader(validation, [['hello']], vocab, batch_size=32, seed=42, shuffle=False)
        self.assertEqual(next(iter(valid_loader))['inputs']['input_ids'].tolist(), [[2]])

    def test_phobert_loader_dynamic_padding_keeps_row_ids_out_of_features(self):
        class TinyTokenizer:
            bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
            truncation_side = 'right'

            def __call__(self, texts, **kwargs):
                rows = [[0, 4, 2] if len(text) < 2 else [0, 4, 5, 2] for text in texts]
                width = max(map(len, rows))
                ids = [row + [1] * (width - len(row)) for row in rows]
                mask = [[int(value != 1) for value in row] for row in ids]
                return {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(mask)}

        train = [{'data': 'first', **{a: '' for a in ASPECTS}, 'source_split': 'train', 'source_row_index': 91},
                 {'data': 'second', **{a: '' for a in ASPECTS}, 'source_split': 'train', 'source_row_index': 92}]
        tokenizer = TinyTokenizer()
        loader = make_phobert_loader(train, ['a', 'bb'], tokenizer, batch_size=16, seed=42, shuffle=True)
        batch = next(iter(loader))
        self.assertEqual(set(batch['inputs']), {'input_ids', 'attention_mask'})
        self.assertEqual(batch['inputs']['input_ids'].shape, (2, 4))
        validation = [{**train[0], 'source_split': 'val'}]
        valid_loader = make_phobert_loader(validation, ['a'], tokenizer, batch_size=16, seed=42, shuffle=False)
        self.assertEqual(next(iter(valid_loader))['gold_rows'][0]['source_row_index'], 91)

    def test_phobert_batch_preserves_eos_after_right_truncation(self):
        class TinyTokenizer:
            bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
            truncation_side = 'right'

            def __call__(self, texts, **kwargs):
                limit = kwargs['max_length']
                rows = []
                for _text in texts:
                    ids = [0] + [4, 5] * 70 + [2]
                    ids = ids[:limit - 1] + [2]
                    rows.append(ids)
                return {'input_ids': torch.tensor(rows), 'attention_mask': torch.ones(len(rows), len(rows[0]), dtype=torch.long)}

        tokenizer = TinyTokenizer()
        batch = encode_phobert_batch(['xin chao xin chao xin chao'], tokenizer, already_segmented=True)
        self.assertEqual(batch['input_ids'].shape[1], 128)
        self.assertEqual(batch['input_ids'][0, -1].item(), 2)
        self.assertEqual(batch['attention_mask'][0, -1].item(), 1)

    def test_bigru_and_conv_outputs_are_padding_invariant_and_finite(self):
        torch.manual_seed(4)
        for use_conv in (False, True):
            model = RecurrentABSA(vocab_size=12, use_conv=use_conv).eval()
            short = torch.tensor([[2, 3, 4, 0, 0]])
            long = torch.tensor([[2, 3, 4, 5, 6]])
            batch = model(torch.cat([short, long], dim=0))
            isolated = model(short[:, :3])
            self.assertEqual(tuple(batch.aspect_logits.shape), (2, 8))
            self.assertEqual(tuple(batch.sentiment_logits.shape), (2, 7, 3))
            self.assertTrue(torch.allclose(batch.aspect_logits[0], isolated.aspect_logits[0], atol=1e-6))
            self.assertTrue(torch.allclose(batch.sentiment_logits[0], isolated.sentiment_logits[0], atol=1e-6))
            self.assertTrue(torch.isfinite(batch.aspect_logits).all())
            self.assertTrue(torch.isfinite(batch.sentiment_logits).all())

    def test_registered_recurrent_factories_keep_architecture_and_backward(self):
        from training.targets import encode_batch
        for candidate, expected_conv in (('BIGRU-01', False), ('BIGRU-02', True)):
            model = build_recurrent(candidate, vocab_size=20)
            self.assertEqual(model.embedding.embedding_dim, 128)
            self.assertEqual(model.gru.hidden_size, 128)
            self.assertEqual(model.gru.num_layers, 1)
            self.assertEqual(model.use_conv, expected_conv)
            inputs = torch.tensor([[2, 3, 0], [2, 4, 5]])
            target_rows = [{'data': 'a', **{a: '' for a in ASPECTS}} for _ in range(2)]
            encoded = encode_batch(target_rows)
            output = model(inputs)
            loss, _ = compute_multitask_loss(output.aspect_logits, output.sentiment_logits,
                                             torch.as_tensor(encoded['aspect_targets']),
                                             torch.as_tensor(encoded['sentiment_targets']),
                                             torch.as_tensor(encoded['sentiment_mask']))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(model.embedding.weight.grad)

    def test_registered_optimizer_defaults_and_gpu_launch_boundary(self):
        bigru = build_recurrent('BIGRU-01', vocab_size=12)
        optimizer, scheduler, _ = build_neural_optimizer(bigru, 'BIGRU-01', total_optimizer_steps=4)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]['lr'], .001)
        self.assertIsNone(scheduler)
        with self.assertRaisesRegex(RuntimeError, 'requires the authorized CUDA launch'):
            fit_neural_candidate(None, [], [], None, None, 'BIGRU-01', 42,
                                 tempfile.gettempdir(), {}, device='cpu')

    def test_phobert_optimizer_uses_registered_lr_decay_groups_and_scheduler(self):
        from transformers import RobertaConfig, RobertaModel
        config = RobertaConfig(vocab_size=32, hidden_size=24, num_hidden_layers=1,
                               num_attention_heads=4, intermediate_size=32,
                               max_position_embeddings=16, pad_token_id=1)
        model = build_phobert(RobertaModel(config), 'PHOBERT-02')
        optimizer, scheduler, _ = build_neural_optimizer(model, 'PHOBERT-02', total_optimizer_steps=10)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual([group['lr'] for group in optimizer.param_groups], [2e-5, 2e-5])
        self.assertEqual([group['weight_decay'] for group in optimizer.param_groups], [.01, 0.0])
        self.assertIsNotNone(scheduler)

    def test_phobert_adapter_uses_bos_and_attention_mask_with_random_config(self):
        from transformers import RobertaConfig, RobertaModel
        torch.manual_seed(7)
        config = RobertaConfig(vocab_size=64, hidden_size=24, num_hidden_layers=1,
                               num_attention_heads=4, intermediate_size=32,
                               max_position_embeddings=32, pad_token_id=1, bos_token_id=0,
                               eos_token_id=2)
        model = PhoBERTABSA(RobertaModel(config), dropout=.1).eval()
        ids = torch.tensor([[0, 5, 2, 1, 1], [0, 8, 9, 2, 1]])
        mask = ids.ne(config.pad_token_id).long()
        output = model(ids, mask)
        isolated = model(ids[:1, :3], mask[:1, :3])
        self.assertEqual(tuple(output.aspect_logits.shape), (2, 8))
        self.assertEqual(tuple(output.sentiment_logits.shape), (2, 7, 3))
        self.assertTrue(torch.isfinite(output.aspect_logits).all())
        self.assertTrue(torch.isfinite(output.sentiment_logits).all())
        self.assertTrue(torch.allclose(output.aspect_logits[0], isolated.aspect_logits[0], atol=1e-6))
        self.assertTrue(torch.allclose(output.sentiment_logits[0], isolated.sentiment_logits[0], atol=1e-6))
        self.assertEqual(build_phobert(RobertaModel(config), 'PHOBERT-01').dropout.p, .1)
        with self.assertRaisesRegex(ValueError, 'right padding'):
            model(ids, torch.tensor([[1, 0, 1, 0, 0], [1, 1, 1, 1, 0]]))

    def test_tfidf_probability_columns_are_reordered_to_frozen_sentiment_order(self):
        class Estimator:
            classes_ = np.asarray(['negative', 'neutral', 'positive'])

            @staticmethod
            def predict_proba(_matrix):
                return np.asarray([[.7, .2, .1]])

        model = TfidfLogisticABSA('TFIDF-01')
        aligned = model._aligned_probabilities(Estimator(), type('Matrix', (), {'shape': (1, 4)})(),
                                               SENTIMENTS, 'synthetic')
        self.assertTrue(np.allclose(aligned, [[.1, .2, .7]]))

    def test_tfidf_candidate_has_aligned_binary_and_conditional_probabilities(self):
        for candidate in ('TFIDF-01', 'TFIDF-02'):
            model = TfidfLogisticABSA(candidate).fit(tiny_rows())
            aspect, conditional = model.predict_probabilities(['son dep mau synthetic review'])
            self.assertEqual(aspect.shape, (1, 8))
            self.assertEqual(conditional.shape, (1, 7, 3))
            self.assertTrue(np.isfinite(aspect).all())
            self.assertTrue(np.isfinite(conditional).all())
            self.assertTrue(np.allclose(conditional.sum(axis=2), 1.0))
            self.assertEqual(model.sentiment_class_order, SENTIMENTS)
            self.assertEqual(model.feature_count_, model.aspect_heads[0].n_features_in_)

    def test_tfidf_rejects_missing_registered_sentiment_class(self):
        rows = tiny_rows()
        rows = [r for r in rows if r['shipping'] != 'negative']
        with self.assertRaisesRegex(ValueError, 'missing registered sentiments'):
            TfidfLogisticABSA('TFIDF-01').fit(rows)

    def test_checkpoint_round_trip_checks_hash_and_prevents_overwrite(self):
        model = torch.nn.Linear(3, 2)
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.json'
            metadata = {'candidate_id': 'BIGRU-01', 'seed': 42, 'epoch': 1, 'threshold': .5,
                        'validation_metrics': {'joint_macro_f1': None}, 'preprocessing': {'synthetic': True},
                        'dataset_version': 'synthetic', 'protocol_id': 'absa_data_evaluation_v1',
                        'protocol_sha256': hashlib.sha256((ROOT / 'protocols/evaluation_protocol_v1.yaml').read_bytes()).hexdigest(),
                        'registration_id': 'absa_model_comparison_registration_v1',
                        'registration_sha256': hashlib.sha256((ROOT / 'protocols/experiments/experiment_registration_v1.json').read_bytes()).hexdigest(),
                        'code_sha256': training_code_sha256(), 'model_family': 'bigru'}
            checkpoint_hash = save_checkpoint(path, serialize_torch_state(state), metadata)
            restored_bytes, metadata = load_checkpoint(path, expected_sha256=checkpoint_hash)
            restored = restore_torch_state(restored_bytes)
            model2 = torch.nn.Linear(3, 2)
            model2.load_state_dict(restored)
            self.assertTrue(torch.equal(model(torch.ones(1, 3)), model2(torch.ones(1, 3))))
            self.assertEqual(metadata['seed'], 42)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, serialize_torch_state(state), metadata)
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_checkpoint(path, expected_sha256=checkpoint_hash)
            with self.assertRaisesRegex(ValueError, 'expected checkpoint SHA-256'):
                load_checkpoint(path)

    def test_estimator_serialization_round_trip(self):
        from sklearn.dummy import DummyClassifier
        estimator = DummyClassifier(strategy='constant', constant=1).fit([[0], [1]], [1, 1])
        restored = restore_estimator(serialize_estimator(estimator))
        self.assertEqual(restored.classes_.tolist(), [1])
        self.assertEqual(restored.predict([[0]]).tolist(), [1])

    def test_synthetic_seed_repeats_predictions_on_cpu(self):
        seed_everything(2026, deterministic_cuda=False)
        first = RecurrentABSA(vocab_size=10).eval()(torch.tensor([[2, 3, 4]])).aspect_logits.detach()
        seed_everything(2026, deterministic_cuda=False)
        second = RecurrentABSA(vocab_size=10).eval()(torch.tensor([[2, 3, 4]])).aspect_logits.detach()
        self.assertTrue(torch.equal(first, second))

    def test_prediction_export_is_evaluator_compatible(self):
        rows = [{'source_split': 'val', 'source_row_index': 0,
                 **{a: ('positive' if a == 'shipping' else '') for a in ASPECTS}}]
        preds = [{'source_split': 'val', 'source_row_index': 0, 'aspects': ['shipping'],
                  'conditional_sentiments': {a: ('positive' if a == 'shipping' else 'neutral') for a in CORE_ASPECTS}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'val.jsonl'
            write_predictions(path, rows, preds, 'val')
            parsed = json.loads(path.read_text().strip())
        self.assertEqual(parsed, preds[0])


if __name__ == '__main__':
    unittest.main()
