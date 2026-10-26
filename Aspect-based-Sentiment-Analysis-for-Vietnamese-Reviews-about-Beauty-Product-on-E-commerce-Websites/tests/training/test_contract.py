import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise ImportError('BLOCKED: install the registered modeling lock in Python 3.13.9 before TRAIN-01 tests') from exc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from absa_protocol import ASPECTS, CORE_ASPECTS, JOINT_LABELS, SENTIMENTS
from training.dataset import load_split
from training.targets import encode_batch
from training.decoding import decode_probabilities, select_global_threshold
from training.losses import compute_multitask_loss
from training.trainer import aggregate_candidate_results, fit_tfidf_validation


def row(text='', **labels):
    return {'data': text, **{a: labels.get(a, '') for a in ASPECTS}}


class TrainingContractTests(unittest.TestCase):
    def test_dataset_preserves_row_identity_and_literal_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.csv'
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['data', *ASPECTS])
                writer.writeheader()
                writer.writerow(row('NAN', others='neutral', shipping='positive'))
            rows = load_split(path, 'train')
        self.assertEqual(rows[0]['data'], 'NAN')
        self.assertEqual((rows[0]['source_split'], rows[0]['source_row_index']), ('train', 0))
        self.assertEqual(ASPECTS, ('stayingpower','texture','smell','price','others','colour','shipping','packing'))
        self.assertEqual(CORE_ASPECTS, ('stayingpower','texture','smell','price','colour','shipping','packing'))
        self.assertEqual(SENTIMENTS, ('positive','neutral','negative'))
        self.assertEqual(len(JOINT_LABELS), 21)

    def test_blank_is_masked_not_neutral_and_others_coexists(self):
        encoded = encode_batch([row(shipping='positive', others='neutral')])
        self.assertEqual(encoded['aspect_targets'].shape, (1, 8))
        self.assertEqual(encoded['sentiment_targets'].shape, (1, 7))
        self.assertEqual(encoded['sentiment_mask'].shape, (1, 7))
        self.assertEqual(encoded['aspect_targets'][0, ASPECTS.index('others')], 1)
        self.assertEqual(encoded['aspect_targets'][0, ASPECTS.index('shipping')], 1)
        self.assertEqual(encoded['sentiment_mask'][0, CORE_ASPECTS.index('shipping')], 1)
        self.assertEqual(encoded['sentiment_mask'][0, CORE_ASPECTS.index('price')], 0)

    def test_neural_loss_matches_bce_plus_masked_ce_and_zero_mask_is_finite(self):
        import torch
        aspect_logits = torch.zeros((2, 8), requires_grad=True)
        sentiment_logits = torch.zeros((2, 7, 3), requires_grad=True)
        aspect_targets = torch.zeros((2, 8)); aspect_targets[0, 0] = 1
        sentiment_targets = torch.zeros((2, 7), dtype=torch.long); sentiment_targets[0, 0] = 2
        sentiment_mask = torch.zeros((2, 7), dtype=torch.bool); sentiment_mask[0, 0] = True
        total, parts = compute_multitask_loss(aspect_logits, sentiment_logits, aspect_targets,
                                              sentiment_targets, sentiment_mask)
        self.assertAlmostEqual(parts['aspect'].item(), np.log(2), places=6)
        self.assertAlmostEqual(parts['sentiment'].item(), np.log(3), places=6)
        self.assertAlmostEqual(total.item(), np.log(2) + np.log(3), places=6)
        total.backward()
        self.assertTrue(torch.isfinite(aspect_logits.grad).all())
        zero, parts = compute_multitask_loss(aspect_logits, sentiment_logits, aspect_targets,
                                             sentiment_targets, torch.zeros_like(sentiment_mask))
        self.assertEqual(parts['sentiment'].item(), 0.0)
        self.assertTrue(torch.isfinite(zero))

    def test_decoder_uses_registered_order_and_inclusive_threshold(self):
        aspect = np.full(8, .1); aspect[ASPECTS.index('shipping')] = .5
        conditional = np.zeros((7, 3)); conditional[CORE_ASPECTS.index('shipping')] = [.2, .7, .1]
        result = decode_probabilities(aspect, conditional, threshold=.5,
                                      source_split='val', source_row_index=4)
        self.assertEqual(result['aspects'], ['shipping'])
        self.assertEqual(result['conditional_sentiments']['shipping'], 'neutral')
        self.assertEqual(set(result['conditional_sentiments']), set(CORE_ASPECTS))

    def test_threshold_selection_delegates_fixed21_metrics_and_registered_ties(self):
        from evaluate_predictions import align_predictions
        gold = [row(shipping='positive'), row(price='negative')]
        probabilities = np.full((2, 8), .1)
        probabilities[:, ASPECTS.index('shipping')] = [.4, .1]
        probabilities[:, ASPECTS.index('price')] = [.1, .4]
        conditional = np.zeros((2, 7, 3)); conditional[:, :, 0] = 1
        selection = select_global_threshold(gold, probabilities, conditional, 'val')
        self.assertIn(selection['threshold'], (.3, .4, .5, .6, .7))
        self.assertIn('joint_macro_f1', selection)
        self.assertIn('joint_micro_f1', selection)

    def test_candidate_selection_uses_all_registered_seeds_then_micro_then_id(self):
        rows = []
        for candidate_id, macro, micro in (('TFIDF-01', .5, .7), ('TFIDF-02', .5, .8)):
            for seed in (42, 2026, 3407):
                rows.append({'candidate_id': candidate_id, 'seed': seed,
                             'val_joint_macro_f1': macro, 'val_joint_micro_f1': micro})
        selected, summary = aggregate_candidate_results(rows, family='tfidf_logistic')
        self.assertEqual(selected, 'TFIDF-02')
        self.assertEqual(summary['TFIDF-02']['seeds'], [42, 2026, 3407])

    def test_tfidf_validation_helper_fits_train_only_and_uses_evaluator(self):
        train = []
        for i, sentiment in enumerate(SENTIMENTS):
            labels = {a: sentiment for a in CORE_ASPECTS}
            labels['others'] = 'neutral' if i == 1 else ''
            train.append(row(f'synthetic son mau {sentiment} {i}', **labels))
        train += [row(f'empty synthetic {i}') for i in range(3)]
        for i, sample in enumerate(train): sample.update(source_split='train', source_row_index=i)
        validation = [row('synthetic son mau positive', shipping='positive')]
        validation[0].update(source_split='val', source_row_index=0)
        result = fit_tfidf_validation(train, validation, 'TFIDF-01')
        self.assertEqual(result['candidate_id'], 'TFIDF-01')
        self.assertIn(result['threshold'], (.3, .4, .5, .6, .7))
        self.assertEqual(result['validation_metrics']['joint']['reviews'], 1)


if __name__ == '__main__':
    unittest.main()
