"""Focused TEST-02B quantitative checks; never fit or infer models."""
import hashlib
import json
import math
import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.test02b.core import (  # noqa: E402
    decompose_contributions, sample_summary, validate_pair_counts,
    validate_seed_matrix,
)
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_predictions import evaluate_file  # noqa: E402


class Test02BQuantitativeTests(unittest.TestCase):
    def test_frozen_evaluator_matches_sealed_metrics_without_mutating_inputs(self):
        run_id = 'TFIDF-02-seed-42'
        run_dir = ROOT / 'absa-runs-final-test' / run_id / 'final_test'
        prediction = run_dir / 'test_predictions.jsonl'
        metric_path = run_dir / 'test_metrics.json'
        summary_path = ROOT / 'absa-runs-final-test' / 'final_test_summary.json'
        protocol_paths = [ROOT / 'data/data_test.csv', ROOT / 'protocols/evaluation_protocol_v1.yaml',
                          ROOT / 'protocols/freeze_lock_v1.json', ROOT / 'protocols/data_audit_manifest.csv',
                          prediction, metric_path, summary_path]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protocol_paths}
        actual = evaluate_file(prediction, 'test', ROOT)
        expected = json.loads(metric_path.read_text())
        summary = json.loads(summary_path.read_text())
        saved = next(row for row in summary['families']['TFIDF-02']['per_seed']
                     if row['seed'] == 42)
        self.assertEqual(actual, expected)
        self.assertEqual(actual['reports'], saved['reports'])
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in protocol_paths})

    def test_sample_summary_uses_ddof_one(self):
        self.assertEqual(sample_summary([1.0, 2.0, 3.0]), (2.0, 1.0))

    def test_candidate_seed_matrix_is_complete_and_unique(self):
        from src.analysis.test02b.core import CANDIDATES, SEEDS
        keys = [(candidate, seed) for candidate in CANDIDATES for seed in SEEDS]
        validate_seed_matrix(keys)
        with self.assertRaises(ValueError):
            validate_seed_matrix(keys[:-1])
        with self.assertRaises(ValueError):
            validate_seed_matrix(keys + [keys[0]])

    def test_pair_counts_and_zero_support_convention(self):
        validate_pair_counts({'support': 0, 'predicted_positive': 0, 'tp': 0,
                              'fp': 0, 'fn': 0, 'recall': None, 'precision': None})
        with self.assertRaises(ValueError):
            validate_pair_counts({'support': 0, 'predicted_positive': 0, 'tp': 0,
                                  'fp': 0, 'fn': 0, 'recall': 0, 'precision': None})
        with self.assertRaises(ValueError):
            validate_pair_counts({'support': 2, 'predicted_positive': 1, 'tp': 1,
                                  'fp': 0, 'fn': 0, 'recall': 1, 'precision': 1})

    def test_fixed21_decomposition_reconciles_by_sentiment(self):
        labels = [f'a{i}:{sent}' for i in range(7) for sent in
                  ('positive', 'neutral', 'negative')]
        left = {label: {'f1': .8} for label in labels}
        right = {label: {'f1': .5} for label in labels}
        rows = decompose_contributions(left, right, labels, 'L minus R', 42)
        self.assertEqual([row['sentiment'] for row in rows],
                         ['positive', 'neutral', 'negative'])
        self.assertTrue(all(math.isclose(row['fixed21_contribution'], .1)
                            for row in rows))
        self.assertAlmostEqual(sum(r['fixed21_contribution'] for r in rows), .3)


if __name__ == '__main__':
    unittest.main()
