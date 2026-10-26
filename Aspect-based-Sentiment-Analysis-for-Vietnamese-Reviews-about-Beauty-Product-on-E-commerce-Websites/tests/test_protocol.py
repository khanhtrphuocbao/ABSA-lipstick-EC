"""Hand-calculated acceptance cases; no training or source data mutation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from absa_protocol import (ASPECTS, CORE_ASPECTS, JOINT_LABELS, SENTIMENTS,
                           audit_key, encode_targets, score_sets, evaluate_rows,
                           placeholder_candidate)


def gold(**labels):
    return {a: labels.get(a, '') for a in ASPECTS}


def prediction(aspects, **sentiments):
    return {'aspects': aspects, 'conditional_sentiments': {
        a: sentiments.get(a, 'positive') for a in CORE_ASPECTS}}


class ProtocolTests(unittest.TestCase):
    def test_fixed_absent_labels_are_not_silently_omitted(self):
        r = score_sets([{'a'}], [{'a'}], ['a', 'b'])
        self.assertEqual(r['macro_f1'], .5)
        self.assertEqual(r['observed_support_macro_f1'], 1)
        self.assertEqual(r['per_label']['b']['f1'], 0)
        self.assertIsNone(r['per_label']['b']['recall'])

    def test_gold_positive_but_no_prediction(self):
        r = score_sets([{'a'}], [set()], ['a'])
        self.assertEqual(r['per_label']['a']['fn'], 1)
        self.assertEqual(r['per_label']['a']['recall'], 0)
        self.assertIsNone(r['per_label']['a']['precision'])
        self.assertEqual(r['macro_f1'], 0)

    def test_wrong_aspect_creates_joint_fp_and_fn(self):
        r = evaluate_rows([gold(shipping='negative')],
                          [prediction(['packing'], shipping='negative', packing='negative')])
        self.assertEqual(r['joint']['per_label']['shipping:negative']['fn'], 1)
        self.assertEqual(r['joint']['per_label']['packing:negative']['fp'], 1)
        self.assertEqual(r['conditional']['per_aspect']['shipping']['macro_f1'], 1/3)

    def test_others_and_shipping_can_coexist(self):
        target = encode_targets(gold(others='neutral', shipping='positive'))
        self.assertEqual(target['aspects'], {'others', 'shipping'})
        self.assertEqual(target['joint'], {'shipping:positive'})
        r = evaluate_rows([gold(others='neutral', shipping='positive')],
                          [prediction(['others', 'shipping'])])
        self.assertEqual(r['others']['f1'], 1)
        self.assertEqual(r['joint']['macro_f1'], 1/21)

    def test_wrong_sentiment_and_no_aspect_are_different_failures(self):
        r = evaluate_rows([gold(price='negative')], [prediction(['price'])])
        self.assertEqual(r['joint']['per_label']['price:negative']['fn'], 1)
        self.assertEqual(r['joint']['per_label']['price:positive']['fp'], 1)
        self.assertEqual(r['aspect_detection']['per_label']['price']['f1'], 1)

    def test_empty_subset_is_not_a_zero_performance_claim(self):
        r = score_sets([], [], JOINT_LABELS)
        self.assertEqual(r['status'], 'empty_subset')
        self.assertIsNone(r['macro_f1'])

    def test_reject_unknown_labels_missing_outputs_and_length_mismatch(self):
        with self.assertRaises(ValueError):
            score_sets([{'other'}], [set()], ['a'])
        with self.assertRaises(ValueError):
            evaluate_rows([gold()], [])
        with self.assertRaises(ValueError):
            evaluate_rows([gold()], [{'aspects': [], 'conditional_sentiments': {}}])
        with self.assertRaises(ValueError):
            encode_targets(gold(price='Positive'))
        with self.assertRaises(ValueError):
            evaluate_rows([gold()], [prediction(['shipping', 'shipping'])])

    def test_audit_levels_preserve_different_semantics(self):
        raw = '  Ma\u0300u_đẹp!!!  KHÔ\nquá :( '
        self.assertEqual(audit_key(raw, 'L0'), raw)
        self.assertEqual(audit_key(raw, 'L1'), 'ma\u0300u_đẹp!!! khô quá :(')
        self.assertEqual(audit_key(raw, 'L2'), 'màu đẹp khô quá')
        self.assertEqual(audit_key('!!!', 'L2'), '')
        self.assertEqual(audit_key('đẹpppp ko 😊', 'L1'), 'đẹpppp ko 😊')

    def test_placeholder_is_candidate_not_missing_or_neutral(self):
        self.assertTrue(placeholder_candidate('NAN'))
        self.assertTrue(placeholder_candidate(' NaN '))
        self.assertTrue(placeholder_candidate('NULL'))
        self.assertFalse(placeholder_candidate(''))
        self.assertFalse(placeholder_candidate('son NAN đẹp'))
        self.assertEqual(encode_targets(gold())['joint'], set())
        self.assertNotIn('others:neutral', JOINT_LABELS)
        self.assertEqual(len(ASPECTS), 8)
        self.assertEqual(len(JOINT_LABELS), 21)

    def test_micro_weighted_and_observed_support_hand_calculation(self):
        r = score_sets([{'a'}, {'a'}, {'b'}], [{'a'}, {'b'}, {'b'}], ['a', 'b', 'c'])
        self.assertAlmostEqual(r['macro_f1'], 4/9)
        self.assertAlmostEqual(r['micro']['f1'], 2/3)
        self.assertAlmostEqual(r['weighted']['f1'], 2/3)
        self.assertAlmostEqual(r['observed_support_macro_f1'], 2/3)


if __name__ == '__main__':
    unittest.main()
