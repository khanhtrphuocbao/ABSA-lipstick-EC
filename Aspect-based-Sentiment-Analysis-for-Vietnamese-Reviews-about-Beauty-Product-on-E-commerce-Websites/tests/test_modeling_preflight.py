import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from modeling_preflight import ceiling_metadata,length_statistics,prepare_text
from absa_protocol import JOINT_LABELS,score_sets

class ModelingPreflightTests(unittest.TestCase):
    def test_ceiling_is_interpretation_of_existing_fixed_label_metric(self):
        supports={label:0 for label in JOINT_LABELS};supports[JOINT_LABELS[0]]=2
        m=ceiling_metadata(supports,2)
        actual=score_sets([{JOINT_LABELS[0]}]*2,[{JOINT_LABELS[0]}]*2,JOINT_LABELS)
        self.assertEqual(m['support_only_upper_bound'],actual['macro_f1'])
        self.assertEqual(m['supported_labels'],1)
        self.assertEqual(len(m['zero_support_labels']),20)
        self.assertIsNone(ceiling_metadata(supports,0)['support_only_upper_bound'])

    def test_lengths_include_specials_and_strictly_greater_than_limit(self):
        r=length_statistics([2,128,129,256,257])
        self.assertEqual(r['p50'],129)
        self.assertEqual(r['truncation']['128']['reviews'],3)
        self.assertEqual(r['truncation']['256']['reviews'],1)

    def test_model_preparation_retains_case_punctuation_emoji_negation(self):
        self.assertEqual(prepare_text('  Ma\u0300u  đẹp!!!\nKHÔ quá :( 😊 ko  '),'Màu đẹp!!! KHÔ quá :( 😊 ko')

if __name__=='__main__':unittest.main()
