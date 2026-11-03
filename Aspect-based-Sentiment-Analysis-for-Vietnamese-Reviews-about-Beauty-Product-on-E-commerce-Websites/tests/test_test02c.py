import unittest
import csv
from collections import defaultdict
from src.analysis.test02c.__main__ import events_for, feature_flags, CORE_ASPECTS, ASPECTS
from src.analysis.test02c.supplement import stats
from evaluate_predictions import align_predictions

class ErrorAnalysisTests(unittest.TestCase):
    def gold(self, **kw):
        return {'source_split':'test','source_row_index':'0', **{a:kw.get(a,'') for a in ASPECTS}}
    def pred(self, aspects, **kw):
        return {'source_split':'test','source_row_index':0,'aspects':aspects,
                'conditional_sentiments':{a:kw.get(a,'positive') for a in CORE_ASPECTS}}
    def test_polarity_and_missed_are_distinct(self):
        g=self.gold(shipping='negative')
        e=events_for(g,self.pred(['shipping'],shipping='positive'),'A',42)
        self.assertEqual(len(e),1); self.assertEqual(e[0]['outcome'],'CORRECT_ASPECT_WRONG_SENTIMENT')
        e=events_for(g,self.pred([],shipping='negative'),'A',42)
        self.assertEqual(e[0]['outcome'],'MISSED_ASPECT'); self.assertEqual(e[0]['conditional_sentiment'],'negative')
    def test_spurious_is_separate(self):
        e=events_for(self.gold(shipping='negative'),self.pred(['packing']),'A',42)
        self.assertEqual({r['outcome'] for r in e},{'MISSED_ASPECT','SPURIOUS_ASPECT'})
    def test_duplicate_or_missing_prediction_id_rejected(self):
        g=[self.gold()]; p=self.pred([])
        with self.assertRaises(ValueError): align_predictions(g,[p,p],'test')
        with self.assertRaises(ValueError): align_predictions(g,[],'test')
    def test_feature_flags_are_observations(self):
        f=feature_flags('không đẹp 😊',self.gold(price='negative',colour='positive'))
        self.assertTrue(f['negation']); self.assertTrue(f['emoji_or_emoticon']); self.assertTrue(f['mixed_gold_sentiment'])
    def test_sample_standard_deviation(self):
        self.assertEqual(stats([1,2,3]), (2,1))
        self.assertEqual(stats([1,2]), (None,None))
    def test_confusion_rows_reconcile_support_and_definitions(self):
        from src.analysis.test02c.__main__ import ROOT
        path=ROOT/'reports/experiments/test_02/confusion_summary.csv'
        groups=defaultdict(list)
        with path.open(newline='',encoding='utf8') as handle:
            for row in csv.DictReader(handle):
                if row['definition'] in ('conditional','detection_gated'):
                    groups[tuple(row[k] for k in ('candidate','seed','aspect','definition','gold_sentiment'))].append(row)
        self.assertEqual(len(groups),3*3*7*2*3)
        for key,rows in groups.items():
            support=int(rows[0]['gold_support'])
            self.assertEqual(sum(int(r['count']) for r in rows),support)
            self.assertEqual(len(rows),3 if key[3]=='conditional' else 4)
            for row in rows:
                self.assertEqual(int(row['denominator']),support)
                if support: self.assertAlmostEqual(float(row['rate']),int(row['count'])/support)
                else: self.assertEqual(row['rate'],'')
    def test_diagnostic_coverage_and_zero_support(self):
        from src.analysis.test02c.__main__ import ROOT
        expected={'original_test':1623,'test_nonoverlap_l1':1592,'test_nonoverlap_l2':1585,'test_without_placeholder':1615,'test_placeholder_only':8}
        import json
        with (ROOT/'reports/experiments/test_02/diagnostic_subset_comparison.csv').open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),45)
        for row in rows:
            self.assertEqual(int(row['reviews']),expected[row['subset']])
            self.assertEqual(len(json.loads(row['label_support'])),21)
            if row['subset']=='test_placeholder_only':
                self.assertEqual(sum(json.loads(row['label_support']).values()),0)
                self.assertEqual(float(row['joint_macro_f1']),0)

if __name__=='__main__': unittest.main()
