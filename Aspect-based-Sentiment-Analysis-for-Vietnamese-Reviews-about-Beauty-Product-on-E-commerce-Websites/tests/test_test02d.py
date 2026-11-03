"""Synthetic and evidence-based reporting contract checks."""
import csv,json,math,tempfile,unittest
from pathlib import Path
from src.analysis.test02d.sources import ROOT,OUT,CANDIDATES,SEEDS,SENTIMENTS,LABELS,mean_sd,matrix,read_csv,load,near
from src.analysis.test02d.__main__ import stale
from src.analysis.test02b.core import validate_seed_matrix

class TestReportingContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=load(ROOT.parent/'absa-run/validation_aggregate_assumed_equivalence.json')
 def test_seed_coverage_duplicate_and_missing(self):
  keys=[(c,s) for c in CANDIDATES for s in SEEDS];validate_seed_matrix(keys)
  for v in (keys[:-1],keys+[keys[0]]):
   with self.assertRaises(ValueError):validate_seed_matrix(v)
 def test_schema_and_duplicate_keys(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.csv';p.write_text('a\n1\n')
   with self.assertRaises(ValueError):read_csv(p,['missing'])
  with self.assertRaises(ValueError):matrix([{'a':'1'},{'a':'1'}],['a'])
 def test_mean_and_null_handling(self):
  self.assertEqual(mean_sd([1,2,3]),(2,1));self.assertEqual(mean_sd([1,None,3]),(None,None))
 def test_summary_and_seed_reconciliation(self):
  for c in CANDIDATES:
   ref=self.data['summary']['families'][c]['aggregate']['original_test']['joint']['macro_f1'];v=[float(self.data['model'][c,str(s)]['joint_macro_f1']) for s in SEEDS];mean,sd=mean_sd(v)
   self.assertTrue(near(mean,ref['mean']));self.assertTrue(near(sd,ref['sample_sd']))
 def test_validation_hash_mismatch_blocks_only_validation(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'bad.json';p.write_text('{}');data=load(p)
   self.assertEqual(data['validation']['status'],'BLOCKED');self.assertEqual(len(data['model']),9)
 def test_fixed21_and_decomposition(self):
  self.assertEqual(len(LABELS),21);self.assertEqual(LABELS[0],'stayingpower:positive');self.assertEqual(LABELS[-1],'packing:negative')
  for a,b in [('PHOBERT-02','BIGRU-02'),('PHOBERT-02','TFIDF-02'),('BIGRU-02','TFIDF-02')]:
   for seed in SEEDS:
    parts=sum(float(self.data['decomposition'][f'{a} minus {b}',str(seed),s]['fixed21_contribution']) for s in SENTIMENTS)
    gap=float(self.data['model'][a,str(seed)]['joint_macro_f1'])-float(self.data['model'][b,str(seed)]['joint_macro_f1'])
    self.assertTrue(near(parts,gap))
 def test_diagnostic_zero_support(self):
  for c in CANDIDATES:
   for s in SEEDS:
    row=self.data['diagnostics'][c,str(s),'test_placeholder_only'];self.assertEqual(sum(json.loads(row['label_support']).values()),0);self.assertEqual(float(row['joint_macro_f1']),0)
 def test_confusion_denominator_and_pooled_sentiment(self):
  for c in CANDIDATES:
   for seed in SEEDS:
    ev=[r for r in self.data['events'] if r['candidate']==c and r['seed']==str(seed)]
    self.assertEqual(sum(r['event_type']=='GOLD_PRESENT' for r in ev),2969)
    self.assertEqual(1623*7-2969,8392)
    for sentiment in SENTIMENTS:
     row=self.data['sentiment'][c,str(seed),sentiment];ref=next(r['reports']['original_test']['joint_pooled_per_sentiment'][sentiment] for r in self.data['summary']['families'][c]['per_seed'] if r['seed']==seed)
     self.assertEqual(int(row['tp']),ref['tp']);self.assertTrue(near(row['f1'],ref['f1']))
 def test_stale_detection(self):
  from src.analysis.test02d.__main__ import generator_sha
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'source';p.write_text('a')
   from src.analysis.test02d.sources import sha
   m={'figures':[{'figure_id':'FIG-01','generator_sha256':generator_sha(),'sources':[{'path':str(p),'sha256':sha(p)}],'outputs':[]}]}
   self.assertEqual(stale(m),[]);p.write_text('b');self.assertTrue(stale(m))
 def test_manifest_completeness(self):
  p=OUT/'figure_manifest.json'
  if not p.exists():self.skipTest('figures not yet generated')
  m=json.loads(p.read_text());self.assertEqual(len(m['figures']),8)
  for fig in m['figures']:
   for key in ('figure_id','title','sources','filters','candidate_order','seed_order','metric_definition','aggregation','units','outputs','generator_sha256','environment','caption','limitations','status'):self.assertIn(key,fig)

if __name__=='__main__':unittest.main()
