"""Verify manuscript traceability and frozen-source consistency."""
import json,re,statistics,unittest
from src.analysis.test02d.sources import ROOT,OUT,SEEDS,CANDIDATES,load,near
from src.analysis.test02d_reporting import documents,VAL,snapshot
class ManuscriptTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.before=snapshot();cls.data=load(VAL);cls.outputs=documents(cls.data)
 def test_outputs_deterministic_and_sources_unchanged(self):
  self.assertEqual(self.outputs,documents(self.data));self.assertEqual(self.before,snapshot())
  for name,raw in self.outputs.items():self.assertEqual((OUT/name).read_bytes(),raw)
 def test_claim_aggregation_and_complete_seeds(self):
  claims=json.loads(self.outputs['test_02d_claims.json'])
  self.assertGreater(len(claims),80)
  for claim in claims.values():
   self.assertEqual(claim['seeds'],list(SEEDS));self.assertEqual(len(claim['values']),3)
   self.assertTrue(near(claim['mean'],statistics.mean(claim['values'])*claim['scale']))
   self.assertTrue(near(claim['sample_sd'],statistics.stdev(claim['values'])*claim['scale']))
   self.assertTrue((OUT/claim['source']).is_file())
 def test_secondary_metrics_match_sealed_reports(self):
  for c in CANDIDATES:
   for seed in SEEDS:
    report=next(r['reports']['original_test'] for r in self.data['summary']['families'][c]['per_seed'] if r['seed']==seed)
    row=self.data['model'][c,str(seed)]
    self.assertTrue(near(row['joint_weighted_f1'],report['joint']['weighted']['f1']))
    self.assertTrue(near(row['observed_support_macro_f1'],report['joint']['observed_support_macro_f1']))
 def test_figure_and_local_source_links(self):
  text=self.outputs['results_and_discussion.md'].decode()
  for i in range(1,9):self.assertIn(f'**FIG-{i:02d}.**',text)
  for name in ('results_and_discussion.md','threats_to_validity.md'):
   for link in re.findall(r'\]\(([^)]+)\)',self.outputs[name].decode()):
    self.assertTrue((OUT/link).exists(),link)
 def test_privacy_and_qualitative_reference(self):
  import csv
  with (OUT/'representative_errors_reviewed.csv').open(encoding='utf8',newline='') as handle:
   examples=list(csv.DictReader(handle))
  for line in self.outputs['results_and_discussion.md'].decode().splitlines():
   if line.startswith('| test:'):
    cells=[x.strip() for x in line.split('|')[1:-1]]
    idx=int(cells[0].split(':')[1])
    match=[r for r in examples if int(r['source_row_index'])==idx and r['aspect']==cells[1] and r['candidate']==cells[3]]
    self.assertTrue(match)
    self.assertEqual(cells[2],match[0]['review_excerpt'])
    self.assertEqual(match[0]['privacy_status'],'REVIEWED_NO_VISIBLE_IDENTIFIER')
    self.assertEqual(cells[4:],[match[0]['gold_sentiment'] or 'absent',match[0]['conditional_sentiment'],match[0]['outcome']])
if __name__=='__main__':unittest.main()
