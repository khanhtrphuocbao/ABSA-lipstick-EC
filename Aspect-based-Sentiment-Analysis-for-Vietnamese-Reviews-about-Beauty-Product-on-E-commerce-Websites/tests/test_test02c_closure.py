"""Integration checks for the review and immutable evidence."""
import csv, hashlib, json, subprocess, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/experiments/test_02'
class ClosureTests(unittest.TestCase):
    def test_review_coverage_and_reproducibility(self):
        with (OUT/'representative_errors_reviewed.csv').open(newline='',encoding='utf8') as f: rows=list(csv.DictReader(f))
        self.assertEqual(len({(r['category'],r['sample_rank']) for r in rows}),50)
        self.assertTrue(all(r['privacy_status']=='REVIEWED_NO_VISIBLE_IDENTIFIER' and r['review_excerpt'] and r['interpretation'] for r in rows))
        names=['representative_errors_reviewed.csv','linguistic_manual_review.csv','test_02c_closure_report.md','test_02c_closure_integrity.json']
        def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
        before={n:sha(OUT/n) for n in names}
        proc=subprocess.run([sys.executable,'-B','-m','src.analysis.test02c.closure'],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(before,{n:sha(OUT/n) for n in names})
        old=json.loads((OUT/'test_02c_integrity.json').read_text())
        for name,digest in old['input_sha256'].items(): self.assertEqual(sha(ROOT/name),digest,name)
if __name__=='__main__': unittest.main()
