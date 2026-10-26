import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preflight_validation import verify_bytes, compute, ROOT, MANIFEST
from evaluate_predictions import align_predictions


class PreflightTests(unittest.TestCase):
    def test_artifact_tampering_fails_instead_of_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.csv').write_bytes(b'wrong\n')
            with self.assertRaisesRegex(ValueError, 'drifted'):
                verify_bytes(root, {'manifest.csv': b'correct\n'})
            self.assertEqual((root / 'manifest.csv').read_bytes(), b'wrong\n')

    def test_predictions_are_aligned_by_source_id_not_input_order(self):
        manifest = [{'source_split': 'test', 'source_row_index': '0'},
                    {'source_split': 'test', 'source_row_index': '1'}]
        p0 = {'source_split': 'test', 'source_row_index': 0}
        p1 = {'source_split': 'test', 'source_row_index': 1}
        _, ordered = align_predictions(manifest, [p1, p0], 'test')
        self.assertEqual(ordered, [p0, p1])
        for bad in ([p0], [p0, p0], [p0, {'source_split': 'val', 'source_row_index': 1}]):
            with self.assertRaises(ValueError):
                align_predictions(manifest, bad, 'test')

    def test_source_tampering_rejected_by_fingerprint(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'protocols').mkdir()
            (root / 'data').mkdir()
            (root / 'protocols/evaluation_protocol_v1.yaml').write_bytes(
                (ROOT / 'protocols/evaluation_protocol_v1.yaml').read_bytes())
            (root / 'data/full_data.csv').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
                compute(root)


if __name__ == '__main__':
    unittest.main()
