import json
import sys
import unittest
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from portable_verification import verify_portable


class PortableVerificationTests(unittest.TestCase):
    def test_portable_hash_check_does_not_claim_exact_runtime_reproduction(self):
        result = verify_portable(ROOT, expected_commit=None, check_git=False)
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['exact_runtime_reproduced'])
        self.assertEqual(result['gate_runtime']['python'], '3.15.1')
        self.assertEqual(result['modeling_runtime']['python'], '3.13.9')

    def test_colab_notebook_is_nine_safe_ordered_cells_without_fit(self):
        notebook_path = ROOT / 'notebooks/colab_training.ipynb'
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        cells = notebook['cells']
        headings = [cell['metadata'].get('purpose') for cell in cells]
        self.assertEqual(headings, [f'CELL {i}' for i in range(1, 10)])
        source = '\n'.join(''.join(cell['source']) for cell in cells)
        for index, cell in enumerate(cells, start=1):
            ast.parse(''.join(cell['source']), filename=f'CELL {index}')
        self.assertIn('COMMIT_SHA', source)
        self.assertIn('synthetic_smoke', source)
        self.assertIn('portable_verification', source)
        self.assertNotIn('train_registered_experiments(', source)
        self.assertNotIn('from_pretrained(', source)
        self.assertIn('GOOGLE_DRIVE', source)


if __name__ == '__main__':
    unittest.main()
