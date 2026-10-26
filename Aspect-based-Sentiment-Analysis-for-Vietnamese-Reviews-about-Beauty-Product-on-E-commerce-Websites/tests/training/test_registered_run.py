import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training.assets import resolve_java_runtime
from src.training.restore_runtime_assets import restore_asset_items
from src.training.registered_run import (
    aggregate_completed_runs,
    official_fit_paths,
    require_run_authorization,
    validate_startup_environment,
)
from src.training.models.tfidf_lr import TfidfLogisticABSA


ROOT = Path(__file__).resolve().parents[2]


class RegisteredRunTests(unittest.TestCase):
    def test_fit_paths_exclude_full_and_test_csvs(self):
        paths = official_fit_paths(ROOT / 'data')
        self.assertEqual(set(paths), {'train', 'val'})
        self.assertEqual(paths['train'].name, 'data_train.csv')
        self.assertEqual(paths['val'].name, 'data_val.csv')
        self.assertNotIn('test', paths)
        self.assertNotIn('full', paths)

    def test_tfidf_heads_receive_the_registered_run_seed(self):
        model = TfidfLogisticABSA('TFIDF-01', random_state=2026)
        self.assertEqual(model._make_logistic().random_state, 2026)

    def test_colab_java_runtime_is_accepted_only_at_registered_release(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            binary = home / 'bin/java'
            binary.parent.mkdir()
            binary.write_text('#!/bin/sh\necho \'openjdk version "17.0.21"\' >&2\necho \'Temurin-17.0.21+9\' >&2\n',
                              encoding='utf-8')
            binary.chmod(0o755)
            with patch.dict(os.environ, {'JAVA_HOME': str(home)}), patch('src.training.assets.shutil.which', return_value=None):
                report = resolve_java_runtime(home / 'assets', ROOT)
            self.assertEqual(report['release'], 'jdk-17.0.21+9')
            self.assertEqual(report['path'], str(binary.resolve()))

    def test_colab_java_runtime_rejects_a_different_release(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            binary = home / 'bin/java'
            binary.parent.mkdir()
            binary.write_text('#!/bin/sh\necho \'openjdk version "21.0.1"\' >&2\n', encoding='utf-8')
            binary.chmod(0o755)
            with patch.dict(os.environ, {'JAVA_HOME': str(home)}), patch('src.training.assets.shutil.which', return_value=None):
                with self.assertRaisesRegex(ValueError, r'17.0.21\+9'):
                    resolve_java_runtime(home / 'assets', ROOT)

    def test_runtime_asset_restore_is_hash_checked_and_write_once(self):
        import hashlib
        payload = b'synthetic pinned asset bytes'
        item = {'relative_path': 'synthetic/model.bin', 'url': 'https://example.invalid/model.bin',
                'sha256': hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            written = restore_asset_items(directory, [item], fetch_bytes=lambda _: payload)
            self.assertEqual(Path(written[0]).read_bytes(), payload)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                restore_asset_items(directory, [dict(item, sha256='0' * 64)], fetch_bytes=lambda _: b'new bytes')

    def test_registered_run_requires_a_manifest_bound_authorization(self):
        manifest = {'run_id': 'synthetic-42', 'candidate_id': 'TFIDF-01', 'seed': 42}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'authorization'):
                require_run_authorization(path, 'synthetic-42')
            (Path(directory) / 'run_authorization.json').write_text(
                json.dumps({'run_id': 'synthetic-42', 'manifest_sha256': 'bad'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '(?i)manifest'):
                require_run_authorization(path, 'synthetic-42')

    def test_startup_requires_seed_and_cublas_environment_before_cuda(self):
        with self.assertRaisesRegex(RuntimeError, 'PYTHONHASHSEED=42'):
            validate_startup_environment(42, {})
        with self.assertRaisesRegex(RuntimeError, 'CUBLAS_WORKSPACE_CONFIG'):
            validate_startup_environment(42, {'PYTHONHASHSEED': '42'})
        validate_startup_environment(42, {'PYTHONHASHSEED': '42', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})

    def test_family_aggregation_requires_every_registered_candidate_seed(self):
        partial = [{'status': 'COMPLETED', 'candidate_id': 'TFIDF-01', 'seed': 42,
                    'val_joint_macro_f1': .4, 'val_joint_micro_f1': .5}]
        with self.assertRaisesRegex(ValueError, 'all registered seeds'):
            aggregate_completed_runs('tfidf_logistic', partial)

    def test_family_aggregation_uses_registered_three_seed_tie_break(self):
        results = []
        for candidate, macro, micro in (('TFIDF-01', .6, .7), ('TFIDF-02', .6, .7)):
            for seed in (42, 2026, 3407):
                results.append({'status': 'COMPLETED', 'candidate_id': candidate, 'seed': seed,
                                'val_joint_macro_f1': macro, 'val_joint_micro_f1': micro})
        selected, summary = aggregate_completed_runs('tfidf_logistic', results)
        self.assertEqual(selected, 'TFIDF-01')
        self.assertEqual(summary['TFIDF-01']['seeds'], [42, 2026, 3407])


if __name__ == '__main__':
    unittest.main()
